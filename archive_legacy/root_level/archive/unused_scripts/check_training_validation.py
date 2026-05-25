"""Training & Validation Integrity Checker.

Scans every run folder under RUN_ROOT and verifies that:
  1. config.yaml exists
  2. logs/train.log exists
  3. validation/val_metrics.csv exists
  4. checkpoints/best.pth exists
  5. validation happened at every expected interval (from config)
  6. no validation rows show all-background collapse
  7. best_metrics.json F1 matches the CSV maximum (if file exists)

Outputs
-------
  results/training_validation_checks/check_summary.csv
  results/training_validation_checks/check_summary.md
  results/training_validation_checks/check_details.json

Status codes
------------
  PASS  — all required artefacts present, all intervals found, no collapse
  WARN  — optional files missing / intervals incomplete (may be mid-run) / collapse
  FAIL  — config, validation CSV, or checkpoint missing; or validation never ran

Usage
-----
    cd /storage2/ChangeDetection/MV/MambaRefine-CD
    python scripts/check_training_validation.py

Change RUN_ROOT below to target a different output subtree.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
RUN_ROOT = "outputs"
# RUN_ROOT = "outputs/benchmark_runs"
# RUN_ROOT = "outputs/rf_stability_phase"
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[1]
_RUN_ROOT   = REPO_ROOT / RUN_ROOT
RESULTS_DIR = REPO_ROOT / "results" / "training_validation_checks"

# Required columns in val_metrics.csv
REQUIRED_COLS  = {"iteration", "f1", "precision", "recall", "oa"}
OPTIONAL_COLS  = {"iou", "miou", "boundary_f1", "edge_iou", "pred_positive_ratio", "gt_positive_ratio"}
# A run is "possibly collapsed" at a row if any of these conditions hold
COLLAPSE_THRESHOLD_PPR = 0.001   # pred_positive_ratio below this → suspicious


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_load_yaml(path: Path) -> dict | None:
    """Load a YAML file without requiring PyYAML to be on the path."""
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    # Minimal regex-based fallback for simple key: value pairs
    result: dict[str, Any] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s{0,4}(\w+):\s*(.+)$", line)
            if m:
                k, v = m.group(1).strip(), m.group(2).strip()
                try:
                    result[k] = int(v)
                except ValueError:
                    try:
                        result[k] = float(v)
                    except ValueError:
                        result[k] = v
    except OSError:
        pass
    return result


def _read_config(run_dir: Path) -> dict:
    """Read the run-level config.yaml; return {} if not present."""
    p = run_dir / "config.yaml"
    if not p.exists():
        return {}
    return _try_load_yaml(p) or {}


def _get_training_params(cfg: dict) -> tuple[int, int]:
    """Extract (max_iterations, validate_every) from config with sensible defaults."""
    tc = cfg.get("training", cfg)   # top-level fallback for flat configs
    max_iter   = int(tc.get("max_iterations", 50000))
    val_every  = int(tc.get("validate_every",  5000))
    return max_iter, val_every


def _read_val_csv(path: Path) -> list[dict]:
    """Read val_metrics.csv; return list of row dicts (empty list on error)."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows
    except OSError:
        return []


def _read_best_metrics_json(run_dir: Path) -> dict | None:
    p = run_dir / "validation" / "best_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Per-run checker ───────────────────────────────────────────────────────────

def check_run(run_dir: Path) -> dict:
    """Analyse one run directory and return a structured result dict."""
    result: dict[str, Any] = {
        "run":             run_dir.name,
        "run_path":        str(run_dir),
        "status":          "PASS",
        "notes":           [],
        "warnings":        [],
        "failures":        [],
        "missing_iters":   [],
        "extra_iters":     [],
        "collapse_iters":  [],
        "best_f1":         None,
        "best_iter":       None,
        "max_iter":        None,
        "val_every":       None,
        "recorded_iters":  [],
        "expected_iters":  [],
        "has_config":      False,
        "has_train_log":   False,
        "has_val_csv":     False,
        "has_checkpoint":  False,
        "optional_cols_missing": [],
    }

    # ── 1. config.yaml ────────────────────────────────────────────────────────
    cfg = _read_config(run_dir)
    if not (run_dir / "config.yaml").exists():
        result["failures"].append("config.yaml missing")
    else:
        result["has_config"] = True

    # ── 2. logs/train.log ─────────────────────────────────────────────────────
    log_path = run_dir / "logs" / "train.log"
    if not log_path.exists():
        result["failures"].append("logs/train.log missing")
    else:
        result["has_train_log"] = True

    # ── 3. validation/val_metrics.csv ─────────────────────────────────────────
    val_csv = run_dir / "validation" / "val_metrics.csv"
    if not val_csv.exists():
        result["failures"].append("validation/val_metrics.csv missing")
    else:
        result["has_val_csv"] = True

    # ── 4. checkpoints/best.pth ───────────────────────────────────────────────
    ckpt_path = run_dir / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        result["failures"].append("checkpoints/best.pth missing")
    else:
        result["has_checkpoint"] = True

    # ── Early exit if critical artefacts missing ───────────────────────────────
    if result["failures"]:
        result["status"] = "FAIL"
        return result

    # ── Parse expected intervals from config ──────────────────────────────────
    max_iter, val_every = _get_training_params(cfg)
    result["max_iter"]  = max_iter
    result["val_every"] = val_every
    expected_iters = set(range(val_every, max_iter + 1, val_every))
    result["expected_iters"] = sorted(expected_iters)

    # ── Parse recorded intervals from CSV ─────────────────────────────────────
    rows = _read_val_csv(val_csv)

    if not rows:
        result["failures"].append("val_metrics.csv is empty or unreadable")
        result["status"] = "FAIL"
        return result

    # Column check
    all_cols = set(rows[0].keys())
    missing_req = REQUIRED_COLS - all_cols
    if missing_req:
        result["failures"].append(f"val_metrics.csv missing required columns: {sorted(missing_req)}")
        result["status"] = "FAIL"
        return result

    missing_opt = OPTIONAL_COLS - all_cols
    result["optional_cols_missing"] = sorted(missing_opt)
    if missing_opt:
        result["warnings"].append(f"optional columns absent: {sorted(missing_opt)}")

    # Parse iteration column — may be named 'iteration' or contain it as first col
    iter_col = "iteration"
    if iter_col not in all_cols:
        # Try first column
        iter_col = rows[0] and list(rows[0].keys())[0]

    recorded_iters = set()
    best_f1   = -1.0
    best_iter = None

    for row in rows:
        raw_iter = row.get(iter_col, "")
        it = _safe_float(raw_iter)
        if it is None:
            continue
        it_int = int(it)
        recorded_iters.add(it_int)

        f1v = _safe_float(row.get("f1", ""))
        if f1v is not None and f1v > best_f1:
            best_f1   = f1v
            best_iter = it_int

        # ── Collapse check ────────────────────────────────────────────────────
        collapsed = False
        f1   = _safe_float(row.get("f1", ""))
        prec = _safe_float(row.get("precision", ""))
        rec  = _safe_float(row.get("recall", ""))
        ppr  = _safe_float(row.get("pred_positive_ratio", ""))

        if f1 is not None and f1 == 0.0:
            collapsed = True
        elif (prec is not None and prec == 0.0) and (rec is not None and rec == 0.0):
            collapsed = True
        elif ppr is not None and ppr < COLLAPSE_THRESHOLD_PPR:
            collapsed = True

        if collapsed:
            result["collapse_iters"].append(it_int)

    result["recorded_iters"] = sorted(recorded_iters)
    result["best_f1"]        = round(best_f1, 6) if best_f1 >= 0 else None
    result["best_iter"]      = best_iter

    # ── Interval gap analysis ─────────────────────────────────────────────────
    missing = sorted(expected_iters - recorded_iters)
    extra   = sorted(recorded_iters - expected_iters)
    result["missing_iters"] = missing
    result["extra_iters"]   = extra

    if missing:
        # Determine if training is incomplete vs truly missing validations
        # If the last recorded iter < max_iter, training may be mid-run → WARN
        last_recorded = max(recorded_iters) if recorded_iters else 0
        if last_recorded < max_iter:
            result["warnings"].append(
                f"Training appears incomplete (last recorded iter={last_recorded}). "
                f"Missing: {missing}"
            )
        else:
            result["warnings"].append(f"Missing validation intervals: {missing}")

    if extra:
        result["notes"].append(f"Extra validation intervals (not in expected set): {extra}")

    # ── Collapse summary ──────────────────────────────────────────────────────
    if result["collapse_iters"]:
        result["warnings"].append(
            f"COLLAPSE WARNING at iterations: {result['collapse_iters']}"
        )

    # ── best_metrics.json cross-check ─────────────────────────────────────────
    best_json = _read_best_metrics_json(run_dir)
    if best_json is not None:
        json_f1   = _safe_float(best_json.get("f1") or best_json.get("best_f1"))
        json_iter = best_json.get("iteration") or best_json.get("best_iteration")
        if json_f1 is not None and best_f1 >= 0:
            if abs(json_f1 - best_f1) > 1e-4:
                result["warnings"].append(
                    f"best_metrics.json F1={json_f1:.4f} differs from CSV max F1={best_f1:.4f}"
                )
    else:
        result["notes"].append("validation/best_metrics.json not found (optional)")

    # ── Final status ──────────────────────────────────────────────────────────
    if result["failures"]:
        result["status"] = "FAIL"
    elif result["warnings"]:
        result["status"] = "WARN"
    else:
        result["status"] = "PASS"

    return result


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_runs(root: Path) -> list[Path]:
    """Find all run_* directories anywhere under root (up to 4 levels deep)."""
    runs: list[Path] = []
    for depth in range(1, 5):
        pattern = "/".join(["*"] * depth)
        for p in sorted(root.glob(pattern)):
            if p.is_dir() and p.name.startswith("run_"):
                # Avoid double-counting nested runs
                if not any(p.is_relative_to(r) for r in runs):
                    runs.append(p)
    return sorted(runs)


# ── Output formatters ─────────────────────────────────────────────────────────

_STATUS_EMOJI = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}

def _fmt_iters(iters: list[int]) -> str:
    if not iters:
        return "—"
    if len(iters) <= 4:
        return ", ".join(str(i) for i in iters)
    return f"{iters[0]}…{iters[-1]} ({len(iters)} total)"


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    cols = [
        ("Run", 40),
        ("Status", 6),
        ("Missing Val Iters", 30),
        ("Best F1", 8),
        ("Best Iter", 9),
        ("Collapse", 8),
        ("Notes", 45),
    ]
    sep = "+" + "+".join("-" * (w + 2) for _, w in cols) + "+"
    hdr = "| " + " | ".join(f"{n:<{w}}" for n, w in cols) + " |"

    print()
    print(sep)
    print(hdr)
    print(sep)

    for r in results:
        emoji    = _STATUS_EMOJI.get(r["status"], "?")
        status   = f"{emoji} {r['status']}"
        missing  = _fmt_iters(r.get("missing_iters", []))
        best_f1  = f"{r['best_f1']:.4f}" if r.get("best_f1") is not None else "—"
        best_it  = str(r["best_iter"]) if r.get("best_iter") else "—"
        collapse = str(len(r.get("collapse_iters", []))) + " iter(s)" if r.get("collapse_iters") else "—"

        notes_list = r.get("failures", []) + r.get("warnings", []) + r.get("notes", [])
        notes = "; ".join(notes_list)[:45] if notes_list else "—"

        row_vals = [r["run"], status, missing, best_f1, best_it, collapse, notes]
        row = "| " + " | ".join(f"{str(v):<{w}}" for v, (_, w) in zip(row_vals, cols)) + " |"
        print(row)

    print(sep)
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n  Total: {len(results)} runs  |  ✓ PASS: {pass_count}  ⚠ WARN: {warn_count}  ✗ FAIL: {fail_count}")
    print()


def save_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "run", "status", "has_config", "has_train_log", "has_val_csv",
        "has_checkpoint", "max_iter", "val_every",
        "expected_iters_count", "recorded_iters_count",
        "missing_iters_count", "missing_iters",
        "collapse_iters_count", "collapse_iters",
        "best_f1", "best_iter",
        "optional_cols_missing", "warnings", "failures",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([
                r["run"],
                r["status"],
                r["has_config"],
                r["has_train_log"],
                r["has_val_csv"],
                r["has_checkpoint"],
                r.get("max_iter", ""),
                r.get("val_every", ""),
                len(r.get("expected_iters", [])),
                len(r.get("recorded_iters", [])),
                len(r.get("missing_iters", [])),
                _fmt_iters(r.get("missing_iters", [])),
                len(r.get("collapse_iters", [])),
                _fmt_iters(r.get("collapse_iters", [])),
                r.get("best_f1", ""),
                r.get("best_iter", ""),
                "; ".join(r.get("optional_cols_missing", [])),
                "; ".join(r.get("warnings", [])),
                "; ".join(r.get("failures", [])),
            ])


def save_markdown(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Validation Check Report",
        "",
        "| Run | Status | Missing Iters | Best F1 | Best Iter | Collapses | Notes |",
        "|-----|--------|---------------|---------|-----------|-----------|-------|",
    ]
    for r in results:
        emoji    = _STATUS_EMOJI.get(r["status"], "?")
        status   = f"{emoji} {r['status']}"
        missing  = _fmt_iters(r.get("missing_iters", []))
        best_f1  = f"{r['best_f1']:.4f}" if r.get("best_f1") is not None else "—"
        best_it  = str(r["best_iter"]) if r.get("best_iter") else "—"
        collapse_n = len(r.get("collapse_iters", []))
        collapse = f"{collapse_n}" if collapse_n > 0 else "—"
        notes_list = r.get("failures", []) + r.get("warnings", [])
        notes = "; ".join(notes_list)[:80] if notes_list else "—"
        lines.append(
            f"| {r['run']} | {status} | {missing} | {best_f1} | {best_it} | {collapse} | {notes} |"
        )

    lines += [
        "",
        "## Status Legend",
        "",
        "| Status | Meaning |",
        "|--------|---------|",
        "| ✓ PASS | All artefacts present, all expected validation intervals recorded, no collapse |",
        "| ⚠ WARN | Optional files missing, some intervals missing (may be incomplete run), collapse detected |",
        "| ✗ FAIL | config.yaml / val_metrics.csv / best.pth absent, or validation never ran |",
        "",
        "## Expected Validation Intervals",
        "",
        "Derived from `config.yaml`:",
        "```",
        "training:",
        "  max_iterations: 50000",
        "  validate_every: 5000",
        "# → expects validation at: 5000, 10000, 15000, …, 50000",
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_json(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _RUN_ROOT.exists():
        print(f"[WARN] RUN_ROOT does not exist: {_RUN_ROOT}")
        print("       No runs found — nothing to check.")
        print("       Start a training run first with: python scripts/train.py")
        # Still write empty reports so the script is verifiably runnable
        results: list[dict] = []
    else:
        runs = discover_runs(_RUN_ROOT)
        if not runs:
            print(f"[WARN] No run_* directories found under: {_RUN_ROOT}")
            results = []
        else:
            print(f"Found {len(runs)} run(s) under {_RUN_ROOT}\n")
            results = [check_run(r) for r in runs]

    # ── Print to stdout ───────────────────────────────────────────────────────
    if results:
        print_summary(results)

        # Print per-run details for failures and warnings
        for r in results:
            if r["status"] in ("FAIL", "WARN"):
                print(f"  [{r['status']}] {r['run']}")
                for f in r.get("failures", []):
                    print(f"    FAIL : {f}")
                for w in r.get("warnings", []):
                    print(f"    WARN : {w}")
                for n in r.get("notes", []):
                    print(f"    NOTE : {n}")
                if r.get("collapse_iters"):
                    print(f"    COLLAPSE at iters: {r['collapse_iters']}")
                print()
    else:
        print("No runs to check.")

    # ── Save reports ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = RESULTS_DIR / "check_summary.csv"
    md_path   = RESULTS_DIR / "check_summary.md"
    json_path = RESULTS_DIR / "check_details.json"

    save_csv(results, csv_path)
    save_markdown(results, md_path)
    save_json(results, json_path)

    print(f"Reports saved to:")
    print(f"  {csv_path.relative_to(REPO_ROOT)}")
    print(f"  {md_path.relative_to(REPO_ROOT)}")
    print(f"  {json_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
