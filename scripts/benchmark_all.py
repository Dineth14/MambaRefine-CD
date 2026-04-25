"""Benchmark all datasets with a single script.

Reads configs/global_config.yaml, evaluates each checkpoint on its
matching dataset, then generates:

  outputs/benchmark_runs/summary/
    benchmark_results.csv
    benchmark_results.md
    latex_tables/
      core_table.tex
      boundary_table.tex
      generalization_table.tex
    generalization_summary.json
    generalization_summary.md

No CLI arguments needed.

Run:
    conda activate mamba_new
    cd MambaRefine-CD
    python scripts/benchmark_all.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from utils.config                   import load_config
from utils.seed                     import set_seed
from data.dataset_builder           import build_test_loader
from models.cd_model                import build_model
from training.evaluator             import Evaluator
from training.checkpoint            import peek as peek_ckpt
from training.generalization_metrics import (
    compute_generalization,
    save_generalization_report,
)

_METRIC_KEYS = [
    "mf1", "f1_0", "f1_1", "miou", "iou_0", "iou_1",
    "precision_1", "recall_1", "oa",
    "boundary_f1", "edge_iou", "pred_positive_ratio", "gt_positive_ratio",
]

# Aliases: result dict may use old key names; this resolves to canonical value
_KEY_ALIAS = {
    "mf1":         ("mf1",),
    "f1_0":        ("f1_0",),
    "f1_1":        ("f1_1", "f1"),
    "miou":        ("miou",),
    "iou_0":       ("iou_0",),
    "iou_1":       ("iou_1", "iou"),
    "precision_1": ("precision_1", "precision"),
    "recall_1":    ("recall_1",    "recall"),
    "oa":          ("oa",),
    "boundary_f1":         ("boundary_f1",),
    "edge_iou":            ("edge_iou",),
    "pred_positive_ratio": ("pred_positive_ratio",),
    "gt_positive_ratio":   ("gt_positive_ratio",),
}


def _resolve(res: dict, key: str) -> str:
    """Get value from result dict, trying canonical key then aliases."""
    for k in _KEY_ALIAS.get(key, (key,)):
        if k in res and isinstance(res[k], (int, float)):
            return res[k]
    return ""


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def _latex_row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\"


def _latex_table(caption: str, label: str, header: list[str], rows: list[list[str]]) -> str:
    ncol = len(header)
    col_fmt = "l" + "c" * (ncol - 1)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        _latex_row(header),
        r"\midrule",
    ]
    for row in rows:
        lines.append(_latex_row(row))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _fmt(v, decimals: int = 4) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    suite = cfg.benchmark
    set_seed(int(cfg.experiment.seed))

    model_name  = suite.model_name
    out_dir     = ROOT / suite.output_dir
    latex_dir   = out_dir / "latex_tables"
    main_ds     = suite.main_dataset
    eval_split  = suite.eval_split

    out_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.hardware.device if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    amp = bool(cfg.hardware.mixed_precision)

    checkpoints   = suite.checkpoints
    datasets_run  = list(suite.datasets)
    ds_catalog    = cfg.datasets_catalog

    # Build a lightweight eval-only cfg for Evaluator
    eval_base_cfg = {
        "evaluation":      cfg.evaluation.to_dict(),
        "boundary_metrics": cfg.boundary_metrics.to_dict(),
    }

    all_results: dict = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ds_name in datasets_run:
        # Find the matching checkpoint key (case-insensitive prefix match)
        ckpt_path = None
        for key, path in checkpoints.items():
            if key.lower() in ds_name.lower() or ds_name.lower().startswith(key.lower()):
                ckpt_path = path
                break

        if ckpt_path is None:
            print(f"  [SKIP] {ds_name}: no checkpoint configured (set in global_config.yaml -> benchmark.checkpoints)")
            continue
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (ROOT / ckpt_path).resolve()
        if not ckpt_path.exists():
            print(f"  [SKIP] {ds_name}: checkpoint not found: {ckpt_path}")
            continue

        ds_cfg = None
        for key, value in ds_catalog.items():
            catalog_name = str(value.get("name", key))
            if key.lower() in ds_name.lower() or ds_name.lower().startswith(key.lower()) or catalog_name.lower() == ds_name.lower():
                ds_cfg = value
                break
        if ds_cfg is None:
            print(f"  [SKIP] {ds_name}: no dataset configured in global_config.yaml")
            continue

        print(f"  [{ds_name}] Loading dataset and model ...")
        loader_cfg = cfg.to_dict()
        loader_cfg["dataset"] = dict(ds_cfg)
        loader_cfg["evaluation"]["split"] = eval_split

        loader = build_test_loader(loader_cfg)

        model = build_model(cfg).to(device)
        ckpt_info = peek_ckpt(ckpt_path)
        model.load_state_dict(ckpt_info["model"], strict=True)
        model.eval()

        evaluator = Evaluator(eval_base_cfg, device)
        results   = evaluator.evaluate(model, loader, dataset_name=ds_name, amp=amp)
        evaluator.print_table(results, title=f"  ── {ds_name} ──")

        all_results[ds_name] = results
        print()

    if not all_results:
        print("No datasets were evaluated. Check benchmark.checkpoints in global_config.yaml.")
        return

    # ── Save benchmark_results.csv ────────────────────────────────────────────
    csv_path = out_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model"] + _METRIC_KEYS)
        for ds, res in all_results.items():
            w.writerow(
                [ds, model_name]
                + [_fmt(_resolve(res, k), 4) for k in _METRIC_KEYS]
            )
    print(f"Saved CSV        → {csv_path}")

    # ── Save benchmark_results.md ─────────────────────────────────────────────
    md_path = out_dir / "benchmark_results.md"
    header  = ["Dataset"] + [k.upper().replace("_", " ") for k in _METRIC_KEYS]
    lines   = [
        "# Benchmark Results",
        "",
        f"Model: **{model_name}**   |   Timestamp: {timestamp}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for ds, res in all_results.items():
        row = [ds] + [_fmt(_resolve(res, k), 4) for k in _METRIC_KEYS]
        lines.append("| " + " | ".join(row) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown   → {md_path}")

    # ── Save literature_comparison.csv ────────────────────────────────────────
    lit_csv_path = out_dir / "literature_comparison.csv"
    with open(lit_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Dataset", "mF1", "mIoU", "OA"])
        for ds, res in all_results.items():
            w.writerow([
                model_name, ds,
                _fmt(_resolve(res, "mf1"),  4),
                _fmt(_resolve(res, "miou"), 4),
                _fmt(_resolve(res, "oa"),   4),
            ])
    print(f"Saved lit CSV    → {lit_csv_path}")

    # ── Save practical_change_metrics.csv ─────────────────────────────────────
    prac_csv_path = out_dir / "practical_change_metrics.csv"
    with open(prac_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Dataset", "F1_1", "IoU_1", "Precision_1", "Recall_1", "Boundary_F1", "Edge_IoU"])
        for ds, res in all_results.items():
            w.writerow([
                model_name, ds,
                _fmt(_resolve(res, "f1_1"),        4),
                _fmt(_resolve(res, "iou_1"),       4),
                _fmt(_resolve(res, "precision_1"), 4),
                _fmt(_resolve(res, "recall_1"),    4),
                _fmt(_resolve(res, "boundary_f1"), 4),
                _fmt(_resolve(res, "edge_iou"),    4),
            ])
    print(f"Saved prac CSV   → {prac_csv_path}")

    # ── LaTeX tables ──────────────────────────────────────────────────────────
    # 1. Core benchmark table — literature-style mean metrics
    core_keys  = ["mf1", "f1_1", "miou", "iou_1", "precision_1", "recall_1", "oa"]
    core_rows  = [
        [ds] + [_fmt(_resolve(all_results[ds], k), 4) for k in core_keys]
        for ds in all_results
    ]
    core_tex   = _latex_table(
        caption = f"Core Change Detection Results — {model_name}",
        label   = "tab:core_results",
        header  = ["Dataset", "mF1", "F1\\_1", "mIoU", "IoU\\_1", "Prec\\_1", "Rec\\_1", "OA"],
        rows    = core_rows,
    )
    tex_path   = latex_dir / "core_table.tex"
    tex_path.write_text(core_tex)
    print(f"Saved LaTeX core → {tex_path}")

    # 2. Boundary table
    bnd_keys  = ["boundary_f1", "edge_iou", "pred_positive_ratio", "gt_positive_ratio"]
    bnd_rows  = [
        [ds] + [_fmt(_resolve(all_results[ds], k), 4) for k in bnd_keys]
        for ds in all_results
    ]
    bnd_tex   = _latex_table(
        caption = f"Boundary-Aware Metrics — {model_name}",
        label   = "tab:boundary_results",
        header  = ["Dataset", "Bnd F1", "Edge IoU", "Pred Ratio", "GT Ratio"],
        rows    = bnd_rows,
    )
    (latex_dir / "boundary_table.tex").write_text(bnd_tex)
    print(f"Saved LaTeX bnd  → {latex_dir / 'boundary_table.tex'}")

    # 3. Literature comparison LaTeX table (mean metrics, for paper comparison)
    lit_keys  = ["mf1", "miou", "oa"]
    lit_rows  = [
        [ds] + [_fmt(_resolve(all_results[ds], k), 4) for k in lit_keys]
        for ds in all_results
    ]
    lit_tex   = _latex_table(
        caption = f"Literature Comparison — Mean Metrics — {model_name}",
        label   = "tab:literature_comparison",
        header  = ["Dataset", "mF1", "mIoU", "OA"],
        rows    = lit_rows,
    )
    (latex_dir / "literature_comparison.tex").write_text(lit_tex)
    print(f"Saved LaTeX lit  → {latex_dir / 'literature_comparison.tex'}")

    # 4. Change-class detailed table
    cc_keys  = ["f1_1", "iou_1", "precision_1", "recall_1", "boundary_f1", "edge_iou"]
    cc_rows  = [
        [ds] + [_fmt(_resolve(all_results[ds], k), 4) for k in cc_keys]
        for ds in all_results
    ]
    cc_tex   = _latex_table(
        caption = f"Change-Class Metrics (Class 1) — {model_name}",
        label   = "tab:change_class_comparison",
        header  = ["Dataset", "F1\\_1", "IoU\\_1", "Prec\\_1", "Rec\\_1", "Bnd F1", "Edge IoU"],
        rows    = cc_rows,
    )
    (latex_dir / "change_class_comparison.tex").write_text(cc_tex)
    print(f"Saved LaTeX cc   → {latex_dir / 'change_class_comparison.tex'}")

    # ── Generalization metrics ────────────────────────────────────────────────
    gen_stats = compute_generalization(all_results, main_dataset=main_ds)
    save_generalization_report(gen_stats, all_results, out_dir)

    # 5. Generalization LaTeX table
    ds_list  = gen_stats.get("datasets", [])
    mf1_vals = [_fmt(_resolve(all_results[ds], "mf1"), 4) for ds in ds_list]
    mean_mf1 = _fmt(gen_stats.get("mean", {}).get("mf1") or gen_stats.get("mean", {}).get("f1", ""), 4)
    std_mf1  = _fmt(gen_stats.get("std",  {}).get("mf1") or gen_stats.get("std",  {}).get("f1", ""), 4)
    gen_tex  = _latex_table(
        caption = "Generalization Across Datasets",
        label   = "tab:generalization",
        header  = ["Model"] + ds_list + ["Mean mF1", "Std mF1"],
        rows    = [[model_name] + mf1_vals + [mean_mf1, std_mf1]],
    )
    (latex_dir / "generalization_table.tex").write_text(gen_tex)
    print(f"Saved LaTeX gen  → {latex_dir / 'generalization_table.tex'}")

    print(f"\nAll benchmark outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
