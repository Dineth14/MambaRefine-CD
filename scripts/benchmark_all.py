"""Benchmark all datasets with a single script.

Reads configs/benchmark_suite.yaml, evaluates each checkpoint on its
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

from utils.config_loader            import load_config
from utils.seed                     import set_seed
from data.dataset_builder           import build_test_loader
from models.cd_model                import build_model
from training.evaluator             import Evaluator
from training.checkpoint            import peek as peek_ckpt
from training.generalization_metrics import (
    compute_generalization,
    save_generalization_report,
)

# ── Benchmark suite config ─────────────────────────────────────────────────
SUITE_PATH = "configs/benchmark_suite.yaml"
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_KEYS = [
    "f1", "iou", "miou", "precision", "recall", "oa",
    "boundary_f1", "edge_iou", "pred_positive_ratio", "gt_positive_ratio",
]


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
    suite_path = ROOT / SUITE_PATH
    if not suite_path.exists():
        raise FileNotFoundError(f"Benchmark suite config not found: {suite_path}")

    suite = load_config(suite_path)["benchmark"]
    set_seed(42)

    model_name  = suite.get("model_name", "model")
    out_dir     = ROOT / suite.get("output_dir", "outputs/benchmark_runs/summary")
    latex_dir   = out_dir / "latex_tables"
    main_ds     = suite.get("main_dataset", "LEVIR-CD")
    eval_split  = suite.get("eval_split", "test")

    out_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        suite.get("hardware", {}).get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
    amp = bool(suite.get("hardware", {}).get("mixed_precision", True))

    checkpoints   = suite.get("checkpoints", {})
    ds_cfg_paths  = suite.get("dataset_configs", {})
    datasets_run  = suite.get("datasets", list(checkpoints.keys()))

    # Build a lightweight eval-only cfg for Evaluator
    eval_base_cfg = {
        "evaluation":      suite.get("evaluation", {"threshold": 0.5}),
        "boundary_metrics": suite.get("boundary_metrics", {"enabled": True}),
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
            print(f"  [SKIP] {ds_name}: no checkpoint configured (set in benchmark_suite.yaml)")
            continue
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (ROOT / ckpt_path).resolve()
        if not ckpt_path.exists():
            print(f"  [SKIP] {ds_name}: checkpoint not found: {ckpt_path}")
            continue

        # Find matching dataset config
        ds_cfg_path = None
        for key, path in ds_cfg_paths.items():
            if key.lower() in ds_name.lower() or ds_name.lower().startswith(key.lower()):
                ds_cfg_path = path
                break
        if ds_cfg_path is None:
            print(f"  [SKIP] {ds_name}: no dataset_config configured")
            continue

        print(f"  [{ds_name}] Loading dataset and model ...")
        ds_full_cfg = load_config(ROOT / ds_cfg_path)

        # Build merged cfg for this dataset
        model_cfg = {"model": suite["model"], "hardware": suite.get("hardware", {}), **eval_base_cfg}
        loader_cfg = {**ds_full_cfg, **model_cfg, "training": {"batch_size": 8}}

        loader = build_test_loader(loader_cfg)

        model = build_model({**ds_full_cfg, **model_cfg}).to(device)
        ckpt_info = peek_ckpt(ckpt_path)
        model.load_state_dict(ckpt_info["model"], strict=True)
        model.eval()

        evaluator = Evaluator(eval_base_cfg, device)
        results   = evaluator.evaluate(model, loader, dataset_name=ds_name, amp=amp)
        evaluator.print_table(results, title=f"  ── {ds_name} ──")

        all_results[ds_name] = results
        print()

    if not all_results:
        print("No datasets were evaluated. Check checkpoint paths in benchmark_suite.yaml.")
        return

    # ── Save benchmark_results.csv ────────────────────────────────────────────
    csv_path = out_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model"] + _METRIC_KEYS)
        for ds, res in all_results.items():
            w.writerow(
                [ds, model_name]
                + [_fmt(res.get(k, ""), 4) for k in _METRIC_KEYS]
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
        row = [ds] + [_fmt(res.get(k, ""), 4) for k in _METRIC_KEYS]
        lines.append("| " + " | ".join(row) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown   → {md_path}")

    # ── LaTeX tables ──────────────────────────────────────────────────────────
    # 1. Core benchmark table
    core_keys  = ["f1", "iou", "miou", "precision", "recall", "oa"]
    core_rows  = [
        [ds] + [_fmt(all_results[ds].get(k, ""), 4) for k in core_keys]
        for ds in all_results
    ]
    core_tex   = _latex_table(
        caption = f"Core Change Detection Results — {model_name}",
        label   = "tab:core_results",
        header  = ["Dataset", "F1", "IoU", "mIoU", "Prec", "Recall", "OA"],
        rows    = core_rows,
    )
    tex_path   = latex_dir / "core_table.tex"
    tex_path.write_text(core_tex)
    print(f"Saved LaTeX core → {tex_path}")

    # 2. Boundary table
    bnd_keys  = ["boundary_f1", "edge_iou", "pred_positive_ratio", "gt_positive_ratio"]
    bnd_rows  = [
        [ds] + [_fmt(all_results[ds].get(k, ""), 4) for k in bnd_keys]
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

    # ── Generalization metrics ────────────────────────────────────────────────
    gen_stats = compute_generalization(all_results, main_dataset=main_ds)
    save_generalization_report(gen_stats, all_results, out_dir)

    # 3. Generalization LaTeX table
    ds_list = gen_stats.get("datasets", [])
    f1_vals = [_fmt(all_results[ds].get("f1", ""), 4) for ds in ds_list]
    mean_f1 = _fmt(gen_stats.get("mean", {}).get("f1", ""), 4)
    std_f1  = _fmt(gen_stats.get("std",  {}).get("f1", ""), 4)
    gen_header = [model_name] + ds_list + ["Mean F1", "Std F1"]
    gen_row    = [model_name] + f1_vals + [mean_f1, std_f1]
    gen_tex    = _latex_table(
        caption = "Generalization Across Datasets",
        label   = "tab:generalization",
        header  = ["Model"] + ds_list + ["Mean F1", "Std F1"],
        rows    = [gen_row],
    )
    (latex_dir / "generalization_table.tex").write_text(gen_tex)
    print(f"Saved LaTeX gen  → {latex_dir / 'generalization_table.tex'}")

    print(f"\nAll benchmark outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
