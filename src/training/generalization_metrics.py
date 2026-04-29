"""Generalization metrics across multiple benchmark datasets.

Given a results dict ``{dataset_name: metric_dict}``, computes:

  * Generalization gap  : delta between main dataset and each other dataset
  * Performance variance: variance of F1 (and other metrics) across datasets
  * Mean performance    : mean F1, IoU, boundary_F1 across datasets

Saves to:
    outputs/benchmark_runs/summary/generalization_summary.json
    outputs/benchmark_runs/summary/generalization_summary.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

# ── Core metrics tracked for generalization analysis ──────────────────────────
_TRACKED = ["f1", "iou", "miou", "boundary_f1", "edge_iou"]


def compute_generalization(
    results: Dict[str, dict],
    main_dataset: str = "DSIFN-CD",
) -> dict:
    """Compute generalization statistics across datasets.

    Args:
        results:      ``{dataset_name: metric_dict}`` from benchmark_all.py.
        main_dataset: reference dataset for computing generalization gap.

    Returns:
        dict with keys:
          gaps        – ``{metric: {dataset: gap_value}}``
          variance    – ``{metric: variance}``
          mean        – ``{metric: mean}``
          std         – ``{metric: std}``
          datasets    – list of dataset names evaluated
    """
    datasets = list(results.keys())
    if not datasets:
        return {}

    out: dict = {
        "datasets":     datasets,
        "main_dataset": main_dataset,
        "gaps":         {},
        "variance":     {},
        "std":          {},
        "mean":         {},
    }

    for metric in _TRACKED:
        values = {
            ds: results[ds][metric]
            for ds in datasets
            if metric in results[ds]
        }
        if not values:
            continue

        vals = list(values.values())
        mean = sum(vals) / len(vals)
        var  = sum((v - mean) ** 2 for v in vals) / len(vals)
        std  = math.sqrt(var)

        out["mean"][metric]     = round(mean, 6)
        out["variance"][metric] = round(var, 6)
        out["std"][metric]      = round(std, 6)

        # Generalization gap: main_dataset - each_other_dataset
        main_val = values.get(main_dataset)
        if main_val is not None:
            gaps = {}
            for ds, v in values.items():
                if ds != main_dataset:
                    gaps[ds] = round(main_val - v, 6)
            out["gaps"][metric] = gaps

    return out


def save_generalization_report(
    gen_stats: dict,
    results:   Dict[str, dict],
    save_dir:  Path,
) -> None:
    """Save generalization statistics to JSON and Markdown.

    Args:
        gen_stats: output of ``compute_generalization()``.
        results:   original per-dataset metric dicts.
        save_dir:  directory to write files into.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ────────────────────────────────────────────────────────────────
    json_path = save_dir / "generalization_summary.json"
    payload   = {"per_dataset": results, "generalization": gen_stats}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved generalization JSON → {json_path}")

    # ── Markdown ────────────────────────────────────────────────────────────
    md_path  = save_dir / "generalization_summary.md"
    datasets = gen_stats.get("datasets", [])
    main_ds  = gen_stats.get("main_dataset", "")

    lines: list[str] = [
        "# Generalization Summary",
        "",
        f"Main dataset (reference): **{main_ds}**",
        "",
        "## Per-Dataset Results",
        "",
    ]

    # Table header
    tracked = [m for m in _TRACKED if any(m in results[ds] for ds in datasets)]
    header  = "| Dataset | " + " | ".join(m.upper() for m in tracked) + " |"
    sep     = "| --- | " + " | ".join(["---"] * len(tracked)) + " |"
    lines += [header, sep]
    for ds in datasets:
        row = f"| {ds} | "
        row += " | ".join(
            f"{results[ds].get(m, float('nan')):.4f}" for m in tracked
        )
        row += " |"
        lines.append(row)

    lines += ["", "## Mean ± Std Across Datasets", ""]
    lines.append("| Metric | Mean | Std | Variance |")
    lines.append("| --- | --- | --- | --- |")
    for m in tracked:
        mean = gen_stats.get("mean", {}).get(m, float("nan"))
        std  = gen_stats.get("std",  {}).get(m, float("nan"))
        var  = gen_stats.get("variance", {}).get(m, float("nan"))
        lines.append(f"| {m.upper()} | {mean:.4f} | {std:.4f} | {var:.6f} |")

    if gen_stats.get("gaps"):
        lines += ["", "## Generalization Gaps (Main → Other)", ""]
        lines.append("| Metric | Dataset | Gap |")
        lines.append("| --- | --- | --- |")
        for metric, gaps in gen_stats["gaps"].items():
            for ds, gap in gaps.items():
                sign = "+" if gap >= 0 else ""
                lines.append(f"| {metric.upper()} | {ds} | {sign}{gap:.4f} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved generalization MD   → {md_path}")
