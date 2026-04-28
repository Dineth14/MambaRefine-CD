#!/usr/bin/env python3
"""Generate bar and drop charts for ablation results."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "outputs" / "ablation" / "ablation_summary.csv"
PLOT_DIR = ROOT / "outputs" / "ablation" / "plots"


def _to_float(value: str) -> float | None:
    if value in {"", "None", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        labels = [row["method"] for row in dataset_rows]
        f1_values = [_to_float(row["F1_1"]) or 0.0 for row in dataset_rows]
        delta_values = [0.0 if row["experiment"] == "baseline" else (_to_float(row["delta_F1"]) or 0.0) for row in dataset_rows]
        slug = dataset.lower().replace("/", "-").replace(" ", "_")

        plt.figure(figsize=(10, 4.5))
        plt.bar(labels, f1_values, color="#2563eb")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("F1_1")
        plt.title(f"{dataset} ablation F1")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{slug}_f1_bar.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 4.5))
        colors = ["#dc2626" if value < 0 else "#059669" for value in delta_values]
        plt.bar(labels, delta_values, color=colors)
        plt.axhline(0.0, color="#111827", linewidth=1)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("ΔF1")
        plt.title(f"{dataset} ablation ΔF1")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{slug}_delta_f1.png", dpi=200)
        plt.close()

    print(f"Wrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()