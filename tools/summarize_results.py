"""Scans outputs/ for test_metrics.json files.

Writes outputs/summary/results.csv with columns:
    run_name, dataset, temporal_mode, F1, IoU, Precision, Recall, OA, iteration
Usage: python tools/summarize_results.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    root = Path("outputs")
    rows = []
    for metrics_path in sorted(root.glob("run_*/metrics/test_metrics.json")):
        run_dir = metrics_path.parents[1]
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        cfg_path = run_dir / "config.yaml"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        rows.append({
            "run_name": run_dir.name,
            "dataset": cfg.get("data", {}).get("dataset_name", ""),
            "temporal_mode": cfg.get("ablation", {}).get("temporal_input_mode", ""),
            "F1": metrics.get("F1", ""),
            "IoU": metrics.get("IoU", ""),
            "Precision": metrics.get("Precision", ""),
            "Recall": metrics.get("Recall", ""),
            "OA": metrics.get("OA", ""),
            "iteration": metrics.get("iteration", metrics.get("best_iteration", "")),
        })

    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.csv"
    fields = ["run_name", "dataset", "temporal_mode", "F1", "IoU", "Precision", "Recall", "OA", "iteration"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
