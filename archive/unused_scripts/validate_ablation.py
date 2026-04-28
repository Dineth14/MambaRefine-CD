#!/usr/bin/env python3
"""Validate ablation summary outputs."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = ROOT / "outputs" / "ablation"
SUMMARY_CSV = ABLATION_ROOT / "ablation_summary.csv"


def _to_float(value: str) -> float | None:
    if value in {"", "None", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checks": {},
        "errors": [],
    }
    report["checks"]["summary_csv_exists"] = SUMMARY_CSV.exists()
    if SUMMARY_CSV.exists():
        rows = list(csv.DictReader(SUMMARY_CSV.open("r", encoding="utf-8", newline="")))
        if not rows:
            report["errors"].append("No ablation rows found in outputs/ablation/ablation_summary.csv")
        datasets = sorted({row["dataset"] for row in rows})
        report["checks"]["datasets"] = datasets
        for dataset in datasets:
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            baseline = next((row for row in dataset_rows if row["experiment"] == "baseline"), None)
            if baseline is None:
                report["errors"].append(f"{dataset}: baseline exists check failed")
                continue
            baseline_f1 = _to_float(baseline["F1_1"])
            if baseline_f1 is None:
                report["errors"].append(f"{dataset}: baseline F1_1 missing")
                continue
            for row in dataset_rows:
                for key in ["F1_1", "IoU_1", "OA", "Precision_1", "Recall_1"]:
                    if _to_float(row[key]) is None:
                        report["errors"].append(f"{dataset}/{row['experiment']}: missing {key}")
                eval_path = ROOT / row["eval_file"]
                if not eval_path.exists():
                    report["errors"].append(f"{dataset}/{row['experiment']}: missing eval file")
                if row["experiment"] == "baseline":
                    continue
                observed = _to_float(row["delta_F1"])
                expected = (_to_float(row["F1_1"]) or 0.0) - baseline_f1
                if observed is None or abs(observed - expected) > 1e-6:
                    report["errors"].append(f"{dataset}/{row['experiment']}: Delta_F1 computed incorrectly")
    else:
        report["errors"].append("Missing outputs/ablation/ablation_summary.csv")

    report["checks"]["baseline_exists"] = not any("baseline exists" in item for item in report["errors"])
    report["checks"]["missing_metrics"] = not any("missing" in item for item in report["errors"])
    report["checks"]["delta_f1_correct"] = not any("Delta_F1 computed incorrectly" in item for item in report["errors"])
    report["status"] = "PASS" if not report["errors"] else "FAIL"

    json_path = ABLATION_ROOT / "ablation_validation.json"
    md_path = ABLATION_ROOT / "ablation_validation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Ablation Validation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Checks",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Errors"])
    if report["errors"]:
        lines.extend(f"- {item}" for item in report["errors"])
    else:
        lines.append("- None")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(report["status"])


if __name__ == "__main__":
    main()