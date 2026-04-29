#!/usr/bin/env python3
"""Export the DSIFN-CD ablation table with paper metrics only."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ABLATIONS = [
    ("a0_fpn_baseline", "A0 Baseline"),
    ("a1_mambavision_fpn", "A1 MambaVision"),
    ("a2_mambavision_drbi", "A2 +D-RBI"),
    ("a3_mambavision_drbi_signed", "A3 +SignedDiff"),
    ("a4_mambavision_drbi_arf", "A4 +ARF"),
    ("a5_mambavision_drbi_arf_boundary", "A5 +Boundary"),
    ("a6_full", "A6 Full"),
]

MAIN_FIELDS = ["Pre", "Rec", "F1", "IoU", "OA"]

SUMMARY_KEY_MAP = {
    "Precision": "Pre",
    "Recall": "Rec",
    "F1": "F1",
    "IoU": "IoU",
    "OA": "OA",
}


def _latest_run(ablation: str) -> Path:
    root = REPO / "outputs" / "dsifn" / ablation
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run_* directory found for {ablation} under {root}")
    return runs[-1]


def _read_metrics_json(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(raw[k]) for k in MAIN_FIELDS if k in raw}


def _read_metrics_csv(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and set(MAIN_FIELDS).issubset(reader.fieldnames):
            rows = list(reader)
            if not rows:
                return {}
            row = rows[-1]
            return {k: float(row[k]) for k in MAIN_FIELDS}
        f.seek(0)
        reader = csv.reader(f)
        rows = list(reader)
    values: dict[str, float] = {}
    for row in rows[1:]:
        if len(row) >= 2 and row[0] in MAIN_FIELDS:
            values[row[0]] = float(row[1])
    return values


def _read_summary(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(r"^([A-Za-z ]+)\s*:\s*([0-9.]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        key = SUMMARY_KEY_MAP.get(match.group(1).strip())
        if key is not None:
            value = float(match.group(2))
            values[key] = value * 100.0 if value <= 1.0 else value
    return values


def _load_metrics(run_dir: Path) -> dict[str, float]:
    candidates = [
        run_dir / "test_results" / "test_metrics.json",
        run_dir / "test_results" / "test_metrics.csv",
        run_dir / "test_results" / "test_summary.txt",
        run_dir / "metrics.json",
        run_dir / "metrics.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".json":
            values = _read_metrics_json(path)
        elif path.suffix == ".csv":
            values = _read_metrics_csv(path)
        else:
            values = _read_summary(path)
        if set(MAIN_FIELDS).issubset(values):
            return values
    raise FileNotFoundError(f"No complete DSIFN main-metric result found under {run_dir}")


def _load_boundary_f1(run_dir: Path) -> float | None:
    summary = run_dir / "test_results" / "test_summary.txt"
    if not summary.exists():
        return None
    pattern = re.compile(r"^Boundary F1\s*:\s*([0-9.]+)")
    for line in summary.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            value = float(match.group(1))
            return value * 100.0 if value <= 1.0 else value
    return None


def _round(v: float) -> str:
    return f"{v:.2f}"


def _sanity_check(rows: list[dict[str, str]]) -> None:
    metric_vectors = [tuple(row[k] for k in MAIN_FIELDS) for row in rows]
    if len(set(metric_vectors)) == 1:
        raise RuntimeError("All DSIFN ablation results are identical; stop and debug config application.")
    baseline = float(rows[0]["F1"])
    full = float(rows[-1]["F1"])
    if full <= baseline:
        raise RuntimeError(f"Full model F1 ({full:.2f}) does not outperform baseline F1 ({baseline:.2f}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DSIFN-CD ablation CSV.")
    parser.add_argument("--output", default="results/dsifn_ablation.csv")
    parser.add_argument("--boundary_output", default="results/dsifn_boundary_analysis.csv")
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    boundary_rows: list[dict[str, str]] = []
    missing: list[str] = []

    for ablation, label in ABLATIONS:
        try:
            run_dir = _latest_run(ablation)
            metrics = _load_metrics(run_dir)
        except FileNotFoundError as exc:
            if args.allow_incomplete:
                missing.append(str(exc))
                continue
            raise
        rows.append(
            {
                "Dataset": "DSIFN",
                "Model": label,
                **{key: _round(metrics[key]) for key in MAIN_FIELDS},
            }
        )
        boundary = _load_boundary_f1(run_dir)
        if boundary is not None:
            boundary_rows.append({"Model": label, "Boundary F1": _round(boundary)})

    if not rows:
        raise RuntimeError("No DSIFN ablation rows were exported.")
    if len(rows) == len(ABLATIONS):
        _sanity_check(rows)
    elif not args.allow_incomplete:
        raise RuntimeError("Incomplete DSIFN ablation table.")

    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Model", *MAIN_FIELDS])
        writer.writeheader()
        writer.writerows(rows)

    boundary_path = REPO / args.boundary_output
    if boundary_rows:
        boundary_path.parent.mkdir(parents=True, exist_ok=True)
        with boundary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Boundary F1"])
            writer.writeheader()
            writer.writerows(boundary_rows)

    print(f"Saved main table: {out_path}")
    if boundary_rows:
        print(f"Saved boundary analysis: {boundary_path}")
    if missing:
        print("Incomplete export; missing results:")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
