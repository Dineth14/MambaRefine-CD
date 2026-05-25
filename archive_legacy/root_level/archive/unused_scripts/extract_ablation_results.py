#!/usr/bin/env python3
"""Extract ablation results into summary files and website assets."""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = ROOT / "outputs" / "ablation"
WEBSITE_TABLES = ROOT / "website" / "assets" / "tables"

DISPLAY_NAMES = {
    "baseline": "Full Model",
    "no_drbi": "w/o D-RBI",
    "no_boundary_branch": "w/o Boundary Branch",
    "no_rf_decoder": "w/o RF Decoder",
    "no_boundary_refinement": "w/o Refinement",
    "no_difference_features": "w/o Difference Features",
    "alpha_zero": "alpha = 0.0",
    "alpha_high": "alpha = 0.1",
    "fixed_dilation": "Fixed Dilation",
    "bce_only": "BCE only",
    "bce_dice": "BCE + Dice",
    "full_loss": "Full Loss",
}

CSV_COLUMNS = [
    "dataset",
    "experiment",
    "method",
    "F1_1",
    "IoU_1",
    "OA",
    "Precision_1",
    "Recall_1",
    "Boundary_F1",
    "best_threshold",
    "delta_F1",
    "eval_file",
]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "tbd", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _metric_text(value: float | None) -> str:
    return "TBD" if value is None else f"{value:.4f}"


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def _find_metrics_file(experiment_dir: Path) -> Path | None:
    for candidate in (experiment_dir / "eval_metrics.json", experiment_dir / "metrics.json"):
        if candidate.exists():
            return candidate
    recursive = sorted(experiment_dir.glob("runs/*/test_results/test_metrics.json"))
    return recursive[-1] if recursive else None


def _metric(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in payload:
            value = _to_float(payload.get(key))
            if value is not None:
                return value
    nested = payload.get("metrics")
    if isinstance(nested, dict):
        for key in keys:
            if key in nested:
                value = _to_float(nested.get(key))
                if value is not None:
                    return value
    return None


def _format_delta(value: float | None) -> str:
    if value is None:
        return "—"
    if value > 1e-9:
        return f"↑{value:.4f}"
    if value < -1e-9:
        return f"↓{abs(value):.4f}"
    return "0.0000"


def _load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not ABLATION_ROOT.exists():
        return rows
    for dataset_dir in sorted(path for path in ABLATION_ROOT.iterdir() if path.is_dir() and path.name != "plots"):
        dataset = dataset_dir.name.upper().replace("_", "-")
        for experiment_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            metrics_path = _find_metrics_file(experiment_dir)
            if metrics_path is None:
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "dataset": dataset,
                    "experiment": experiment_dir.name,
                    "method": DISPLAY_NAMES.get(experiment_dir.name, experiment_dir.name),
                    "F1_1": _metric(payload, "F1_1", "f1_1", "f1"),
                    "IoU_1": _metric(payload, "IoU_1", "iou_1", "iou"),
                    "OA": _metric(payload, "OA", "oa"),
                    "Precision_1": _metric(payload, "Precision_1", "precision_1", "precision"),
                    "Recall_1": _metric(payload, "Recall_1", "recall_1", "recall"),
                    "Boundary_F1": _metric(payload, "Boundary F1", "boundary_f1"),
                    "best_threshold": _metric(payload, "best_threshold", "threshold"),
                    "eval_file": _safe_rel(metrics_path),
                }
            )
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    for dataset_rows in grouped.values():
        baseline = next((row for row in dataset_rows if row["experiment"] == "baseline"), None)
        baseline_f1 = None if baseline is None else baseline["F1_1"]
        for row in dataset_rows:
            row["delta_F1"] = None if baseline_f1 is None or row["F1_1"] is None else row["F1_1"] - baseline_f1
        dataset_rows.sort(key=lambda item: (item["experiment"] != "baseline", item["method"]))
    return grouped


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            for key in ["F1_1", "IoU_1", "OA", "Precision_1", "Recall_1", "Boundary_F1", "best_threshold", "delta_F1"]:
                if isinstance(serializable.get(key), float):
                    serializable[key] = round(serializable[key], 6)
            writer.writerow({key: serializable.get(key) for key in CSV_COLUMNS})


def _build_md(grouped: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Ablation Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Ablation study evaluates the contribution of each component by removing or modifying it.",
        "",
    ]
    for dataset, rows in grouped.items():
        lines.extend([
            f"## {dataset}",
            "",
            "| Method | F1 | IoU | OA | ΔF1 |",
            "|--------|----|-----|----|-----|",
        ])
        for row in rows:
            delta = "—" if row["experiment"] == "baseline" else _format_delta(row["delta_F1"])
            lines.append(
                f"| {row['method']} | {_metric_text(row['F1_1'])} | {_metric_text(row['IoU_1'])} | {_metric_text(row['OA'])} | {delta} |"
            )
        ablations = [row for row in rows if row["experiment"] != "baseline" and row["delta_F1"] is not None]
        if ablations:
            biggest = min(ablations, key=lambda item: item["delta_F1"])
            least = max(ablations, key=lambda item: item["delta_F1"])
            lines.extend([
                "",
                f"Most important module: {biggest['method']} ({_format_delta(biggest['delta_F1'])}).",
                f"Least important module: {least['method']} ({_format_delta(least['delta_F1'])}).",
            ])
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_tex(grouped: dict[str, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for dataset, rows in grouped.items():
        lines.extend([
            f"% {dataset}",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Method & F1 & IoU & OA & $\\Delta$F1 \\\\",
            "\\hline",
        ])
        for row in rows:
            delta = "--" if row["experiment"] == "baseline" else _format_delta(row["delta_F1"])
            lines.append(
                f"{row['method']} & {_metric_text(row['F1_1'])} & {_metric_text(row['IoU_1'])} & {_metric_text(row['OA'])} & {delta} \\\\")
        lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines) + "\n"


def main() -> None:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    WEBSITE_TABLES.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    grouped = _group_rows(rows)

    csv_path = ABLATION_ROOT / "ablation_summary.csv"
    md_path = ABLATION_ROOT / "ablation_summary.md"
    tex_path = ABLATION_ROOT / "ablation_summary.tex"
    website_csv_path = WEBSITE_TABLES / "ablation_summary.csv"

    _write_csv(csv_path, rows)
    md_path.write_text(_build_md(grouped), encoding="utf-8")
    tex_path.write_text(_build_tex(grouped), encoding="utf-8")
    shutil.copy2(csv_path, website_csv_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {website_csv_path}")


if __name__ == "__main__":
    main()