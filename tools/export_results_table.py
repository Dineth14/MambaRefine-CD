"""Export experiment results to a formatted table.

Reads all metrics.json files under the outputs/ folder and writes
a consolidated CSV and Markdown table.

Columns:
  - Binary CD (LEVIR/WHU/DSIFN): Dataset | Experiment | Pre | Rec | F1 | IoU | OA | Params(M) | FLOPs(G)
  - SECOND:                       Dataset | Experiment | OA  | mIoU | SeK | Fscd | Params(M) | FLOPs(G)

Usage:
    python tools/export_results_table.py --outputs_root outputs/
    python tools/export_results_table.py --outputs_root outputs/ --format markdown
    python tools/export_results_table.py --outputs_root outputs/levir/  --out results/levir_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_BINARY_COLS = ["Pre", "Rec", "F1", "IoU", "OA"]
_SECOND_COLS = ["OA", "mIoU", "SeK", "Fscd"]

_SECOND_DATASETS = {"second", "scd"}


def _is_second(record: dict) -> bool:
    return record.get("dataset", "").lower() in _SECOND_DATASETS


def _cols_for(record: dict) -> list[str]:
    return _SECOND_COLS if _is_second(record) else _BINARY_COLS


def _load_record(metrics_path: Path) -> dict:
    """Load metrics.json and optional params_flops.json from the same directory."""
    record: dict = {}
    with open(metrics_path) as f:
        data = json.load(f)
    record.update(data)

    # Try params_flops.json
    pf_path = metrics_path.parent / "params_flops.json"
    if pf_path.exists():
        with open(pf_path) as f:
            pf = json.load(f)
        record["Params(M)"] = round(pf.get("params_M", float("nan")), 2)
        record["FLOPs(G)"]  = round(pf.get("flops_G", float("nan")), 2)
    else:
        record["Params(M)"] = "N/A"
        record["FLOPs(G)"]  = "N/A"

    # Infer dataset and experiment from path
    parts = metrics_path.parts
    # Heuristic: outputs/<dataset>/<experiment>/metrics.json
    if len(parts) >= 4:
        record.setdefault("dataset",    parts[-3])
        record.setdefault("experiment", parts[-2])
    else:
        record.setdefault("dataset",    "unknown")
        record.setdefault("experiment", metrics_path.parent.name)

    return record


def _to_row(record: dict, cols: list[str]) -> list[str]:
    row = [record.get("dataset", ""), record.get("experiment", "")]
    for c in cols:
        val = record.get(c, "N/A")
        if isinstance(val, float):
            row.append(f"{val:.2f}")
        else:
            row.append(str(val))
    row.append(str(record.get("Params(M)", "N/A")))
    row.append(str(record.get("FLOPs(G)", "N/A")))
    return row


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export results to a table.")
    parser.add_argument("--outputs_root", default="outputs/")
    parser.add_argument("--out",    type=str, default=None)
    parser.add_argument("--format", choices=["csv", "markdown"], default="csv")
    parser.add_argument("--split",  default="test", help="Which metrics file suffix to look for.")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    suffix = f"metrics_{args.split}.json"

    # Collect all matching metrics files
    metrics_files = sorted(root.rglob(suffix))
    # Also look for plain metrics.json
    if not metrics_files:
        metrics_files = sorted(root.rglob("metrics.json"))

    if not metrics_files:
        logger.warning(f"No metrics files found under {root}")
        return

    records = []
    for mf in metrics_files:
        try:
            records.append(_load_record(mf))
        except Exception as e:
            logger.warning(f"Skipping {mf}: {e}")

    if not records:
        logger.warning("No valid records.")
        return

    # Separate binary and SECOND records
    binary_recs = [r for r in records if not _is_second(r)]
    second_recs = [r for r in records if _is_second(r)]

    def _write(recs: list[dict], metric_cols: list[str], label: str) -> None:
        if not recs:
            return
        headers = ["Dataset", "Experiment"] + metric_cols + ["Params(M)", "FLOPs(G)"]
        rows    = [_to_row(r, metric_cols) for r in recs]

        out_path = Path(args.out) if args.out else root / f"results_{label}.{args.format}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "csv":
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
        else:
            table = _markdown_table(headers, rows)
            with open(out_path, "w") as f:
                f.write(table + "\n")

        logger.info(f"Saved {label} table ({len(recs)} rows) → {out_path}")

    _write(binary_recs, _BINARY_COLS, "binary")
    _write(second_recs, _SECOND_COLS, "second")


if __name__ == "__main__":
    main()
