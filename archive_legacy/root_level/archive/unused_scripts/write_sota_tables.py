#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sota_eval.table_writer import DATASETS, write_csv, write_json, write_latex, write_markdown

OUT_ROOT = ROOT / "outputs" / "sota_reproduced_eval"
TABLE_DIR = OUT_ROOT / "tables"
WEBSITE_DATA = ROOT / "website" / "assets" / "data"
OURS_EFF = WEBSITE_DATA / "ours_efficiency.json"


def _ours_efficiency() -> dict:
    if not OURS_EFF.exists():
        return {}
    try:
        return json.loads(OURS_EFF.read_text(encoding="utf-8")).get("metrics", {})
    except Exception:
        return {}


def _metric_row(model_name: str, dataset_name: str, payload: dict, status_payload: dict | None, ours_eff: dict) -> dict:
    metrics = payload.get("metrics", {}) if payload else {}
    status = (status_payload or {}).get("status", payload.get("status", "FAILED") if payload else "FAILED")
    checkpoint = payload.get("checkpoint_path") if payload else None
    params_m = ours_eff.get("total_params_millions", "") if model_name == "MambaRefine-CD" else ""
    flops_g = ours_eff.get("flops_gmac", "") if model_name == "MambaRefine-CD" else ""
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Params_M": params_m,
        "FLOPs_G": flops_g,
        "mF1": metrics.get("mF1", ""),
        "F1_1": metrics.get("F1_1", ""),
        "mIoU": metrics.get("mIoU", ""),
        "IoU_1": metrics.get("IoU_1", ""),
        "Precision_1": metrics.get("Precision_1", ""),
        "Recall_1": metrics.get("Recall_1", ""),
        "OA": metrics.get("OA", ""),
        "Boundary_F1": metrics.get("Boundary F1", ""),
        "Edge_IoU": metrics.get("Edge IoU", ""),
        "Threshold": metrics.get("best_threshold", ""),
        "Status": status,
        "Source": checkpoint or (status_payload or {}).get("reason", ""),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    WEBSITE_DATA.mkdir(parents=True, exist_ok=True)
    dataset_rows = {dataset: [] for dataset in DATASETS}
    reproduced_records = []
    ours_eff = _ours_efficiency()

    for model_dir in sorted(OUT_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name in {"logs", "tables", "reports"}:
            continue
        for dataset_name in DATASETS:
            run_dir = model_dir / dataset_name
            eval_path = run_dir / "eval_metrics.json"
            status_path = run_dir / "status.json"
            payload = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None
            status_payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
            row = _metric_row(model_dir.name, dataset_name, payload, status_payload, ours_eff)
            dataset_rows[dataset_name].append(row)
            reproduced_records.append(row)

    for dataset_name, rows in dataset_rows.items():
        slug = dataset_name.lower().replace("-", "_")
        write_csv(TABLE_DIR / f"{slug}_comparison.csv", rows)
        write_markdown(TABLE_DIR / f"{slug}_comparison.md", rows)
    all_rows = [row for rows in dataset_rows.values() for row in rows]
    write_csv(TABLE_DIR / "all_binary_cd_comparison.csv", all_rows)
    write_latex(TABLE_DIR / "paper_table_binary_cd.tex", all_rows)

    website_json = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "records": reproduced_records,
    }
    write_json(WEBSITE_DATA / "reproduced_sota_results.json", website_json)
    with (WEBSITE_DATA / "reproduced_sota_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reproduced_records[0].keys()) if reproduced_records else [])
        if reproduced_records:
            writer.writeheader()
            writer.writerows(reproduced_records)
    print(f"Wrote {TABLE_DIR}")
    print(f"Wrote {WEBSITE_DATA / 'reproduced_sota_results.json'}")


if __name__ == "__main__":
    main()
