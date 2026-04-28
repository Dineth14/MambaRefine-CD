#!/usr/bin/env python3
"""Validate Mamba-CD protocol assets and website wiring.

No CLI args.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
WEBSITE_DATA = WEBSITE / "assets" / "data"
OUTPUT_DIR = ROOT / "outputs" / "website_validation"
DATASETS = ["LEVIR-CD", "WHU-CD", "DSIFN-CD"]
METRIC_COLUMNS = ["Pre (%)", "Rec (%)", "F1 (%)", "IoU (%)", "OA (%)"]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"tbd", "null", "none", "nan", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_valid_missing(value: Any) -> bool:
    return value is None or value == "TBD"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ours_path = WEBSITE_DATA / "mambacd_protocol_ours.json"
    paper_path = WEBSITE_DATA / "mambacd_paper_comparison.json"
    website_index = WEBSITE / "index.html"

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checks": {},
        "datasets": {},
        "errors": [],
        "warnings": [],
    }

    report["checks"]["mambacd_protocol_ours_exists"] = ours_path.exists()
    report["checks"]["mambacd_paper_comparison_exists"] = paper_path.exists()
    report["checks"]["website_index_exists"] = website_index.exists()

    if not (ours_path.exists() and paper_path.exists() and website_index.exists()):
        missing = [
            str(path.relative_to(ROOT))
            for path in (ours_path, paper_path, website_index)
            if not path.exists()
        ]
        report["errors"].append(f"Missing required files: {', '.join(missing)}")
    else:
        ours_payload = _load_json(ours_path)
        paper_payload = _load_json(paper_path)
        index_html = website_index.read_text(encoding="utf-8")

        mapping_tokens = ["mambacd-mapping-table", "precision_1", "recall_1", "F1_1", "IoU_1", "OA"]
        report["checks"]["website_contains_metric_mapping_table"] = all(token in index_html for token in mapping_tokens)
        if not report["checks"]["website_contains_metric_mapping_table"]:
            report["errors"].append("Website metric mapping table is missing required keys.")

        report["checks"]["all_datasets_present"] = True
        report["checks"]["ours_rows_use_f1_1_not_mf1"] = True
        report["checks"]["comparison_values_are_percent_format"] = True
        report["checks"]["missing_values_are_tbd_or_null"] = True

        ours_results = ours_payload.get("results", {})
        paper_datasets = paper_payload.get("datasets", {})

        for dataset in DATASETS:
            dataset_errors: list[str] = []
            ours_row = ours_results.get(dataset)
            table_rows = paper_datasets.get(dataset, [])
            dataset_report = {
                "ours_result_found": ours_row is not None,
                "table_row_count": len(table_rows),
                "ours_table_row_found": False,
                "selected_split": ours_row.get("split") if isinstance(ours_row, dict) else None,
                "selected_source": ours_row.get("source_file") if isinstance(ours_row, dict) else None,
                "errors": dataset_errors,
            }

            if ours_row is None or not table_rows:
                report["checks"]["all_datasets_present"] = False
                dataset_errors.append("Missing selected result or table rows for dataset.")
            else:
                ours_table_row = next((row for row in table_rows if row.get("Method") == "MambaRefine-CD"), None)
                dataset_report["ours_table_row_found"] = ours_table_row is not None
                if ours_table_row is None:
                    report["checks"]["all_datasets_present"] = False
                    dataset_errors.append("Missing MambaRefine-CD row in comparison table.")
                else:
                    expected_pairs = {
                        "Pre (%)": round((_to_float(ours_row.get("precision_1")) or 0.0) * 100.0, 2) if _to_float(ours_row.get("precision_1")) is not None else "TBD",
                        "Rec (%)": round((_to_float(ours_row.get("recall_1")) or 0.0) * 100.0, 2) if _to_float(ours_row.get("recall_1")) is not None else "TBD",
                        "F1 (%)": round((_to_float(ours_row.get("F1_1")) or 0.0) * 100.0, 2) if _to_float(ours_row.get("F1_1")) is not None else "TBD",
                        "IoU (%)": round((_to_float(ours_row.get("IoU_1")) or 0.0) * 100.0, 2) if _to_float(ours_row.get("IoU_1")) is not None else "TBD",
                        "OA (%)": round((_to_float(ours_row.get("OA")) or 0.0) * 100.0, 2) if _to_float(ours_row.get("OA")) is not None else "TBD",
                    }
                    for column, expected in expected_pairs.items():
                        actual_numeric = _to_float(ours_table_row.get(column))
                        if isinstance(expected, str):
                            if ours_table_row.get(column) != "TBD":
                                report["checks"]["missing_values_are_tbd_or_null"] = False
                                dataset_errors.append(f"Expected {column} to be TBD for {dataset}.")
                        else:
                            if actual_numeric is None or abs(actual_numeric - expected) > 1e-6:
                                report["checks"]["ours_rows_use_f1_1_not_mf1"] = False
                                dataset_errors.append(f"{column} does not match extracted protocol metric for {dataset}.")

            for row in table_rows:
                for column in METRIC_COLUMNS:
                    value = row.get(column)
                    numeric = _to_float(value)
                    if numeric is not None:
                        if numeric <= 1.0 or numeric > 100.0:
                            report["checks"]["comparison_values_are_percent_format"] = False
                            dataset_errors.append(f"{row.get('Method')} {column} is not in percent format: {value}")
                        if numeric == 0.0:
                            report["checks"]["missing_values_are_tbd_or_null"] = False
                            dataset_errors.append(f"{row.get('Method')} {column} uses fake zero instead of TBD/null.")
                    elif not _is_valid_missing(value):
                        report["checks"]["missing_values_are_tbd_or_null"] = False
                        dataset_errors.append(f"{row.get('Method')} {column} uses invalid missing value {value!r}.")

            report["datasets"][dataset] = dataset_report
            report["errors"].extend(f"{dataset}: {message}" for message in dataset_errors)

    status = "PASS" if not report["errors"] else "FAIL"
    report["status"] = status

    json_path = OUTPUT_DIR / "mambacd_protocol_validation.json"
    md_path = OUTPUT_DIR / "mambacd_protocol_validation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Mamba-CD Protocol Validation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Status: **{status}**",
        "",
        "## Checks",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Dataset Summary"])
    for dataset in DATASETS:
        info = report["datasets"].get(dataset, {})
        lines.append(f"- `{dataset}`: rows=`{info.get('table_row_count')}`, ours_row=`{info.get('ours_table_row_found')}`, split=`{info.get('selected_split')}`")
    lines.extend(["", "## Errors"])
    if report["errors"]:
        lines.extend([f"- {item}" for item in report["errors"]])
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings"])
    if report["warnings"]:
        lines.extend([f"- {item}" for item in report["warnings"]])
    else:
        lines.append("- None")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(status)


if __name__ == "__main__":
    main()
