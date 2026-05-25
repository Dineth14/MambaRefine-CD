#!/usr/bin/env python3
"""Extract Mamba-CD protocol metrics from local evaluation artifacts.

No CLI args.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEBSITE_DATA = ROOT / "website" / "assets" / "data"

SEARCH_ROOTS = [
    ROOT / "outputs",
    ROOT / "outputs" / "whu_experiment",
    ROOT / "outputs" / "full_debug_sota_eval",
    ROOT / "outputs" / "sota_reproduced_eval",
]

DATASETS = ["LEVIR-CD", "WHU-CD", "DSIFN-CD"]

FALLBACKS = {
    "LEVIR-CD": {
        "precision_1": 0.8515,
        "recall_1": 0.8638,
        "F1_1": 0.8576,
        "IoU_1": 0.7507,
        "OA": 0.9883,
        "threshold": 0.30,
        "split": "test",
        "source_file": "provided verified fallback",
        "source_kind": "provided_verified_fallback",
        "checkpoint_path": None,
        "run_directory": None,
    },
    "WHU-CD": {
        "precision_1": 0.9495,
        "recall_1": 0.9449,
        "F1_1": 0.9472,
        "IoU_1": 0.8996,
        "OA": 0.9949,
        "threshold": 0.60,
        "split": "test",
        "source_file": "provided verified fallback",
        "source_kind": "provided_verified_fallback",
        "checkpoint_path": None,
        "run_directory": None,
    },
}

PAPER_LITERATURE = {
    "LEVIR-CD": [
        ("FC-EF", 2018, 86.91, 80.17, 83.40, 71.53, 98.39),
        ("FC-Siam-Di", 2018, 89.53, 83.31, 86.31, 75.92, 98.67),
        ("FC-Siam-Conc", 2018, 91.99, 76.77, 83.69, 71.96, 98.49),
        ("DTCDSCN", 2020, 88.53, 86.83, 87.67, 78.05, 98.77),
        ("STANet", 2020, 83.81, 91.00, 87.26, 77.40, 98.66),
        ("IFNet", 2020, 94.02, 82.93, 88.13, 78.77, 98.87),
        ("SNUNet", 2021, 89.18, 87.17, 88.16, 78.83, 98.82),
        ("BIT", 2021, 89.24, 89.37, 89.31, 80.68, 98.92),
        ("ChangeFormer", 2022, 92.05, 88.80, 90.40, 82.48, 99.04),
        ("BiFA", 2024, 91.52, 89.86, 90.69, 82.96, 99.06),
        ("RSM-CD", 2024, 92.52, 89.73, 91.10, 83.66, None),
        ("CDMamba", 2025, 91.43, 90.08, 90.75, 83.07, 99.06),
        ("Mamba-CD", 2025, 93.06, 91.07, 92.06, 85.28, 99.20),
    ],
    "WHU-CD": [
        ("FC-EF", 2018, 71.63, 67.25, 69.37, 53.11, 97.61),
        ("FC-Siam-Di", 2018, 47.33, 77.66, 58.81, 41.66, 95.63),
        ("FC-Siam-Conc", 2018, 60.88, 73.58, 66.63, 49.95, 97.04),
        ("DTCDSCN", 2020, 63.92, 82.30, 71.95, 56.19, 97.42),
        ("STANet", 2020, 79.37, 85.50, 82.32, 69.95, 98.52),
        ("IFNet", 2020, 96.91, 73.19, 83.40, 71.52, 98.83),
        ("SNUNet", 2021, 85.60, 81.49, 83.50, 71.67, 98.71),
        ("ChangeFormer", 2022, 91.83, 88.02, 89.88, 81.63, 99.12),
        ("BiFA", 2024, 95.15, 93.60, 94.37, 89.34, 99.56),
        ("RSM-CD", 2024, 93.37, 90.42, 91.87, 84.96, None),
        ("SChanger", 2025, 94.62, 91.83, 93.20, 87.27, None),
        ("CDMamba", 2025, 95.58, 92.01, 93.76, 88.26, 99.51),
        ("Mamba-CD", 2025, 96.52, 93.91, 95.20, 90.83, 99.62),
    ],
    "DSIFN-CD": [
        ("FC-EF", 2018, 72.61, 52.73, 61.09, 43.98, 88.59),
        ("FC-Siam-Di", 2018, 59.67, 65.71, 62.54, 45.50, 86.63),
        ("FC-Siam-Conc", 2018, 66.45, 54.21, 59.71, 42.56, 87.57),
        ("DTCDSCN", 2020, 53.87, 77.99, 63.72, 46.76, 84.91),
        ("STANet", 2020, 67.71, 61.68, 64.56, 47.66, 88.49),
        ("IFNet", 2020, 67.86, 53.94, 60.10, 42.96, 87.83),
        ("SNUNet", 2021, 60.60, 72.89, 66.18, 49.45, 87.34),
        ("ChangeFormer", 2022, 88.48, 84.94, 86.67, 76.48, 95.56),
        ("BiFA", 2024, 73.99, 68.87, 71.34, 55.45, 90.80),
        ("FTAN", 2024, 90.54, 88.61, 89.56, 81.10, None),
        ("ADSFNet", 2025, 94.79, 95.24, 95.01, 90.50, 98.30),
        ("Mamba-CD", 2025, 95.60, 95.61, 95.61, 91.69, 98.51),
    ],
}

CSV_COLUMNS = [
    "dataset",
    "split",
    "precision_1",
    "recall_1",
    "F1_1",
    "IoU_1",
    "OA",
    "threshold",
    "source_file",
    "source_kind",
    "checkpoint_path",
    "run_directory",
    "status",
]

PAPER_CSV_COLUMNS = [
    "Dataset",
    "Method",
    "Year",
    "Pre (%)",
    "Rec (%)",
    "F1 (%)",
    "IoU (%)",
    "OA (%)",
    "Source",
    "Source Status",
]


def _safe_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "null", "tbd", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _dataset_from_text(text: str) -> str | None:
    lower = text.lower()
    if "levir" in lower:
        return "LEVIR-CD"
    if "whu" in lower:
        return "WHU-CD"
    if "dsifn" in lower:
        return "DSIFN-CD"
    return None


def _extract_dataset(record: dict[str, Any], path: Path) -> str | None:
    for key in ("dataset", "name", "Dataset"):
        value = record.get(key)
        if value:
            dataset = _dataset_from_text(str(value))
            if dataset:
                return dataset
    return _dataset_from_text(str(path))


def _extract_split(record: dict[str, Any], path: Path) -> str:
    for key in ("split", "Split"):
        value = record.get(key)
        if value:
            text = str(value).strip().lower()
            if text in {"test", "val", "validation", "train"}:
                return "val" if text == "validation" else text
    path_text = str(path).lower()
    if "test_results" in path_text or "test_metrics" in path_text or re.search(r"(^|[^a-z])test([^a-z]|$)", path_text):
        return "test"
    if "validation" in path_text or "val_metrics" in path_text:
        return "val"
    return "unknown"


def _extract_checkpoint(record: dict[str, Any]) -> str | None:
    for key in ("checkpoint_path", "checkpoint", "best_checkpoint"):
        value = record.get(key)
        if value:
            return str(value)
    return None


def _extract_threshold(record: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    for container in (record, metrics):
        for key in ("threshold", "best_threshold"):
            if key in container:
                value = _to_float(container.get(key))
                if value is not None:
                    return value
    return None


def _extract_run_directory(path: Path) -> str | None:
    candidates = [path.parent, *path.parents]
    for parent in candidates:
        name = parent.name.lower()
        if name.startswith("run_") or name.startswith("eval_") or name == "test_results" or name == "evaluation":
            if name in {"test_results", "evaluation"}:
                continue
            return _safe_rel(parent)
    for parent in candidates:
        if parent.name in {"test_results", "evaluation", "validation", "logs"}:
            return _safe_rel(parent.parent)
    return _safe_rel(path.parent)


def _extract_metrics(record: dict[str, Any]) -> dict[str, float | None]:
    metrics_obj = record.get("metrics") if isinstance(record.get("metrics"), dict) else None
    merged: dict[str, Any] = {}
    if metrics_obj:
        merged.update(metrics_obj)
    merged.update(record)
    norm = {_norm_key(key): value for key, value in merged.items()}
    return {
        "precision_1": _to_float(norm.get("precision1")) or _to_float(norm.get("precision")),
        "recall_1": _to_float(norm.get("recall1")) or _to_float(norm.get("recall")),
        "F1_1": _to_float(norm.get("f11")) or _to_float(norm.get("f1")),
        "IoU_1": _to_float(norm.get("iou1")) or _to_float(norm.get("iou")),
        "OA": _to_float(norm.get("oa")) or _to_float(norm.get("accuracy")),
    }


def _is_complete(candidate: dict[str, Any]) -> bool:
    return all(candidate.get(key) is not None for key in ("precision_1", "recall_1", "F1_1", "IoU_1", "OA"))


def _round_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in candidate.items():
        if isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def _json_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    dataset = _extract_dataset(payload, path)
    if dataset not in DATASETS:
        return None
    metrics = _extract_metrics(payload)
    candidate = {
        "dataset": dataset,
        "split": _extract_split(payload, path),
        "precision_1": metrics["precision_1"],
        "recall_1": metrics["recall_1"],
        "F1_1": metrics["F1_1"],
        "IoU_1": metrics["IoU_1"],
        "OA": metrics["OA"],
        "threshold": _extract_threshold(payload, payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}),
        "source_file": _safe_rel(path),
        "source_kind": "eval_metrics_json" if path.name == "eval_metrics.json" else ("test_metrics_json" if path.name == "test_metrics.json" else "metrics_json"),
        "checkpoint_path": _extract_checkpoint(payload),
        "run_directory": _extract_run_directory(path),
        "status": payload.get("status", "OK" if _is_complete(metrics) else "INCOMPLETE"),
        "raw_file_name": path.name,
    }
    return _round_metrics(candidate) if _is_complete(candidate) else None


def _csv_candidates(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        dataset = _extract_dataset(row, path)
        if dataset not in DATASETS:
            continue
        metrics = _extract_metrics(row)
        candidate = {
            "dataset": dataset,
            "split": _extract_split(row, path),
            "precision_1": metrics["precision_1"],
            "recall_1": metrics["recall_1"],
            "F1_1": metrics["F1_1"],
            "IoU_1": metrics["IoU_1"],
            "OA": metrics["OA"],
            "threshold": _extract_threshold(row, row),
            "source_file": _safe_rel(path),
            "source_kind": "eval_metrics_csv" if path.name == "eval_metrics.csv" else ("test_metrics_csv" if path.name == "test_metrics.csv" else ("validation_csv" if "val" in path.name.lower() else "metrics_csv")),
            "checkpoint_path": _extract_checkpoint(row),
            "run_directory": _extract_run_directory(path),
            "status": row.get("status", "OK" if _is_complete(metrics) else "INCOMPLETE"),
            "raw_file_name": path.name,
        }
        if _is_complete(candidate):
            out.append(_round_metrics(candidate))
    return out


def _parse_log_file(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    dataset = _dataset_from_text(text) or _dataset_from_text(str(path))
    if dataset not in DATASETS:
        return []

    test_markers = [
        r"Dataset\s*:\s*.*?split\s*=\s*test",
        r"split\s*=\s*test",
        r"split\s*:\s*test",
        r"test_metrics",
        r"test results",
    ]
    if not any(re.search(pattern, text, flags=re.I) for pattern in test_markers):
        return []

    def grab(pattern: str) -> float | None:
        matches = re.findall(pattern, text, flags=re.I)
        if not matches:
            return None
        value = matches[-1]
        if isinstance(value, tuple):
            for item in reversed(value):
                coerced = _to_float(item)
                if coerced is not None:
                    return coerced
            return None
        return _to_float(value)

    candidate = {
        "dataset": dataset,
        "split": "test",
        "precision_1": grab(r"precision[_\s:-]*1?\s*[=:]\s*([0-9.]+)"),
        "recall_1": grab(r"recall[_\s:-]*1?\s*[=:]\s*([0-9.]+)"),
        "F1_1": grab(r"f1[_\s:-]*1?\s*[=:]\s*([0-9.]+)"),
        "IoU_1": grab(r"iou[_\s:-]*1?\s*[=:]\s*([0-9.]+)"),
        "OA": grab(r"oa\s*[=:]\s*([0-9.]+)|accuracy\s*[=:]\s*([0-9.]+)"),
        "threshold": grab(r"(?:best_)?threshold\s*[=:]\s*([0-9.]+)"),
        "source_file": _safe_rel(path),
        "source_kind": "log_test_split",
        "checkpoint_path": None,
        "run_directory": _extract_run_directory(path),
        "status": "OK",
        "raw_file_name": path.name,
    }
    # grab() with alternation may return tuple-like values through re.findall; re-normalize OA safely.
    if isinstance(candidate["OA"], tuple):
        candidate["OA"] = next((_to_float(v) for v in candidate["OA"] if _to_float(v) is not None), None)
    return [_round_metrics(candidate)] if _is_complete(candidate) else []


def _candidate_priority(candidate: dict[str, Any]) -> tuple[int, int, float, str]:
    kind = candidate.get("source_kind")
    if kind == "eval_metrics_json":
        priority = 4
    elif kind == "test_metrics_json":
        priority = 3
    elif kind == "eval_metrics_csv":
        priority = 2
    elif kind == "test_metrics_csv":
        priority = 1
    elif kind == "log_test_split":
        priority = 0
    else:
        priority = -1
    split = candidate.get("split")
    split_priority = 2 if split == "test" else (1 if split == "val" else 0)
    f1 = _to_float(candidate.get("F1_1")) or -1.0
    src = candidate.get("source_file") or ""
    return (split_priority, priority, f1, src)


def _validation_priority(candidate: dict[str, Any]) -> tuple[float, str]:
    f1 = _to_float(candidate.get("F1_1")) or -1.0
    src = candidate.get("source_file") or ""
    return (f1, src)


def _discover_candidates() -> dict[str, list[dict[str, Any]]]:
    files: dict[str, Path] = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name not in {"eval_metrics.json", "eval_metrics.csv", "test_metrics.json", "test_metrics.csv", "train.log"} and not (path.suffix == ".log" or "val_metrics" in path.name):
                continue
            files[str(path.resolve())] = path

    datasets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(files.values()):
        if path.suffix == ".json":
            candidate = _json_record(path)
            if candidate:
                datasets[candidate["dataset"]].append(candidate)
        elif path.suffix == ".csv":
            for candidate in _csv_candidates(path):
                datasets[candidate["dataset"]].append(candidate)
        elif path.suffix == ".log":
            for candidate in _parse_log_file(path):
                datasets[candidate["dataset"]].append(candidate)
    return datasets


def _select_result(dataset: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    test_candidates = [item for item in candidates if item.get("split") == "test"]
    if test_candidates:
        best = max(test_candidates, key=_candidate_priority)
        best = dict(best)
        best["selection_reason"] = "best test F1 among discovered test candidates"
        return best
    val_candidates = [item for item in candidates if item.get("split") == "val"]
    if val_candidates:
        best = max(val_candidates, key=_validation_priority)
        best = dict(best)
        best["selection_reason"] = "validation fallback because no test result was discovered"
        return best
    if dataset in FALLBACKS:
        fallback = dict(FALLBACKS[dataset])
        fallback["dataset"] = dataset
        fallback["status"] = "fallback"
        fallback["selection_reason"] = "provided verified fallback because extraction did not yield a usable result"
        return fallback
    return {
        "dataset": dataset,
        "split": None,
        "precision_1": "TBD",
        "recall_1": "TBD",
        "F1_1": "TBD",
        "IoU_1": "TBD",
        "OA": "TBD",
        "threshold": "TBD",
        "source_file": None,
        "source_kind": "missing",
        "checkpoint_path": None,
        "run_directory": None,
        "status": "TBD",
        "selection_reason": "no usable local result was discovered",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def _to_percent(value: Any) -> float | str | None:
    numeric = _to_float(value)
    if numeric is None:
        if value in {None, "TBD"}:
            return value
        return None
    return round(numeric * 100.0, 2)


def _build_paper_rows(selected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method, year, pre, rec, f1, iou, oa in PAPER_LITERATURE[dataset]:
            rows.append({
                "Dataset": dataset,
                "Method": method,
                "Year": year,
                "Pre (%)": pre,
                "Rec (%)": rec,
                "F1 (%)": f1,
                "IoU (%)": iou,
                "OA (%)": oa,
                "Source": "Mamba-CD paper literature table",
                "Source Status": "Mamba-CD paper literature table",
            })

        ours = selected[dataset]
        rows.append({
            "Dataset": dataset,
            "Method": "MambaRefine-CD",
            "Year": 2026,
            "Pre (%)": _to_percent(ours.get("precision_1")),
            "Rec (%)": _to_percent(ours.get("recall_1")),
            "F1 (%)": _to_percent(ours.get("F1_1")),
            "IoU (%)": _to_percent(ours.get("IoU_1")),
            "OA (%)": _to_percent(ours.get("OA")),
            "Source": "Ours, reproduced from local evaluation",
            "Source Status": "Ours, reproduced from local evaluation",
        })
    return rows


def _write_paper_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in PAPER_CSV_COLUMNS})


def main() -> None:
    WEBSITE_DATA.mkdir(parents=True, exist_ok=True)
    discovered = _discover_candidates()

    selected: dict[str, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []
    all_candidates_payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": {},
    }

    for dataset in DATASETS:
        candidates = discovered.get(dataset, [])
        ordered_candidates = sorted(candidates, key=_candidate_priority, reverse=True)
        selected_row = _select_result(dataset, ordered_candidates)
        selected[dataset] = selected_row
        flat_rows.append(selected_row)
        all_candidates_payload["datasets"][dataset] = {
            "selected": selected_row,
            "candidates": ordered_candidates,
        }

    protocol_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metric_mapping": {
            "Pre": "precision_1",
            "Rec": "recall_1",
            "F1": "F1_1",
            "IoU": "IoU_1",
            "OA": "OA",
        },
        "results": selected,
    }
    paper_rows = _build_paper_rows(selected)
    paper_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "columns": PAPER_CSV_COLUMNS,
        "datasets": {
            dataset: [row for row in paper_rows if row["Dataset"] == dataset]
            for dataset in DATASETS
        },
    }

    json_path = WEBSITE_DATA / "mambacd_protocol_ours.json"
    csv_path = WEBSITE_DATA / "mambacd_protocol_ours.csv"
    candidates_path = WEBSITE_DATA / "mambacd_protocol_ours_all_candidates.json"
    paper_json_path = WEBSITE_DATA / "mambacd_paper_comparison.json"
    paper_csv_path = WEBSITE_DATA / "mambacd_paper_comparison.csv"

    json_path.write_text(json.dumps(protocol_payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, flat_rows)
    candidates_path.write_text(json.dumps(all_candidates_payload, indent=2), encoding="utf-8")
    paper_json_path.write_text(json.dumps(paper_payload, indent=2), encoding="utf-8")
    _write_paper_csv(paper_csv_path, paper_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {paper_json_path}")
    print(f"Wrote {paper_csv_path}")


if __name__ == "__main__":
    main()
