#!/usr/bin/env python3
"""Extract external SOTA numbers from local sources only.

Searches the local repo for model-specific files and emits conservative
structured records. Missing values remain TBD.
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

MODELS = {
    "ChangeFormer": ["changeformer"],
    "BIT": ["bit"],
    "SNUNet": ["snunet"],
    "STANet": ["stanet"],
    "Mamba-CD": ["mamba-cd", "mambacd"],
}

DATASETS = ["LEVIR-CD", "WHU-CD", "DSIFN-CD", "SECOND"]
ALLOWED_ROOTS = [
    ROOT / "external",
    ROOT / "external_weights",
    ROOT / "papers",
    ROOT / "docs",
    ROOT / "outputs" / "full_debug_sota_eval",
    ROOT / "website" / "assets" / "data",
]

METRICS_BY_DATASET = {
    "LEVIR-CD": ["params", "flops", "F1_1", "IoU_1", "OA"],
    "WHU-CD": ["params", "flops", "F1_1", "IoU_1", "OA"],
    "DSIFN-CD": ["params", "flops", "F1_1", "IoU_1", "OA"],
    "SECOND": ["params", "flops", "OA", "Fscd", "mIoU", "SeK"],
}


def _dataset_from_text(text: str) -> str | None:
    lower = text.lower()
    if "levir" in lower:
        return "LEVIR-CD"
    if "whu" in lower:
        return "WHU-CD"
    if "dsifn" in lower:
        return "DSIFN-CD"
    if "second" in lower:
        return "SECOND"
    return None


def _metric_definition(metric: str) -> str:
    return metric if metric in {"F1_1", "mF1", "OA", "IoU_1", "Fscd", "mIoU", "SeK"} else "unknown"


def _metric_regex(metric: str) -> re.Pattern[str]:
    aliases = {
        "params": [r"params?", r"parameters?"],
        "flops": [r"flops?", r"gflops?"],
        "F1_1": [r"f1_1", r"f1-1", r"f1change", r"changef1"],
        "IoU_1": [r"iou_1", r"iou-1", r"changeiou"],
        "OA": [r"\boa\b", r"overall accuracy"],
        "Fscd": [r"fscd"],
        "mIoU": [r"miou"],
        "SeK": [r"\bsek\b"],
    }
    pattern = "|".join(aliases[metric])
    return re.compile(rf"({pattern})[^0-9\-]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".json", ".csv", ".md", ".txt", ".log", ".tex"}:
        return "literature"
    if suffix == ".pdf":
        return "manual_needed"
    return "missing"


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def _allowed_files() -> list[Path]:
    files: list[Path] = []
    for root in ALLOWED_ROOTS:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".txt", ".log", ".tex", ".pdf"}:
                files.append(path)
    return files


def _path_mentions_model(path: Path, aliases: list[str]) -> bool:
    parts = [part.lower() for part in path.parts]
    path_text = "/".join(parts)
    return any(alias in path_text for alias in aliases)


def _line_mentions_model(line: str, aliases: list[str]) -> bool:
    lower = line.lower()
    return any(alias in lower for alias in aliases)


def _line_mentions_dataset(line: str, dataset: str) -> bool:
    needle = dataset.lower().replace("-cd", "").replace("-", "")
    lower = line.lower().replace("-", "")
    return needle in lower


def _extract_from_text(path: Path, dataset: str, metric: str, aliases: list[str]) -> dict[str, Any]:
    entry = {
        "value": "TBD",
        "metric": metric,
        "dataset": dataset,
        "model": None,
        "source_file": None,
        "source_type": "missing",
        "metric_definition": _metric_definition(metric),
    }

    if path.suffix.lower() == ".pdf":
        entry["source_file"] = _safe_rel(path)
        entry["source_type"] = "manual_needed"
        return entry

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return entry

    model_scoped = _path_mentions_model(path, aliases)
    regex = _metric_regex(metric)
    lines = text.splitlines()

    for idx, line in enumerate(lines):
        context_lines = [line]
        if idx + 1 < len(lines):
            context_lines.append(lines[idx + 1])
        if idx > 0:
            context_lines.insert(0, lines[idx - 1])
        context = " ".join(context_lines)

        # Only trust model-specific local sources:
        # 1. files/directories already scoped to the model, or
        # 2. a local table/row line that explicitly mentions the model alias.
        if not model_scoped and not _line_mentions_model(context, aliases):
            continue
        if not _line_mentions_dataset(context, dataset):
            continue

        match = regex.search(context)
        if not match:
            continue

        value = match.group(2)
        entry["value"] = float(value)
        entry["source_file"] = _safe_rel(path)
        entry["source_type"] = _source_type(path)
        return entry

    return entry


def main() -> None:
    WEBSITE_DATA.mkdir(parents=True, exist_ok=True)
    files = _allowed_files()

    records: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = defaultdict(dict)

    for model, aliases in MODELS.items():
        model_files = [path for path in files if _path_mentions_model(path, aliases)]
        sources[model] = {
            "matched_files": [_safe_rel(path) for path in model_files],
            "notes": (
                "No model-scoped local files found. Generic docs are intentionally ignored."
                if not model_files
                else "Model-scoped local files found."
            ),
        }

        for dataset in DATASETS:
            for metric in METRICS_BY_DATASET[dataset]:
                best = {
                    "value": "TBD",
                    "metric": metric,
                    "dataset": dataset,
                    "model": model,
                    "source_file": None,
                    "source_type": "missing",
                    "metric_definition": _metric_definition(metric),
                }
                for path in model_files:
                    candidate = _extract_from_text(path, dataset, metric, aliases)
                    candidate["model"] = model
                    if candidate["source_type"] == "literature" and candidate["value"] != "TBD":
                        best = candidate
                        break
                    if candidate["source_type"] == "manual_needed" and best["source_type"] == "missing":
                        best = candidate
                records.append(best)

    json_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records": records,
    }
    (WEBSITE_DATA / "external_sota_results.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    (WEBSITE_DATA / "external_sota_sources.json").write_text(
        json.dumps(
            {
                "generated_at": json_payload["generated_at"],
                "sources": sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (WEBSITE_DATA / "external_sota_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "dataset", "metric", "value", "source_file", "source_type", "metric_definition"],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {WEBSITE_DATA / 'external_sota_results.json'}")
    print(f"Wrote {WEBSITE_DATA / 'external_sota_results.csv'}")
    print(f"Wrote {WEBSITE_DATA / 'external_sota_sources.json'}")


if __name__ == "__main__":
    main()
