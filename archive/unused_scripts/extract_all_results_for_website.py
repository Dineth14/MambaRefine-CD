#!/usr/bin/env python3
"""Extract verified MambaRefine-CD results for the website.

Scans local output folders and builds a normalized dataset-centric summary.
No CLI args.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
WEBSITE_DATA = ROOT / "website" / "assets" / "data"

DATASETS = ["LEVIR-CD", "WHU-CD", "DSIFN-CD", "SECOND"]

FALLBACK_VERIFIED = {
    "LEVIR-CD": {
        "split": "test",
        "mF1": 0.9258,
        "F1_1": 0.8576,
        "F1_0": 0.9939,
        "mIoU": 0.8693,
        "IoU_1": 0.7507,
        "IoU_0": 0.9879,
        "Precision_1": 0.8515,
        "Recall_1": 0.8638,
        "OA": 0.9883,
        "Boundary F1": 0.8632,
        "Edge IoU": 0.5899,
        "threshold": 0.30,
        "source_file": "provided verified log",
        "source_kind": "provided_verified_log",
    },
    "WHU-CD": {
        "split": "test",
        "mF1": 0.9723,
        "F1_1": 0.9472,
        "F1_0": 0.9973,
        "mIoU": 0.9472,
        "IoU_1": 0.8996,
        "IoU_0": 0.9947,
        "Precision_1": 0.9495,
        "Recall_1": 0.9449,
        "OA": 0.9949,
        "Boundary F1": 0.8966,
        "Edge IoU": 0.6533,
        "threshold": 0.60,
        "source_file": "provided verified log",
        "source_kind": "provided_verified_log",
    },
}

CSV_COLUMNS = [
    "dataset",
    "split",
    "checkpoint_path",
    "run_directory",
    "mF1",
    "F1_1",
    "F1_0",
    "mIoU",
    "IoU_1",
    "IoU_0",
    "Precision_1",
    "Recall_1",
    "OA",
    "Boundary F1",
    "Edge IoU",
    "threshold",
    "params",
    "FLOPs",
    "GPU memory",
    "FPS",
    "source_file",
    "source_kind",
]


def _round_or_none(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def _coerce_float(value: Any) -> float | None:
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
    if "second" in lower:
        return "SECOND"
    return None


def _find_dataset(candidate: dict[str, Any], path: Path) -> str | None:
    for key in ("dataset", "name"):
        if key in candidate:
            found = _dataset_from_text(str(candidate[key]))
            if found:
                return found
    return _dataset_from_text(str(path))


def _split_score(split: str) -> int:
    split = split.lower()
    if split == "test":
        return 3
    if split == "val":
        return 2
    if split == "train":
        return 1
    return 0


def _source_kind_score(kind: str) -> int:
    order = {
        "eval_metrics_json": 6,
        "test_metrics_json": 5,
        "eval_metrics_csv": 4,
        "test_metrics_csv": 3,
        "validation_csv": 2,
        "provided_verified_log": 1,
    }
    return order.get(kind, 0)


def _timestamp_from_path(path: Path) -> str | None:
    match = re.search(r"20\d{6}_\d{6}", str(path))
    return match.group(0) if match else None


def _safe_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def _find_related_model_info(result_path: Path, checkpoint_path: str | None) -> Path | None:
    for parent in [result_path.parent, *result_path.parents]:
        candidate = parent / "model_info.json"
        if candidate.exists():
            return candidate
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            ckpt = (ROOT / ckpt).resolve()
        run_dir = ckpt.parent.parent
        candidate = run_dir / "model_info.json"
        if candidate.exists():
            return candidate
    return None


def _find_related_efficiency(result_path: Path) -> Path | None:
    candidates = []
    ours_eff = WEBSITE_DATA / "ours_efficiency.json"
    latest_eff = OUTPUTS / "model_efficiency" / "latest_efficiency.json"
    for path in (ours_eff, latest_eff):
        if path.exists():
            candidates.append(path)
    return candidates[0] if candidates else None


def _parse_model_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}
    return {
        "params": _coerce_float(data.get("total_params")),
        "trainable_params": _coerce_float(data.get("trainable_params")),
        "model_variant": data.get("variant"),
        "model_info_source": _safe_rel(path),
    }


def _parse_efficiency(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}
    metrics = data.get("metrics", data)
    return {
        "FLOPs": metrics.get("flops_gmac")
        or metrics.get("flops")
        or metrics.get("gflops")
        or metrics.get("flops_text"),
        "GPU memory": metrics.get("peak_forward_memory_mb")
        or metrics.get("peak_train_step_memory_mb")
        or metrics.get("peak_memory_mb"),
        "FPS": metrics.get("fps") or metrics.get("throughput_fps") or metrics.get("images_per_second"),
        "efficiency_source": _safe_rel(path),
    }


def _normalize_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    norm = {_norm_key(k): v for k, v in raw.items()}
    out = {
        "mF1": _coerce_float(norm.get("mf1")),
        "F1_1": _coerce_float(norm.get("f11")) or _coerce_float(norm.get("f1")),
        "F1_0": _coerce_float(norm.get("f10")),
        "mIoU": _coerce_float(norm.get("miou")),
        "IoU_1": _coerce_float(norm.get("iou1")) or _coerce_float(norm.get("iou")),
        "IoU_0": _coerce_float(norm.get("iou0")),
        "Precision_1": _coerce_float(norm.get("precision1")) or _coerce_float(norm.get("precision")),
        "Recall_1": _coerce_float(norm.get("recall1")) or _coerce_float(norm.get("recall")),
        "OA": _coerce_float(norm.get("oa")),
        "Boundary F1": _coerce_float(norm.get("boundaryf1")),
        "Edge IoU": _coerce_float(norm.get("edgeiou")),
        "threshold": _coerce_float(norm.get("bestthreshold")) or _coerce_float(norm.get("threshold")),
        "Fscd": _coerce_float(norm.get("fscd")),
        "SeK": _coerce_float(norm.get("sek")),
    }
    return out


@dataclass
class Candidate:
    dataset: str
    split: str
    run_directory: str | None
    checkpoint_path: str | None
    source_file: str
    source_kind: str
    timestamp: str | None
    metrics: dict[str, Any]
    params: Any = None
    FLOPs: Any = None
    gpu_memory: Any = None
    fps: Any = None
    extra: dict[str, Any] | None = None

    def selection_key(self) -> tuple[int, int, float, str]:
        return (
            _split_score(self.split),
            _source_kind_score(self.source_kind),
            float(self.metrics.get("F1_1") or -1.0),
            self.timestamp or "",
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "dataset": self.dataset,
            "split": self.split,
            "run_directory": self.run_directory,
            "checkpoint_path": self.checkpoint_path,
            "source_file": self.source_file,
            "source_kind": self.source_kind,
            "timestamp": self.timestamp,
            "params": self.params,
            "FLOPs": self.FLOPs,
            "GPU memory": self.gpu_memory,
            "FPS": self.fps,
        }
        payload.update({k: _round_or_none(v) for k, v in self.metrics.items()})
        if self.extra:
            payload.update(self.extra)
        return payload


def _build_candidate_from_json(path: Path, source_kind: str) -> Candidate | None:
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    metrics = data.get("metrics", data)
    if not isinstance(metrics, dict):
        return None
    dataset = _find_dataset(data, path)
    if not dataset:
        return None
    split = str(data.get("split") or ("test" if "test_results" in str(path) else "val")).lower()
    checkpoint_path = data.get("checkpoint")
    run_dir = path.parent
    if "test_results" in str(path):
        run_dir = path.parent.parent
    model_info = _parse_model_info(_find_related_model_info(path, checkpoint_path))
    efficiency = _parse_efficiency(_find_related_efficiency(path))
    normalized_metrics = _normalize_metrics(metrics)
    if normalized_metrics.get("threshold") is None:
        normalized_metrics["threshold"] = _coerce_float(data.get("threshold")) or _coerce_float(data.get("best_threshold"))

    return Candidate(
        dataset=dataset,
        split=split,
        run_directory=_safe_rel(run_dir),
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        source_file=_safe_rel(path) or str(path),
        source_kind=source_kind,
        timestamp=str(data.get("timestamp") or _timestamp_from_path(path) or ""),
        metrics=normalized_metrics,
        params=model_info.get("params"),
        FLOPs=efficiency.get("FLOPs"),
        gpu_memory=efficiency.get("GPU memory"),
        fps=efficiency.get("FPS"),
    )


def _build_candidates_from_csv(path: Path, source_kind: str) -> list[Candidate]:
    rows = _read_csv_rows(path)
    candidates: list[Candidate] = []
    for row in rows:
        dataset = _find_dataset(row, path)
        if not dataset:
            continue
        split = str(row.get("split") or ("test" if "test_results" in str(path) else "val")).lower()
        checkpoint_path = row.get("checkpoint") or row.get("checkpoint_path")
        run_dir = path.parent
        if "test_results" in str(path):
            run_dir = path.parent.parent
        model_info = _parse_model_info(_find_related_model_info(path, checkpoint_path))
        efficiency = _parse_efficiency(_find_related_efficiency(path))
        timestamp = row.get("timestamp") or _timestamp_from_path(path) or ""
        if "iteration" in row and row.get("iteration"):
            timestamp = f"{timestamp}_iter_{row['iteration']}"
        candidates.append(
            Candidate(
                dataset=dataset,
                split=split,
                run_directory=_safe_rel(run_dir),
                checkpoint_path=checkpoint_path,
                source_file=_safe_rel(path) or str(path),
                source_kind=source_kind,
                timestamp=str(timestamp),
                metrics=_normalize_metrics(row),
                params=model_info.get("params"),
                FLOPs=efficiency.get("FLOPs"),
                gpu_memory=efficiency.get("GPU memory"),
                fps=efficiency.get("FPS"),
                extra={"iteration": _coerce_float(row.get("iteration"))},
            )
        )
    return candidates


def _fallback_candidate(dataset: str) -> Candidate:
    fallback = FALLBACK_VERIFIED[dataset]
    return Candidate(
        dataset=dataset,
        split=str(fallback["split"]),
        run_directory=None,
        checkpoint_path=None,
        source_file=str(fallback["source_file"]),
        source_kind=str(fallback["source_kind"]),
        timestamp=None,
        metrics={k: v for k, v in fallback.items() if k in {"mF1", "F1_1", "F1_0", "mIoU", "IoU_1", "IoU_0", "Precision_1", "Recall_1", "OA", "Boundary F1", "Edge IoU", "threshold"}},
        params=None,
    )


def main() -> None:
    WEBSITE_DATA.mkdir(parents=True, exist_ok=True)

    patterns = {
        "eval_metrics_json": list(OUTPUTS.rglob("eval_metrics.json")),
        "test_metrics_json": list(OUTPUTS.rglob("test_metrics.json")),
        "eval_metrics_csv": list(OUTPUTS.rglob("eval_metrics.csv")),
        "test_metrics_csv": list(OUTPUTS.rglob("test_metrics.csv")),
        "validation_csv": list(OUTPUTS.rglob("val_metrics.csv")),
    }

    all_candidates: dict[str, list[Candidate]] = defaultdict(list)

    for path in patterns["eval_metrics_json"]:
        candidate = _build_candidate_from_json(path, "eval_metrics_json")
        if candidate:
            all_candidates[candidate.dataset].append(candidate)
    for path in patterns["test_metrics_json"]:
        candidate = _build_candidate_from_json(path, "test_metrics_json")
        if candidate:
            all_candidates[candidate.dataset].append(candidate)
    for path in patterns["eval_metrics_csv"]:
        for candidate in _build_candidates_from_csv(path, "eval_metrics_csv"):
            all_candidates[candidate.dataset].append(candidate)
    for path in patterns["test_metrics_csv"]:
        for candidate in _build_candidates_from_csv(path, "test_metrics_csv"):
            all_candidates[candidate.dataset].append(candidate)
    for path in patterns["validation_csv"]:
        for candidate in _build_candidates_from_csv(path, "validation_csv"):
            all_candidates[candidate.dataset].append(candidate)

    best_results: dict[str, dict[str, Any]] = {}
    all_payload: dict[str, list[dict[str, Any]]] = {}

    for dataset in DATASETS:
        candidates = all_candidates.get(dataset, [])
        if not candidates and dataset in FALLBACK_VERIFIED:
            candidates = [_fallback_candidate(dataset)]
        candidates = sorted(candidates, key=lambda item: item.selection_key(), reverse=True)
        if candidates:
            best_results[dataset] = candidates[0].as_dict()
        else:
            best_results[dataset] = {
                "dataset": dataset,
                "split": None,
                "run_directory": None,
                "checkpoint_path": None,
                "source_file": None,
                "source_kind": "missing",
                "timestamp": None,
                "mF1": None,
                "F1_1": None,
                "F1_0": None,
                "mIoU": None,
                "IoU_1": None,
                "IoU_0": None,
                "Precision_1": None,
                "Recall_1": None,
                "OA": None,
                "Boundary F1": None,
                "Edge IoU": None,
                "threshold": None,
                "params": None,
                "FLOPs": None,
                "GPU memory": None,
                "FPS": None,
            }
        all_payload[dataset] = [candidate.as_dict() for candidate in candidates]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": best_results,
        "summary_cards": {
            "best_whu_f1_1": best_results["WHU-CD"].get("F1_1"),
            "best_whu_boundary_f1": best_results["WHU-CD"].get("Boundary F1"),
            "levir_f1_1": best_results["LEVIR-CD"].get("F1_1"),
            "params_millions": _round_or_none(
                (
                    (_coerce_float(best_results["WHU-CD"].get("params")) or _coerce_float(best_results["LEVIR-CD"].get("params")))
                    / 1e6
                )
                if (_coerce_float(best_results["WHU-CD"].get("params")) or _coerce_float(best_results["LEVIR-CD"].get("params")))
                else None
            ),
            "params_source": best_results["WHU-CD"].get("source_file") or best_results["LEVIR-CD"].get("source_file"),
        },
    }

    (WEBSITE_DATA / "ours_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (WEBSITE_DATA / "ours_results_all_candidates.json").write_text(
        json.dumps(
            {
                "generated_at": payload["generated_at"],
                "candidates": all_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (WEBSITE_DATA / "ours_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for dataset in DATASETS:
            row = best_results[dataset]
            writer.writerow(
                {
                    "dataset": dataset,
                    "split": row.get("split"),
                    "checkpoint_path": row.get("checkpoint_path"),
                    "run_directory": row.get("run_directory"),
                    "mF1": row.get("mF1"),
                    "F1_1": row.get("F1_1"),
                    "F1_0": row.get("F1_0"),
                    "mIoU": row.get("mIoU"),
                    "IoU_1": row.get("IoU_1"),
                    "IoU_0": row.get("IoU_0"),
                    "Precision_1": row.get("Precision_1"),
                    "Recall_1": row.get("Recall_1"),
                    "OA": row.get("OA"),
                    "Boundary F1": row.get("Boundary F1"),
                    "Edge IoU": row.get("Edge IoU"),
                    "threshold": row.get("threshold"),
                    "params": row.get("params"),
                    "FLOPs": row.get("FLOPs"),
                    "GPU memory": row.get("GPU memory"),
                    "FPS": row.get("FPS"),
                    "source_file": row.get("source_file"),
                    "source_kind": row.get("source_kind"),
                }
            )

    print(f"Wrote {WEBSITE_DATA / 'ours_results.json'}")
    print(f"Wrote {WEBSITE_DATA / 'ours_results.csv'}")
    print(f"Wrote {WEBSITE_DATA / 'ours_results_all_candidates.json'}")


if __name__ == "__main__":
    main()
