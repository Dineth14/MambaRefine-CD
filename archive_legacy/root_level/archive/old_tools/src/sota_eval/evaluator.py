from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from data.dataset_builder import build_test_loader
from training.boundary_metrics import BoundaryMetrics
from training.metrics import StreamingMetrics
from utils.visualization import save_prediction_grid

from .base_adapter import AdapterUnavailableError


CANONICAL_COLUMNS = [
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
    "Pred positive ratio",
    "GT positive ratio",
    "best_threshold",
]


def _dataset_eval_cfg(base_cfg: dict, dataset_name: str, dataset_cfg: dict, batch_size: int | None = None) -> dict:
    hardware = base_cfg.get("hardware", {})
    evaluation = base_cfg.get("evaluation", {})
    batch_size = int(batch_size if batch_size is not None else evaluation.get("batch_size", 8))
    return {
        "experiment": {"seed": int(base_cfg.get("experiment", {}).get("seed", 42))},
        "hardware": {
            "device": hardware.get("device", "cuda"),
        },
        "training": {"batch_size": batch_size},
        "validation": {"batch_size": batch_size},
        "evaluation": {
            "split": dataset_cfg.get("split", "test"),
            "threshold": 0.5,
        },
        "dataset": {
            "name": dataset_name,
            "root": dataset_cfg["root"],
            "image_size": int(dataset_cfg.get("image_size", 256)),
            "num_workers": int(evaluation.get("num_workers", 0)),
            "pin_memory": bool(evaluation.get("pin_memory", True)),
            "persistent_workers": False,
            "prefetch_factor": 2,
            "task_type": dataset_cfg.get("task_type", "binary"),
            "mode": dataset_cfg.get("mode", "binary"),
        },
        "boundary_metrics": {"enabled": True, "boundary_width": 3, "tolerance": 2},
    }


def _threshold_values(cfg: dict) -> list[float]:
    values = cfg.get("evaluation", {}).get("threshold_values", [0.5])
    return [float(v) for v in values]


def _select_metric_key(name: str) -> str:
    key = str(name).strip().lower()
    mapping = {"f1_1": "f1_1", "mf1": "mf1", "miou": "miou", "iou_1": "iou_1", "oa": "oa"}
    return mapping.get(key, "f1_1")


def _canonicalize(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    return {
        "mF1": round(float(result.get("mf1", 0.0)), 6),
        "F1_1": round(float(result.get("f1_1", result.get("f1", 0.0))), 6),
        "F1_0": round(float(result.get("f1_0", 0.0)), 6),
        "mIoU": round(float(result.get("miou", 0.0)), 6),
        "IoU_1": round(float(result.get("iou_1", result.get("iou", 0.0))), 6),
        "IoU_0": round(float(result.get("iou_0", 0.0)), 6),
        "Precision_1": round(float(result.get("precision_1", result.get("precision", 0.0))), 6),
        "Recall_1": round(float(result.get("recall_1", result.get("recall", 0.0))), 6),
        "OA": round(float(result.get("oa", 0.0)), 6),
        "Boundary F1": round(float(result.get("boundary_f1", 0.0)), 6),
        "Edge IoU": round(float(result.get("edge_iou", 0.0)), 6),
        "Pred positive ratio": round(float(result.get("pred_positive_ratio", 0.0)), 6),
        "GT positive ratio": round(float(result.get("gt_positive_ratio", 0.0)), 6),
        "best_threshold": threshold,
    }


def _compute_metrics(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float, boundary_cfg: dict) -> dict[str, Any]:
    pixel = StreamingMetrics(threshold=threshold)
    boundary = BoundaryMetrics(
        boundary_width=int(boundary_cfg.get("boundary_width", 3)),
        tolerance=int(boundary_cfg.get("tolerance", 2)),
        threshold=threshold,
    )
    pixel.update(probabilities, labels)
    boundary.update(probabilities, labels)
    result = pixel.compute()
    result.update(boundary.compute())
    return result


def evaluate_with_adapter(
    adapter,
    cfg: dict,
    dataset_cfg: dict,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = adapter.dataset_name
    available, reason = adapter.is_available()
    if not available:
        raise AdapterUnavailableError(reason)

    model = adapter.build_model()
    checkpoint_info = adapter.load_checkpoint(model)
    model.eval()

    all_probs = []
    all_labels = []
    saved_batch = None
    amp = bool(cfg.get("hardware", {}).get("mixed_precision", True)) and adapter.device.type == "cuda"
    base_batch_size = int(cfg.get("evaluation", {}).get("batch_size", 8))
    attempt_batch_sizes = [base_batch_size] if base_batch_size == 1 else [base_batch_size, 1]
    last_error: Exception | None = None
    eval_cfg = None

    for attempt_batch_size in attempt_batch_sizes:
        all_probs = []
        all_labels = []
        saved_batch = None
        eval_cfg = _dataset_eval_cfg(cfg, dataset_name, dataset_cfg, batch_size=attempt_batch_size)
        loader = build_test_loader(eval_cfg)
        try:
            for batch in loader:
                label = batch.get("change_mask", batch.get("label", batch.get("mask")))
                if label is None:
                    raise AdapterUnavailableError("batch does not contain a binary change label")
                prepared = adapter.preprocess_batch(batch)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=amp):
                        output = adapter.forward(model, prepared)
                    prob = adapter.logits_to_change_prob(output).detach().cpu()
                all_probs.append(prob)
                all_labels.append(label.detach().cpu().float().unsqueeze(1) if label.ndim == 3 else label.detach().cpu().float())
                if saved_batch is None:
                    saved_batch = batch
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
            if "out of memory" not in str(exc).lower() or attempt_batch_size == 1:
                raise
            if adapter.device.type == "cuda":
                torch.cuda.empty_cache()
            continue
    if last_error is not None:
        raise last_error

    probabilities = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metric_key = _select_metric_key(cfg.get("evaluation", {}).get("threshold_select_metric", "F1_1"))
    best_threshold = 0.5
    best_metrics = None
    best_score = float("-inf")
    for threshold in _threshold_values(cfg):
        metrics = _compute_metrics(probabilities, labels, threshold, eval_cfg["boundary_metrics"])
        score = float(metrics.get(metric_key, float("-inf")))
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score
    if best_metrics is None:
        raise AdapterUnavailableError("threshold sweep did not produce metrics")

    metrics_out = _canonicalize(best_metrics, best_threshold)
    payload = {
        "status": "OK",
        "model": adapter.model_name,
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": adapter.checkpoint_path,
        "normalization": adapter.last_normalization_used,
        "expected_output_type": adapter.expected_output_type,
        "checkpoint_info": checkpoint_info,
        "metrics": metrics_out,
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "eval_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "model", *CANONICAL_COLUMNS])
        writer.writerow([dataset_name, adapter.model_name, *[metrics_out[column] for column in CANONICAL_COLUMNS]])

    if bool(cfg.get("evaluation", {}).get("save_qualitative", True)) and saved_batch is not None:
        pred = (probabilities[: int(cfg.get("evaluation", {}).get("num_qualitative_samples", 16))] > best_threshold).float()
        label = labels[: pred.shape[0]]
        count = min(4, pred.shape[0])
        save_prediction_grid(saved_batch["image_a"][:count], saved_batch["image_b"][:count], label[:count], pred[:count], output_dir / "qualitative_grid.png", count=count)

    (output_dir / "status.json").write_text(
        json.dumps({"status": "OK", "stage": "eval", "reason": "", "checkpoint_path": adapter.checkpoint_path}, indent=2),
        encoding="utf-8",
    )
    return payload
