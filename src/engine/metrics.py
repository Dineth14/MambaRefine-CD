"""Changed-class binary change detection metrics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_BOUNDARY_TOLERANCE = 2


def mask_to_boundary(mask: torch.Tensor, tolerance: int = DEFAULT_BOUNDARY_TOLERANCE) -> torch.Tensor:
    """ORBIT-compatible dilation-minus-erosion contour band."""
    binary = (mask.float() > 0.5).float()
    if binary.ndim == 3:
        binary = binary.unsqueeze(1)
    kernel_size = int(tolerance) * 2 + 1
    dilated = F.max_pool2d(binary, kernel_size, stride=1, padding=kernel_size // 2)
    eroded = -F.max_pool2d(-binary, kernel_size, stride=1, padding=kernel_size // 2)
    return (dilated - eroded).clamp(0, 1)


def boundary_prf_batch(
    pred_binary: torch.Tensor,
    gt_binary: torch.Tensor,
    tolerance: int = DEFAULT_BOUNDARY_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image ORBIT-compatible boundary precision, recall, and F1."""
    pred_boundary = mask_to_boundary(pred_binary, tolerance)
    gt_boundary = mask_to_boundary(gt_binary, tolerance)
    kernel_size = int(tolerance) * 2 + 1
    gt_dilated = F.max_pool2d(
        gt_boundary, kernel_size, stride=1, padding=kernel_size // 2
    )
    pred_dilated = F.max_pool2d(
        pred_boundary, kernel_size, stride=1, padding=kernel_size // 2
    )
    spatial_dims = tuple(range(1, pred_boundary.ndim))
    pred_count = pred_boundary.sum(dim=spatial_dims)
    gt_count = gt_boundary.sum(dim=spatial_dims)
    precision = (pred_boundary * gt_dilated).sum(dim=spatial_dims) / (pred_count + 1e-6)
    recall = (gt_boundary * pred_dilated).sum(dim=spatial_dims) / (gt_count + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    both_empty = (pred_count == 0) & (gt_count == 0)
    one_empty = (pred_count == 0) ^ (gt_count == 0)
    precision = torch.where(both_empty, torch.ones_like(precision), precision)
    recall = torch.where(both_empty, torch.ones_like(recall), recall)
    f1 = torch.where(both_empty, torch.ones_like(f1), f1)
    precision = torch.where(one_empty, torch.zeros_like(precision), precision)
    recall = torch.where(one_empty, torch.zeros_like(recall), recall)
    f1 = torch.where(one_empty, torch.zeros_like(f1), f1)
    return precision, recall, f1


def _flatten_binary(x) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.detach().float().reshape(-1)


def binary_metrics_from_arrays(pred_binary, gt_binary, mean_sigmoid_prob=None) -> dict:
    pred = (_flatten_binary(pred_binary) > 0.5)
    gt = (_flatten_binary(gt_binary) > 0.5)
    eps = 1e-6
    tp = float((pred & gt).sum().item())
    fp = float((pred & ~gt).sum().item())
    tn = float((~pred & ~gt).sum().item())
    fn = float((~pred & gt).sum().item())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    total = max(tp + fp + tn + fn, 1.0)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
        "OA": oa,
        "pred_positive_ratio": (tp + fp) / total,
        "gt_positive_ratio": (tp + fn) / total,
        "mean_sigmoid_prob": float(mean_sigmoid_prob) if mean_sigmoid_prob is not None else 0.0,
    }


class BinaryMetricAccumulator:
    def __init__(
        self,
        *,
        compute_boundary: bool = True,
        boundary_tolerance: int = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> None:
        self.tp = self.fp = self.tn = self.fn = 0.0
        self.sum_prob = 0.0
        self.total = 0.0
        self.compute_boundary = bool(compute_boundary)
        self.boundary_tolerance = int(boundary_tolerance)
        self.boundary_precision_sum = 0.0
        self.boundary_recall_sum = 0.0
        self.boundary_f1_sum = 0.0
        self.boundary_count = 0

    def update(self, logits, mask, threshold: float) -> None:
        probs = torch.sigmoid(logits.detach().float())
        self.update_probs(probs, mask, threshold)

    def update_probs(self, probs, mask, threshold: float) -> None:
        probs = probs.detach().float()
        pred = probs >= float(threshold)
        gt = mask.detach().float() > 0.5
        self.tp += float((pred & gt).sum().item())
        self.fp += float((pred & ~gt).sum().item())
        self.tn += float((~pred & ~gt).sum().item())
        self.fn += float((~pred & gt).sum().item())
        self.sum_prob += float(probs.sum().item())
        self.total += float(probs.numel())
        if self.compute_boundary:
            boundary_precision, boundary_recall, boundary_f1 = boundary_prf_batch(
                pred.float(), gt.float(), self.boundary_tolerance
            )
            self.boundary_precision_sum += float(boundary_precision.sum().item())
            self.boundary_recall_sum += float(boundary_recall.sum().item())
            self.boundary_f1_sum += float(boundary_f1.sum().item())
            self.boundary_count += int(boundary_f1.numel())

    def compute(self) -> dict:
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        oa = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + eps)
        total = max(self.total, 1.0)
        metrics = {
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
            "OA": oa,
            "pred_positive_ratio": (self.tp + self.fp) / total,
            "gt_positive_ratio": (self.tp + self.fn) / total,
            "mean_sigmoid_prob": self.sum_prob / total,
        }
        if self.compute_boundary:
            count = max(self.boundary_count, 1)
            metrics.update({
                "BoundaryPrecision": self.boundary_precision_sum / count,
                "BoundaryRecall": self.boundary_recall_sum / count,
                "BoundaryF1": self.boundary_f1_sum / count,
                "boundary_tolerance_px": self.boundary_tolerance,
                "boundary_sample_count": self.boundary_count,
            })
        return metrics
