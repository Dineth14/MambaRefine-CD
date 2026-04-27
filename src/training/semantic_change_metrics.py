"""Utilities for future semantic change detection metrics.

These helpers are intentionally decoupled from the current binary trainer.
They are safe to import today, and can be wired into future semantic-mode
training once the model supports multi-class outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def semantic_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute a dense ``[C, C]`` confusion matrix for semantic labels."""
    pred = pred.detach().view(-1).long()
    target = target.detach().view(-1).long()
    valid = (target != ignore_index) & (pred >= 0) & (pred < num_classes)
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64)
    indices = target * num_classes + pred
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).to(torch.float64)


def semantic_iou_per_class(confmat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return IoU for each class from a semantic confusion matrix."""
    confmat = confmat.to(torch.float64)
    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp
    return tp / (tp + fp + fn + eps)


def semantic_miou(confmat: torch.Tensor, eps: float = 1e-6) -> float:
    """Return mean IoU across semantic classes."""
    ious = semantic_iou_per_class(confmat, eps=eps)
    return float(ious.mean().item()) if ious.numel() else 0.0


def binary_change_f1_from_semantics(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> dict:
    """Compute binary change metrics from semantic label pairs."""
    pred_change = (pred_a != pred_b)
    target_change = (target_a != target_b)
    valid = (target_a != ignore_index) & (target_b != ignore_index)
    pred_change = pred_change[valid]
    target_change = target_change[valid]
    if pred_change.numel() == 0:
        return {"precision_1": 0.0, "recall_1": 0.0, "f1_1": 0.0, "iou_1": 0.0}
    pred = pred_change.long()
    target = target_change.long()
    tp = float((pred * target).sum().item())
    fp = float((pred * (1 - target)).sum().item())
    fn = float(((1 - pred) * target).sum().item())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"precision_1": precision, "recall_1": recall, "f1_1": f1, "iou_1": iou}


def semantic_sek_placeholder(*_, **__) -> None:
    """Placeholder for SeK until the repo adopts a validated implementation.

    TODO: wire a tested SeK implementation once semantic change training and
    evaluation are fully supported end-to-end.
    """
    return None


@dataclass
class SemanticChangeMetrics:
    """Streaming container for future semantic change evaluation."""

    num_classes: int
    ignore_index: int = 255
    confmat_a: Optional[torch.Tensor] = None
    confmat_b: Optional[torch.Tensor] = None

    def update(
        self,
        pred_a: torch.Tensor,
        pred_b: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
    ) -> None:
        cm_a = semantic_confusion_matrix(pred_a, target_a, self.num_classes, self.ignore_index)
        cm_b = semantic_confusion_matrix(pred_b, target_b, self.num_classes, self.ignore_index)
        self.confmat_a = cm_a if self.confmat_a is None else self.confmat_a + cm_a
        self.confmat_b = cm_b if self.confmat_b is None else self.confmat_b + cm_b

    def compute(self) -> dict:
        if self.confmat_a is None or self.confmat_b is None:
            return {
                "semantic_iou_per_class_t1": [],
                "semantic_iou_per_class_t2": [],
                "semantic_miou_t1": 0.0,
                "semantic_miou_t2": 0.0,
                "sek": None,
            }
        iou_a = semantic_iou_per_class(self.confmat_a)
        iou_b = semantic_iou_per_class(self.confmat_b)
        return {
            "semantic_iou_per_class_t1": [float(v) for v in iou_a.tolist()],
            "semantic_iou_per_class_t2": [float(v) for v in iou_b.tolist()],
            "semantic_miou_t1": semantic_miou(self.confmat_a),
            "semantic_miou_t2": semantic_miou(self.confmat_b),
            "sek": semantic_sek_placeholder(),
            "confusion_matrix_t1": self.confmat_a.tolist(),
            "confusion_matrix_t2": self.confmat_b.tolist(),
        }