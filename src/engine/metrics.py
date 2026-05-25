"""Changed-class binary change detection metrics."""
from __future__ import annotations

import torch


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
    def __init__(self) -> None:
        self.tp = self.fp = self.tn = self.fn = 0.0
        self.sum_prob = 0.0
        self.total = 0.0

    def update(self, logits, mask, threshold: float) -> None:
        probs = torch.sigmoid(logits.detach().float())
        pred = probs >= float(threshold)
        gt = mask.detach().float() > 0.5
        self.tp += float((pred & gt).sum().item())
        self.fp += float((pred & ~gt).sum().item())
        self.tn += float((~pred & ~gt).sum().item())
        self.fn += float((~pred & gt).sum().item())
        self.sum_prob += float(probs.sum().item())
        self.total += float(probs.numel())

    def compute(self) -> dict:
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        oa = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + eps)
        total = max(self.total, 1.0)
        return {
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
