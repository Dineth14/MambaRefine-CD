"""Streaming binary change-detection metrics.

Accumulated across batches; call compute() for final results.

Metrics returned
----------------
f1, iou, miou, precision, recall, oa,
boundary_f1, pred_positive_ratio, gt_positive_ratio

Notes
-----
* ``iou``  = change-class IoU  =  TP / (TP + FP + FN)
* ``miou`` = mean IoU over both classes
           = (IoU_change + IoU_nochange) / 2
           where IoU_nochange = TN / (TN + FP + FN)
* ``boundary_f1`` is a streaming approximation using max-pool dilation.
  For tolerance-aware boundary F1 and edge IoU, use BoundaryMetrics from
  boundary_metrics.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class StreamingMetrics:
    def __init__(self, threshold: float = 0.5, boundary_dil: int = 3) -> None:
        self.thr   = threshold
        self.bdil  = boundary_dil
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0.0
        self.bnd_tp = self.bnd_fp = self.bnd_fn = 0.0
        self.total_pred_pos = self.total_gt_pos = self.total_px = 0.0

    def _boundary(self, mask: torch.Tensor) -> torch.Tensor:
        k = self.bdil * 2 + 1
        return (F.max_pool2d(mask, k, stride=1, padding=k // 2) - mask).clamp(0, 1)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        probs  = torch.sigmoid(preds) if not (0 <= preds.min() and preds.max() <= 1) else preds
        pred_b = (probs > self.thr).float()
        if pred_b.dim() == 3:
            pred_b = pred_b.unsqueeze(1)
        tgt = targets.float()
        if tgt.dim() == 3:
            tgt = tgt.unsqueeze(1)

        p, t = pred_b.view(-1).long(), tgt.view(-1).long()
        self.tp += (p * t).sum().item()
        self.fp += (p * (1 - t)).sum().item()
        self.fn += ((1 - p) * t).sum().item()
        self.tn += ((1 - p) * (1 - t)).sum().item()

        self.total_pred_pos += float(p.sum().item())
        self.total_gt_pos   += float(t.sum().item())
        self.total_px       += float(p.numel())

        bnd = self._boundary(tgt)
        self.bnd_tp += (pred_b * bnd * tgt).sum().item()
        self.bnd_fp += (pred_b * bnd * (1 - tgt)).sum().item()
        self.bnd_fn += ((1 - pred_b) * bnd * tgt).sum().item()

    def compute(self) -> dict:
        eps = 1e-6
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        iou_change   = tp / (tp + fp + fn + eps)
        iou_nochange = tn / (tn + fp + fn + eps)
        miou         = (iou_change + iou_nochange) / 2.0

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        oa        = (tp + tn) / (tp + tn + fp + fn + eps)

        bnd_p  = self.bnd_tp / (self.bnd_tp + self.bnd_fp + eps)
        bnd_r  = self.bnd_tp / (self.bnd_tp + self.bnd_fn + eps)
        bnd_f1 = 2 * bnd_p * bnd_r / (bnd_p + bnd_r + eps)

        return {
            "f1":                  f1,
            "iou":                 iou_change,
            "miou":                miou,
            "precision":           precision,
            "recall":              recall,
            "oa":                  oa,
            "boundary_f1":         bnd_f1,
            "pred_positive_ratio": self.total_pred_pos / max(self.total_px, 1.0),
            "gt_positive_ratio":   self.total_gt_pos   / max(self.total_px, 1.0),
        }
