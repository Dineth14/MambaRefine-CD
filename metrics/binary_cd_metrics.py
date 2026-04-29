"""Binary change detection metrics for WHU-CD and DSIFN-CD.

Computes ONLY the metrics used in the Mamba-CD paper:
  Pre, Rec, F1, IoU, OA

No other metrics are computed, logged, or returned.

Usage
-----
    m = BinaryMetrics(threshold=0.5)
    m.update(pred_logits, gt_mask)
    results = m.compute()   # dict with keys: Pre, Rec, F1, IoU, OA
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class BinaryMetrics:
    """Streaming binary change-detection metrics.

    Accumulates confusion-matrix entries across batches.
    Call compute() after processing all batches.

    Args:
        threshold: sigmoid threshold for binarizing logits (default 0.5).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.thr = threshold
        self.reset()

    def reset(self) -> None:
        self.tp: float = 0.0
        self.fp: float = 0.0
        self.fn: float = 0.0
        self.tn: float = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate confusion-matrix entries.

        Args:
            preds:   logits [B, 1, H, W] or [B, H, W] or probabilities in [0,1].
            targets: binary ground-truth [B, 1, H, W] or [B, H, W], values 0/1.
        """
        # Convert logits to binary predictions
        if preds.min() < 0 or preds.max() > 1:
            probs = torch.sigmoid(preds)
        else:
            probs = preds
        pred_b = (probs > self.thr).float()

        targets = targets.float()
        if targets.numel() and targets.max() > 1:
            targets = (targets > 127).float()
        else:
            targets = (targets > 0.5).float()

        # Flatten
        p = pred_b.view(-1)
        t = targets.view(-1)

        # Confusion matrix entries
        self.tp += float((p * t).sum().item())
        self.fp += float((p * (1.0 - t)).sum().item())
        self.fn += float(((1.0 - p) * t).sum().item())
        self.tn += float(((1.0 - p) * (1.0 - t)).sum().item())

    def compute(self) -> dict[str, float]:
        """Compute final metrics.

        Returns:
            dict with keys: Pre, Rec, F1, IoU, OA
            Values are percentages in [0, 100].
        """
        eps = 1e-6
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        # Pre = TP / (TP + FP + eps)
        pre = tp / (tp + fp + eps)

        # Rec = TP / (TP + FN + eps)
        rec = tp / (tp + fn + eps)

        # F1  = 2 * Pre * Rec / (Pre + Rec + eps)
        f1 = 2.0 * pre * rec / (pre + rec + eps)

        # IoU = TP / (TP + FP + FN + eps)
        iou = tp / (tp + fp + fn + eps)

        # OA  = (TP + TN) / (TP + TN + FP + FN + eps)
        oa = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            "Pre": round(pre * 100.0, 4),
            "Rec": round(rec * 100.0, 4),
            "F1":  round(f1  * 100.0, 4),
            "IoU": round(iou * 100.0, 4),
            "OA":  round(oa  * 100.0, 4),
        }
