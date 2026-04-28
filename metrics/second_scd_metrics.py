"""SECOND semantic change detection metrics.

Returns only OA, mIoU, SeK, and Fscd, scaled to percentages.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


_EPS = 1e-6


def _safe_div(num: float, den: float, eps: float = _EPS) -> float:
    return 0.0 if abs(float(den)) <= eps else float(num) / float(den)


def _dense_confmat(pred: torch.Tensor, target: torch.Tensor, num_classes: int, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred = pred.detach().reshape(-1).long()
    target = target.detach().reshape(-1).long()
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    if valid_mask is not None:
        valid = valid & valid_mask.detach().reshape(-1).bool()
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64)
    bins = torch.bincount(pred * num_classes + target, minlength=num_classes * num_classes)
    return bins.reshape(num_classes, num_classes).to(torch.float64)


def _cohen_kappa(confmat: torch.Tensor, eps: float = _EPS) -> float:
    confmat = confmat.to(torch.float64)
    total = float(confmat.sum().item())
    if total <= 0:
        return 0.0
    po = float(torch.diag(confmat).sum().item()) / (total + eps)
    row = confmat.sum(dim=1)
    col = confmat.sum(dim=0)
    pe = float((row * col).sum().item()) / (total * total + eps)
    if abs(1.0 - pe) <= eps:
        return 1.0 if abs(po - 1.0) <= eps else 0.0
    return _safe_div(po - pe, 1.0 - pe, eps)


def _scd_labels(pred1: torch.Tensor, pred2: torch.Tensor, gt1: torch.Tensor, gt2: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if gt1.ndim == 4 and gt1.shape[1] == 1:
        gt1 = gt1[:, 0]
    if gt2.ndim == 4 and gt2.shape[1] == 1:
        gt2 = gt2[:, 0]
    if pred1.ndim == 4 and pred1.shape[1] == 1:
        pred1 = pred1[:, 0]
    if pred2.ndim == 4 and pred2.shape[1] == 1:
        pred2 = pred2[:, 0]

    valid = (gt1 != ignore_index) & (gt2 != ignore_index)
    gt_change = (gt1 != gt2) & valid
    pred_change = (pred1 != pred2) & valid

    zero = torch.zeros_like(gt1)
    gt_scd1 = torch.where(gt_change, gt1, zero)
    gt_scd2 = torch.where(gt_change, gt2, zero)
    pred_scd1 = torch.where(pred_change, pred1, zero)
    pred_scd2 = torch.where(pred_change, pred2, zero)
    return pred_scd1, pred_scd2, gt_scd1, gt_scd2


def _compute_scores(hist: torch.Tensor, eps: float = _EPS) -> dict[str, float]:
    """Compute SECOND SCD scores from a global SCD confusion matrix.

    The matrix is built over the semantic-change label stream used by common
    SECOND/Mamba-FCS style evaluation: unchanged pixels are mapped to class 0,
    and changed pixels keep their semantic class ID. OA is therefore the
    overall accuracy of this semantic-change stream over both timestamps.

    SeK follows the semantic-kappa convention used in SCD literature: remove
    the no-change true-negative dominance at hist[0, 0], compute kappa on the
    remaining semantic-change matrix, then scale it by exp(IoU_change) / e.
    This is deliberately not ordinary kappa over the full matrix.
    """
    hist = hist.to(torch.float64)
    total = float(hist.sum().item())
    correct = float(torch.diag(hist).sum().item())
    eps = float(eps)
    oa = _safe_div(correct, total, eps)

    hist_fg = hist[1:, 1:]
    c2 = hist.new_zeros((2, 2))
    c2[0, 0] = hist[0, 0]
    c2[0, 1] = hist.sum(dim=1)[0] - hist[0, 0]
    c2[1, 0] = hist.sum(dim=0)[0] - hist[0, 0]
    c2[1, 1] = hist_fg.sum()
    iou_nochange = _safe_div(float(c2[0, 0].item()), float((c2[0].sum() + c2[:, 0].sum() - c2[0, 0]).item()), eps)
    iou_change = _safe_div(float(c2[1, 1].item()), float((c2[1].sum() + c2[:, 1].sum() - c2[1, 1]).item()), eps)
    miou = 0.5 * (iou_nochange + iou_change)

    hist_n0 = hist.clone()
    hist_n0[0, 0] = 0.0
    kappa_n0 = _cohen_kappa(hist_n0, eps)
    sek = (kappa_n0 * math.exp(iou_change)) / math.e

    sc_correct = float(torch.diag(hist_fg).sum().item())
    pred_changed = float(hist.sum(dim=1)[1:].sum().item())
    gt_changed = float(hist.sum(dim=0)[1:].sum().item())
    precision = _safe_div(sc_correct, pred_changed, eps)
    recall = _safe_div(sc_correct, gt_changed, eps)
    fscd = _safe_div(2.0 * precision * recall, precision + recall, eps)

    return {"OA": oa, "mIoU": miou, "SeK": sek, "Fscd": fscd}


class SECONDSCDMetrics:
    """Streaming SECOND SCD metrics from timestamp-wise semantic predictions."""

    def __init__(self, num_classes: int = 7, ignore_index: int = 255, threshold: float = 0.5) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self._hist = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    def update(
        self,
        pred_sem1: torch.Tensor,
        pred_sem2: torch.Tensor,
        gt_sem1: torch.Tensor,
        gt_sem2: torch.Tensor,
        change_mask: Optional[torch.Tensor] = None,
    ) -> None:
        del change_mask
        gt1 = gt_sem1.long()
        gt2 = gt_sem2.long()
        if gt1.ndim == 4 and gt1.shape[1] == 1:
            gt1 = gt1[:, 0]
        if gt2.ndim == 4 and gt2.shape[1] == 1:
            gt2 = gt2[:, 0]
        pred_scd1, pred_scd2, gt_scd1, gt_scd2 = _scd_labels(
            pred_sem1.long(),
            pred_sem2.long(),
            gt1,
            gt2,
            self.ignore_index,
        )
        valid = (gt1 != self.ignore_index) & (gt2 != self.ignore_index)
        self._hist += _dense_confmat(pred_scd1, gt_scd1, self.num_classes, valid)
        self._hist += _dense_confmat(pred_scd2, gt_scd2, self.num_classes, valid)

    def compute(self) -> dict[str, float]:
        if float(self._hist.sum().item()) <= 0.0:
            return {"OA": 0.0, "mIoU": 0.0, "SeK": 0.0, "Fscd": 0.0}
        raw = _compute_scores(self._hist)
        return {key: round(value * 100.0, 4) for key, value in raw.items()}
