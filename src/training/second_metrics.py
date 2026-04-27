"""SECOND semantic change detection metrics.

Supports two evaluation levels:

* Binary-compatible mode for binary change logits.
* Semantic mode for future semantic predictions at both timestamps.

The current model is binary-only, but this accumulator is designed so the
evaluator can start reporting SECOND-specific metrics now and reuse the same
API once semantic predictions are added later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


def _safe_div(num: float, den: float, eps: float = 1e-6) -> float:
    if abs(float(den)) <= eps:
        return 0.0
    return float(num) / float(den)


def _cohen_kappa_from_confmat(confmat: torch.Tensor, eps: float = 1e-6) -> float:
    confmat = confmat.to(torch.float64)
    total = float(confmat.sum().item())
    if total <= 0:
        return 0.0
    po = float(torch.diag(confmat).sum().item()) / (total + eps)
    row = confmat.sum(dim=1)
    col = confmat.sum(dim=0)
    pe = float((row * col).sum().item()) / ((total * total) + eps)
    if abs(1.0 - pe) <= eps:
        return 1.0 if abs(po - 1.0) <= eps else 0.0
    return _safe_div(po - pe, 1.0 - pe, eps)


def _semantic_confmat(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    pred = pred.detach().view(-1).long()
    target = target.detach().view(-1).long()
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    pred = pred[valid]
    target = target[valid]
    pred = pred[(pred >= 0) & (pred < num_classes)]
    target = target[: pred.numel()]
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64)
    inds = target * num_classes + pred
    bins = torch.bincount(inds, minlength=num_classes * num_classes)
    return bins.reshape(num_classes, num_classes).to(torch.float64)


def _iou_from_binary_confusion(tp: float, fp: float, fn: float) -> float:
    return _safe_div(tp, tp + fp + fn)


@dataclass
class SECONDMetrics:
    num_classes: int = 7
    ignore_index: int = 255
    compute_sek: bool = True
    sek_binary_fallback: bool = False
    notes: list[str] = field(default_factory=list)

    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0
    semantic_correct_changed: float = 0.0
    pred_change_total: float = 0.0
    gt_change_total: float = 0.0

    semantic_correct_total: float = 0.0
    semantic_valid_total: float = 0.0
    binary_valid_total: float = 0.0
    confmat_t1: torch.Tensor | None = None
    confmat_t2: torch.Tensor | None = None
    sek_confmat: torch.Tensor | None = None

    def update(
        self,
        *,
        change_pred: torch.Tensor,
        change_gt: torch.Tensor,
        ignore_mask: Optional[torch.Tensor] = None,
        pred_label_t1: Optional[torch.Tensor] = None,
        pred_label_t2: Optional[torch.Tensor] = None,
        label_t1: Optional[torch.Tensor] = None,
        label_t2: Optional[torch.Tensor] = None,
    ) -> None:
        change_pred = change_pred.detach().bool()
        change_gt = change_gt.detach().bool()

        if ignore_mask is not None:
            valid = ~ignore_mask.detach().bool()
        else:
            valid = torch.ones_like(change_gt, dtype=torch.bool)

        if label_t1 is not None and label_t2 is not None:
            valid = valid & (label_t1.detach() != self.ignore_index) & (label_t2.detach() != self.ignore_index)

        pred = change_pred[valid]
        gt = change_gt[valid]

        self.tp += float((pred & gt).sum().item())
        self.fp += float((pred & ~gt).sum().item())
        self.fn += float((~pred & gt).sum().item())
        self.tn += float((~pred & ~gt).sum().item())
        self.binary_valid_total += float(pred.numel())
        self.pred_change_total += float(pred.sum().item())
        self.gt_change_total += float(gt.sum().item())

        semantic_available = all(x is not None for x in (pred_label_t1, pred_label_t2, label_t1, label_t2))
        if not semantic_available:
            return

        pred_label_t1 = pred_label_t1.detach().long()
        pred_label_t2 = pred_label_t2.detach().long()
        label_t1 = label_t1.detach().long()
        label_t2 = label_t2.detach().long()

        valid_t1 = (label_t1 != self.ignore_index)
        valid_t2 = (label_t2 != self.ignore_index)
        valid_sem = valid_t1 & valid_t2

        correct_t1 = (pred_label_t1 == label_t1) & valid_t1
        correct_t2 = (pred_label_t2 == label_t2) & valid_t2
        self.semantic_correct_total += float(correct_t1.sum().item() + correct_t2.sum().item())
        self.semantic_valid_total += float(valid_t1.sum().item() + valid_t2.sum().item())

        cm_t1 = _semantic_confmat(pred_label_t1, label_t1, self.num_classes, self.ignore_index)
        cm_t2 = _semantic_confmat(pred_label_t2, label_t2, self.num_classes, self.ignore_index)
        self.confmat_t1 = cm_t1 if self.confmat_t1 is None else self.confmat_t1 + cm_t1
        self.confmat_t2 = cm_t2 if self.confmat_t2 is None else self.confmat_t2 + cm_t2

        changed_valid = valid_sem & change_gt
        changed_correct = changed_valid & change_pred & (pred_label_t1 == label_t1) & (pred_label_t2 == label_t2)
        self.semantic_correct_changed += float(changed_correct.sum().item())

        if self.compute_sek and changed_valid.any():
            sek_gt = label_t2[changed_valid]
            sek_pred = pred_label_t2[changed_valid]
            sek_cm = _semantic_confmat(sek_pred, sek_gt, self.num_classes, self.ignore_index)
            self.sek_confmat = sek_cm if self.sek_confmat is None else self.sek_confmat + sek_cm

    def _binary_scores(self) -> dict:
        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = _safe_div(self.tp, self.tp + self.fn)
        binary_f1 = _safe_div(2.0 * precision * recall, precision + recall)
        binary_iou = _iou_from_binary_confusion(self.tp, self.fp, self.fn)
        iou_nochange = _safe_div(self.tn, self.tn + self.fp + self.fn)
        oa_binary = _safe_div(self.tp + self.tn, self.tp + self.tn + self.fp + self.fn)
        miou_binary = (binary_iou + iou_nochange) / 2.0
        return {
            "oa_binary": oa_binary,
            "precision": precision,
            "recall": recall,
            "binary_F1": binary_f1,
            "binary_IoU": binary_iou,
            "IoU_change": binary_iou,
            "IoU_nochange": iou_nochange,
            "binary_mIoU": miou_binary,
            "pred_positive_ratio": _safe_div(self.pred_change_total, self.binary_valid_total),
            "gt_positive_ratio": _safe_div(self.gt_change_total, self.binary_valid_total),
        }

    def compute(self) -> dict:
        binary_scores = self._binary_scores()
        semantic_available = self.confmat_t1 is not None and self.confmat_t2 is not None

        fscd = binary_scores["binary_F1"]
        oa = binary_scores["oa_binary"]
        miou = binary_scores["binary_mIoU"]
        semantic_miou: float | None = None
        sek: float | None = None
        class_ious: list[float] = []
        binary_kappa: float | None = None

        if semantic_available:
            combined = self.confmat_t1 + self.confmat_t2
            tp = torch.diag(combined)
            fp = combined.sum(dim=0) - tp
            fn = combined.sum(dim=1) - tp
            ious = tp / (tp + fp + fn + 1e-6)
            class_ious = [float(v) for v in ious.tolist()]
            semantic_miou = float(ious.mean().item()) if ious.numel() else 0.0
            oa = _safe_div(self.semantic_correct_total, self.semantic_valid_total)
            miou = semantic_miou

            precision_scd = _safe_div(self.semantic_correct_changed, self.pred_change_total)
            recall_scd = _safe_div(self.semantic_correct_changed, self.gt_change_total)
            fscd = _safe_div(2.0 * precision_scd * recall_scd, precision_scd + recall_scd)

            if self.compute_sek:
                if self.sek_confmat is not None and float(self.sek_confmat.sum().item()) > 0:
                    sek = _cohen_kappa_from_confmat(self.sek_confmat)
                else:
                    self.notes.append("SeK requires semantic predictions on changed pixels.")
        else:
            self.notes.append("Semantic Fscd requires semantic predictions; using binary Fscd.")
            if self.compute_sek:
                if self.sek_binary_fallback:
                    confmat = torch.tensor([[self.tn, self.fp], [self.fn, self.tp]], dtype=torch.float64)
                    binary_kappa = _cohen_kappa_from_confmat(confmat)
                    self.notes.append("SeK unavailable; reporting binary_kappa separately because sek_binary_fallback=true.")
                else:
                    self.notes.append("SeK requires semantic predictions.")

        return {
            "OA": oa,
            "Fscd": fscd,
            "F1scd": fscd,
            "mIoU": miou,
            "SeK": sek,
            "binary_F1": binary_scores["binary_F1"],
            "binary_IoU": binary_scores["binary_IoU"],
            "precision": binary_scores["precision"],
            "recall": binary_scores["recall"],
            "pred_positive_ratio": binary_scores["pred_positive_ratio"],
            "gt_positive_ratio": binary_scores["gt_positive_ratio"],
            "semantic_mIoU": semantic_miou,
            "IoU_change": binary_scores["IoU_change"],
            "IoU_nochange": binary_scores["IoU_nochange"],
            "class_IoUs": class_ious,
            "binary_kappa": binary_kappa,
            "notes": " ".join(dict.fromkeys(self.notes)).strip(),
        }