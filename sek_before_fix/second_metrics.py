"""TRUE SECOND semantic change detection metrics."""
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


def _dense_confmat(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred = pred.detach().view(-1).long()
    target = target.detach().view(-1).long()
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    if valid_mask is not None:
        valid = valid & valid_mask.detach().view(-1).bool()
    pred = pred[valid]
    target = target[valid]
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
    eps: float = 1e-6
    notes: list[str] = field(default_factory=list)

    tp_binary_head: float = 0.0
    fp_binary_head: float = 0.0
    fn_binary_head: float = 0.0
    tn_binary_head: float = 0.0
    tp_semantic_change: float = 0.0
    fp_semantic_change: float = 0.0
    fn_semantic_change: float = 0.0
    tn_semantic_change: float = 0.0
    semantic_correct_total: float = 0.0
    semantic_valid_total: float = 0.0
    valid_pixels: float = 0.0
    confmat_t1: torch.Tensor | None = None
    confmat_t2: torch.Tensor | None = None
    sek_target_confmat: torch.Tensor | None = None
    sek_transition_confmat: torch.Tensor | None = None

    def update(
        self,
        *,
        change_pred_binary: Optional[torch.Tensor],
        change_pred_semantic: Optional[torch.Tensor],
        change_gt: torch.Tensor,
        pred_sem1: Optional[torch.Tensor] = None,
        pred_sem2: Optional[torch.Tensor] = None,
        gt_sem1: Optional[torch.Tensor] = None,
        gt_sem2: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        change_gt = change_gt.detach().bool()

        if valid_mask is None:
            valid = torch.ones_like(change_gt, dtype=torch.bool)
        else:
            valid = valid_mask.detach().bool()
        if gt_sem1 is not None and gt_sem2 is not None:
            valid = valid & (gt_sem1.detach() != self.ignore_index) & (gt_sem2.detach() != self.ignore_index)

        gt = change_gt[valid]
        self.valid_pixels += float(gt.numel())

        if change_pred_binary is not None:
            pred_binary = change_pred_binary.detach().bool()[valid]
            self.tp_binary_head += float((pred_binary & gt).sum().item())
            self.fp_binary_head += float((pred_binary & ~gt).sum().item())
            self.fn_binary_head += float((~pred_binary & gt).sum().item())
            self.tn_binary_head += float((~pred_binary & ~gt).sum().item())

        if change_pred_semantic is not None:
            pred_sem_change = change_pred_semantic.detach().bool()[valid]
            self.tp_semantic_change += float((pred_sem_change & gt).sum().item())
            self.fp_semantic_change += float((pred_sem_change & ~gt).sum().item())
            self.fn_semantic_change += float((~pred_sem_change & gt).sum().item())
            self.tn_semantic_change += float((~pred_sem_change & ~gt).sum().item())

        semantic_available = all(x is not None for x in (pred_sem1, pred_sem2, gt_sem1, gt_sem2))
        if not semantic_available:
            self.notes.append("Semantic predictions unavailable; SECOND metrics are binary-compatible only.")
            return

        pred_sem1 = pred_sem1.detach().long()
        pred_sem2 = pred_sem2.detach().long()
        gt_sem1 = gt_sem1.detach().long()
        gt_sem2 = gt_sem2.detach().long()

        correct_t1 = (pred_sem1 == gt_sem1) & valid
        correct_t2 = (pred_sem2 == gt_sem2) & valid
        valid_count = float(valid.sum().item())
        self.semantic_correct_total += float(correct_t1.sum().item() + correct_t2.sum().item())
        self.semantic_valid_total += 2.0 * valid_count

        cm_t1 = _dense_confmat(pred_sem1, gt_sem1, self.num_classes, valid)
        cm_t2 = _dense_confmat(pred_sem2, gt_sem2, self.num_classes, valid)
        self.confmat_t1 = cm_t1 if self.confmat_t1 is None else self.confmat_t1 + cm_t1
        self.confmat_t2 = cm_t2 if self.confmat_t2 is None else self.confmat_t2 + cm_t2

        changed_valid = valid & change_gt
        if self.compute_sek and bool(changed_valid.any().item()):
            sek_target = _dense_confmat(pred_sem2, gt_sem2, self.num_classes, changed_valid)
            self.sek_target_confmat = sek_target if self.sek_target_confmat is None else self.sek_target_confmat + sek_target
            transition_classes = self.num_classes * self.num_classes
            gt_transition = gt_sem1 * self.num_classes + gt_sem2
            pred_transition = pred_sem1 * self.num_classes + pred_sem2
            sek_transition = _dense_confmat(pred_transition, gt_transition, transition_classes, changed_valid)
            self.sek_transition_confmat = sek_transition if self.sek_transition_confmat is None else self.sek_transition_confmat + sek_transition

    def _binary_scores(self, tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
        precision = _safe_div(tp, tp + fp, self.eps)
        recall = _safe_div(tp, tp + fn, self.eps)
        binary_f1 = _safe_div(2.0 * precision * recall, precision + recall)
        binary_iou = _iou_from_binary_confusion(tp, fp, fn)
        iou_nochange = _safe_div(tn, tn + fp + fn, self.eps)
        oa_binary = _safe_div(tp + tn, tp + tn + fp + fn, self.eps)
        miou_binary = (binary_iou + iou_nochange) / 2.0
        return {
            "oa": oa_binary,
            "precision": precision,
            "recall": recall,
            "f1": binary_f1,
            "iou": binary_iou,
            "IoU_change": binary_iou,
            "IoU_nochange": iou_nochange,
            "miou": miou_binary,
        }

    def compute(self) -> dict:
        binary_head_scores = self._binary_scores(
            self.tp_binary_head,
            self.fp_binary_head,
            self.fn_binary_head,
            self.tn_binary_head,
        )
        semantic_change_scores = self._binary_scores(
            self.tp_semantic_change,
            self.fp_semantic_change,
            self.fn_semantic_change,
            self.tn_semantic_change,
        )
        semantic_available = self.confmat_t1 is not None and self.confmat_t2 is not None

        oa_semantic: float | None = None
        oa = binary_head_scores["oa"]
        fscd = binary_head_scores["f1"]
        miou = binary_head_scores["miou"]
        semantic_miou: float | None = None
        sek_target: float | None = None
        sek_transition: float | None = None
        class_ious: list[float] = []

        if semantic_available:
            combined = self.confmat_t1 + self.confmat_t2
            tp = torch.diag(combined)
            fp = combined.sum(dim=0) - tp
            fn = combined.sum(dim=1) - tp
            ious = tp / (tp + fp + fn + self.eps)
            class_ious = [float(v) for v in ious.tolist()]
            semantic_miou = float(ious.mean().item()) if ious.numel() else 0.0
            oa_semantic = _safe_div(self.semantic_correct_total, self.semantic_valid_total, self.eps)
            oa = oa_semantic
            miou = semantic_miou
            if self.compute_sek:
                if self.sek_target_confmat is not None and float(self.sek_target_confmat.sum().item()) > 0:
                    sek_target = _cohen_kappa_from_confmat(self.sek_target_confmat, self.eps)
                else:
                    self.notes.append("No changed valid pixels for SeK_target.")
                if self.sek_transition_confmat is not None and float(self.sek_transition_confmat.sum().item()) > 0:
                    sek_transition = _cohen_kappa_from_confmat(self.sek_transition_confmat, self.eps)
                else:
                    self.notes.append("No changed valid pixels for SeK_transition.")
        else:
            self.notes.append("Semantic SECOND metrics unavailable; using binary-compatible OA/Fscd/mIoU only.")

        return {
            "OA": oa,
            "OA_semantic": oa_semantic,
            "OA_change": binary_head_scores["oa"],
            "Fscd": fscd,
            "F1scd": fscd,
            "Fscd_binary_head": binary_head_scores["f1"],
            "Fscd_semantic_change": semantic_change_scores["f1"],
            "mIoU": miou,
            "binary_mIoU_change": binary_head_scores["miou"],
            "binary_mIoU_semantic_change": semantic_change_scores["miou"],
            "SeK": sek_target,
            "SeK_target": sek_target,
            "SeK_transition": sek_transition,
            "binary_F1": binary_head_scores["f1"],
            "binary_IoU": binary_head_scores["iou"],
            "precision": binary_head_scores["precision"],
            "recall": binary_head_scores["recall"],
            "precision_change": binary_head_scores["precision"],
            "recall_change": binary_head_scores["recall"],
            "precision_change_semantic": semantic_change_scores["precision"],
            "recall_change_semantic": semantic_change_scores["recall"],
            "semantic_mIoU": semantic_miou,
            "IoU_change": binary_head_scores["IoU_change"],
            "IoU_nochange": binary_head_scores["IoU_nochange"],
            "IoU_change_semantic": semantic_change_scores["IoU_change"],
            "IoU_nochange_semantic": semantic_change_scores["IoU_nochange"],
            "class_IoUs": class_ious,
            "second_eval_level": "semantic_change" if semantic_available else "binary-compatible",
            "notes": " ".join(dict.fromkeys(self.notes)).strip(),
        }