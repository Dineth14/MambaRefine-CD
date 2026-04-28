"""SECOND semantic change detection loss."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.second_scd_metrics import _compute_scores as _mambafcs_scdd_scores
from metrics.second_scd_metrics import _dense_confmat


def _expand_mask(reference: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.float32)
    mask = valid_mask.float()
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape != reference.shape:
        mask = torch.broadcast_to(mask, reference.shape)
    return mask


def _masked_mean(values: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    mask = _expand_mask(values, valid_mask)
    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        return values.new_zeros(())
    return (values * mask).sum() / denom


def _binary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    smooth: float = 1.0,
) -> torch.Tensor:
    logits = torch.clamp(logits.float(), -20.0, 20.0)
    targets = targets.float()
    mask = _expand_mask(targets, valid_mask)
    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        return logits.new_zeros(())
    probs = torch.sigmoid(logits)
    numerator = 2.0 * (probs * targets * mask).sum() + smooth
    denominator = (probs * mask).sum() + (targets * mask).sum() + smooth
    return 1.0 - numerator / denominator


def _binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    logits = torch.clamp(logits.float(), -20.0, 20.0)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    return _masked_mean(((1.0 - pt) ** gamma) * bce, valid_mask)


class SecondSemanticChangeLoss(nn.Module):
    """Joint loss for binary change and timestamp-wise semantic prediction."""

    def __init__(
        self,
        *,
        num_classes: int,
        ignore_index: int,
        change_loss_weight: float = 1.0,
        semantic_loss_weight: float = 0.5,
        consistency_loss_weight: float = 0.2,
        sek_loss_weight: float = 0.3,
        dice_weight: float = 1.0,
        focal_weight: float = 0.2,
        focal_gamma: float = 1.5,
        ce_weight: float = 1.0,
        sek_eps: float = 1e-6,
        consistency_detach_semantic: bool = True,
        consistency_loss_type: str = "bce",
        safe_on_invalid: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.change_loss_weight = float(change_loss_weight)
        self.semantic_loss_weight = float(semantic_loss_weight)
        self.consistency_loss_weight = float(consistency_loss_weight)
        self.sek_loss_weight = float(sek_loss_weight)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.ce_weight = float(ce_weight)
        self.sek_eps = float(sek_eps)
        self.consistency_detach_semantic = bool(consistency_detach_semantic)
        self.consistency_loss_type = str(consistency_loss_type).lower()
        self.safe_on_invalid = bool(safe_on_invalid)
        self.latest_stats: dict[str, float | bool] = {}

    def _mambafcs_eval_sek_loss(
        self,
        change_logits: torch.Tensor,
        sem_logits_t1: torch.Tensor,
        sem_logits_t2: torch.Tensor,
        label_a: torch.Tensor,
        label_b: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, float, float]:
        with torch.no_grad():
            gate = torch.sigmoid(torch.clamp(change_logits.float(), -20.0, 20.0)) > 0.5
            if gate.ndim == 4 and gate.shape[1] == 1:
                gate = gate[:, 0]

            pred_t1 = torch.argmax(sem_logits_t1.float(), dim=1)
            pred_t2 = torch.argmax(sem_logits_t2.float(), dim=1)
            pred_t1 = pred_t1.clone()
            pred_t2 = pred_t2.clone()
            pred_t1[~gate] = 0
            pred_t2[~gate] = 0

            valid_t1 = label_a != self.ignore_index
            valid_t2 = label_b != self.ignore_index
            if valid_mask is not None:
                if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
                    valid_mask = valid_mask[:, 0]
                valid_t1 = valid_t1 & valid_mask.bool()
                valid_t2 = valid_t2 & valid_mask.bool()

            hist_t1 = _dense_confmat(pred_t1, label_a, self.num_classes, valid_t1)
            hist_t2 = _dense_confmat(pred_t2, label_b, self.num_classes, valid_t2)
            scores_t1 = _mambafcs_scdd_scores(hist_t1, self.sek_eps)
            scores_t2 = _mambafcs_scdd_scores(hist_t2, self.sek_eps)
            average_sek = 0.5 * (scores_t1["SeK"] + scores_t2["SeK"])
            average_iou = 0.5 * (scores_t1["mIoU"] + scores_t2["mIoU"])

        average_sek_tensor = change_logits.new_tensor(average_sek, dtype=torch.float32)
        average_iou_tensor = change_logits.new_tensor(average_iou, dtype=torch.float32)
        sek_loss = -torch.log((average_sek_tensor + 1.0) / 2.0 + 1e-6) - 0.5 * torch.log(average_iou_tensor + 1e-6)
        return torch.clamp(sek_loss, min=0.0), average_sek, average_iou

    def _semantic_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits.float(), target.long(), ignore_index=self.ignore_index, reduction="none")
        valid = (target != self.ignore_index).float()
        return _masked_mean(ce, valid)

    def forward(self, outputs: dict[str, torch.Tensor | None], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        change_logits = outputs["change_logits"]
        sem_logits_t1 = outputs["sem_logits_t1"]
        sem_logits_t2 = outputs["sem_logits_t2"]
        if change_logits is None or sem_logits_t1 is None or sem_logits_t2 is None:
            raise ValueError("SecondSemanticChangeLoss requires change_logits, sem_logits_t1, and sem_logits_t2.")

        change_target = batch["change_mask"].float()
        label_a = batch["label_a"].long()
        label_b = batch["label_b"].long()
        valid_mask = batch.get("valid_mask")

        dice_loss = _binary_dice_loss(change_logits, change_target, valid_mask)
        focal_loss = _binary_focal_loss(change_logits, change_target, valid_mask, gamma=self.focal_gamma)
        change_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        sem_ce_t1 = self._semantic_ce(sem_logits_t1, label_a)
        sem_ce_t2 = self._semantic_ce(sem_logits_t2, label_b)
        semantic_ce_loss = self.ce_weight * (sem_ce_t1 + sem_ce_t2)

        sek_loss, average_sek, average_iou = self._mambafcs_eval_sek_loss(
            change_logits=change_logits,
            sem_logits_t1=sem_logits_t1,
            sem_logits_t2=sem_logits_t2,
            label_a=label_a,
            label_b=label_b,
            valid_mask=valid_mask,
        )
        sek_was_sanitized = False
        if not torch.isfinite(sek_loss):
            if self.safe_on_invalid:
                sek_loss = change_logits.new_zeros(())
                sek_was_sanitized = True
            else:
                raise FloatingPointError("SECOND semantic SeK loss became NaN/Inf.")

        change_logits = torch.clamp(change_logits.float(), -20.0, 20.0)
        p_change = torch.sigmoid(change_logits)
        p_sem_1 = torch.softmax(sem_logits_t1.float(), dim=1)
        p_sem_2 = torch.softmax(sem_logits_t2.float(), dim=1)
        semantic_change_prob = 1.0 - (p_sem_1 * p_sem_2).sum(dim=1, keepdim=True)
        if self.consistency_detach_semantic:
            semantic_change_prob = semantic_change_prob.detach()
        if self.consistency_loss_type == "mse":
            consistency_raw = (p_change - semantic_change_prob) ** 2
        else:
            consistency_raw = F.binary_cross_entropy_with_logits(
                change_logits,
                semantic_change_prob.clamp(0.0, 1.0),
                reduction="none",
            )
        consistency_loss = _masked_mean(consistency_raw, valid_mask)

        total = (
            self.change_loss_weight * change_loss
            + self.semantic_loss_weight * semantic_ce_loss
            + self.sek_loss_weight * sek_loss
            + self.consistency_loss_weight * consistency_loss
        )
        self.latest_stats = {
            "total_loss": float(total.detach().item()),
            "change_loss": float(change_loss.detach().item()),
            "semantic_ce_loss": float(semantic_ce_loss.detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "sek_loss": float(sek_loss.detach().item()),
            "dice_loss": float(dice_loss.detach().item()),
            "focal_loss": float(focal_loss.detach().item()),
            "soft_kappa": float(average_sek),
            "average_sek": float(average_sek),
            "average_iou": float(average_iou),
            "sek_was_sanitized": sek_was_sanitized,
        }
        return total
