"""Loss functions for change detection.

Supported loss types (config: loss.type)
-----------------------------------------
bce_dice         BCE + Dice
dice_focal       Dice + Focal
dice_focal_sek   Dice + Focal + SeK-inspired soft-kappa surrogate
second_semantic_cd   SECOND semantic change loss
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.sek_loss import SeKLoss


def _resolve_valid_mask(
    reference: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.float32)
    mask = valid_mask.float()
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape != reference.shape:
        mask = torch.broadcast_to(mask, reference.shape)
    return mask


def _masked_mean(values: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    mask = _resolve_valid_mask(values, valid_mask)
    denom = mask.sum()
    if float(denom.item()) <= 0.0:
        return values.new_zeros(())
    return (values * mask).sum() / denom


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def _boundary_map(values: torch.Tensor) -> torch.Tensor:
    kx, ky = _sobel_kernels(values.device, values.dtype)
    grad_x = F.conv2d(values, kx, padding=1)
    grad_y = F.conv2d(values, ky, padding=1)
    return torch.clamp(torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6), 0.0, 1.0)


def _boundary_l1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_edges = _boundary_map(torch.sigmoid(torch.clamp(logits.float(), -20.0, 20.0)))
    target_edges = _boundary_map(targets.float())
    return _masked_mean(torch.abs(pred_edges - target_edges), valid_mask)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = torch.clamp(logits.float(), -20.0, 20.0)
        targets = targets.float()
        valid = _resolve_valid_mask(targets, valid_mask)
        denom_valid = valid.sum()
        if float(denom_valid.item()) <= 0.0:
            return logits.new_zeros(())
        p = torch.sigmoid(logits)
        num = 2.0 * (p * targets * valid).sum() + self.smooth
        den = (p * valid).sum() + (targets * valid).sum() + self.smooth
        return 1.0 - num / den


class FocalLoss(nn.Module):
    """Binary focal loss with optional validity masking."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = torch.clamp(logits.float(), -20.0, 20.0)
        targets = targets.float()
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_w = (1.0 - p_t) ** self.gamma
        loss = focal_w * bce_raw

        if self.reduction == "sum":
            valid = _resolve_valid_mask(loss, valid_mask)
            return (loss * valid).sum()
        if self.reduction == "none":
            return loss
        return _masked_mean(loss, valid_mask)


class BCEDiceLoss(nn.Module):
    """BCE + Dice. Returns (total, bce, dice) for compatibility."""

    def __init__(self, bce_w: float = 1.0, dice_w: float = 1.0, boundary_w: float = 0.0) -> None:
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.boundary_w = boundary_w
        self.dice = DiceLoss()
        self.latest_stats: dict[str, float | str | bool] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.clamp(logits.float(), -20.0, 20.0)
        targets = targets.float()
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = _masked_mean(bce_raw, valid_mask)
        dice = self.dice(logits, targets, valid_mask=valid_mask)
        boundary = _boundary_l1_loss(logits, targets, valid_mask=valid_mask) if self.boundary_w > 0.0 else logits.new_zeros(())
        total = self.bce_w * bce + self.dice_w * dice + self.boundary_w * boundary
        self.latest_stats = {
            "total_loss": float(total.detach().item()),
            "bce_loss": float(bce.detach().item()),
            "dice_loss": float(dice.detach().item()),
            "boundary_loss": float(boundary.detach().item()),
            "focal_loss": 0.0,
            "sek_loss": 0.0,
            "soft_kappa": 0.0,
            "primary_name": "bce_loss",
            "sek_name": "",
            "sek_was_sanitized": False,
        }
        return total, bce, dice


class DiceFocalLoss(nn.Module):
    """Dice + Focal. Returns (total, focal, dice) for compatibility."""

    def __init__(
        self,
        dice_w: float = 1.0,
        focal_w: float = 0.3,
        focal_gamma: float = 1.5,
    ) -> None:
        super().__init__()
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.latest_stats: dict[str, float | str | bool] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dice = self.dice(logits, targets, valid_mask=valid_mask)
        focal = self.focal(logits, targets, valid_mask=valid_mask)
        total = self.dice_w * dice + self.focal_w * focal
        self.latest_stats = {
            "total_loss": float(total.detach().item()),
            "dice_loss": float(dice.detach().item()),
            "focal_loss": float(focal.detach().item()),
            "sek_loss": 0.0,
            "soft_kappa": 0.0,
            "primary_name": "focal_loss",
            "sek_name": "",
            "sek_was_sanitized": False,
        }
        return total, focal, dice


class DiceFocalSeKLoss(nn.Module):
    """Dice + Focal + SeK-inspired soft-kappa surrogate.

    Binary mode uses a binary soft-kappa / SeK-inspired loss.
    Semantic mode uses a general semantic soft-kappa surrogate and leaves
    separated semantic SeK as a future TODO.
    """

    def __init__(
        self,
        *,
        dice_w: float = 1.0,
        focal_w: float = 0.2,
        sek_w: float = 0.05,
        focal_gamma: float = 1.5,
        sek_mode: str = "binary",
        sek_eps: float = 1e-6,
        sek_separate_nochange: bool = False,
        num_classes: int = 7,
        ignore_index: int = 255,
        safe_on_invalid: bool = True,
    ) -> None:
        super().__init__()
        self.dice_w = float(dice_w)
        self.focal_w = float(focal_w)
        self.sek_w = float(sek_w)
        self.sek_mode = str(sek_mode).lower()
        self.safe_on_invalid = bool(safe_on_invalid)
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.sek = SeKLoss(
            mode=self.sek_mode,
            num_classes=num_classes,
            ignore_index=ignore_index,
            eps=sek_eps,
            separate_nochange=sek_separate_nochange,
        )
        self.latest_stats: dict[str, float | str | bool] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_semantic_mode = self.sek_mode == "semantic" or logits.shape[1] > 1
        if use_semantic_mode:
            dice = logits.new_zeros(())
            focal = logits.new_zeros(())
        else:
            dice = self.dice(logits, targets, valid_mask=valid_mask)
            focal = self.focal(logits, targets, valid_mask=valid_mask)
        sek_result = self.sek(logits, targets, valid_mask=valid_mask)
        sek_loss = sek_result.loss
        soft_kappa = sek_result.soft_kappa
        sek_was_sanitized = False

        if not torch.isfinite(sek_loss) or not torch.isfinite(soft_kappa):
            if self.safe_on_invalid:
                sek_loss = logits.new_zeros(())
                soft_kappa = logits.new_zeros(())
                sek_was_sanitized = True
            else:
                raise FloatingPointError("SeK-inspired loss became NaN/Inf and safe_on_invalid=false.")

        total = self.dice_w * dice + self.focal_w * focal + self.sek_w * sek_loss
        self.latest_stats = {
            "total_loss": float(total.detach().item()),
            "dice_loss": float(dice.detach().item()),
            "focal_loss": float(focal.detach().item()),
            "sek_loss": float(sek_loss.detach().item()),
            "soft_kappa": float(soft_kappa.detach().item()),
            "sek_mode": self.sek_mode,
            "primary_name": "focal_loss",
            "sek_name": "binary soft-kappa / SeK-inspired loss" if self.sek_mode == "binary" else "semantic soft-kappa / SeK-style surrogate",
            "sek_was_sanitized": sek_was_sanitized,
            "change_agreement": (
                None
                if sek_result.change_agreement is None
                else float(sek_result.change_agreement.detach().item())
            ),
        }
        return total, focal, dice


def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config."""
    lc = cfg.get("loss", {})
    dc = cfg.get("dataset", {})
    mc = cfg.get("model", {})
    tc = cfg.get("training", {})
    kind = str(lc.get("type", "bce_dice")).lower().replace("-", "_")

    if kind == "second_semantic_cd":
        from training.second_loss import SecondSemanticChangeLoss
        second_cfg = lc.get("second", {}) or {}

        return SecondSemanticChangeLoss(
            num_classes=int(dc.get("num_classes", mc.get("semantic_num_classes", 7))),
            ignore_index=int(second_cfg.get("ignore_index", dc.get("ignore_index", 255))),
            change_loss_weight=float(second_cfg.get("change_weight", lc.get("change_loss_weight", 1.0))),
            semantic_loss_weight=float(second_cfg.get("sem_ce_weight", lc.get("semantic_loss_weight", 0.5))),
            semantic_dice_weight=float(second_cfg.get("sem_dice_weight", lc.get("semantic_dice_weight", 0.0))),
            consistency_loss_weight=float(second_cfg.get("consistency_weight", lc.get("consistency_loss_weight", 0.2))),
            sek_loss_weight=float(lc.get("sek_loss_weight", 0.3)),
            dice_weight=float(lc.get("dice_weight", 1.0)),
            focal_weight=float(lc.get("focal_weight", 0.2)),
            focal_gamma=float(lc.get("focal_gamma", 1.5)),
            ce_weight=float(lc.get("ce_weight", 1.0)),
            sek_eps=float(lc.get("sek_eps", 1e-6)),
            consistency_detach_semantic=bool(lc.get("consistency_detach_semantic", True)),
            consistency_loss_type=str(lc.get("consistency_loss_type", "bce")),
            safe_on_invalid=bool(tc.get("skip_nan_steps", True)),
        )

    if kind == "dice_focal_sek":
        return DiceFocalSeKLoss(
            dice_w=float(lc.get("dice_weight", 1.0)),
            focal_w=float(lc.get("focal_weight", 0.2)),
            sek_w=float(lc.get("sek_weight", 0.05)),
            focal_gamma=float(lc.get("focal_gamma", 1.5)),
            sek_mode=str(lc.get("sek_mode", dc.get("mode", "binary"))),
            sek_eps=float(lc.get("sek_eps", 1e-6)),
            sek_separate_nochange=bool(lc.get("sek_separate_nochange", False)),
            num_classes=int(dc.get("num_classes", mc.get("semantic_num_classes", 7))),
            ignore_index=int(dc.get("ignore_index", 255)),
            safe_on_invalid=bool(tc.get("skip_nan_steps", True)),
        )

    if kind == "dice_focal":
        return DiceFocalLoss(
            dice_w=float(lc.get("dice_weight", 1.0)),
            focal_w=float(lc.get("focal_weight", 0.3)),
            focal_gamma=float(lc.get("focal_gamma", 1.5)),
        )

    return BCEDiceLoss(
        bce_w=float(lc.get("bce_weight", 1.0)),
        dice_w=float(lc.get("dice_weight", 1.0)),
        boundary_w=float(lc.get("boundary_weight", 0.0)),
    )
