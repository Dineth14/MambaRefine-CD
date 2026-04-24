"""Loss functions for binary change detection.

Supported loss types (config: loss.type)
-----------------------------------------
bce_dice   (default) — BCE + Dice
dice_focal             — Dice + Focal loss

Config keys
-----------
    loss:
      type: bce_dice         # bce_dice | dice_focal
      bce_weight:  1.0       # used by bce_dice
      dice_weight: 1.0
      focal_weight: 0.3      # used by dice_focal
      focal_gamma:  1.5
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).view(-1)
        t = targets.float().view(-1)
        num = 2.0 * (p * t).sum() + self.smooth
        den = p.sum() + t.sum() + self.smooth
        return 1.0 - num / den


class FocalLoss(nn.Module):
    """Binary focal loss.

    loss = -[ t·(1-p)^γ·log(p) + (1-t)·p^γ·log(1-p) ]

    Args:
        gamma: focusing parameter (higher = more focus on hard examples).
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p       = torch.sigmoid(logits)
        # p_t = p when target=1, (1-p) when target=0
        p_t     = p * targets + (1.0 - p) * (1.0 - targets)
        focal_w = (1.0 - p_t) ** self.gamma
        loss    = focal_w * bce_raw

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCEDiceLoss(nn.Module):
    """BCE + Dice.  Returns (total, bce, dice) for logging convenience."""

    def __init__(self, bce_w: float = 1.0, dice_w: float = 1.0) -> None:
        super().__init__()
        self.bce_w  = bce_w
        self.dice_w = dice_w
        self.bce    = nn.BCEWithLogitsLoss()
        self.dice   = DiceLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce   = self.bce(logits, targets.float())
        dice  = self.dice(logits, targets)
        total = self.bce_w * bce + self.dice_w * dice
        return total, bce, dice


class DiceFocalLoss(nn.Module):
    """Dice + Focal loss.  Returns (total, focal, dice) for logging convenience.

    Recommended config:
        dice_weight: 1.0
        focal_weight: 0.3
        focal_gamma: 1.5
    """

    def __init__(
        self,
        dice_w: float = 1.0,
        focal_w: float = 0.3,
        focal_gamma: float = 1.5,
    ) -> None:
        super().__init__()
        self.dice_w  = dice_w
        self.focal_w = focal_w
        self.dice    = DiceLoss()
        self.focal   = FocalLoss(gamma=focal_gamma)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dice  = self.dice(logits, targets)
        focal = self.focal(logits, targets)
        total = self.dice_w * dice + self.focal_w * focal
        # Return as (total, focal, dice) — focal takes the role of "bce" in logging
        return total, focal, dice


def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config.

    Supported types: 'bce_dice' (default), 'dice_focal'.
    """
    lc   = cfg.get("loss", {})
    kind = str(lc.get("type", "bce_dice")).lower().replace("-", "_")

    if kind == "dice_focal":
        return DiceFocalLoss(
            dice_w      = float(lc.get("dice_weight",  1.0)),
            focal_w     = float(lc.get("focal_weight", 0.3)),
            focal_gamma = float(lc.get("focal_gamma",  1.5)),
        )

    # Default: bce_dice (backward compatible)
    return BCEDiceLoss(
        bce_w  = float(lc.get("bce_weight",  1.0)),
        dice_w = float(lc.get("dice_weight", 1.0)),
    )
