"""BCE + Dice combined loss for binary change detection."""
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


class BCEDiceLoss(nn.Module):
    """Returns (total, bce, dice) for logging convenience."""

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


def build_loss(cfg: dict) -> BCEDiceLoss:
    lc = cfg.get("loss", {})
    return BCEDiceLoss(
        bce_w=float(lc.get("bce_weight", 1.0)),
        dice_w=float(lc.get("dice_weight", 1.0)),
    )
