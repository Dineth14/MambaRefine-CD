"""Binary change detection losses.

Supported loss types (config: loss.final.type)
-----------------------------------------------
    bce_dice     BCE + Dice
    bce          Binary Cross Entropy only
    dice         Dice only
    focal_dice   Focal + Dice

Total binary CD loss:
    L = L_final + coarse_weight * L_coarse + boundary_weight * L_boundary

Config keys
-----------
    loss.final.type:        bce_dice
    loss.coarse.enabled:    true
    loss.coarse.weight:     0.4
    loss.boundary.enabled:  true
    loss.boundary.weight:   0.1
    loss.boundary.type:     bce_dice  # used for boundary supervision
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is available
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(ch: int) -> nn.GroupNorm:
    for g in range(min(32, ch), 0, -1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------

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
        logits  = torch.clamp(logits.float(), -20.0, 20.0)
        targets = targets.float()
        probs   = torch.sigmoid(logits)

        if valid_mask is not None:
            mask = valid_mask.float().view(-1)
            p = probs.view(-1) * mask
            t = targets.view(-1) * mask
        else:
            p = probs.view(-1)
            t = targets.view(-1)

        numerator   = 2.0 * (p * t).sum() + self.smooth
        denominator = p.sum() + t.sum() + self.smooth
        return 1.0 - numerator / denominator


# ---------------------------------------------------------------------------
# BCE + Dice loss
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_w  = bce_weight
        self.dice_w = dice_weight
        self.dice   = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits  = torch.clamp(logits.float(), -20.0, 20.0)
        targets = targets.float()

        if valid_mask is not None:
            mask = valid_mask.float()
            while mask.ndim < logits.ndim:
                mask = mask.unsqueeze(1)
            bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * mask).sum() / (mask.sum() + 1e-6)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        return self.bce_w * bce + self.dice_w * self.dice(logits, targets, valid_mask)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits  = torch.clamp(logits.float(), -20.0, 20.0)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt    = probs * targets + (1.0 - probs) * (1.0 - targets)
        fl    = ((1.0 - pt) ** self.gamma) * bce
        if valid_mask is not None:
            mask = valid_mask.float()
            while mask.ndim < fl.ndim:
                mask = mask.unsqueeze(1)
            return (fl * mask).sum() / (mask.sum() + 1e-6)
        return fl.mean()


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.focal_w = focal_weight
        self.dice_w  = dice_weight
        self.focal   = FocalLoss(gamma)
        self.dice    = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (
            self.focal_w * self.focal(logits, targets, valid_mask)
            + self.dice_w * self.dice(logits, targets, valid_mask)
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_binary_loss(cfg: dict) -> nn.Module:
    """Build a binary CD loss from config (loss.final section)."""
    loss_cfg  = cfg.get("loss", {})
    final_cfg = loss_cfg.get("final", loss_cfg)   # fallback to root loss cfg
    ltype     = str(final_cfg.get("type", "bce_dice")).lower()

    if ltype == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(final_cfg.get("bce_weight", 1.0)),
            dice_weight=float(final_cfg.get("dice_weight", 1.0)),
        )
    elif ltype == "bce":
        return BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
    elif ltype == "dice":
        return BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)
    elif ltype in ("focal_dice", "dice_focal"):
        return FocalDiceLoss(
            focal_weight=float(final_cfg.get("focal_weight", 1.0)),
            dice_weight=float(final_cfg.get("dice_weight", 1.0)),
            gamma=float(final_cfg.get("focal_gamma", 2.0)),
        )
    else:
        raise ValueError(f"Unknown binary loss type: {ltype!r}")
