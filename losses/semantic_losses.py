"""Semantic change detection losses for SECOND.

Implements configurable CE + Dice + optional SeK loss.

Config keys (loss.semantic section)
---------------------------------------
    ce_weight:    1.0
    dice_weight:  1.0
    sek_weight:   0.1
    use_sek:      true
    ignore_index: 255
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CE loss (supports ignore_index)
# ---------------------------------------------------------------------------

class SemanticCELoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W]
            targets: [B, H, W] long
        """
        return self.ce(logits.float(), targets.long())


# ---------------------------------------------------------------------------
# Semantic Dice loss
# ---------------------------------------------------------------------------

class SemanticDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.smooth        = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W]
            targets: [B, H, W] long
        """
        B, C, H, W = logits.shape
        probs = torch.softmax(logits.float(), dim=1)           # [B, C, H, W]
        tgt   = targets.long().view(-1)
        valid = tgt != self.ignore_index
        flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, C) # [N, C]
        flat_probs = flat_probs[valid]
        tgt_valid  = tgt[valid]
        if tgt_valid.numel() == 0:
            return logits.new_zeros(())
        one_hot = torch.zeros(tgt_valid.shape[0], C, device=logits.device)
        one_hot.scatter_(1, tgt_valid.unsqueeze(1), 1.0)
        num   = 2.0 * (flat_probs * one_hot).sum(0) + self.smooth
        den   = flat_probs.sum(0) + one_hot.sum(0) + self.smooth
        return (1.0 - num / den).mean()


# ---------------------------------------------------------------------------
# Combined semantic loss
# ---------------------------------------------------------------------------

class SemanticChangeLoss(nn.Module):
    """Combined CE + Dice + optional SeK loss for SECOND SCD.

    Args:
        num_classes:  Number of semantic classes.
        ignore_index: Ignore label.
        ce_weight:    CE loss weight.
        dice_weight:  Dice loss weight.
        sek_weight:   SeK loss weight (0 disables SeK).
        use_sek:      Whether to include SeK in the loss.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        sek_weight: float = 0.1,
        use_sek: bool = True,
    ) -> None:
        super().__init__()
        self.ce_w    = ce_weight
        self.dice_w  = dice_weight
        self.sek_w   = sek_weight
        self.use_sek = use_sek and sek_weight > 0.0

        self.ce   = SemanticCELoss(num_classes, ignore_index)
        self.dice = SemanticDiceLoss(num_classes, ignore_index)

        if self.use_sek:
            # Re-use existing SeK loss from src/
            try:
                from training.sek_loss import binary_soft_kappa_loss
                self._sek_fn = binary_soft_kappa_loss
            except ImportError:
                self.use_sek = False
                self._sek_fn = None
        else:
            self._sek_fn = None

    def forward(
        self,
        logits_t1: torch.Tensor,
        logits_t2: torch.Tensor,
        gt_t1: torch.Tensor,
        gt_t2: torch.Tensor,
        change_logits: Optional[torch.Tensor] = None,
        change_gt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total semantic CD loss.

        Args:
            logits_t1:     [B, C, H, W] semantic logits at t1.
            logits_t2:     [B, C, H, W] semantic logits at t2.
            gt_t1:         [B, H, W] long semantic labels at t1.
            gt_t2:         [B, H, W] long semantic labels at t2.
            change_logits: [B, 1, H, W] optional binary change logits for SeK.
            change_gt:     [B, H, W] optional binary change GT.
        """
        # Per-timestamp CE + Dice
        l_ce   = self.ce(logits_t1, gt_t1) + self.ce(logits_t2, gt_t2)
        l_dice = self.dice(logits_t1, gt_t1) + self.dice(logits_t2, gt_t2)
        total  = self.ce_w * l_ce + self.dice_w * l_dice

        # Optional SeK loss on binary change
        if self.use_sek and self._sek_fn is not None and change_logits is not None and change_gt is not None:
            sek_result = self._sek_fn(change_logits, change_gt.float())
            l_sek = sek_result.loss
            total = total + self.sek_w * l_sek

        return total


def build_semantic_loss(cfg: dict, num_classes: int) -> SemanticChangeLoss:
    """Build a semantic CD loss from config."""
    sem_cfg = cfg.get("loss", {}).get("semantic", {})
    return SemanticChangeLoss(
        num_classes=num_classes,
        ignore_index=int(cfg.get("dataset", {}).get("ignore_index", 255)),
        ce_weight=float(sem_cfg.get("ce_weight", 1.0)),
        dice_weight=float(sem_cfg.get("dice_weight", 1.0)),
        sek_weight=float(sem_cfg.get("sek_weight", 0.1)),
        use_sek=bool(sem_cfg.get("use_sek", True)),
    )
