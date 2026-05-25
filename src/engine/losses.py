"""Loss functions for MambaRefine-CD. All prediction inputs are logits."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(logits, mask):
    return F.binary_cross_entropy_with_logits(logits.float(), mask.float())


def dice_loss(logits, mask, smooth: float = 1.0):
    probs = torch.sigmoid(logits.float())
    mask = mask.float()
    numerator = 2.0 * (probs * mask).sum() + smooth
    denominator = probs.sum() + mask.sum() + smooth
    return 1.0 - numerator / denominator


def boundary_loss(boundary_logits, mask):
    return bce_loss(boundary_logits, mask) + dice_loss(boundary_logits, mask)


def residual_regularization(residual):
    return residual.float().abs().mean()


class ComputeLoss:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, outputs, mask):
        lc = self.cfg.loss
        final_logits = outputs["logits"]
        main_logits = outputs["main_logits"]
        boundary_logits = outputs["boundary_logits"]
        residual = outputs["residual"]

        loss_bce = bce_loss(final_logits, mask)
        loss_dice = dice_loss(final_logits, mask)
        loss_aux = bce_loss(main_logits, mask) + dice_loss(main_logits, mask)
        loss_boundary = boundary_loss(boundary_logits, mask)
        loss_residual = residual_regularization(residual)
        total = (
            float(lc.bce_weight) * loss_bce
            + float(lc.dice_weight) * loss_dice
            + float(lc.aux_weight) * loss_aux
            + float(lc.boundary_weight) * loss_boundary
            + float(lc.residual_reg_weight) * loss_residual
        )
        return total, {
            "loss_total": float(total.detach().item()),
            "loss_bce": float(loss_bce.detach().item()),
            "loss_dice": float(loss_dice.detach().item()),
            "loss_aux": float(loss_aux.detach().item()),
            "loss_boundary": float(loss_boundary.detach().item()),
            "loss_residual_reg": float(loss_residual.detach().item()),
        }


def build_loss(cfg):
    return ComputeLoss(cfg)
