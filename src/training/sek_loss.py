"""Differentiable SeK-inspired surrogate losses.

Notes
-----
* True SeK is an evaluation metric derived from semantic confusion matrices.
* The functions below provide differentiable soft-kappa surrogates for training.
* Binary mode is a SeK-inspired approximation only. It is not true semantic SeK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_valid_mask(
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


def _safe_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


@dataclass
class SoftKappaResult:
    loss: torch.Tensor
    soft_kappa: torch.Tensor
    observed_agreement: torch.Tensor
    expected_agreement: torch.Tensor
    change_agreement: Optional[torch.Tensor] = None


def binary_soft_kappa_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    separate_nochange: bool = False,
) -> SoftKappaResult:
    """Binary soft-kappa / SeK-inspired loss for change detection."""
    logits = torch.clamp(logits.float(), -20.0, 20.0)
    target = target.float()
    valid = _as_valid_mask(target, valid_mask)
    valid_sum = valid.sum()
    if float(valid_sum.item()) <= 0.0:
        one = logits.new_tensor(1.0)
        return SoftKappaResult(
            loss=_safe_zero(logits),
            soft_kappa=one,
            observed_agreement=one,
            expected_agreement=_safe_zero(logits),
            change_agreement=one if separate_nochange else None,
        )

    p1 = torch.sigmoid(logits)
    p0 = 1.0 - p1
    y1 = target
    y0 = 1.0 - target

    tp = (p1 * y1 * valid).sum()
    tn = (p0 * y0 * valid).sum()
    fp = (p1 * y0 * valid).sum()
    fn = (p0 * y1 * valid).sum()

    total = tp + tn + fp + fn + eps
    po = (tp + tn) / total

    pred_pos = tp + fp
    pred_neg = tn + fn
    gt_pos = tp + fn
    gt_neg = tn + fp
    pe = (pred_pos * gt_pos + pred_neg * gt_neg) / ((total * total) + eps)
    soft_kappa = torch.clamp((po - pe) / (1.0 - pe + eps), -1.0, 1.0)

    change_agreement = None
    score = soft_kappa
    if separate_nochange:
        po_change = tp / (tp + fp + fn + eps)
        score = 0.5 * soft_kappa + 0.5 * po_change
        change_agreement = po_change

    loss = torch.clamp(1.0 - score, 0.0, 2.0)
    return SoftKappaResult(
        loss=loss,
        soft_kappa=soft_kappa,
        observed_agreement=po,
        expected_agreement=pe,
        change_agreement=change_agreement,
    )


def semantic_soft_sek_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    eps: float = 1e-6,
) -> SoftKappaResult:
    """General semantic soft-kappa surrogate.

    This uses a soft confusion matrix over all semantic classes. A separated
    semantic SeK variant that excludes no-change classes remains a future TODO.
    """
    logits = logits.float()
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    target = target.long()

    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if not bool(valid.any().item()):
        one = logits.new_tensor(1.0)
        return SoftKappaResult(
            loss=_safe_zero(logits),
            soft_kappa=one,
            observed_agreement=one,
            expected_agreement=_safe_zero(logits),
        )

    probs = F.softmax(torch.clamp(logits, -20.0, 20.0), dim=1)
    probs = probs.permute(0, 2, 3, 1)[valid]
    target_valid = target[valid]
    one_hot = F.one_hot(target_valid, num_classes=num_classes).float()

    confmat = probs.transpose(0, 1) @ one_hot
    total = confmat.sum()
    po = torch.trace(confmat) / (total + eps)
    row = confmat.sum(dim=1)
    col = confmat.sum(dim=0)
    pe = (row * col).sum() / ((total * total) + eps)
    soft_kappa = torch.clamp((po - pe) / (1.0 - pe + eps), -1.0, 1.0)
    loss = torch.clamp(1.0 - soft_kappa, 0.0, 2.0)
    return SoftKappaResult(
        loss=loss,
        soft_kappa=soft_kappa,
        observed_agreement=po,
        expected_agreement=pe,
    )


class SeKLoss(nn.Module):
    """Wrapper for binary soft-kappa and future semantic soft-kappa losses."""

    def __init__(
        self,
        mode: str = "binary",
        num_classes: int = 7,
        ignore_index: int = 255,
        eps: float = 1e-6,
        separate_nochange: bool = False,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.separate_nochange = bool(separate_nochange)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> SoftKappaResult:
        if self.mode == "semantic":
            return semantic_soft_sek_loss(
                logits=logits,
                target=target,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                eps=self.eps,
            )
        return binary_soft_kappa_loss(
            logits=logits,
            target=target,
            valid_mask=valid_mask,
            eps=self.eps,
            separate_nochange=self.separate_nochange,
        )
