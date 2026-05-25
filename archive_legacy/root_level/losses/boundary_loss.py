"""Boundary-supervised loss for change detection.

Generates a boundary GT from the binary change mask using Sobel edge detection
(or morphological dilation, configurable), then supervises the boundary branch.

Config keys (loss.boundary section)
---------------------------------------
    enabled:     true
    weight:      0.1
    type:        bce_dice   # loss type for boundary supervision
    target_type: sobel      # 'sobel' | 'morph'
    edge_width:  3          # dilation radius for morphological edge

Total boundary loss:
    L_edge = BCEWithLogitsLoss(boundary_logits, boundary_gt)
           + DiceLoss(boundary_logits, boundary_gt)

Final change prediction loss:
    L = L_final + coarse_weight * L_coarse + boundary_weight * L_edge
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Extract soft boundary map from a binary mask using Sobel gradients.

    Args:
        mask: [B, 1, H, W] float binary mask (values 0/1).
    Returns:
        [B, 1, H, W] boundary map, values in [0, 1].
    """
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                       device=mask.device, dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                       device=mask.device, dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(mask.float(), kx, padding=1)
    gy = F.conv2d(mask.float(), ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return torch.clamp(mag, 0.0, 1.0)


def _morph_boundary(mask: torch.Tensor, edge_width: int = 3) -> torch.Tensor:
    """Extract boundary via morphological dilation minus original mask.

    Args:
        mask:       [B, 1, H, W] float binary mask.
        edge_width: kernel radius for max-pool dilation.
    Returns:
        [B, 1, H, W] boundary map, values in {0, 1}.
    """
    k = edge_width * 2 + 1
    dilated = F.max_pool2d(mask.float(), k, stride=1, padding=k // 2)
    return torch.clamp(dilated - mask.float(), 0.0, 1.0)


def make_boundary_gt(
    mask: torch.Tensor,
    target_type: str = "sobel",
    edge_width: int = 3,
) -> torch.Tensor:
    """Create a boundary ground-truth map from a binary change mask.

    Args:
        mask:        [B, 1, H, W] float mask (0/1).
        target_type: 'sobel' or 'morph'.
        edge_width:  dilation radius (used only when target_type='morph').
    Returns:
        [B, 1, H, W] boundary map.
    """
    if target_type == "morph":
        return _morph_boundary(mask, edge_width)
    return _sobel_boundary(mask)


class BoundaryLoss(nn.Module):
    """Boundary-supervised loss.

    Computes BCE + Dice between predicted boundary logits and GT boundary map.
    The GT boundary is derived from the binary change mask via Sobel or
    morphological dilation.

    Args:
        loss_type:   'bce_dice' | 'bce' | 'dice'.
        target_type: 'sobel' | 'morph' for boundary GT generation.
        edge_width:  dilation radius (morph only).
    """

    def __init__(
        self,
        loss_type: str = "bce_dice",
        target_type: str = "sobel",
        edge_width: int = 3,
    ) -> None:
        super().__init__()
        self.loss_type   = loss_type
        self.target_type = target_type
        self.edge_width  = edge_width

    def forward(
        self,
        boundary_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute boundary supervision loss.

        Args:
            boundary_logits: [B, 1, H, W] predicted boundary logits.
            gt_mask:         [B, 1, H, W] binary change GT (0/1 float).
            valid_mask:      optional [B, H, W] bool valid pixel mask.
        Returns:
            Scalar loss.
        """
        gt_mask  = gt_mask.float()
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.unsqueeze(1)

        boundary_gt = make_boundary_gt(
            gt_mask, self.target_type, self.edge_width
        ).detach()   # GT is not differentiable

        logits = torch.clamp(boundary_logits.float(), -20.0, 20.0)

        # Compute loss components
        if valid_mask is not None:
            vmask = valid_mask.float()
            while vmask.ndim < logits.ndim:
                vmask = vmask.unsqueeze(1)
            bce = (F.binary_cross_entropy_with_logits(logits, boundary_gt, reduction="none") * vmask).sum() / (vmask.sum() + 1e-6)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, boundary_gt)

        if self.loss_type == "bce":
            return bce

        # Dice component
        probs = torch.sigmoid(logits)
        p, t = probs.view(-1), boundary_gt.view(-1)
        dice = 1.0 - (2.0 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)

        if self.loss_type == "dice":
            return dice
        return bce + dice


def build_boundary_loss(cfg: dict) -> Optional[BoundaryLoss]:
    """Build a boundary loss from config, or return None if disabled."""
    bnd_cfg = cfg.get("loss", {}).get("boundary", {})
    if not bool(bnd_cfg.get("enabled", True)):
        return None
    return BoundaryLoss(
        loss_type=str(bnd_cfg.get("type", "bce_dice")),
        target_type=str(bnd_cfg.get("target_type", "sobel")),
        edge_width=int(bnd_cfg.get("edge_width", 3)),
    )
