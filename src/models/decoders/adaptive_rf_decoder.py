"""Adaptive RF decoder with data-driven dilation attention.

Each encoder stage uses parallel dilated convolution branches.
Fusion weights are predicted per-image via channel attention (GAP → FC →
softmax), giving a dynamic effective receptive field with no deformable ops.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNGELU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2 * d, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class _AdaptiveDilationBlock(nn.Module):
    """Softmax-gated mixture of dilated branches with channel attention."""

    def __init__(self, in_ch: int, out_ch: int, rates: List[int]) -> None:
        super().__init__()
        self.n        = len(rates)
        self.branches = nn.ModuleList([_ConvBNGELU(in_ch, out_ch, k=3, d=r) for r in rates])
        hidden        = max(in_ch // 4, self.n)
        self.attn     = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        branches = torch.stack([b(x) for b in self.branches], dim=1)   # B,n,C,H,W
        w = F.softmax(self.attn(x), dim=1).view(B, self.n, 1, 1, 1)
        return (w * branches).sum(dim=1)


class AdaptiveRFDecoder(nn.Module):
    """FPN decoder where each scale uses an adaptive-dilation RF block."""

    def __init__(
        self,
        channels: List[int],
        out_channels: int = 256,
        dilation_rates: Optional[List[int]] = None,
        **_kwargs,
    ) -> None:
        super().__init__()
        rates = dilation_rates or [1, 2, 4, 8]
        self.proj   = nn.ModuleList([_ConvBNGELU(c * 2, out_channels, k=1) for c in channels])
        self.arf    = nn.ModuleList([_AdaptiveDilationBlock(out_channels, out_channels, rates) for _ in channels])
        self.smooth = nn.ModuleList([_ConvBNGELU(out_channels, out_channels, k=3) for _ in channels])
        self.head   = nn.Sequential(
            _ConvBNGELU(out_channels, out_channels // 2, k=3),
            nn.Conv2d(out_channels // 2, 1, 1),
        )

    def forward(
        self,
        feats_a: List[torch.Tensor],
        feats_b: List[torch.Tensor],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, None]:
        fused = [
            arf(proj(torch.cat([torch.abs(fa - fb), fa + fb], dim=1)))
            for fa, fb, proj, arf in zip(feats_a, feats_b, self.proj, self.arf)
        ]
        top: Optional[torch.Tensor] = None
        for feat, smooth in zip(reversed(fused), reversed(self.smooth)):
            top = smooth(feat) if top is None else smooth(
                F.interpolate(top, size=feat.shape[-2:], mode="bilinear", align_corners=False) + feat
            )
        logits = F.interpolate(self.head(top), size=out_size, mode="bilinear", align_corners=False)
        return logits, None
