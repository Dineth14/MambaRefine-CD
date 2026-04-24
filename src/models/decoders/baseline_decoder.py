"""Baseline FPN change-detection decoder.

Fuses bi-temporal features via (abs_diff, sum) at each scale,
then builds a top-down FPN and produces a binary change logit map.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNGELU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class BaselineDecoder(nn.Module):
    """Simple FPN decoder with absolute-difference + sum feature fusion."""

    def __init__(self, channels: List[int], out_channels: int = 256, **_kwargs) -> None:
        super().__init__()
        self.proj   = nn.ModuleList([_ConvBNGELU(c * 2, out_channels, k=1) for c in channels])
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
            proj(torch.cat([torch.abs(fa - fb), fa + fb], dim=1))
            for fa, fb, proj in zip(feats_a, feats_b, self.proj)
        ]
        top: Optional[torch.Tensor] = None
        for feat, smooth in zip(reversed(fused), reversed(self.smooth)):
            top = feat if top is None else smooth(
                F.interpolate(top, size=feat.shape[-2:], mode="bilinear", align_corners=False) + feat
            )
            if top is feat:   # first iteration — still need smooth
                top = smooth(top)
        logits = F.interpolate(self.head(top), size=out_size, mode="bilinear", align_corners=False)
        return logits, None
