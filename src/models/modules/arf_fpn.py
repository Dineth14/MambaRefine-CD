"""Adaptive RF FPN decoder."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.heads import ConvNormGELU


class _AdaptiveDilationBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: List[int]) -> None:
        super().__init__()
        self.n = len(rates)
        self.branches = nn.ModuleList([ConvNormGELU(in_ch, out_ch, k=3, d=r) for r in rates])
        hidden = max(in_ch // 4, self.n)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        branches = torch.stack([branch(x) for branch in self.branches], dim=1)
        weights = F.softmax(self.attn(x), dim=1).view(b, self.n, 1, 1, 1)
        return (weights * branches).sum(dim=1)


class ARFFPN(nn.Module):
    def __init__(self, in_channels: List[int], decoder_channels: int = 128, dilation_rates: List[int] | None = None) -> None:
        super().__init__()
        rates = dilation_rates or [1, 2, 4, 8]
        self.proj = nn.ModuleList([ConvNormGELU(c, decoder_channels, k=1) for c in in_channels])
        self.arf = nn.ModuleList([_AdaptiveDilationBlock(decoder_channels, decoder_channels, rates) for _ in in_channels])
        self.smooth = nn.ModuleList([ConvNormGELU(decoder_channels, decoder_channels, k=3) for _ in in_channels])
        self.head = nn.Sequential(
            ConvNormGELU(decoder_channels, decoder_channels // 2, k=3),
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )

    def _fpn(self, feats: list[torch.Tensor]) -> torch.Tensor:
        top = None
        for feat, smooth in zip(reversed(feats), reversed(self.smooth)):
            if top is None:
                top = smooth(feat)
            else:
                top = smooth(F.interpolate(top, size=feat.shape[-2:], mode="bilinear", align_corners=False) + feat)
        return top

    def forward(self, region_feats: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        fused = [arf(proj(feat)) for feat, proj, arf in zip(region_feats, self.proj, self.arf)]
        top = self._fpn(fused)
        return F.interpolate(self.head(top), size=out_size, mode="bilinear", align_corners=False)
