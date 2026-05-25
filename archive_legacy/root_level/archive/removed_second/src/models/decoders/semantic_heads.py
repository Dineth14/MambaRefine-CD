"""Lightweight semantic decoder heads for semantic change detection."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class _ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )


class LightweightSemanticHead(nn.Module):
    """Shared semantic decoder head for timestamp-wise class prediction.

    The head consumes a multi-scale encoder feature list, projects each scale to a
    common width, performs top-down fusion, and predicts per-pixel semantic logits.
    """

    def __init__(
        self,
        channels: Sequence[int],
        num_classes: int,
        hidden_channels: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        drop = nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                _group_norm(hidden_channels),
                nn.GELU(),
            )
            for in_channels in channels
        ])
        self.smooth = nn.ModuleList([
            _ConvGNAct(hidden_channels, hidden_channels, kernel_size=3)
            for _ in channels
        ])
        self.classifier = nn.Sequential(
            _ConvGNAct(hidden_channels, hidden_channels, kernel_size=3),
            drop,
            nn.Conv2d(hidden_channels, num_classes, 1),
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        out_size: tuple[int, int],
    ) -> torch.Tensor:
        fused = [proj(feature) for proj, feature in zip(self.proj, features)]
        top: torch.Tensor | None = None
        for feature, smooth in zip(reversed(fused), reversed(self.smooth)):
            if top is None:
                top = smooth(feature)
                continue
            top = F.interpolate(top, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            top = smooth(feature + top)
        assert top is not None
        logits = self.classifier(top)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
