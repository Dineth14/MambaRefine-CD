from __future__ import annotations

import torch.nn as nn


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvNormGELU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2 * d, dilation=d, bias=False),
            group_norm(out_ch),
            nn.GELU(),
        )


def depthwise_pointwise(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, k, padding=k // 2, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        group_norm(out_ch),
        nn.GELU(),
    )
