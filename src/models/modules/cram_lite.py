"""CRAMLite — Change-Region-Aware attention Module (Lightweight).

Inspired by Mamba-CD's change-region-aware mechanism, but implemented
as a pure-CNN spatial attention module for GPU-friendliness.
No transformer attention, no multi-head self-attention.

Architecture (per stage)
------------------------
Input: F_i  [B, C, H, W]  (difference feature from D-RBI or decoder)

1. DW Conv 3x3  (channel-wise spatial mixing)
2. GroupNorm
3. GELU
4. PW Conv 1x1  (channel refinement)
5. GroupNorm
6. GELU
7. PW Conv 1x1 -> 1 channel  (spatial attention map)
8. Sigmoid -> A_i in [0, 1]

Attention application (residual):
    F_out = F_i * (1 + alpha * A_i)

where alpha is a learnable scalar initialized to the config default.
The residual formulation ensures that at alpha=0 (or when A_i=0.5) the
module is a near-identity pass-through.

Config keys (under model.cram_lite)
-------------------------------------
    enabled:        true
    alpha:          0.5   # initial residual scale
    apply_stages:   [0, 1, 2]   # which D-RBI output scales to apply at
    attention_type: spatial  # only 'spatial' is supported
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm choosing the largest num_groups <= 32 that divides channels."""
    for g in range(min(32, channels), 0, -1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


class CRAMLite(nn.Module):
    """Lightweight change-region-aware spatial attention.

    Args:
        channels:       Number of input/output channels.
        alpha_init:     Initial value for the learnable residual scale.
    """

    def __init__(self, channels: int, alpha_init: float = 0.5) -> None:
        super().__init__()
        # Stage 1: depthwise 3x3 spatial mixing
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm1 = _group_norm(channels)
        # Stage 2: pointwise channel refinement
        self.pw1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = _group_norm(channels)
        # Stage 3: collapse to 1-channel spatial attention
        self.pw2 = nn.Conv2d(channels, 1, 1, bias=True)
        self.act = nn.GELU()
        # Learnable residual scale (initialized from config)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input difference feature.
        Returns:
            [B, C, H, W] attention-modulated feature.
        """
        # Compute spatial attention map A in [0,1]
        a = self.act(self.norm1(self.dw(x)))
        a = self.act(self.norm2(self.pw1(a)))
        a = torch.sigmoid(self.pw2(a))           # [B, 1, H, W]
        # Residual attention: F_out = F_in * (1 + alpha * A)
        return x * (1.0 + self.alpha * a)


class CRAMLiteBank(nn.Module):
    """Collection of CRAMLite modules for multiple encoder stages.

    Args:
        channels_list:  Channel count at each stage (e.g. [80, 160, 320, 640]).
        apply_stages:   Which stage indices to apply CRAM-lite at.
        alpha_init:     Initial residual scale.
    """

    def __init__(
        self,
        channels_list: list[int],
        apply_stages: list[int],
        alpha_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.apply_stages = set(apply_stages)
        mods: dict[str, CRAMLite] = {}
        for i in apply_stages:
            if 0 <= i < len(channels_list):
                mods[str(i)] = CRAMLite(channels_list[i], alpha_init)
        self.modules_dict = nn.ModuleDict(mods)

    def apply(self, features: list[torch.Tensor]) -> list[torch.Tensor]:  # type: ignore[override]
        """Apply CRAMLite to the specified stages.

        Args:
            features: List of [B, C_i, H_i, W_i] tensors (one per scale).
        Returns:
            List of same-shape tensors with attention applied at configured stages.
        """
        out = []
        for i, f in enumerate(features):
            key = str(i)
            if key in self.modules_dict:
                out.append(self.modules_dict[key](f))
            else:
                out.append(f)
        return out
