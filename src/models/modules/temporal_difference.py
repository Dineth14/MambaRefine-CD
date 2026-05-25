"""Bitemporal difference features.

Modes:
    abs_only    : D = |F2 - F1|             channel_multiplier = 1
    signed_only : D = F2 - F1              channel_multiplier = 1
    abs_signed  : D = Cat(|F2-F1|, F2-F1) channel_multiplier = 2
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalDifference(nn.Module):
    VALID_MODES = ("abs_only", "signed_only", "abs_signed")

    def __init__(self, mode: str = "abs_signed"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"temporal_input_mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode

    @property
    def channel_multiplier(self) -> int:
        return 2 if self.mode == "abs_signed" else 1

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        if feat_a.shape != feat_b.shape:
            raise ValueError(f"TemporalDifference shape mismatch: {tuple(feat_a.shape)} vs {tuple(feat_b.shape)}")
        diff = feat_b - feat_a
        if self.mode == "abs_only":
            return diff.abs()
        if self.mode == "signed_only":
            return diff
        return torch.cat([diff.abs(), diff], dim=1)
