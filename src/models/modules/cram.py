"""Cross-temporal attention for MambaRefine-CD.

CRAM refines each timestamp's feature map with context attended from the other
timestamp. Key/value tokens are adaptively pooled on large feature maps to keep
memory bounded while query tokens remain at the native spatial resolution.
"""
from __future__ import annotations

from math import sqrt
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.heads import group_norm


def _valid_heads(channels: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(channels)))
    while channels % heads != 0:
        heads -= 1
    return max(heads, 1)


def _pooled_size(height: int, width: int, max_tokens: int) -> tuple[int, int]:
    tokens = height * width
    if tokens <= max_tokens:
        return height, width
    scale = sqrt(float(max_tokens) / float(tokens))
    pooled_h = max(1, int(round(height * scale)))
    pooled_w = max(1, int(round(width * scale)))
    while pooled_h * pooled_w > max_tokens:
        if pooled_h >= pooled_w and pooled_h > 1:
            pooled_h -= 1
        elif pooled_w > 1:
            pooled_w -= 1
        else:
            break
    return pooled_h, pooled_w


class CrossTemporalAttention(nn.Module):
    """Bidirectional cross-attention for one encoder stage."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        max_kv_tokens: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_heads = _valid_heads(self.channels, num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.max_kv_tokens = max(1, int(max_kv_tokens))

        self.q = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.k = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.v = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1, bias=False),
            group_norm(self.channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _pool_kv_input(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled_h, pooled_w = _pooled_size(h, w, self.max_kv_tokens)
        if (pooled_h, pooled_w) == (h, w):
            return x
        return F.adaptive_avg_pool2d(x, (pooled_h, pooled_w))

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)

    def _attend(self, query_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = query_feat.shape
        context_feat = self._pool_kv_input(context_feat)

        q = self._tokens(self.q(query_feat))
        k = self._tokens(self.k(context_feat))
        v = self._tokens(self.v(context_feat))

        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1).to(v.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(b, self.channels, h, w)
        return self.out(out)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feat_a.shape != feat_b.shape:
            raise ValueError(f"CRAM shape mismatch: {tuple(feat_a.shape)} vs {tuple(feat_b.shape)}")
        a_from_b = self._attend(feat_a, feat_b)
        b_from_a = self._attend(feat_b, feat_a)
        return feat_a + self.gamma * a_from_b, feat_b + self.gamma * b_from_a


class CRAMBank(nn.Module):
    """Applies CRAM independently to every encoder scale."""

    def __init__(
        self,
        channels: List[int],
        num_heads: int = 4,
        max_kv_tokens: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossTemporalAttention(
                channels=c,
                num_heads=num_heads,
                max_kv_tokens=max_kv_tokens,
                dropout=dropout,
            )
            for c in channels
        ])

    def forward(
        self,
        feats_a: list[torch.Tensor],
        feats_b: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if len(feats_a) != len(self.blocks) or len(feats_b) != len(self.blocks):
            raise ValueError(
                f"CRAM expected {len(self.blocks)} feature scales, "
                f"got {len(feats_a)} and {len(feats_b)}."
            )
        refined_a: list[torch.Tensor] = []
        refined_b: list[torch.Tensor] = []
        for feat_a, feat_b, block in zip(feats_a, feats_b, self.blocks):
            out_a, out_b = block(feat_a, feat_b)
            refined_a.append(out_a)
            refined_b.append(out_b)
        return refined_a, refined_b
