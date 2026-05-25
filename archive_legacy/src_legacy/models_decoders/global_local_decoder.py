"""Global-Local RF decoder.

Two-branch architecture separating fine spatial detail (local) from
semantic context (global), blended via a learned sigmoid gate.

Local branch  — stages 0–1: preserves boundary detail.
Global branch — stages 2–3 + global average-pool context.
Gate          — sigmoid-weighted blend of both branches.
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


class GlobalLocalDecoder(nn.Module):
    """FPN with separate local (shallow) and global (deep) paths + gate fusion."""

    def __init__(self, channels: List[int], out_channels: int = 256, **_kwargs) -> None:
        super().__init__()
        assert len(channels) == 4, "GlobalLocalDecoder expects exactly 4 encoder stages"

        # Local path — stages 0, 1
        self.local_proj   = nn.ModuleList([_ConvBNGELU(c * 2, out_channels, k=1) for c in channels[:2]])
        self.local_smooth = nn.ModuleList([_ConvBNGELU(out_channels, out_channels)  for _ in channels[:2]])

        # Global path — stages 2, 3
        self.global_proj   = nn.ModuleList([_ConvBNGELU(c * 2, out_channels, k=1) for c in channels[2:]])
        self.global_ctx    = _ConvBNGELU(channels[3], out_channels, k=1)
        self.global_smooth = nn.ModuleList([_ConvBNGELU(out_channels, out_channels)  for _ in channels[2:]])

        # Gated fusion
        self.gate = nn.Sequential(
            _ConvBNGELU(out_channels * 2, out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            _ConvBNGELU(out_channels, out_channels // 2),
            nn.Conv2d(out_channels // 2, 1, 1),
        )

    def _fpn(self, fused: List[torch.Tensor], smooths) -> torch.Tensor:
        top: Optional[torch.Tensor] = None
        for feat, smooth in zip(reversed(fused), reversed(smooths)):
            top = smooth(feat) if top is None else smooth(
                F.interpolate(top, size=feat.shape[-2:], mode="bilinear", align_corners=False) + feat
            )
        return top  # type: ignore[return-value]

    def forward(
        self,
        feats_a: List[torch.Tensor],
        feats_b: List[torch.Tensor],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, None]:
        # Local branch
        local_fused = [
            proj(torch.cat([torch.abs(fa - fb), fa + fb], dim=1))
            for fa, fb, proj in zip(feats_a[:2], feats_b[:2], self.local_proj)
        ]
        local_top = self._fpn(local_fused, self.local_smooth)

        # Global branch — inject global-pool context at deepest stage
        ctx = self.global_ctx(F.adaptive_avg_pool2d(feats_a[3] + feats_b[3], 1))
        global_fused = []
        for i, (fa, fb, proj) in enumerate(zip(feats_a[2:], feats_b[2:], self.global_proj)):
            f = proj(torch.cat([torch.abs(fa - fb), fa + fb], dim=1))
            if i == 1:
                f = f + ctx.expand_as(f)
            global_fused.append(f)
        global_top = self._fpn(global_fused, self.global_smooth)

        # Align to same spatial size then gate
        target_size = local_top.shape[-2:]
        global_top  = F.interpolate(global_top, size=target_size, mode="bilinear", align_corners=False)
        gate        = self.gate(torch.cat([local_top, global_top], dim=1))
        merged      = gate * local_top + (1.0 - gate) * global_top

        logits = F.interpolate(self.head(merged), size=out_size, mode="bilinear", align_corners=False)
        return logits, None
