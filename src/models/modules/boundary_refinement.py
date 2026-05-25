"""Boundary residual refinement."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.heads import ConvNormGELU, depthwise_pointwise


class SobelEdge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx.to(x.dtype), padding=1)
        gy = F.conv2d(x, self.ky.to(x.dtype), padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)


class BoundaryRefinement(nn.Module):
    def __init__(self, boundary_channels: List[int], decoder_channels: int = 128, use_depthwise: bool = True) -> None:
        super().__init__()
        self.proj = nn.ModuleList([ConvNormGELU(c, decoder_channels, k=1) for c in boundary_channels])
        self.smooth = ConvNormGELU(decoder_channels, decoder_channels, k=3)
        self.sobel = SobelEdge()
        conv = depthwise_pointwise if use_depthwise else ConvNormGELU
        self.boundary_head = nn.Sequential(
            conv(decoder_channels, decoder_channels, k=3),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        self.refine_head = nn.Sequential(
            conv(decoder_channels + 2, decoder_channels, k=3),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)

    def _aggregate(self, boundary_feats: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        agg = None
        for feat, proj in zip(boundary_feats, self.proj):
            x = F.interpolate(proj(feat), size=out_size, mode="bilinear", align_corners=False)
            agg = x if agg is None else agg + x
        return self.smooth(agg / max(len(boundary_feats), 1))

    def forward(self, main_logits: torch.Tensor, boundary_feats: list[torch.Tensor], out_size: tuple[int, int] | None = None):
        out_size = out_size or main_logits.shape[-2:]
        b_feat = self._aggregate(boundary_feats, out_size)
        boundary_logits = self.boundary_head(b_feat)
        edge = self.sobel(torch.sigmoid(main_logits))
        raw_delta = self.refine_head(torch.cat([b_feat, main_logits, edge], dim=1))
        residual = 0.1 * torch.tanh(raw_delta)
        final_logits = main_logits + residual
        return final_logits, boundary_logits, residual
