"""Differential Region-Boundary Interaction.

Input:
    D_in^s [B, C_in, H, W]

Processing:
    D^s = diff_projection(D_in^s)
    R^s = D^s * G_r^s
    B^s = D^s * G_b^s
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.heads import depthwise_pointwise, group_norm

_GATE_LOGIT_CLAMP = 8.0
_SOBEL_MAG_CLAMP = 10.0


def _gate_mlp(channels: int, hidden_ratio: float = 0.25) -> nn.Sequential:
    hidden = max(int(channels * hidden_ratio), 1)
    return nn.Sequential(
        nn.Conv2d(channels, hidden, 1, bias=True),
        nn.GELU(),
        nn.Conv2d(hidden, channels, 1, bias=True),
    )


class SobelGradient(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(x.dtype)
        ky = self.ky.to(x.dtype)
        gx = F.conv2d(x, kx, padding=1, groups=self.groups)
        gy = F.conv2d(x, ky, padding=1, groups=self.groups)
        return torch.clamp(torch.sqrt(gx ** 2 + gy ** 2 + 1e-6), 0.0, _SOBEL_MAG_CLAMP)


class DRBI(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_depthwise: bool = True,
        gate_hidden_ratio: float = 0.25,
        region_gate_min: float = 0.2,
        region_gate_max: float = 0.8,
        boundary_gate_min: float = 0.0,
        boundary_gate_max: float = 0.4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.region_gate_min = float(region_gate_min)
        self.region_gate_max = float(region_gate_max)
        self.boundary_gate_min = float(boundary_gate_min)
        self.boundary_gate_max = float(boundary_gate_max)

        self.diff_projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            group_norm(out_channels),
            nn.GELU(),
        )
        self.spatial = depthwise_pointwise(out_channels, out_channels) if use_depthwise else nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            group_norm(out_channels),
            nn.GELU(),
        )
        self.region_gate = _gate_mlp(out_channels, gate_hidden_ratio)
        self.sobel = SobelGradient(out_channels)
        self.boundary_gate = _gate_mlp(out_channels, gate_hidden_ratio)

    def forward(self, d_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if d_in.shape[1] != self.in_channels:
            raise RuntimeError(f"D-RBI expected {self.in_channels} input channels, got {d_in.shape[1]}")
        d = self.diff_projection(d_in)
        d = self.spatial(d)

        g_r_logits = torch.clamp(self.region_gate(d), -_GATE_LOGIT_CLAMP, _GATE_LOGIT_CLAMP)
        g_r = self.region_gate_min + (self.region_gate_max - self.region_gate_min) * torch.sigmoid(g_r_logits)
        region = d * g_r

        edge = self.sobel(d)
        g_b_logits = torch.clamp(self.boundary_gate(edge), -_GATE_LOGIT_CLAMP, _GATE_LOGIT_CLAMP)
        g_b = self.boundary_gate_min + (self.boundary_gate_max - self.boundary_gate_min) * torch.sigmoid(g_b_logits)
        boundary = d * g_b
        return region, boundary
