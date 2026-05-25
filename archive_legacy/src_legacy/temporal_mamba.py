"""Temporal Mamba fusion module for change detection.

Treats (F1, F2) as a 2-step temporal sequence and processes each spatial
position through a Mamba SSM, then returns a single [B, C, H, W] change
representation from the last timestep.

Usage
-----
    fuser = TemporalMamba(dim=80)                # one instance per encoder scale
    H = fuser(f1, f2)                            # [B, C, H, W]

Shapes at each stage
---------------------
    f1, f2          : [B, C, H, W]     (individual encoder feature maps)
    stacked         : [B, 2, C, H, W]  (T=2 temporal sequence)
    permuted        : [B, H, W, 2, C]  (spatial axes before sequence)
    x_seq           : [B*H*W, T=2, C]  (each pixel is a sequence of length 2)
    h_seq           : [B*H*W, T=2, C]  (Mamba output, same layout)
    h_last          : [B*H*W, C]       (last-timestep representation)
    H               : [B, C, H, W]     (reshaped back; residual +F2 applied)

Memory notes
------------
* x_seq is never stored as a field — it lives only inside forward().
* Mamba's CUDA scan is causal in T, so gradient flows correctly.
* For T=2 the sequence is minimal but still exercises the SSM's state.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba


class TemporalMamba(nn.Module):
    """Per-pixel temporal Mamba fusion for a pair of feature maps.

    Args:
        dim:      Channel dimension C (must match the encoder output channels
                  for the target scale).
        d_state:  Mamba SSM state dimension. Default: 16.
        d_conv:   Mamba 1-D convolution width. Default: 4.
        expand:   Inner expansion factor for Mamba. Default: 2.
        residual: If True, adds F2 as a skip connection to the output.
                  This stabilises early training (output starts near F2).
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.dim      = dim
        self.residual = residual

        # Layer-norm applied before Mamba (pre-norm convention)
        self.norm  = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f1: Pre-change features  [B, C, H, W]
            f2: Post-change features [B, C, H, W]

        Returns:
            H: Change representation [B, C, H, W]
        """
        B, C, H, W = f1.shape

        # ── 1. Stack into temporal sequence ───────────────────────────
        # [B, 2, C, H, W]
        x = torch.stack([f1, f2], dim=1)

        # ── 2. Rearrange: spatial dims first, then temporal, then channel
        # [B, H, W, 2, C]
        x = x.permute(0, 3, 4, 1, 2)

        # ── 3. Flatten spatial batch dimension for Mamba ───────────────
        # [B*H*W, T=2, C]
        x_seq = x.reshape(B * H * W, 2, C)

        # ── 4. Pre-norm + Mamba SSM ────────────────────────────────────
        x_seq = self.norm(x_seq)
        h_seq = self.mamba(x_seq)   # [B*H*W, T=2, C]

        # ── 5. Take last timestep (most recent context) ────────────────
        h_last = h_seq[:, -1, :]    # [B*H*W, C]

        # ── 6. Reshape back to spatial map ────────────────────────────
        H_map = h_last.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # ── 7. Optional residual connection (stabilises training) ──────
        if self.residual:
            H_map = H_map + f2

        return H_map
