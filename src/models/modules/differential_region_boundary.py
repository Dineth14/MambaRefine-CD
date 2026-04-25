"""Differential Region-Boundary Interaction (D-RBI) Module.

Numerical-stability design
--------------------------
1. Pre-normalisation (pre_norm=True, default): F1 and F2 are each passed
   through an independent GroupNorm BEFORE computing differences/products.
   This prevents large-magnitude encoder features from exploding in the
   product term F1n * F2n.

2. Product scaling (product_scale, default 0.25): the product term is
   scaled down before concatenation because F1n * F2n can still have
   variance ~C even after normalisation.

3. Sobel clamp: gradient magnitude is clamped to [0, 10] after sqrt to
   prevent large values from saturating the gate MLP.

4. Gate logit clamp: gate logits are hard-clamped to [-8, 8] before
   sigmoid, so gates are always in a finite, well-behaved range.

5. Gate bounds: region_gate_max <= 0.8, boundary_gate_max <= 0.4 by default,
   preventing full-pass gates that allow gradient explosion.

Safe ablation defaults
----------------------
    pre_norm      = True
    use_product   = False  (re-enable carefully once stable)
    product_scale = 0.25
    boundary_gate_max = 0.4

Re-enabling product later
--------------------------
    difference:
      use_product: true
      product_scale: 0.1   # start conservative
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_GATE_LOGIT_CLAMP = 8.0   # hard clamp for gate logits before sigmoid
_SOBEL_MAG_CLAMP  = 10.0  # hard clamp for Sobel gradient magnitude


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with num_groups <= 32 that evenly divides channels."""
    for g in range(min(32, channels), 0, -1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)  # fallback: LayerNorm-style


def _dw_pw(in_ch: int, out_ch: int) -> nn.Sequential:
    """Depthwise 3x3 + pointwise 1x1, GroupNorm, GELU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        _group_norm(out_ch),
        nn.GELU(),
    )


def _pw(in_ch: int, out_ch: int) -> nn.Sequential:
    """Pointwise 1x1, GroupNorm, GELU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        _group_norm(out_ch),
        nn.GELU(),
    )


def _gate_mlp(channels: int, hidden_ratio: float) -> nn.Sequential:
    """Lightweight 1x1 -> GELU -> 1x1 gate MLP."""
    hidden = max(int(channels * hidden_ratio), 1)
    return nn.Sequential(
        nn.Conv2d(channels, hidden, 1, bias=True),
        nn.GELU(),
        nn.Conv2d(hidden, channels, 1, bias=True),
    )


# ---------------------------------------------------------------------------
# Fixed Sobel gradient (not learnable)
# ---------------------------------------------------------------------------

class _SobelGradient(nn.Module):
    """Fixed depthwise Sobel magnitude for a C-channel feature map.

    Stability: magnitude is clamped to [0, _SOBEL_MAG_CLAMP] after sqrt.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        kx = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        ky = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(x.dtype)
        ky = self.ky.to(x.dtype)
        gx = F.conv2d(x, kx, padding=1, groups=self.groups)
        gy = F.conv2d(x, ky, padding=1, groups=self.groups)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return torch.clamp(grad_mag, 0.0, _SOBEL_MAG_CLAMP)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DifferentialRegionBoundaryInteraction(nn.Module):
    """Per-scale D-RBI fusion for a pair of bi-temporal feature maps.

    Args:
        in_channels       : C -- input channel width (encoder output at this scale)
        out_channels      : C_out -- output/internal channel width
        use_depthwise     : use depthwise-separable convolutions (default True)
        gate_hidden_ratio : gate MLP hidden width ratio (default 0.25)
        region_gate_min   : minimum region gate value
        region_gate_max   : maximum region gate value (keep <= 0.8 for stability)
        boundary_gate_min : minimum boundary gate value
        boundary_gate_max : maximum boundary gate value (keep <= 0.4 for stability)
        use_product       : include F1n*F2n in input (default False -- numerically risky)
        product_scale     : scale factor applied to product term (default 0.25)
        use_absdiff       : include |F2n-F1n| in input (default True)
        pre_norm          : apply GroupNorm to F1,F2 before diff/product (default True)
        use_region_gate   : apply learnable region gate G_r (ablation switch)
        use_boundary_gate : apply Sobel-conditioned gate G_b (ablation switch)
        return_debug      : if True, include gates in output dict
    """

    def __init__(
        self,
        in_channels       : int,
        out_channels      : int   = 256,
        use_depthwise     : bool  = True,
        gate_hidden_ratio : float = 0.25,
        region_gate_min   : float = 0.2,
        region_gate_max   : float = 0.8,
        boundary_gate_min : float = 0.0,
        boundary_gate_max : float = 0.4,
        use_product       : bool  = False,
        product_scale     : float = 0.25,
        use_absdiff       : bool  = True,
        pre_norm          : bool  = True,
        use_region_gate   : bool  = True,
        use_boundary_gate : bool  = True,
        return_debug      : bool  = False,
    ) -> None:
        super().__init__()

        self.out_ch            = out_channels
        self.region_gate_min   = region_gate_min
        self.region_gate_max   = region_gate_max
        self.boundary_gate_min = boundary_gate_min
        self.boundary_gate_max = boundary_gate_max
        self.use_product       = use_product
        self.product_scale     = product_scale
        self.use_absdiff       = use_absdiff
        self.pre_norm          = pre_norm
        self.use_region_gate   = use_region_gate
        self.use_boundary_gate = use_boundary_gate
        self.return_debug      = return_debug

        # Optional pre-normalization of raw encoder features
        if pre_norm:
            self.norm_f1 = _group_norm(in_channels)
            self.norm_f2 = _group_norm(in_channels)

        # Number of input streams: always F1n, F2n; optionally |diff|, product
        n_streams   = 2 + int(use_absdiff) + int(use_product)
        in_ch_total = in_channels * n_streams

        # Step 1: bottleneck compression
        conv_fn = _dw_pw if use_depthwise else _pw
        self.compress = nn.Sequential(
            nn.Conv2d(in_ch_total, out_channels, 1, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )

        # Step 2: spatial refinement
        self.spatial = conv_fn(out_channels, out_channels)

        # Step 3: region gate psi_r
        if use_region_gate:
            self.psi_r = _gate_mlp(out_channels, gate_hidden_ratio)

        # Step 4: boundary gradient + gate psi_b
        if use_boundary_gate:
            self.sobel = _SobelGradient(out_channels)
            self.psi_b = _gate_mlp(out_channels, gate_hidden_ratio)

        self._checked = False

    # ------------------------------------------------------------------
    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> dict:
        """
        Args:
            f1: [B, C, H, W]  pre-change feature map
            f2: [B, C, H, W]  post-change feature map

        Returns:
            dict with keys:
                "diff"    : D  [B, C_out, H, W]
                "region"  : R  [B, C_out, H, W]  = G_r * D
                "boundary": B  [B, C_out, H, W]  = G_b * D
            If return_debug=True also:
                "region_gate"   : G_r [B, C_out, H, W]
                "boundary_gate" : G_b [B, C_out, H, W]
        """
        if not self._checked:
            assert f1.shape == f2.shape, (
                f"D-RBI: F1/F2 shape mismatch: {f1.shape} vs {f2.shape}"
            )
            self._checked = True

        # 1. Pre-normalise encoder features
        if self.pre_norm:
            f1n = self.norm_f1(f1)
            f2n = self.norm_f2(f2)
        else:
            f1n, f2n = f1, f2

        # 2. Build input concat
        parts = [f1n, f2n]
        if self.use_absdiff:
            parts.append(torch.abs(f2n - f1n))
        if self.use_product:
            parts.append(self.product_scale * f1n * f2n)
        x = torch.cat(parts, dim=1)   # [B, n*C, H, W]

        # 3. Compress + spatial refinement
        d = self.compress(x)   # [B, C_out, H, W]
        d = self.spatial(d)    # [B, C_out, H, W]

        # 4. Region gate
        if self.use_region_gate:
            g_r_logits = torch.clamp(self.psi_r(d), -_GATE_LOGIT_CLAMP, _GATE_LOGIT_CLAMP)
            g_r = (
                self.region_gate_min
                + (self.region_gate_max - self.region_gate_min)
                * torch.sigmoid(g_r_logits)
            )
            region = g_r * d
        else:
            g_r    = None
            region = d

        # 5. Boundary gate
        if self.use_boundary_gate:
            grad_mag   = self.sobel(d)   # already clamped inside _SobelGradient
            g_b_logits = torch.clamp(self.psi_b(grad_mag), -_GATE_LOGIT_CLAMP, _GATE_LOGIT_CLAMP)
            g_b = (
                self.boundary_gate_min
                + (self.boundary_gate_max - self.boundary_gate_min)
                * torch.sigmoid(g_b_logits)
            )
            boundary = g_b * d
        else:
            g_b      = None
            boundary = d

        result = {"diff": d, "region": region, "boundary": boundary}
        if self.return_debug:
            result["region_gate"]   = g_r
            result["boundary_gate"] = g_b
        return result
