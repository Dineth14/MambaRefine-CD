"""Adaptive RF decoder with data-driven dilation attention.

Each encoder stage uses parallel dilated convolution branches.
Fusion weights are predicted per-image via channel attention (GAP → FC →
softmax), giving a dynamic effective receptive field with no deformable ops.

When ``use_boundary_residual=True`` (default when boundary_features are
provided), the decoder runs a two-stage pipeline:

    Stage 1 — Coarse prediction via region features through ARF-FPN:
        P_c = CoarseHead(FPN(region_feats))

    Stage 2 — Lightweight boundary residual correction:
        E   = Sobel(sigmoid(P_c))   # boundary uncertainty
        Δ   = BoundaryRefineHead(boundary_feat, P_c, E)
        P_f = P_c + residual_scale * tanh(Δ)   # logits-space correction

If ``boundary_features=None`` the decoder falls back to old abs-diff
fusion for full backward compatibility.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch), ch)


class _ConvBNGELU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2 * d, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


def _dw_sep(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    """Depthwise 3×3 + pointwise 1×1, GroupNorm, GELU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, k, padding=k // 2, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        _group_norm(out_ch),
        nn.GELU(),
    )


class _AdaptiveDilationBlock(nn.Module):
    """Softmax-gated mixture of dilated branches with channel attention."""

    def __init__(self, in_ch: int, out_ch: int, rates: List[int]) -> None:
        super().__init__()
        self.n        = len(rates)
        self.branches = nn.ModuleList([_ConvBNGELU(in_ch, out_ch, k=3, d=r) for r in rates])
        hidden        = max(in_ch // 4, self.n)
        self.attn     = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        branches = torch.stack([b(x) for b in self.branches], dim=1)   # B,n,C,H,W
        w = F.softmax(self.attn(x), dim=1).view(B, self.n, 1, 1, 1)
        return (w * branches).sum(dim=1)


# ── Fixed Sobel for boundary extraction from coarse prob ──────────────────────

class _SobelEdge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(x.dtype)
        ky = self.ky.to(x.dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)


# ── Lightweight Boundary Refinement Head ──────────────────────────────────────

class _BoundaryRefineHead(nn.Module):
    """Produces a delta correction from:
        - upsampled boundary feature  [B, C, H, W]
        - coarse logits P_c           [B, 1, H, W]
        - Sobel edge of sigmoid(P_c)  [B, 1, H, W]

    Input concat: [boundary_feat, P_c, E]  → [B, C+2, H, W]
    Output: delta [B, 1, H, W]
    """

    def __init__(self, in_ch: int, use_depthwise: bool = True) -> None:
        super().__init__()
        self.sobel = _SobelEdge()
        conv_fn    = _dw_sep if use_depthwise else _ConvBNGELU
        self.body  = nn.Sequential(
            conv_fn(in_ch + 2, in_ch, k=3),
            nn.Conv2d(in_ch, 1, 1, bias=True),
        )
        # Init final conv near zero so correction starts small
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, bnd_feat: torch.Tensor, P_c: torch.Tensor) -> torch.Tensor:
        edge  = self.sobel(torch.sigmoid(P_c))   # [B,1,H,W]
        x     = torch.cat([bnd_feat, P_c, edge], dim=1)
        return self.body(x)                       # [B,1,H,W]


# ── Main Decoder ──────────────────────────────────────────────────────────────

class AdaptiveRFDecoder(nn.Module):
    """FPN decoder where each scale uses an adaptive-dilation RF block.

    Args:
        channels              : list of encoder channel widths per scale
        out_channels          : FPN internal width (default 256)
        dilation_rates        : dilation rates for ARF branches
        use_boundary_residual : enable boundary residual correction stage
        residual_scale        : tanh correction scale (default 0.1)
        use_depthwise         : use depthwise-separable convs in refine head

    When D-RBI features are provided (boundary_features is not None):
        • region_features drive the coarse ARF-FPN prediction
        • boundary_features drive the residual correction stage

    When boundary_features is None (old abs-diff path):
        • feats_a / feats_b fused with abs-diff + sum (old behavior)
    """

    def __init__(
        self,
        channels              : List[int],
        out_channels          : int = 256,
        dilation_rates        : Optional[List[int]] = None,
        use_boundary_residual : bool = True,
        residual_scale        : float = 0.1,
        use_depthwise         : bool  = True,
        **_kwargs,
    ) -> None:
        super().__init__()
        rates = dilation_rates or [1, 2, 4, 8]
        self.residual_scale        = residual_scale
        self.use_boundary_residual = use_boundary_residual

        # ── Coarse FPN (region features path) ─────────────────────────
        # proj expects [region_feat] which is already C_out width from D-RBI,
        # but we also need to handle the legacy abs-diff path (2*C input).
        # We handle this by having two projection paths selected at build time;
        # a flag records which mode was built.
        n_ch = channels[0]   # all channels same from D-RBI; detect if already C_out
        # We build for the D-RBI path (in_ch = out_channels per scale).
        self.proj   = nn.ModuleList([_ConvBNGELU(c, out_channels, k=1) for c in channels])
        self.arf    = nn.ModuleList([_AdaptiveDilationBlock(out_channels, out_channels, rates) for _ in channels])
        self.smooth = nn.ModuleList([_ConvBNGELU(out_channels, out_channels, k=3) for _ in channels])

        # Legacy abs-diff path projection (2*C → out_channels)
        self.proj_absdiff = nn.ModuleList([_ConvBNGELU(c * 2, out_channels, k=1) for c in channels])

        self.coarse_head = nn.Sequential(
            _ConvBNGELU(out_channels, out_channels // 2, k=3),
            nn.Conv2d(out_channels // 2, 1, 1),
        )

        # ── Boundary residual head ─────────────────────────────────────
        if use_boundary_residual:
            self.bnd_refine = _BoundaryRefineHead(out_channels, use_depthwise=use_depthwise)

        # Keep legacy head alias for checkpoint compatibility
        self.head = self.coarse_head

    def _fpn(
        self,
        fused: List[torch.Tensor],
    ) -> torch.Tensor:
        """Standard top-down FPN aggregation over fused scale features."""
        top: Optional[torch.Tensor] = None
        for feat, smooth in zip(reversed(fused), reversed(self.smooth)):
            if top is None:
                top = smooth(feat)
            else:
                top = smooth(
                    F.interpolate(top, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                    + feat
                )
        return top   # type: ignore[return-value]

    def forward(
        self,
        feats_a           : List[torch.Tensor],
        feats_b_or_bnd    : Optional[List[torch.Tensor]],
        out_size          : Tuple[int, int],
        boundary_features : Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Two call signatures:

        D-RBI path — pass boundary_features explicitly:
            forward(region_feats, None, out_size, boundary_features=bnd_feats)
            → uses proj (C_out → out_channels) and boundary residual head

        Legacy path (boundary_features=None):
            forward(feats_a, feats_b, out_size)
            → uses proj_absdiff (2C → out_channels), no boundary head
        """
        # ── Detect path via explicit kwarg ─────────────────────────────
        use_drbi_path = boundary_features is not None

        if not use_drbi_path:
            # ── Legacy abs-diff path ──────────────────────────────────
            feats_b = feats_b_or_bnd  # type: ignore[assignment]
            fused = [
                arf(proj(torch.cat([torch.abs(fa - fb), fa + fb], dim=1)))
                for fa, fb, proj, arf in zip(
                    feats_a, feats_b, self.proj_absdiff, self.arf
                )
            ]
            top = self._fpn(fused)
            logits = F.interpolate(
                self.coarse_head(top), size=out_size,
                mode="bilinear", align_corners=False
            )
            return logits, None

        # ── D-RBI path ────────────────────────────────────────────────
        region_feats = feats_a
        bnd_feats    = boundary_features

        # Coarse prediction via region features
        fused = [
            arf(proj(rf))
            for rf, proj, arf in zip(region_feats, self.proj, self.arf)
        ]
        top    = self._fpn(fused)                           # [B, out_ch, H/4, W/4]
        P_c    = F.interpolate(
            self.coarse_head(top), size=out_size,
            mode="bilinear", align_corners=False,
        )   # [B, 1, H, W]

        if not (self.use_boundary_residual and self.bnd_refine is not None):
            return P_c, None

        # Boundary residual correction
        # Use finest scale boundary feature (index 0 = highest res)
        bnd_finest = F.interpolate(
            bnd_feats[0], size=out_size, mode="bilinear", align_corners=False
        )   # [B, C_out, H, W]
        delta  = self.bnd_refine(bnd_finest, P_c)       # [B, 1, H, W]
        P_f    = P_c + self.residual_scale * torch.tanh(delta)
        return P_f, P_c   # return P_c as aux for optional aux loss
