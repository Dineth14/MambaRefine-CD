"""Localization → Refinement Decoder  (MERCon contribution).

Two-stage architecture:

    Stage 1 — Coarse Localization
        P_c = coarse_decoder(multi-scale features)

    Stage 2 — Boundary-Guided Refinement
        E   = edge_extract(P_c)            # Sobel gradient of coarse prob map
        Δ   = refinement_block(P_c, f0, f1, E)   # uses shallow + mid features
        P_f = P_c + Δ                      # RESIDUAL correction

Design rationale
----------------
* The coarse pass localises change regions using all four encoder scales via
  a lightweight FPN.
* Boundary extraction operates on the sigmoid of P_c to obtain a soft edge
  map that highlights uncertain / thin regions.
* The refinement block takes ONLY the two shallowest encoder scales (high
  spatial resolution) together with the coarse prediction and edge map.
  This keeps the module lightweight and focused on boundary quality.
* The residual formulation (P_f = P_c + Δ) keeps the correction small by
  initialising the refinement head's final conv to zero, so training begins
  from the coarse prediction.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared conv primitive ──────────────────────────────────────────────────────

class _ConvBNGELU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )


# ── Sobel edge extractor ───────────────────────────────────────────────────────

class SobelEdge(nn.Module):
    """Differentiable Sobel edge detector on a single-channel probability map.

    Computes |∇P| and clamps to [0, 1].  Weights are fixed (not learned).
    """

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prob: [B, 1, H, W] sigmoid probability map.
        Returns:
            edge: [B, 1, H, W] edge magnitude, values in [0, 1].
        """
        gx = F.conv2d(prob, self.kx, padding=1)
        gy = F.conv2d(prob, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).clamp(0.0, 1.0)


# ── Coarse decoder (lightweight FPN) ──────────────────────────────────────────

class _CoarseFPN(nn.Module):
    def __init__(self, channels: List[int], mid_ch: int) -> None:
        super().__init__()
        self.proj   = nn.ModuleList([_ConvBNGELU(c * 2, mid_ch, k=1) for c in channels])
        self.smooth = nn.ModuleList([_ConvBNGELU(mid_ch, mid_ch) for _ in channels])
        self.head   = nn.Sequential(
            _ConvBNGELU(mid_ch, mid_ch // 2),
            nn.Conv2d(mid_ch // 2, 1, 1),
        )

    def forward(
        self,
        feats_a: List[torch.Tensor],
        feats_b: List[torch.Tensor],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (coarse_logits, fpn_top_feature)."""
        fused = [
            proj(torch.cat([torch.abs(fa - fb), fa + fb], dim=1))
            for fa, fb, proj in zip(feats_a, feats_b, self.proj)
        ]
        top = None
        for feat, smooth in zip(reversed(fused), reversed(self.smooth)):
            top = smooth(feat) if top is None else smooth(
                F.interpolate(top, size=feat.shape[-2:], mode="bilinear", align_corners=False) + feat
            )
        coarse = F.interpolate(self.head(top), size=out_size, mode="bilinear", align_corners=False)
        return coarse, top  # top will be reused by refinement


# ── Refinement block ──────────────────────────────────────────────────────────

class _RefinementBlock(nn.Module):
    """Predicts a residual correction Δ using:

        - coarse logit (1ch)
        - edge map E (1ch)
        - shallow encoder diff feature — stage 0 (c0 * 2 → mid_ch via proj)
        - mid-level encoder diff feature — stage 1 (c1 * 2 → mid_ch via proj)
    """

    def __init__(self, c0: int, c1: int, mid_ch: int) -> None:
        super().__init__()
        # Project shallow and mid encoder features
        self.proj0 = _ConvBNGELU(c0 * 2, mid_ch, k=1)
        self.proj1 = _ConvBNGELU(c1 * 2, mid_ch, k=1)

        # Fuse: coarse(1) + edge(1) + shallow(mid_ch) + mid(mid_ch) → mid_ch
        in_ch = 1 + 1 + mid_ch + mid_ch
        self.fuse = nn.Sequential(
            _ConvBNGELU(in_ch, mid_ch),
            _ConvBNGELU(mid_ch, mid_ch // 2),
        )
        # Final 1×1 conv initialised to zero → residual starts from P_c
        self.delta_conv = nn.Conv2d(mid_ch // 2, 1, 1)
        nn.init.zeros_(self.delta_conv.weight)
        nn.init.zeros_(self.delta_conv.bias)

    def forward(
        self,
        coarse: torch.Tensor,      # [B, 1, H, W]  coarse logits
        edge: torch.Tensor,        # [B, 1, H, W]  Sobel edge map
        fa0: torch.Tensor, fb0: torch.Tensor,   # stage-0 features
        fa1: torch.Tensor, fb1: torch.Tensor,   # stage-1 features
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        # Shallow encoder difference features
        s0 = self.proj0(torch.cat([torch.abs(fa0 - fb0), fa0 + fb0], dim=1))
        s1 = self.proj1(torch.cat([torch.abs(fa1 - fb1), fa1 + fb1], dim=1))

        # Upsample everything to the target resolution
        H, W = out_size
        s0 = F.interpolate(s0, size=(H, W), mode="bilinear", align_corners=False)
        s1 = F.interpolate(s1, size=(H, W), mode="bilinear", align_corners=False)
        # coarse and edge are already at out_size (upsampled in forward())

        cat = torch.cat([coarse, edge, s0, s1], dim=1)
        delta = self.delta_conv(self.fuse(cat))   # [B, 1, H, W]
        # Limit correction magnitude: at most ±0.1 logit units
        # tanh keeps gradient flow; the 0.1 scale keeps corrections small
        delta = 0.1 * torch.tanh(delta)
        return delta


# ── Public decoder ────────────────────────────────────────────────────────────

class RefinementDecoder(nn.Module):
    """Localization → Refinement Decoder.

    Forward pass:
        1. Coarse FPN  →  P_c  (coarse logit map, full resolution)
        2. Sobel edge  →  E    (boundary uncertainty from sigmoid(P_c))
        3. Refinement block → Δ  (residual correction using shallow features)
        4. Final output:  P_f = P_c + Δ

    Returns:
        (P_f, P_c) — final and coarse logits for optional auxiliary loss.
    """

    def __init__(
        self,
        channels: List[int],
        out_channels: int = 256,
        **_kwargs,
    ) -> None:
        super().__init__()
        assert len(channels) == 4, "RefinementDecoder expects 4 encoder scales"
        mid_ch = out_channels

        self.coarse_fpn = _CoarseFPN(channels, mid_ch)
        self.sobel      = SobelEdge()
        self.refine     = _RefinementBlock(channels[0], channels[1], mid_ch)

    def forward(
        self,
        feats_a: List[torch.Tensor],
        feats_b: List[torch.Tensor],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ── Stage 1: coarse localisation ─────────────────────────────────
        P_c, _ = self.coarse_fpn(feats_a, feats_b, out_size)   # [B,1,H,W]

        # ── Stage 2: boundary extraction ─────────────────────────────────
        prob = torch.sigmoid(P_c.detach())                      # detach: no grad through edge
        E    = self.sobel(prob)                                  # [B,1,H,W]

        # ── Stage 3: residual refinement ──────────────────────────────────
        delta = self.refine(
            P_c, E,
            feats_a[0], feats_b[0],
            feats_a[1], feats_b[1],
            out_size,
        )

        P_f = P_c + delta                                        # residual
        return P_f, P_c   # P_c returned for optional auxiliary supervision
