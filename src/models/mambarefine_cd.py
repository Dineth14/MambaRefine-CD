"""MambaRefine-CD.

Bitemporal change detection with shared encoder, temporal difference, D-RBI,
ARF-FPN decoder, and boundary residual refinement.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.modules.temporal_difference import TemporalDifference
from src.models.modules.drbi import DRBI
from src.models.modules.arf_fpn import ARFFPN
from src.models.modules.boundary_refinement import BoundaryRefinement


class MambaRefineCD(nn.Module):
    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder = encoder
        enc_channels = list(encoder.out_channels)
        dec_ch = int(cfg.model.decoder_channels)
        mode = str(cfg.ablation.temporal_input_mode)

        self.temporal_diff = TemporalDifference(mode=mode)
        mult = self.temporal_diff.channel_multiplier
        self.drbi_input_channels = [c * mult for c in enc_channels]

        self.drbi_blocks = nn.ModuleList([
            DRBI(in_channels=enc_channels[s] * mult, out_channels=enc_channels[s])
            for s in range(4)
        ])
        self.decoder = ARFFPN(in_channels=enc_channels, decoder_channels=dec_ch)
        self.boundary_refinement = BoundaryRefinement(
            boundary_channels=enc_channels,
            decoder_channels=dec_ch,
        )

        if bool(cfg.model.freeze_encoder):
            for param in self.encoder.parameters():
                param.requires_grad_(False)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor):
        out_size = image_a.shape[-2:]
        feats_a = self.encoder(image_a)
        feats_b = self.encoder(image_b)

        region_feats = []
        boundary_feats = []
        for s in range(4):
            d_in = self.temporal_diff(feats_a[s], feats_b[s])
            expected = self.drbi_input_channels[s]
            if d_in.shape[1] != expected:
                raise RuntimeError(f"Stage {s} D-RBI input mismatch: got {d_in.shape[1]}, expected {expected}")
            r_s, b_s = self.drbi_blocks[s](d_in)
            region_feats.append(r_s)
            boundary_feats.append(b_s)

        main_logits = self.decoder(region_feats, out_size)
        final_logits, boundary_logits, residual = self.boundary_refinement(
            main_logits,
            boundary_feats,
            out_size,
        )
        return {
            "logits": final_logits,
            "main_logits": main_logits,
            "boundary_logits": boundary_logits,
            "residual": residual,
        }
