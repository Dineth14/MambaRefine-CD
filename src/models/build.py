"""Build MambaRefineCD from config."""
from __future__ import annotations

from src.models.encoders.registry import build_encoder
from src.models.mambarefine_cd import MambaRefineCD


def build_model(cfg) -> MambaRefineCD:
    encoder = build_encoder(cfg)
    print(f"Encoder : {cfg.model.encoder_family}/{cfg.model.encoder_variant}")
    print(f"Channels: {encoder.out_channels}")
    print(f"Temporal: {cfg.ablation.temporal_input_mode}")
    print(f"CRAM enabled: {not bool(getattr(cfg.model, 'disable_cross_attn', False))}")
    print(f"Single-scale fusion: {bool(getattr(cfg.model, 'single_scale_fusion', False))}")
    model = MambaRefineCD(encoder=encoder, cfg=cfg)
    print(f"D-RBI input channels: {model.drbi_input_channels}")
    return model
