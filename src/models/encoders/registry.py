"""Encoder registry."""
from __future__ import annotations


def build_encoder(cfg):
    family = cfg.model.encoder_family.lower()
    variant = cfg.model.encoder_variant.lower()
    pretrained = bool(cfg.model.encoder_pretrained)

    if family == "mambavision":
        from src.models.encoders.mambavision_adapter import MambaVisionAdapter
        return MambaVisionAdapter(variant=variant, pretrained=pretrained)
    if family == "vmamba":
        from src.models.encoders.vmamba_adapter import VMambaAdapter
        return VMambaAdapter(variant=variant, pretrained=pretrained)
    raise ValueError(f"Unknown encoder_family: '{family}'. Choose: mambavision | vmamba")
