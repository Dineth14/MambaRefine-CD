"""Siamese MambaVision change-detection model.

Shared backbone encodes both images independently.
A decoder fuses the resulting multi-scale feature pairs.

Config keys used
----------------
model.backbone   : 'mambavision' (default) — selects the backbone family
model.variant    : 'tiny' | 'tiny2' | 'small' | 'base' | 'large'
model.decoder    : 'baseline' | 'adaptive_rf' | 'refinement' | 'global_local'
model.pretrained : true / false
decoder.channels : FPN internal channel width (default 256)
decoder.dilation_rates : [1,2,4,8] — only used by adaptive_rf decoder
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.mambavision_builder import build as build_backbone
from models.decoders import DECODER_REGISTRY


class SiameseMambaCD(nn.Module):
    """Siamese change-detection model.

    Args:
        cfg: Full experiment config dict.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg   = cfg["model"]
        dec_cfg     = cfg.get("decoder", {})

        variant     = model_cfg.get("variant", "tiny")
        pretrained  = bool(model_cfg.get("pretrained", True))
        dec_name    = model_cfg.get("decoder", "baseline")
        out_ch      = int(dec_cfg.get("channels", 256))

        # ── Shared encoder ─────────────────────────────────────────────
        self.encoder = build_backbone(variant, pretrained=pretrained)
        self.variant = self.encoder.model_name if hasattr(self.encoder, "model_name") else variant
        channels: List[int] = self.encoder.channels

        # ── Decoder ────────────────────────────────────────────────────
        decoder_cls = DECODER_REGISTRY.get(dec_name)
        if decoder_cls is None:
            raise ValueError(
                f"Unknown decoder {dec_name!r}. "
                f"Available: {list(DECODER_REGISTRY)}"
            )
        dec_kwargs = {"channels": channels, "out_channels": out_ch}
        if dec_name == "adaptive_rf":
            dec_kwargs["dilation_rates"] = dec_cfg.get("dilation_rates", [1, 2, 4, 8])

        self.decoder = decoder_cls(**dec_kwargs)

    # ------------------------------------------------------------------
    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        **_,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            img_a: [B, 3, H, W] pre-change image.
            img_b: [B, 3, H, W] post-change image.

        Returns:
            (logits, aux_logits) — aux_logits is None for most decoders;
            RefinementDecoder returns the coarse logit as aux.
        """
        out_size = img_a.shape[-2:]
        feats_a  = self.encoder(img_a)
        feats_b  = self.encoder(img_b)
        return self.decoder(feats_a, feats_b, out_size)


def build_model(cfg: dict) -> SiameseMambaCD:
    return SiameseMambaCD(cfg)
