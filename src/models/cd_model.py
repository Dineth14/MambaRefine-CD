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


class TemporalMambaCDModel(nn.Module):
    """DISABLED — Temporal Mamba mode has been removed due to training instability.

    This class is retained only as a placeholder so that loading old checkpoints
    gives a clear error message rather than an ImportError.
    """

    def __init__(self, cfg: dict) -> None:
        raise ValueError(
            "temporal_mamba mode has been disabled due to training instability. "
            "Set model.mode: dual in global_config.yaml and use the "
            "Differential Region–Boundary Interaction (D-RBI) module instead."
        )

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("TemporalMambaCDModel is disabled.")


class DRBISiameseMambaCD(nn.Module):
    """Dual-branch change detection model with Differential Region–Boundary
    Interaction (D-RBI) feature fusion.

    Architecture
    ------------
    F1 = encoder(img_a)   # shared weights
    F2 = encoder(img_b)   # shared weights

    For each encoder scale i (if difference.enabled):
        D_i = D-RBI(F1_i, F2_i)
        region_feats_i    = D_i["region"]
        boundary_feats_i  = D_i["boundary"]

    logits = AdaptiveRFDecoder(region_feats, boundary_feats, out_size)

    If difference.enabled is False the decoder receives (feats_a, feats_b)
    directly and uses its built-in abs-diff fusion (old baseline path).

    Config keys
    -----------
    model.mode           : "dual"
    model.variant        : backbone variant
    model.decoder        : decoder name
    model.pretrained     : bool
    difference.enabled   : bool  (default True)
    difference.*         : D-RBI hyper-parameters (see DifferentialRegionBoundaryInteraction)
    decoder.*            : decoder hyper-parameters
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        from models.modules.differential_region_boundary import DifferentialRegionBoundaryInteraction

        model_cfg  = cfg["model"]
        dec_cfg    = cfg.get("decoder", {})
        diff_cfg   = cfg.get("difference", {})

        variant    = model_cfg.get("variant", "tiny2")
        pretrained = bool(model_cfg.get("pretrained", True))
        dec_name   = model_cfg.get("decoder", "adaptive_rf")
        out_ch     = int(dec_cfg.get("channels", 256))

        # ── Shared encoder ─────────────────────────────────────────────
        self.encoder = build_backbone(variant, pretrained=pretrained)
        self.variant = getattr(self.encoder, "model_name", variant)
        channels: List[int] = self.encoder.channels   # e.g. [80, 160, 320, 640]

        # ── D-RBI fusion (one per encoder scale) ───────────────────────
        self.use_drbi = bool(diff_cfg.get("enabled", True))
        if self.use_drbi:
            drbi_out  = int(diff_cfg.get("out_channels", out_ch))
            self.diff_modules = nn.ModuleList([
                DifferentialRegionBoundaryInteraction(
                    in_channels       = c,
                    out_channels      = drbi_out,
                    use_depthwise     = bool(diff_cfg.get("use_depthwise",     True)),
                    gate_hidden_ratio = float(diff_cfg.get("gate_hidden_ratio", 0.25)),
                    region_gate_min   = float(diff_cfg.get("region_gate_min",  0.2)),
                    region_gate_max   = float(diff_cfg.get("region_gate_max",  0.8)),
                    boundary_gate_min = float(diff_cfg.get("boundary_gate_min", 0.0)),
                    boundary_gate_max = float(diff_cfg.get("boundary_gate_max", 0.4)),
                    use_product       = bool(diff_cfg.get("use_product",  False)),
                    product_scale     = float(diff_cfg.get("product_scale", 0.25)),
                    use_absdiff       = bool(diff_cfg.get("use_absdiff",  True)),
                    pre_norm          = bool(diff_cfg.get("pre_norm",     True)),
                    use_region_gate   = bool(diff_cfg.get("use_region_gate",   True)),
                    use_boundary_gate = bool(diff_cfg.get("use_boundary_gate", True)),
                    return_debug      = bool(diff_cfg.get("return_debug", False)),
                )
                for c in channels
            ])
            decoder_channels = [drbi_out] * len(channels)
        else:
            self.diff_modules = None
            decoder_channels  = channels

        # ── Decoder ────────────────────────────────────────────────────
        decoder_cls = DECODER_REGISTRY.get(dec_name)
        if decoder_cls is None:
            raise ValueError(
                f"Unknown decoder {dec_name!r}. "
                f"Available: {list(DECODER_REGISTRY)}"
            )
        dec_kwargs: dict = {"channels": decoder_channels, "out_channels": out_ch}
        if dec_name == "adaptive_rf":
            dec_kwargs["dilation_rates"]       = dec_cfg.get("dilation_rates", [1, 2, 4, 8])
            dec_kwargs["use_boundary_residual"] = bool(dec_cfg.get("use_boundary_residual", True))
            dec_kwargs["residual_scale"]        = float(dec_cfg.get("residual_scale", 0.1))
            dec_kwargs["use_depthwise"]         = bool(dec_cfg.get("use_depthwise", True))
        self.decoder = decoder_cls(**dec_kwargs)

    # ------------------------------------------------------------------
    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        **_,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out_size = img_a.shape[-2:]

        feats_a = self.encoder(img_a)   # List[[B, C_i, H_i, W_i]]
        feats_b = self.encoder(img_b)

        if self.use_drbi:
            region_feats   = []
            boundary_feats = []
            for dm, fa, fb in zip(self.diff_modules, feats_a, feats_b):
                out = dm(fa, fb)
                region_feats.append(out["region"])
                boundary_feats.append(out["boundary"])
            return self.decoder(region_feats, None, out_size, boundary_features=boundary_feats)
        else:
            return self.decoder(feats_a, feats_b, out_size)


def build_model(cfg: dict) -> nn.Module:
    """Factory: returns the right model variant based on ``model.mode``.

    Modes
    -----
    ``"dual"`` (default)
        Dual-branch Siamese model.  When ``difference.enabled: true`` (default)
        uses the D-RBI module for feature fusion; otherwise falls back to the
        built-in abs-diff fusion in the decoder.

    ``"temporal_mamba"``
        **DISABLED** — raises ValueError.  Set ``model.mode: dual``.
    """
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    output_mode = str(model_cfg.get("output_mode", "binary")).lower()
    dataset_mode = str(dataset_cfg.get("mode", "binary")).lower()

    if output_mode == "semantic":
        raise NotImplementedError(
            "model.output_mode=semantic is reserved for future semantic change detection. "
            "The current model only supports binary logits. "
            "For SECOND semantic-label training today, set dataset.mode=semantic and keep model.output_mode=binary."
        )

    mode = str(model_cfg.get("mode", "dual")).lower()
    if mode == "temporal_mamba":
        raise ValueError(
            "temporal_mamba mode has been disabled due to training instability. "
            "Set model.mode: dual in global_config.yaml."
        )
    # Legacy class kept for API compatibility
    if mode == "dual":
        diff_enabled = cfg.get("difference", {}).get("enabled", True)
        # Only use DRBISiameseMambaCD when D-RBI is wanted (adaptive_rf + enabled)
        # or always — it handles the fallback internally
        return DRBISiameseMambaCD(cfg)
    raise ValueError(f"Unknown model.mode: {mode!r}. Valid options: dual")

