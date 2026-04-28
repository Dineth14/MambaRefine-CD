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

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.mambavision_builder import build as build_mambavision_backbone
from models.decoders import DECODER_REGISTRY
from models.decoders.semantic_heads import LightweightSemanticHead


class SimpleCNNFeatureExtractor(nn.Module):
    """Small baseline encoder for ablation runs without MambaVision."""

    def __init__(self, channels: List[int] | None = None) -> None:
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        self.channels = channels
        self.model_name = "simple_cnn"
        in_ch = 3
        stages = []
        for idx, out_ch in enumerate(channels):
            stride = 1 if idx == 0 else 2
            stages.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
            )
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


def build_encoder(model_cfg: dict) -> nn.Module:
    backbone = str(model_cfg.get("backbone", "mambavision")).lower()
    if backbone == "mambavision":
        return build_mambavision_backbone(
            model_cfg.get("variant", "tiny"),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
    if backbone in {"baseline", "simple_cnn", "fpn_baseline"}:
        raw_channels = model_cfg.get("baseline_channels", [64, 128, 256, 512])
        return SimpleCNNFeatureExtractor([int(v) for v in raw_channels])
    raise ValueError(f"Unknown model.backbone={backbone!r}. Valid options: mambavision, simple_cnn.")


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
        self.encoder = build_encoder(model_cfg)
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
        from models.modules.cram_lite import CRAMLiteBank

        model_cfg  = cfg["model"]
        dec_cfg    = cfg.get("decoder", {})
        diff_cfg   = cfg.get("difference", {})
        cram_cfg   = cfg.get("model", {}).get("cram_lite", {})
        dataset_cfg = cfg.get("dataset", {})

        variant    = model_cfg.get("variant", "tiny2")
        pretrained = bool(model_cfg.get("pretrained", True))
        dec_name   = model_cfg.get("decoder", "adaptive_rf")
        out_ch     = int(dec_cfg.get("channels", 256))
        self.output_mode = str(model_cfg.get("output_mode", "binary")).lower()
        semantic_head_cfg = model_cfg.get("semantic_head", {}) or {}
        self.enable_semantic_heads = bool(
            semantic_head_cfg.get("enabled", model_cfg.get("enable_semantic_heads", False))
        )
        self.semantic_head_type = str(
            semantic_head_cfg.get("type", model_cfg.get("semantic_head_type", "lightweight"))
        ).lower()
        self.semantic_num_classes = int(
            semantic_head_cfg.get("num_classes", model_cfg.get("semantic_num_classes", dataset_cfg.get("num_classes", 7)))
        )

        # ── Shared encoder ─────────────────────────────────────────────
        self.encoder = build_encoder(model_cfg)
        self.variant = getattr(self.encoder, "model_name", variant)
        channels: List[int] = self.encoder.channels   # e.g. [80, 160, 320, 640]

        if self.output_mode == "semantic_change":
            if not self.enable_semantic_heads:
                raise ValueError(
                    "model.output_mode=semantic_change requires model.enable_semantic_heads=true."
                )
            if self.semantic_head_type != "lightweight":
                raise ValueError(
                    f"Unsupported semantic_head_type={self.semantic_head_type!r}. Only 'lightweight' is implemented."
                )
            self.semantic_head = LightweightSemanticHead(
                channels=channels,
                num_classes=self.semantic_num_classes,
                hidden_channels=int(semantic_head_cfg.get("channels", model_cfg.get("semantic_head_channels", out_ch))),
                dropout=float(semantic_head_cfg.get("dropout", 0.0)),
            )
        else:
            self.semantic_head = None

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
                    use_signed_diff   = bool(diff_cfg.get("use_signed_diff", False)),
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
        self._decoder_accepts_boundary_features = dec_name == "adaptive_rf"

        # ── Optional CRAMLite attention on D-RBI region features ───────
        cram_enabled = bool(cram_cfg.get("enabled", False))
        if cram_enabled and self.use_drbi:
            drbi_ch = int(diff_cfg.get("out_channels", out_ch))
            apply_stages = list(cram_cfg.get("apply_stages", [0, 1, 2]))
            alpha_init   = float(cram_cfg.get("alpha", 0.5))
            self.cram_lite = CRAMLiteBank(
                channels_list=[drbi_ch] * len(channels),
                apply_stages=apply_stages,
                alpha_init=alpha_init,
            )
        else:
            self.cram_lite = None

    # ------------------------------------------------------------------
    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        **_,
    ) -> Any:
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
            # Apply CRAMLite spatial attention to region features if enabled
            if self.cram_lite is not None:
                region_feats = self.cram_lite.apply(region_feats)
            if self._decoder_accepts_boundary_features:
                change_logits, aux_logits = self.decoder(region_feats, None, out_size, boundary_features=boundary_feats)
            else:
                zero_feats = [torch.zeros_like(feat) for feat in region_feats]
                change_logits, aux_logits = self.decoder(region_feats, zero_feats, out_size)
        else:
            change_logits, aux_logits = self.decoder(feats_a, feats_b, out_size)

        if self.output_mode != "semantic_change":
            return change_logits, aux_logits

        if self.semantic_head is None:
            raise RuntimeError("semantic_change output requested but semantic_head is not initialized.")

        sem_logits_t1 = self.semantic_head(feats_a, out_size)
        sem_logits_t2 = self.semantic_head(feats_b, out_size)
        return {
            "change_logits": change_logits,
            "sem_logits_t1": sem_logits_t1,
            "sem_logits_t2": sem_logits_t2,
            "aux_logits": aux_logits,
        }


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

    if output_mode not in {"binary", "semantic_change"}:
        raise ValueError(
            f"Unsupported model.output_mode={output_mode!r}. Valid options: 'binary' or 'semantic_change'."
        )
    if output_mode == "semantic_change" and dataset_mode != "semantic":
        raise ValueError(
            "model.output_mode=semantic_change requires dataset.mode=semantic."
        )

    mode = str(model_cfg.get("mode", "dual")).lower()
    if mode == "temporal_mamba":
        raise ValueError(
            "temporal_mamba mode has been disabled due to training instability. "
            "Set model.mode: dual in global_config.yaml."
        )
    # Legacy class kept for API compatibility
    if mode == "dual":
        model = DRBISiameseMambaCD(cfg)
        from utils.ablation import assert_model_matches_config
        assert_model_matches_config(model, cfg)
        return model
    raise ValueError(f"Unknown model.mode: {mode!r}. Valid options: dual")
