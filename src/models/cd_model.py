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
from torch.utils.checkpoint import checkpoint

from models.backbone.mambavision_builder import build as build_mambavision_backbone
from models.decoders import DECODER_REGISTRY


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

    def forward(self, x: torch.Tensor, *, gradient_checkpointing: bool = False) -> List[torch.Tensor]:
        feats = []
        for stage in self.stages:
            if gradient_checkpointing and self.training and x.requires_grad:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
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
        debug_cfg = cfg.get("debug", {})
        eff_cfg = cfg.get("efficiency", {})
        train_cfg = cfg.get("training", {})

        variant    = model_cfg.get("variant", "tiny2")
        pretrained = bool(model_cfg.get("pretrained", True))
        dec_name   = model_cfg.get("decoder", "adaptive_rf")
        out_ch     = int(dec_cfg.get("channels", 256))
        self.decoder_type = str(dec_name)
        self.ablation_trace_enabled = bool(debug_cfg.get("ablation_trace", False))
        self.gradient_checkpointing = bool(
            eff_cfg.get("gradient_checkpointing", train_cfg.get("gradient_checkpointing", False))
        )
        self._gradient_checkpointing_active = False
        self._last_ablation_trace: dict[str, Any] = {}
        self._last_forward_trace: dict[str, Any] = {}
        self._forward_trace_recorded = False
        self.output_mode = str(model_cfg.get("output_mode", "binary")).lower()
        if self.output_mode != "binary":
            raise ValueError("Only binary change-detection output is active in this cleaned repository.")

        # ── Shared encoder ─────────────────────────────────────────────
        self.encoder = build_encoder(model_cfg)
        self.variant = getattr(self.encoder, "model_name", variant)
        channels: List[int] = self.encoder.channels   # e.g. [80, 160, 320, 640]

        self.semantic_head = None

        # ── D-RBI fusion (one per encoder scale) ───────────────────────
        self.use_drbi = bool(diff_cfg.get("enabled", True))
        self._fusion_terms_used = []
        if self.use_drbi:
            self._fusion_terms_used.append("raw_pair")
            if bool(diff_cfg.get("use_absdiff", True)):
                self._fusion_terms_used.append("abs_diff")
            if bool(diff_cfg.get("use_signed_diff", False)):
                self._fusion_terms_used.append("signed_diff")
            if bool(diff_cfg.get("use_product", False)):
                self._fusion_terms_used.append("feature_product")
        else:
            self._fusion_terms_used.append("abs_diff")
            self._fusion_terms_used.append("feature_sum")
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

        self._loss_terms_used = ["bce", "dice"]
        coarse_cfg = cfg.get("loss", {}).get("coarse", {})
        boundary_cfg = cfg.get("loss", {}).get("boundary", {})
        if isinstance(coarse_cfg, dict) and bool(coarse_cfg.get("enabled", False)):
            self._loss_terms_used.append("coarse")
        if isinstance(boundary_cfg, dict) and bool(boundary_cfg.get("enabled", False)):
            self._loss_terms_used.append("boundary")

    def _encode(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Encode an image, using checkpointing only for known safe stage lists."""
        if (
            self.gradient_checkpointing
            and self.training
            and image.requires_grad
            and isinstance(self.encoder, SimpleCNNFeatureExtractor)
        ):
            self._gradient_checkpointing_active = True
            return self.encoder(image, gradient_checkpointing=True)
        self._gradient_checkpointing_active = False
        return self.encoder(image)

    def _params_m(self, module: Optional[nn.Module] = None, *, trainable_only: bool = False) -> float:
        module = module or self
        params = module.parameters()
        if trainable_only:
            count = sum(p.numel() for p in params if p.requires_grad)
        else:
            count = sum(p.numel() for p in params)
        return round(count / 1e6, 4)

    def _fusion_term_flags(self) -> dict[str, bool]:
        terms = set(self._fusion_terms_used)
        return {
            "raw_pair": "raw_pair" in terms,
            "abs_diff": "abs_diff" in terms,
            "signed_diff": "signed_diff" in terms,
            "feature_product": "feature_product" in terms or "product" in terms,
        }

    def _first_drbi(self) -> Optional[nn.Module]:
        if not self.use_drbi or self.diff_modules is None or len(self.diff_modules) == 0:
            return None
        return self.diff_modules[0]

    def _fusion_input_channels(self) -> Optional[int]:
        dm = self._first_drbi()
        if dm is not None:
            conv = dm.compress[0] if hasattr(dm, "compress") else None
            return int(getattr(conv, "in_channels", 0)) or None
        decoder = getattr(self, "decoder", None)
        projections = getattr(decoder, "proj_absdiff", None)
        if projections:
            first = projections[0][0] if len(projections[0]) > 0 else None
            return int(getattr(first, "in_channels", 0)) or None
        projections = getattr(decoder, "proj", None)
        if projections:
            first = projections[0][0] if len(projections[0]) > 0 else None
            return int(getattr(first, "in_channels", 0)) or None
        return None

    def get_ablation_trace(self) -> dict[str, Any]:
        """Return metadata from the actual modules constructed for this model."""
        decoder = getattr(self, "decoder", None)
        dm = self._first_drbi()
        decoder_type = str(self.decoder_type)
        loss_flags = {name: name in self._loss_terms_used for name in ("bce", "dice", "coarse", "boundary")}
        trace = {
            "variant_name": str(self.variant),
            "encoder_type": "simple_cnn" if isinstance(self.encoder, SimpleCNNFeatureExtractor) else "mambavision",
            "backbone_name": str(getattr(self.encoder, "model_name", self.variant)),
            "fusion_terms_used": self._fusion_term_flags(),
            "fusion_terms_list": list(self._fusion_terms_used),
            "fusion_input_channels": self._fusion_input_channels(),
            "drbi_enabled": bool(self.use_drbi and self.diff_modules is not None),
            "region_gate_enabled": bool(dm is not None and getattr(dm, "use_region_gate", False)),
            "boundary_gate_enabled": bool(dm is not None and getattr(dm, "use_boundary_gate", False)),
            "decoder_type": decoder_type,
            "adaptive_rf_enabled": bool(hasattr(decoder, "arf")),
            "dilation_rates": list(getattr(decoder, "dilation_rates", [])),
            "boundary_residual_enabled": bool(
                getattr(decoder, "use_boundary_residual", False) and hasattr(decoder, "bnd_refine")
            ),
            "cram_lite_enabled": bool(self.cram_lite is not None),
            "loss_terms": loss_flags,
            "loss_terms_list": list(self._loss_terms_used),
            "params_M": self._params_m(),
            "trainable_params_M": self._params_m(trainable_only=True),
            "gradient_checkpointing_requested": bool(self.gradient_checkpointing),
            "gradient_checkpointing_active": bool(self._gradient_checkpointing_active),
        }
        if self._last_ablation_trace:
            trace.update({"last_forward": dict(self._last_ablation_trace)})
        return trace

    def get_forward_trace(self) -> dict[str, Any]:
        """Return the compact first-batch forward-path trace."""
        return dict(self._last_forward_trace)

    def _build_forward_trace(
        self,
        feats_a: List[torch.Tensor],
        feats_b: List[torch.Tensor],
        change_logits: torch.Tensor,
        aux_logits: Optional[torch.Tensor],
        *,
        drbi_called: bool,
        boundary_residual_called: bool,
    ) -> dict[str, Any]:
        decoder = getattr(self, "decoder", None)
        fusion_tensors = list(self._fusion_terms_used)
        return {
            "f1_shapes": [list(feat.shape) for feat in feats_a],
            "f2_shapes": [list(feat.shape) for feat in feats_b],
            "fusion_tensors_concatenated": fusion_tensors,
            "final_fusion_channel_count": self._fusion_input_channels(),
            "drbi_forward_called": bool(drbi_called),
            "adaptive_rf_decoder_forward_called": bool(hasattr(decoder, "arf")),
            "boundary_residual_refinement_applied": bool(boundary_residual_called),
            "decoder_type": str(self.decoder_type),
            "output_shape": list(change_logits.shape),
            "aux_output_shape": list(aux_logits.shape) if torch.is_tensor(aux_logits) else None,
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        **_,
    ) -> Any:
        out_size = img_a.shape[-2:]

        feats_a = self._encode(img_a)   # List[[B, C_i, H_i, W_i]]
        feats_b = self._encode(img_b)
        drbi_called = False
        boundary_residual_called = False

        if self.use_drbi:
            drbi_called = True
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
                boundary_residual_called = bool(
                    getattr(self.decoder, "use_boundary_residual", False)
                    and hasattr(self.decoder, "bnd_refine")
                )
            else:
                zero_feats = [torch.zeros_like(feat) for feat in region_feats]
                change_logits, aux_logits = self.decoder(region_feats, zero_feats, out_size)
        else:
            change_logits, aux_logits = self.decoder(feats_a, feats_b, out_size)

        if self.ablation_trace_enabled and not self._forward_trace_recorded:
            self._last_forward_trace = self._build_forward_trace(
                feats_a,
                feats_b,
                change_logits,
                aux_logits,
                drbi_called=drbi_called,
                boundary_residual_called=boundary_residual_called,
            )
            self._last_ablation_trace = dict(self._last_forward_trace)
            self._forward_trace_recorded = True

        return change_logits, aux_logits


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

    if output_mode != "binary":
        raise ValueError(
            f"Unsupported model.output_mode={output_mode!r}. Only 'binary' is active."
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
