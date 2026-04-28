"""Ablation config diagnostics and fingerprint helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _plain(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def canonical_config(cfg: dict) -> dict:
    data = _plain(cfg)
    data.pop("_meta", None)
    return data


def config_fingerprint(cfg: dict) -> str:
    payload = json.dumps(canonical_config(cfg), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def module_flags(cfg: dict) -> dict[str, Any]:
    model = cfg.get("model", {})
    diff = cfg.get("difference", {})
    dec = cfg.get("decoder", {})
    loss = cfg.get("loss", {})
    loss_boundary = loss.get("boundary", {}) if isinstance(loss.get("boundary", {}), dict) else {}
    ema = cfg.get("ema", {})
    training = cfg.get("training", {})
    backbone = str(model.get("backbone", "mambavision")).lower()
    decoder = str(model.get("decoder", dec.get("type", "baseline"))).lower()
    return {
        "encoder_name": backbone,
        "decoder_name": decoder,
        "mambavision_enabled": backbone == "mambavision",
        "drbi_enabled": bool(diff.get("enabled", True)),
        "signed_diff_enabled": bool(diff.get("use_signed_diff", False)),
        "cram_lite_enabled": bool((model.get("cram_lite", {}) or {}).get("enabled", False)),
        "arf_fpn_enabled": decoder == "adaptive_rf",
        "boundary_refine_enabled": bool(dec.get("use_boundary_residual", False)) and decoder == "adaptive_rf",
        "boundary_loss_enabled": bool(loss_boundary.get("enabled", False)) or float(loss.get("boundary_weight", 0.0)) > 0.0,
        "ema_enabled": bool(ema.get("enabled", training.get("use_ema", False))),
    }


def output_root(cfg: dict) -> str:
    return str(cfg.get("experiment", {}).get("output_root", "outputs"))


def log_startup_config(logger, cfg: dict, config_path: str | Path | None = None) -> dict[str, Any]:
    flags = module_flags(cfg)
    meta = cfg.get("_meta", {}) if isinstance(cfg.get("_meta", {}), dict) else {}
    logger.info("Ablation config summary:")
    logger.info("  config file path        : %s", str(config_path or meta.get("config_path", "unknown")))
    logger.info("  config fingerprint      : %s", meta.get("config_fingerprint", config_fingerprint(cfg)))
    logger.info("  experiment name         : %s", cfg.get("experiment", {}).get("name", "unknown"))
    logger.info("  encoder name            : %s", flags["encoder_name"])
    logger.info("  decoder name            : %s", flags["decoder_name"])
    logger.info("  D-RBI enabled           : %s", str(flags["drbi_enabled"]).lower())
    logger.info("  signed_diff enabled     : %s", str(flags["signed_diff_enabled"]).lower())
    logger.info("  CRAM-lite enabled       : %s", str(flags["cram_lite_enabled"]).lower())
    logger.info("  ARF-FPN enabled         : %s", str(flags["arf_fpn_enabled"]).lower())
    logger.info("  boundary_refine enabled : %s", str(flags["boundary_refine_enabled"]).lower())
    logger.info("  boundary_loss enabled   : %s", str(flags["boundary_loss_enabled"]).lower())
    logger.info("  EMA enabled             : %s", str(flags["ema_enabled"]).lower())
    logger.info("  checkpoint save dir     : %s", output_root(cfg))
    return flags


def _count(module) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def parameter_breakdown(model) -> dict[str, int]:
    decoder = getattr(model, "decoder", None)
    cram = getattr(model, "cram_lite", None)
    diff = getattr(model, "diff_modules", None)
    boundary = getattr(decoder, "bnd_refine", None) if decoder is not None else None
    arf = getattr(decoder, "arf", None) if decoder is not None else None
    return {
        "total_params": _count(model),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder_params": _count(getattr(model, "encoder", None)),
        "decoder_params": _count(decoder),
        "drbi_params": _count(diff),
        "arf_params": _count(arf),
        "cram_lite_params": _count(cram),
        "boundary_refinement_params": _count(boundary),
    }


def log_parameter_breakdown(logger, model) -> dict[str, int]:
    params = parameter_breakdown(model)
    logger.info("Model parameter breakdown:")
    for key in (
        "total_params",
        "trainable_params",
        "encoder_params",
        "decoder_params",
        "drbi_params",
        "arf_params",
        "cram_lite_params",
        "boundary_refinement_params",
    ):
        logger.info("  %-28s: %.4fM (%d)", key, params[key] / 1e6, params[key])
    return params


def assert_model_matches_config(model, cfg: dict) -> None:
    flags = module_flags(cfg)
    diff = getattr(model, "diff_modules", None)
    cram = getattr(model, "cram_lite", None)
    decoder = getattr(model, "decoder", None)
    boundary = getattr(decoder, "bnd_refine", None) if decoder is not None else None
    arf = getattr(decoder, "arf", None) if decoder is not None else None
    errors: list[str] = []
    if flags["drbi_enabled"] and diff is None:
        errors.append("config enables D-RBI but model.diff_modules is missing")
    if not flags["drbi_enabled"] and diff is not None:
        errors.append("config disables D-RBI but model.diff_modules exists")
    if flags["cram_lite_enabled"] and cram is None:
        errors.append("config enables CRAM-lite but model.cram_lite is missing")
    if not flags["cram_lite_enabled"] and cram is not None:
        errors.append("config disables CRAM-lite but model.cram_lite exists")
    if flags["boundary_refine_enabled"] and boundary is None:
        errors.append("config enables boundary refinement but decoder.bnd_refine is missing")
    if not flags["boundary_refine_enabled"] and boundary is not None:
        errors.append("config disables boundary refinement but decoder.bnd_refine exists")
    if flags["arf_fpn_enabled"] and arf is None:
        errors.append("config enables ARF-FPN but decoder.arf is missing")
    if not flags["arf_fpn_enabled"] and arf is not None:
        errors.append("config disables ARF-FPN but decoder.arf exists")
    if errors:
        raise AssertionError("; ".join(errors))


def compare_checkpoint_config(logger, cfg: dict, ckpt: dict, *, strict: bool = True) -> None:
    current_fp = cfg.get("_meta", {}).get("config_fingerprint", config_fingerprint(cfg))
    ckpt_fp = ckpt.get("config_fingerprint")
    ckpt_flags = ckpt.get("module_flags")
    current_flags = module_flags(cfg)
    logger.info("Checkpoint stored experiment name : %s", ckpt.get("experiment_name", ckpt.get("config", {}).get("experiment", {}).get("name", "unknown")))
    logger.info("Checkpoint stored config path     : %s", ckpt.get("config_path", "unknown"))
    logger.info("Checkpoint stored fingerprint     : %s", ckpt_fp or "missing")
    logger.info("Current config fingerprint        : %s", current_fp)
    logger.info("Checkpoint stored module flags    : %s", ckpt_flags or "missing")
    if ckpt_fp and ckpt_fp != current_fp:
        logger.warning("Checkpoint was trained with a different config.")
    if ckpt_flags and ckpt_flags != current_flags:
        msg = f"Checkpoint module flags do not match current config: checkpoint={ckpt_flags}, current={current_flags}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
