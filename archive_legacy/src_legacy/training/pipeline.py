"""Shared training pipeline used by entry scripts."""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from data.factory import build_dataloaders
from models.cd_model import build_model
from training.checkpoint import find_latest, peek as peek_ckpt
from training.final_eval import run_final_test_evaluation
from training.logger import get_logger
from training.losses import build_loss
from training.trainer import Trainer
from utils.config import GLOBAL_CONFIG_PATH
from utils.ablation import config_fingerprint, log_parameter_breakdown, log_startup_config, module_flags
from utils.checkpoint_identity import checkpoint_identity
from utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]


def dataset_run_label(exp_name: str, dataset_name: str) -> str:
    dataset_slug = dataset_name.replace("/", "-").replace(" ", "_")
    if dataset_slug.lower() in exp_name.lower():
        return exp_name
    return f"{exp_name}_{dataset_slug}"


def _slug(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-cd", "")
        .replace("/", "-")
        .replace(" ", "_")
    )


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 10000):
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output directory for {path}")


def _is_ablation_run(cfg: dict, config_source_path: Path | None) -> bool:
    if config_source_path is not None and "configs/ablations" in str(config_source_path).replace("\\", "/"):
        return True
    name = str(cfg.get("experiment", {}).get("name", "")).lower()
    return name.startswith("a") and "_" in name


def _is_safe_same_ablation_resume(cfg: dict, ckpt_meta: dict[str, Any]) -> tuple[bool, str]:
    """Allow resuming the same ablation while blocking cross/full-model leakage."""
    current_exp = str(cfg.get("experiment", {}).get("name", "")).lower()
    ckpt_exp = str(
        ckpt_meta.get("experiment_name")
        or ckpt_meta.get("config", {}).get("experiment", {}).get("name", "")
    ).lower()
    current_fp = str(cfg.get("_meta", {}).get("config_fingerprint", config_fingerprint(cfg)))
    ckpt_fp = str(ckpt_meta.get("config_fingerprint", ""))
    current_flags = module_flags(cfg)
    ckpt_flags = ckpt_meta.get("module_flags")

    if ckpt_exp and ckpt_exp == current_exp:
        return True, "checkpoint experiment name matches current ablation"
    if ckpt_fp and ckpt_fp == current_fp:
        return True, "checkpoint config fingerprint matches current config"
    if isinstance(ckpt_flags, dict) and ckpt_flags == current_flags:
        return True, "checkpoint module flags match current ablation"
    return False, (
        f"checkpoint experiment={ckpt_exp or 'unknown'} fingerprint={ckpt_fp or 'missing'} "
        f"does not match current experiment={current_exp or 'unknown'} fingerprint={current_fp}"
    )


def _resolve_output_dir(cfg: dict, output_dir: Path | None) -> Path:
    exp = cfg.experiment
    ds = cfg.dataset
    training = cfg.get("training", {})
    overwrite = bool(training.get("overwrite_output_dir", False))
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_slug = _slug(str(ds.name))
        variant = str(exp.name)
        seed = int(exp.seed)
        run_name = f"run_{dataset_slug}_{variant}_seed{seed}_{ts}"
        out_dir = ROOT / exp.output_root / run_name
    else:
        out_dir = Path(output_dir)
        if not overwrite and str(exp.name) not in out_dir.name:
            out_dir = out_dir / str(exp.name)
    if out_dir.exists() and not overwrite:
        out_dir = _unique_dir(out_dir)
    return out_dir


def cosine_schedule(optimizer, max_iter: int, warmup: int, eta_min: float = 1e-5):
    def lr_fn(it: int) -> float:
        if it < warmup:
            return max(it / max(warmup, 1), 1e-4)
        prog = (it - warmup) / max(max_iter - warmup, 1)
        return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def run_training_pipeline(
    cfg: dict,
    *,
    output_dir: Path | None = None,
    config_source_path: Path | None = None,
) -> dict[str, Any]:
    exp = cfg.experiment
    tc = cfg.training
    hw = cfg.hardware
    ds = cfg.dataset

    set_seed(int(exp.seed))

    out_dir = _resolve_output_dir(cfg, output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    fingerprint = cfg.get("_meta", {}).get("config_fingerprint", config_fingerprint(cfg))
    (out_dir / "config_fingerprint.txt").write_text(str(fingerprint), encoding="utf-8")
    if config_source_path is not None:
        shutil.copy(config_source_path, out_dir / "config_source.yaml")

    logger = get_logger(exp.name, out_dir / "logs")
    logger.info(f"Experiment : {exp.name}")
    logger.info(f"Dataset    : {ds.name}  mode={ds.get('mode', 'binary')}")
    logger.info(f"Output dir : {out_dir}")
    logger.info(
        "Config     : %s",
        str(config_source_path) if config_source_path is not None else str(GLOBAL_CONFIG_PATH.relative_to(ROOT)),
    )
    log_startup_config(logger, cfg, config_source_path)

    resume_cfg = cfg.resume
    is_ablation = _is_ablation_run(cfg, config_source_path)
    logger.info("resume_checkpoint      : %s", resume_cfg.checkpoint_path if resume_cfg.get("enabled", False) else "none")
    logger.info("backbone_pretrained    : %s", str(bool(cfg.get("model", {}).get("pretrained", False))).lower())
    logger.info("full_model_resume      : %s", str(bool(resume_cfg.get("enabled", False))).lower())

    if resume_cfg.enabled:
        raw = resume_cfg.checkpoint_path
        if not raw:
            found = find_latest(ROOT / exp.output_root)
            if found is None:
                raise FileNotFoundError("resume.checkpoint_path is null and no run_*/latest.pth or best.pth found.")
            raw = str(found)
            logger.info(f"Auto-found checkpoint: {raw}")
            cfg.resume.checkpoint_path = raw
        path = Path(raw)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
            cfg.resume.checkpoint_path = str(path)
        meta = peek_ckpt(path)
        identity = checkpoint_identity(path, meta)
        if is_ablation and not bool(tc.get("allow_resume_for_ablation", False)):
            safe_resume, reason = _is_safe_same_ablation_resume(cfg, meta)
            ckpt_exp = str(identity.get("checkpoint_experiment_name", "")).lower()
            if not safe_resume or ("full" in ckpt_exp and str(exp.name).lower() != ckpt_exp):
                raise RuntimeError(
                    "Ablation resume safety check failed. Same-ablation resumes are allowed, "
                    "but cross-ablation/full-model checkpoint resumes require "
                    "training.allow_resume_for_ablation: true. "
                    f"{reason}. checkpoint_path={path}"
                )
            logger.info("Ablation resume safety check passed: %s", reason)
        (out_dir / "resume_info.json").write_text(
            json.dumps(
                {
                    "source": str(path),
                    "resume_iteration": meta.get("iteration", 0),
                    "best_metric_at_resume": meta.get("best_metric", 0.0),
                    **identity,
                },
                indent=2,
            )
        )
        ckpt_exp = str(identity.get("checkpoint_experiment_name", "")).lower()
        if is_ablation and "full" in ckpt_exp:
            logger.warning("Resume checkpoint appears to come from a full-model run: %s", ckpt_exp)
        logger.info(f"Resuming from iter {meta.get('iteration', 0)}, best={meta.get('best_metric', 0):.4f}")

    device = torch.device(hw.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    logger.info(f"Device : {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train samples : {len(train_loader.dataset)} | Val tiles : {len(val_loader.dataset)}")

    model = build_model(cfg).to(device)
    if bool(cfg.get("efficiency", {}).get("channels_last", False)):
        try:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Channels-last memory format enabled.")
        except Exception as exc:
            cfg.setdefault("efficiency", {})["channels_last"] = False
            logger.warning("channels_last requested but disabled: %s", exc)
    param_info = log_parameter_breakdown(logger, model)
    total_p = param_info["total_params"]
    trainable_p = param_info["trainable_params"]
    backbone_p = param_info["encoder_params"]
    decoder_p = param_info["decoder_params"]
    drbi_p = param_info["drbi_params"]
    semantic_head_p = 0
    decoder = getattr(model, "decoder", None)
    binary_head_p = 0
    for attr in ("head", "coarse_head"):
        head = getattr(decoder, attr, None)
        if head is not None:
            binary_head_p = sum(p.numel() for p in head.parameters())
            break
    semantic_head = getattr(model, "semantic_head", None)
    semantic_head_params = sum(p.numel() for p in semantic_head.parameters()) if semantic_head is not None else 0
    logger.info(f"Variant          : {cfg.model.variant}")
    logger.info(f"Total params     : {total_p / 1e6:.2f}M")
    logger.info(f"Trainable params : {trainable_p / 1e6:.2f}M")
    logger.info(f"Backbone params  : {backbone_p / 1e6:.2f}M")
    logger.info(f"Decoder params   : {decoder_p / 1e6:.2f}M")
    logger.info(f"D-RBI params     : {drbi_p / 1e6:.2f}M")
    logger.info(f"Semantic head    : {semantic_head_p / 1e6:.2f}M")
    logger.info(f"Binary head      : {binary_head_p / 1e6:.2f}M")
    if hasattr(model, "get_ablation_trace"):
        ablation_trace = model.get_ablation_trace()
        ablation_trace["config_path"] = str(config_source_path) if config_source_path is not None else str(GLOBAL_CONFIG_PATH.relative_to(ROOT))
        ablation_trace["run_dir"] = str(out_dir)
        (out_dir / "ablation_trace.json").write_text(json.dumps(ablation_trace, indent=2, default=str), encoding="utf-8")
        logger.info("Ablation trace: %s", json.dumps(ablation_trace, sort_keys=True, default=str))
    (out_dir / "model_info.json").write_text(
        json.dumps(
            {
                "variant": cfg.model.variant,
                "total_params": total_p,
                "trainable_params": trainable_p,
                "backbone_params": backbone_p,
                "decoder_params": decoder_p,
                "drbi_params": drbi_p,
                "semantic_head_params": semantic_head_p,
                "binary_head_params": binary_head_p,
                "arf_params": param_info["arf_params"],
                "cram_lite_params": param_info["cram_lite_params"],
                "boundary_refinement_params": param_info["boundary_refinement_params"],
                "module_flags": module_flags(cfg),
                "config_fingerprint": fingerprint,
                "config_path": str(config_source_path) if config_source_path is not None else str(GLOBAL_CONFIG_PATH.relative_to(ROOT)),
                "output_mode": str(cfg.model.output_mode),
                "encoder_channels": model.encoder.channels,
            },
            indent=2,
        )
    )

    if bool(cfg.model.freeze_backbone):
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        logger.info("Backbone frozen.")

    if bool(cfg.get("efficiency", {}).get("compile", False)):
        if hasattr(torch, "compile"):
            mode = str(cfg.get("efficiency", {}).get("compile_mode", "reduce-overhead"))
            try:
                logger.info("torch.compile enabled (mode=%s). First iterations may be slower.", mode)
                model = torch.compile(model, mode=mode)
            except Exception as exc:
                logger.warning("torch.compile failed; falling back to eager mode: %s", exc)
                cfg.setdefault("efficiency", {})["compile"] = False
        else:
            logger.warning("torch.compile requested but this PyTorch version does not support it.")
            cfg.setdefault("efficiency", {})["compile"] = False

    optimizer = getattr(torch.optim, tc.optimizer)(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(tc.lr),
        weight_decay=float(tc.weight_decay),
    )
    scheduler = cosine_schedule(optimizer, int(tc.max_iterations), int(tc.warmup_iterations))
    loss_fn = build_loss(cfg)
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    trainer = Trainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=out_dir,
        logger=logger,
        writer=writer,
    )
    try:
        trainer.train()
    finally:
        writer.close()

    final_metrics = run_final_test_evaluation(
        cfg=cfg,
        model=model,
        output_dir=out_dir,
        device=device,
        ema=trainer.ema,
        logger=logger,
    )
    logger.info("Done.")
    return {"output_dir": out_dir, "final_metrics": final_metrics}
