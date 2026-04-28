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
from utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]


def dataset_run_label(exp_name: str, dataset_name: str) -> str:
    dataset_slug = dataset_name.replace("/", "-").replace(" ", "_")
    if dataset_slug.lower() in exp_name.lower():
        return exp_name
    return f"{exp_name}_{dataset_slug}"


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

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = dataset_run_label(str(exp.name), str(ds.name))
        run_name = f"run_{ts}_{run_label}"
        out_dir = ROOT / exp.output_root / run_name
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
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

    resume_cfg = cfg.resume
    if resume_cfg.enabled:
        raw = resume_cfg.checkpoint_path
        if not raw:
            found = find_latest(ROOT / exp.output_root)
            if found is None:
                raise FileNotFoundError("resume.checkpoint_path is null and no run_*/best.pth found.")
            raw = str(found)
            logger.info(f"Auto-found checkpoint: {raw}")
            cfg.resume.checkpoint_path = raw
        path = Path(raw)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
            cfg.resume.checkpoint_path = str(path)
        meta = peek_ckpt(path)
        (out_dir / "resume_info.json").write_text(
            json.dumps(
                {
                    "source": str(path),
                    "resume_iteration": meta.get("iteration", 0),
                    "best_metric_at_resume": meta.get("best_metric", 0.0),
                },
                indent=2,
            )
        )
        logger.info(f"Resuming from iter {meta.get('iteration', 0)}, best={meta.get('best_metric', 0):.4f}")

    device = torch.device(hw.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    logger.info(f"Device : {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train samples : {len(train_loader.dataset)} | Val tiles : {len(val_loader.dataset)}")

    model = build_model(cfg).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_p = sum(p.numel() for p in getattr(model, "encoder", torch.nn.Module()).parameters())
    decoder_p = sum(p.numel() for p in getattr(model, "decoder", torch.nn.Module()).parameters())
    drbi_p = sum(p.numel() for p in getattr(model, "diff_modules", torch.nn.ModuleList()).parameters())
    semantic_head_module = getattr(model, "semantic_head", None)
    semantic_head_p = sum(p.numel() for p in semantic_head_module.parameters()) if semantic_head_module is not None else 0
    binary_head_p = 0
    decoder = getattr(model, "decoder", None)
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
