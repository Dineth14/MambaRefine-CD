"""Training entry point.

Configuration is loaded only from configs/global_config.yaml.
No CLI arguments needed.

Usage:
    conda activate mamba_new
    cd mercon_cd_clean
    python scripts/train.py
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.config         import GLOBAL_CONFIG_PATH, load_config
from utils.seed           import set_seed
from data.factory         import build_dataloaders
from models.cd_model      import build_model
from training.losses      import build_loss
from training.trainer     import Trainer
from training.logger      import get_logger
from training.checkpoint  import find_latest, peek as peek_ckpt
from training.final_eval  import run_final_test_evaluation


def _cosine_schedule(optimizer, max_iter: int, warmup: int, eta_min: float = 1e-5):
    def lr_fn(it: int) -> float:
        if it < warmup:
            return max(it / max(warmup, 1), 1e-4)
        prog = (it - warmup) / max(max_iter - warmup, 1)
        return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def main() -> None:
    cfg = load_config()
    exp = cfg.experiment
    tc  = cfg.training
    hw  = cfg.hardware

    set_seed(int(exp.seed))

    # ── Output directory ──────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{ts}_{exp.name}"
    out_dir  = ROOT / exp.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(GLOBAL_CONFIG_PATH, out_dir / "config.yaml")

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = get_logger(exp.name, out_dir / "logs")
    logger.info(f"Experiment : {exp.name}")
    logger.info(f"Output dir : {out_dir}")

    # ── Resume pre-processing ─────────────────────────────────────────────────
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
        # Resolve relative path
        p = Path(raw)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
            cfg.resume.checkpoint_path = str(p)
        meta = peek_ckpt(p)
        resume_info = {
            "source": str(p),
            "resume_iteration": meta.get("iteration", 0),
            "best_metric_at_resume": meta.get("best_metric", 0.0),
        }
        (out_dir / "resume_info.json").write_text(json.dumps(resume_info, indent=2))
        logger.info(f"Resuming from iter {meta.get('iteration', 0)}, best={meta.get('best_metric', 0):.4f}")

    # ── Device ────────────────────────────────────────────────────────────────
    device_str = hw.device
    device     = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    logger.info(f"Device : {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train samples : {len(train_loader.dataset)} | Val tiles : {len(val_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    variant     = cfg.model.variant
    logger.info(f"Variant          : {variant}")
    logger.info(f"Total params     : {total_p / 1e6:.2f}M")
    logger.info(f"Trainable params : {trainable_p / 1e6:.2f}M")
    (out_dir / "model_info.json").write_text(json.dumps({
        "variant": variant,
        "total_params": total_p,
        "trainable_params": trainable_p,
        "encoder_channels": model.encoder.channels,
    }, indent=2))

    if bool(cfg.model.freeze_backbone):
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        logger.info("Backbone frozen.")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    lr        = float(tc.lr)
    wd        = float(tc.weight_decay)
    opt_name  = tc.optimizer
    optimizer = getattr(torch.optim, opt_name)(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd
    )
    max_iter = int(tc.max_iterations)
    warmup   = int(tc.warmup_iterations)
    scheduler = _cosine_schedule(optimizer, max_iter, warmup)

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = build_loss(cfg)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        cfg=cfg, model=model, loss_fn=loss_fn,
        optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader,
        device=device, output_dir=out_dir,
        logger=logger, writer=writer,
    )
    trainer.train()
    writer.close()

    # ── Post-training: automatic test evaluation ──────────────────────────────
    run_final_test_evaluation(
        cfg        = cfg,
        model      = model,
        output_dir = out_dir,
        device     = device,
        ema        = trainer.ema,
        logger     = logger,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
