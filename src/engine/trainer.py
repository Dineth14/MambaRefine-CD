"""Iteration-based trainer for MambaRefine-CD."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cd_dataset import ChangeDetectionDataset
from src.datasets.transforms import get_train_transform, get_val_transform
from src.datasets.verify import verify_dataset
from src.engine.checkpoint import load_checkpoint, save_checkpoint
from src.engine.evaluator import evaluate
from src.engine.logger import get_logger, make_writer
from src.engine.losses import build_loss
from src.models.build import build_model
from src.utils.config import load_config, save_config, to_plain_dict
from src.utils.device import get_device
from src.utils.flops import measure_flops
from src.utils.fps import measure_fps
from src.utils.memory import peak_memory_mb, reset_peak_memory
from src.utils.misc import count_parameters, format_metrics
from src.utils.seed import set_seed


def _run_dir(cfg) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}_{cfg.ablation.id}"
    run_dir = Path(cfg.project.output_root) / name
    for sub in ("checkpoints", "tensorboard", "metrics", "predictions"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_loaders(cfg):
    train_ds = ChangeDetectionDataset(cfg.data.root, cfg.data.train_dir, cfg, transform=get_train_transform(cfg))
    val_ds = ChangeDetectionDataset(cfg.data.root, cfg.data.val_dir, cfg, transform=get_val_transform(cfg))
    test_ds = ChangeDetectionDataset(cfg.data.root, cfg.data.test_dir, cfg, transform=get_val_transform(cfg))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _scheduler(optimizer, cfg):
    total = int(cfg.train.iterations)
    warmup = int(cfg.train.warmup_iters)

    def lr_lambda(step):
        if step < warmup:
            return max(step / max(warmup, 1), 1e-4)
        progress = (step - warmup) / max(total - warmup, 1)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item()))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train():
    cfg = load_config()
    set_seed(int(cfg.project.seed))
    run_dir = _run_dir(cfg)
    save_config(cfg, str(run_dir))
    logger = get_logger(cfg.project.name, run_dir, "train.log")
    (run_dir / "val.log").touch()
    (run_dir / "test.log").touch()
    writer = make_writer(cfg, run_dir)

    logger.info("Verifying dataset ...")
    verify_dataset(cfg)
    _save_json(run_dir / "metrics" / "dataset_verification.json", cfg._dataset_verification)

    device = get_device(cfg)
    train_loader, val_loader, test_loader = _build_loaders(cfg)
    model = build_model(cfg).to(device)
    if bool(cfg.model.freeze_encoder):
        for param in model.encoder.parameters():
            param.requires_grad_(False)
    loss_fn = build_loss(cfg)
    total_params, trainable_params = count_parameters(model)
    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "encoder_out_channels": list(model.encoder.out_channels),
        "drbi_input_channels": list(model.drbi_input_channels),
        "temporal_mode": cfg.ablation.temporal_input_mode,
    }
    (run_dir / "model_summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Params: total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M")
    if bool(cfg.logging.print_flops):
        flops = measure_flops(model, int(cfg.data.image_size), device)
        if flops is not None:
            logger.info(f"FLOPs: {flops:.2f}G")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    scheduler = _scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp) and device.type == "cuda")

    start_iter = 0
    best_metric = float("-inf")
    best_iteration = 0
    best_threshold = float(cfg.eval.threshold)
    if bool(cfg.resume.enabled):
        ckpt = load_checkpoint(cfg.resume.path, model, optimizer if cfg.resume.resume_optimizer else None, scheduler if cfg.resume.resume_scheduler else None)
        start_iter = int(ckpt.get("iteration", 0)) if cfg.resume.resume_iteration else 0
        best_metric = float(ckpt.get("best_metric", best_metric))
        best_iteration = int(ckpt.get("best_iteration", best_iteration))
        best_threshold = float(ckpt.get("best_threshold", best_threshold))

    train_history = []
    val_history = []
    loader_iter = iter(train_loader)
    reset_peak_memory(device)
    model.train()
    pbar = tqdm(range(start_iter, int(cfg.train.iterations)), desc="Training", dynamic_ncols=True)
    for iteration in pbar:
        if cfg.model.unfreeze_after_iters is not None and iteration == int(cfg.model.unfreeze_after_iters):
            for param in model.encoder.parameters():
                param.requires_grad_(True)

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=bool(cfg.train.amp) and device.type == "cuda"):
            outputs = model(image_a, image_b)
            loss, loss_dict = loss_fn(outputs, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step = iteration + 1
        if step % int(cfg.train.log_interval) == 0:
            lr = optimizer.param_groups[0]["lr"]
            entry = {"iteration": step, "lr": lr, "peak_memory_mb": peak_memory_mb(device), **loss_dict}
            train_history.append(entry)
            if writer:
                for key, value in entry.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"train/{key}", value, step)
            logger.info(f"[{step:06d}] loss={loss_dict['loss_total']:.4f} lr={lr:.2e} mem={entry['peak_memory_mb']:.1f}MB")

        if step % int(cfg.train.val_interval) == 0:
            metrics = evaluate(model, val_loader, cfg, "val", sweep_thresholds=bool(cfg.eval.sweep_thresholds_on_val), device=device)
            current = float(metrics["F1"])
            threshold = float(metrics.get("best_threshold", metrics.get("threshold", cfg.eval.threshold)))
            val_history.append({"iteration": step, **metrics})
            if writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"val/{key}", value, step)
            logger.info(f"[VAL {step:06d}] {format_metrics(metrics)} threshold={threshold:.2f}")
            is_better = current > best_metric if bool(cfg.train.higher_is_better) else current < best_metric
            if is_better:
                best_metric = current
                best_iteration = step
                best_threshold = threshold
                ckpt_state = {
                    "iteration": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best_metric,
                    "best_iteration": best_iteration,
                    "best_threshold": best_threshold,
                    "config": to_plain_dict(cfg),
                }
                path = save_checkpoint(ckpt_state, run_dir / "checkpoints", step, best_metric, "F1")
                logger.info(f"Saved best checkpoint: {path}")
            model.train()
            _save_json(run_dir / "metrics" / "train_history.json", train_history)
            _save_json(run_dir / "metrics" / "val_history.json", val_history)

    logger.info(f"Training complete. Best F1={best_metric:.4f} iter={best_iteration} threshold={best_threshold:.2f}")
    best_ckpt = sorted((run_dir / "checkpoints").glob("best_iter_*_F1_*.pth"))
    if best_ckpt:
        load_checkpoint(best_ckpt[-1], model)
    final_val = evaluate(model, val_loader, cfg, "val", threshold=best_threshold, device=device)
    test_metrics = evaluate(model, test_loader, cfg, "test", threshold=best_threshold, device=device, save_dir=run_dir)
    test_metrics["best_val_threshold"] = best_threshold
    test_metrics["test_threshold_used"] = best_threshold
    _save_json(run_dir / "metrics" / "final_val_metrics.json", final_val)
    _save_json(run_dir / "metrics" / "test_metrics.json", test_metrics)
    if bool(cfg.eval.measure_fps):
        fps = measure_fps(model, int(cfg.data.image_size), device)
        _save_json(run_dir / "metrics" / "efficiency.json", {"FPS": fps, "peak_memory_mb": peak_memory_mb(device)})
        logger.info(f"FPS: {fps:.2f}")
    logger.info(f"[TEST] {format_metrics(test_metrics)}")
    if writer:
        writer.close()
    return {"run_dir": str(run_dir), "best_metric": best_metric, "test_metrics": test_metrics}
