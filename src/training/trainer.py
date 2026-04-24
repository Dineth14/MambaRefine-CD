"""Iteration-based training loop for MERCon change detection.

Features
--------
* Mixed-precision (torch.amp)
* tqdm progress bars for training and validation
* Validation metric table printed after every validation pass
* Best-only checkpoint saved to output_dir/checkpoints/best.pth
* TensorBoard scalar logging
* CSV metric history
* Resume support via config (resume.enabled / resume.checkpoint_path)
* Variant mismatch detection on resume
"""
from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from training.losses           import BCEDiceLoss
from training.metrics          import StreamingMetrics
from training.boundary_metrics import BoundaryMetrics
from training.checkpoint       import save as save_ckpt, peek as peek_ckpt
from training.logger           import log_table
from training.ema              import EMA
from utils.visualization       import save_prediction_grid


class Trainer:
    """Iteration-based trainer."""

    def __init__(
        self,
        cfg: dict,
        model: nn.Module,
        loss_fn: BCEDiceLoss,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        logger: logging.Logger,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.cfg          = cfg
        self.model        = model
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.output_dir   = Path(output_dir)
        self.logger       = logger
        self.writer       = writer

        tc = cfg["training"]
        self.max_iter      = int(tc["max_iterations"])
        self.val_every     = int(tc.get("validate_every", 5000))
        self.log_every     = int(tc.get("log_every", 20))
        self.grad_clip     = float(tc.get("gradient_clip", 1.0))
        self.amp           = bool(cfg.get("hardware", {}).get("mixed_precision", True))
        self.aux_weight    = float(cfg.get("decoder", {}).get("aux_weight", 0.4))
        self.threshold     = float(cfg.get("evaluation", {}).get("threshold", 0.5))
        bm_cfg             = cfg.get("boundary_metrics", {})
        self.bnd_width     = int(bm_cfg.get("boundary_width", 3))
        self.bnd_tol       = int(bm_cfg.get("tolerance", 2))
        self.dataset_name  = cfg.get("dataset", {}).get("name", "unknown")

        # EMA
        tc2 = cfg.get("training", {})
        self.use_ema   = bool(tc2.get("use_ema", False))
        self.ema_decay = float(tc2.get("ema_decay", 0.999))
        self.ema: Optional[EMA] = None   # initialised in train()

        ck = cfg.get("checkpoint", {})
        self.monitor      = ck.get("monitor", "f1")
        self.monitor_mode = ck.get("mode", "max")

        vc = cfg.get("validation", {})
        self.save_samples = bool(vc.get("save_samples", True))
        self.sample_count = int(vc.get("sample_count", 16))

        self.best_metric  = float("-inf") if self.monitor_mode == "max" else float("inf")
        self.start_iter   = 0

        self.ckpt_dir   = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.val_dir    = self.output_dir / "validation"
        self.val_csv    = self.val_dir / "val_metrics.csv"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp and device.type == "cuda"
        )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def _resolve_ckpt_path(self, raw: str) -> Path:
        p = Path(raw)
        if not p.is_absolute():
            repo = Path(__file__).resolve().parents[2]
            p = (repo / p).resolve()
        return p

    def _resolve_variant_name(self, v: str) -> str:
        try:
            from models.backbone.mambavision_builder import resolve_name
            return resolve_name(v)
        except Exception:
            return v

    def resume(self, resume_cfg: dict) -> None:
        raw = resume_cfg.get("checkpoint_path")
        if not raw:
            raise ValueError("resume.checkpoint_path is null — set a path or disable resume.")
        ckpt_path = self._resolve_ckpt_path(raw)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        strict = bool(resume_cfg.get("strict", True))
        ckpt   = peek_ckpt(ckpt_path)

        # Variant mismatch check (normalised)
        ckpt_var    = ckpt.get("variant") or ckpt.get("config", {}).get("model", {}).get("variant")
        current_var = self.cfg.get("model", {}).get("variant")
        if ckpt_var and current_var:
            c_canon = self._resolve_variant_name(ckpt_var)
            r_canon = self._resolve_variant_name(current_var)
            if c_canon != r_canon:
                msg = f"Variant mismatch: checkpoint={ckpt_var!r}, current={current_var!r}"
                if strict:
                    raise RuntimeError(msg)
                self.logger.warning(f"WARNING: {msg} — loading anyway (strict=False)")

        self.model.load_state_dict(ckpt["model"], strict=strict)

        if ckpt.get("optimizer") and self.optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if ckpt.get("scheduler") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        self.start_iter  = ckpt.get("iteration", 0)
        self.best_metric = ckpt.get("best_metric", self.best_metric)

        sep = "-" * 52
        self.logger.info(sep)
        self.logger.info("RESUME TRAINING")
        self.logger.info(f"  Checkpoint  : {ckpt_path}")
        self.logger.info(f"  Start Iter  : {self.start_iter}")
        self.logger.info(f"  Best Metric : {self.best_metric:.4f}")
        if ckpt_var:
            self.logger.info(f"  Variant     : {ckpt_var}")
        self.logger.info(sep)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _step(self, batch: dict):
        ia = batch["image_a"].to(self.device, non_blocking=True)
        ib = batch["image_b"].to(self.device, non_blocking=True)
        lb = batch["label"].to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.amp and self.device.type == "cuda"):
            logits, aux = self.model(ia, ib)
            total, bce, dice = self.loss_fn(logits, lb)
            if aux is not None:
                aux_total, _, _ = self.loss_fn(aux, lb)
                total = total + self.aux_weight * aux_total

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        # EMA update after every optimiser step
        if self.ema is not None:
            self.ema.update(self.model)
        return total.item(), bce.item(), dice.item()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, iteration: int) -> dict:
        # Apply EMA weights for validation if enabled
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        pix_metrics = StreamingMetrics(threshold=self.threshold)
        bnd_metrics = BoundaryMetrics(
            boundary_width = self.bnd_width,
            tolerance      = self.bnd_tol,
            threshold      = self.threshold,
        )
        sample_done = False

        val_iter = tqdm(self.val_loader, desc="Validating", leave=False, unit="batch") if _TQDM else self.val_loader
        for batch in val_iter:
            ia = batch["image_a"].to(self.device, non_blocking=True)
            ib = batch["image_b"].to(self.device, non_blocking=True)
            lbl_key = "label" if "label" in batch else "mask"
            lb = batch[lbl_key].to(self.device, non_blocking=True)
            logits, _ = self.model(ia, ib)
            pix_metrics.update(logits, lb)
            bnd_metrics.update(logits, lb)

            if _TQDM:
                m = pix_metrics.compute()
                val_iter.set_postfix(  # type: ignore[union-attr]
                    F1=f"{m['f1']:.3f}",
                    IoU=f"{m['iou']:.3f}",
                    BF1=f"{m.get('boundary_f1', 0):.3f}",
                )

            if self.save_samples and not sample_done:
                n = min(self.sample_count, ia.shape[0])
                save_prediction_grid(ia[:n], ib[:n], lb[:n], logits[:n],
                                     self.sample_dir / f"iter_{iteration:07d}.png", count=n)
                sample_done = True

        # Restore live weights before resuming training
        if self.ema is not None:
            self.ema.restore(self.model)

        self.model.train()
        result = pix_metrics.compute()
        bm     = bnd_metrics.compute()
        result.update(bm)            # tolerance-aware BF1 + edge_iou override
        result["dataset"]          = self.dataset_name
        result["best_threshold"]   = self.threshold  # fixed during training
        result["ema_enabled"]      = self.ema is not None
        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        resume_cfg = self.cfg.get("resume", {})
        if resume_cfg.get("enabled", False):
            self.resume(resume_cfg)

        # Initialise EMA after potential resume so shadow copies loaded weights
        if self.use_ema:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.logger.info(f"EMA enabled (decay={self.ema_decay})")

        self._current_iter = self.start_iter
        self.logger.info(
            f"Training | max_iter={self.max_iter} | "
            f"val_every={self.val_every} | amp={self.amp}"
        )

        csv_cols = [
            "iteration", "dataset",
            "f1", "iou", "miou", "precision", "recall", "oa",
            "boundary_f1", "edge_iou",
            "pred_positive_ratio", "gt_positive_ratio",
            "best_threshold",
        ]
        if not self.val_csv.exists():
            with open(self.val_csv, "w", newline="") as f:
                csv.writer(f).writerow(csv_cols)

        loader_iter = iter(self.train_loader)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = (
            tqdm(total=self.max_iter, initial=self.start_iter, desc="Training", unit="iter")
            if _TQDM else None
        )

        for iteration in range(self.start_iter, self.max_iter):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            self._current_iter = iteration
            loss, bce, dice = self._step(batch)

            if self.scheduler:
                self.scheduler.step()

            if (iteration + 1) % self.log_every == 0:
                lr  = self.optimizer.param_groups[0]["lr"]
                msg = (
                    f"[{iteration+1}/{self.max_iter}] "
                    f"loss={loss:.4f} bce={bce:.4f} dice={dice:.4f} lr={lr:.2e}"
                )
                if pbar:
                    pbar.set_description(msg)
                else:
                    self.logger.info(msg)
                if self.writer:
                    self.writer.add_scalar("train/loss",  loss,  iteration)
                    self.writer.add_scalar("train/bce",   bce,   iteration)
                    self.writer.add_scalar("train/dice",  dice,  iteration)
                    self.writer.add_scalar("train/lr",    lr,    iteration)

            if pbar:
                pbar.update(1)

            if (iteration + 1) % self.val_every == 0:
                vm  = self._validate(iteration + 1)
                self._print_val_block(vm, iteration + 1)
                log_table(self.logger, vm, title="")  # detailed table

                if self.writer:
                    for k, v in vm.items():
                        self.writer.add_scalar(f"val/{k}", v, iteration)

                with open(self.val_csv, "a", newline="") as f:
                    row = [iteration + 1, vm.get("dataset", self.dataset_name)]
                    row += [vm.get(c, 0.0) for c in csv_cols[2:]]
                    csv.writer(f).writerow(row)

                monitor_val = vm.get(self.monitor, vm.get("f1", 0.0))
                is_best = (
                    monitor_val > self.best_metric
                    if self.monitor_mode == "max"
                    else monitor_val < self.best_metric
                )
                if is_best:
                    self.best_metric = monitor_val
                    save_ckpt(
                        self.ckpt_dir / "best.pth",
                        self.model, self.optimizer, self.scheduler,
                        iteration + 1, self.best_metric, self.cfg,
                    )
                    self.logger.info(
                        f"  ✓ New best {self.monitor}={self.best_metric:.4f} saved."
                    )

        if pbar:
            pbar.close()

    # ------------------------------------------------------------------
    # Validation summary block
    # ------------------------------------------------------------------
    def _print_val_block(self, vm: dict, iteration: int) -> None:
        sep = "-" * 42
        self.logger.info(sep)
        self.logger.info(f"Validation Results  (iter {iteration})")
        self.logger.info(f"  Best Threshold  : {vm.get('best_threshold', self.threshold):.2f}")
        self.logger.info(f"  F1              : {vm.get('f1', 0.0):.4f}")
        self.logger.info(f"  IoU (change)    : {vm.get('iou', 0.0):.4f}")
        self.logger.info(f"  mIoU            : {vm.get('miou', 0.0):.4f}")
        self.logger.info(f"  Precision       : {vm.get('precision', 0.0):.4f}")
        self.logger.info(f"  Recall          : {vm.get('recall', 0.0):.4f}")
        self.logger.info(f"  OA              : {vm.get('oa', 0.0):.4f}")
        self.logger.info(f"  Boundary F1     : {vm.get('boundary_f1', 0.0):.4f}")
        ema_tag = " [EMA]" if vm.get("ema_enabled") else ""
        self.logger.info(f"  EMA             : {'on' if vm.get('ema_enabled') else 'off'}")
        self.logger.info(sep)
