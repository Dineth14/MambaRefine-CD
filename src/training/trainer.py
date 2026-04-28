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

from training.metrics          import StreamingMetrics
from training.boundary_metrics import BoundaryMetrics
from training.checkpoint       import save as save_ckpt, peek as peek_ckpt
from training.logger           import log_table
from training.ema              import EMA
from training.model_outputs    import normalize_model_output
from utils.visualization       import save_prediction_grid


# ---------------------------------------------------------------------------
# NaN diagnostic helpers
# ---------------------------------------------------------------------------

def _tstats(t: Optional[torch.Tensor]) -> dict:
    """Return {min, max, std, has_nan, has_inf} for a tensor, or empty dict."""
    if t is None:
        return {}
    with torch.no_grad():
        ft = t.detach().float()
        return {
            "min":     float(ft.min()),
            "max":     float(ft.max()),
            "std":     float(ft.std()),
            "has_nan": bool(torch.isnan(ft).any()),
            "has_inf": bool(torch.isinf(ft).any()),
        }


class _NanDiagWriter:
    """Writes per-iteration NaN diagnostic rows to <log_dir>/nan_debug.csv."""

    COLS = [
        "iteration",
        "f1_min", "f1_max", "f1_std",
        "f2_min", "f2_max", "f2_std",
        "d_min",  "d_max",  "d_std",
        "rg_min", "rg_max", "rg_mean",
        "bg_min", "bg_max", "bg_mean",
        "logit_min", "logit_max", "logit_std",
        "loss",
    ]

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "nan_debug.csv"
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.COLS)

    def write(
        self,
        iteration: int,
        debug_info: dict,
        logits: Optional[torch.Tensor],
        loss_val: float,
    ) -> None:
        s = debug_info   # keys: f1, f2, D, region_gate, boundary_gate
        f1s    = _tstats(s.get("f1"))
        f2s    = _tstats(s.get("f2"))
        ds     = _tstats(s.get("D"))
        rgs    = _tstats(s.get("region_gate"))
        bgs    = _tstats(s.get("boundary_gate"))
        lgts   = _tstats(logits)
        row = [
            iteration,
            f1s.get("min", ""), f1s.get("max", ""), f1s.get("std", ""),
            f2s.get("min", ""), f2s.get("max", ""), f2s.get("std", ""),
            ds.get("min",  ""), ds.get("max",  ""), ds.get("std",  ""),
            rgs.get("min", ""), rgs.get("max", ""), "",   # mean computed below
            bgs.get("min", ""), bgs.get("max", ""), "",
            lgts.get("min",""), lgts.get("max",""), lgts.get("std",""),
            loss_val,
        ]
        # Fill mean for gates
        if s.get("region_gate") is not None:
            row[12] = float(s["region_gate"].detach().float().mean())
        if s.get("boundary_gate") is not None:
            row[15] = float(s["boundary_gate"].detach().float().mean())
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


class Trainer:
    """Iteration-based trainer."""

    def __init__(
        self,
        cfg: dict,
        model: nn.Module,
        loss_fn: nn.Module,
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
        self.non_blocking_transfer = bool(tc.get("non_blocking_transfer", True))
        self.aux_weight    = float(cfg.get("decoder", {}).get("aux_weight", 0.4))
        self.threshold     = float(cfg.get("evaluation", {}).get("threshold", 0.5))
        bm_cfg             = cfg.get("boundary_metrics", {})
        self.bnd_width     = int(bm_cfg.get("boundary_width", 3))
        self.bnd_tol       = int(bm_cfg.get("tolerance", 2))
        self.dataset_name  = cfg.get("dataset", {}).get("name", "unknown")
        self.model_output_mode = str(cfg.get("model", {}).get("output_mode", "binary")).lower()
        self.dataset_mode = str(cfg.get("dataset", {}).get("mode", "binary")).lower()
        self.second_semantic_mode = (
            str(self.dataset_name).upper() == "SECOND"
            and self.model_output_mode == "semantic_change"
            and self.dataset_mode == "semantic"
        )

        # NaN detection / diagnostics
        self.skip_nan_steps   = bool(tc.get("skip_nan_steps", True))
        self.nan_diag_every   = int(tc.get("nan_diag_every", 50))
        self._nan_skipped     = 0

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
        self.log_dir    = self.output_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.nan_diag = _NanDiagWriter(self.log_dir)

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
        ia = batch["image_a"].to(self.device, non_blocking=self.non_blocking_transfer)
        ib = batch["image_b"].to(self.device, non_blocking=self.non_blocking_transfer)
        ignore_mask = batch.get("ignore_mask")
        valid_mask = None
        if ignore_mask is not None and torch.is_tensor(ignore_mask):
            ignore_mask = ignore_mask.to(self.device, non_blocking=self.non_blocking_transfer)
            valid_mask = (ignore_mask <= 0.5).float()
        if self.second_semantic_mode:
            lb = batch["change_mask"].to(self.device, non_blocking=self.non_blocking_transfer)
            if "valid_mask" in batch and torch.is_tensor(batch["valid_mask"]):
                valid_mask = batch["valid_mask"].to(self.device, non_blocking=self.non_blocking_transfer).float()
            semantic_batch = {
                "change_mask": lb,
                "label_a": batch["label_a"].to(self.device, non_blocking=self.non_blocking_transfer),
                "label_b": batch["label_b"].to(self.device, non_blocking=self.non_blocking_transfer),
                "valid_mask": valid_mask,
            }
        else:
            lb = batch["label"].to(self.device, non_blocking=self.non_blocking_transfer)

        # Collect D-RBI debug tensors every nan_diag_every iterations
        iteration   = getattr(self, "_current_iter", 0)
        do_diag     = (iteration % self.nan_diag_every == 0)
        debug_info: dict = {}

        if do_diag:
            debug_info["f1"] = ia   # raw image; real F1 would need hooks
            debug_info["f2"] = ib

        with torch.amp.autocast("cuda", enabled=self.amp and self.device.type == "cuda"):
            outputs = normalize_model_output(self.model(ia, ib))
            logits = torch.clamp(outputs["change_logits"], -20.0, 20.0)
            outputs["change_logits"] = logits
            aux = outputs.get("aux_logits")
            if aux is not None:
                outputs["aux_logits"] = torch.clamp(aux, -20.0, 20.0)

            if self.second_semantic_mode:
                total = self.loss_fn(outputs, semantic_batch)
                loss_stats = dict(getattr(self.loss_fn, "latest_stats", {}))
                if not loss_stats:
                    loss_stats = {
                        "total_loss": float(total.detach().item()),
                        "change_loss": 0.0,
                        "semantic_ce_loss": 0.0,
                        "consistency_loss": 0.0,
                        "sek_loss": 0.0,
                        "dice_loss": 0.0,
                        "focal_loss": 0.0,
                        "soft_kappa": 0.0,
                        "sek_was_sanitized": False,
                    }
            else:
                total, primary_loss, dice = self.loss_fn(logits, lb, valid_mask=valid_mask)
                loss_stats = dict(getattr(self.loss_fn, "latest_stats", {}))
                if aux is not None:
                    aux_total, _, _ = self.loss_fn(outputs["aux_logits"], lb, valid_mask=valid_mask)
                    total = total + self.aux_weight * aux_total
                if not loss_stats:
                    loss_stats = {
                        "total_loss": float(total.detach().item()),
                        "dice_loss": float(dice.detach().item()),
                        "focal_loss": float(primary_loss.detach().item()),
                        "sek_loss": 0.0,
                        "soft_kappa": 0.0,
                        "sek_was_sanitized": False,
                    }

        # ------ Fail-fast NaN/Inf detection --------------------------------
        loss_val = total.item()
        nan_detected = (
            not torch.isfinite(total)
            or not torch.isfinite(logits).all()
        )
        if nan_detected:
            tensor_checks = {
                "ia":     ia,
                "ib":     ib,
                "logits": logits,
                "loss":   total,
            }
            if self.second_semantic_mode:
                if outputs.get("sem_logits_t1") is not None:
                    tensor_checks["sem_logits_t1"] = outputs["sem_logits_t1"]
                if outputs.get("sem_logits_t2") is not None:
                    tensor_checks["sem_logits_t2"] = outputs["sem_logits_t2"]
            for name, t in tensor_checks.items():
                if not torch.isfinite(t).all():
                    self.logger.warning(
                        f"[NaN/Inf @ iter {iteration}] tensor='{name}' "
                        f"min={float(t.float().min()):.4f} "
                        f"max={float(t.float().max()):.4f}"
                    )
            self.nan_diag.write(iteration, debug_info, logits, float("nan"))
            self._nan_skipped += 1
            if self.skip_nan_steps:
                self.optimizer.zero_grad(set_to_none=True)
                self.logger.warning(
                    f"[NaN] Skipping optimizer step (skip #{self._nan_skipped})"
                )
                return {
                    "loss": float("nan"),
                    "dice_loss": float("nan"),
                    "focal_loss": float("nan"),
                    "sek_loss": float("nan"),
                    "soft_kappa": float("nan"),
                    "bce_loss": float("nan"),
                    "change_loss": float("nan"),
                    "semantic_ce_loss": float("nan"),
                    "consistency_loss": float("nan"),
                }
        # ------ End NaN detection ------------------------------------------

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.ema is not None:
            self.ema.update(self.model)

        # Periodic diagnostics (only on healthy steps)
        if do_diag and not nan_detected:
            self.nan_diag.write(iteration, debug_info, logits, loss_val)

        if loss_stats.get("sek_was_sanitized"):
            self.logger.warning(
                "[SeK loss @ iter %s] SeK-inspired loss became NaN/Inf; using sek_loss=0 for this batch.",
                iteration,
            )

        return {
            "loss": loss_val,
            "dice_loss": float(loss_stats.get("dice_loss", 0.0 if self.second_semantic_mode else float(dice.detach().item()))),
            "focal_loss": float(loss_stats.get("focal_loss", 0.0)),
            "sek_loss": float(loss_stats.get("sek_loss", 0.0)),
            "soft_kappa": float(loss_stats.get("soft_kappa", 0.0)),
            "bce_loss": float(loss_stats.get("bce_loss", 0.0)),
            "change_loss": float(loss_stats.get("change_loss", 0.0)),
            "semantic_ce_loss": float(loss_stats.get("semantic_ce_loss", 0.0)),
            "consistency_loss": float(loss_stats.get("consistency_loss", 0.0)),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, iteration: int) -> dict:
        # Apply EMA weights for validation if enabled
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        if self.second_semantic_mode:
            from training.evaluator import Evaluator

            evaluator = Evaluator(self.cfg, self.device, logger=self.logger, save_dir=self.val_dir)
            result = evaluator.evaluate(self.model, self.val_loader, dataset_name=self.dataset_name, amp=self.amp)
            if self.ema is not None:
                self.ema.restore(self.model)
            self.model.train()
            result["ema_enabled"] = self.ema is not None
            return result

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
            ia = batch["image_a"].to(self.device, non_blocking=self.non_blocking_transfer)
            ib = batch["image_b"].to(self.device, non_blocking=self.non_blocking_transfer)
            lbl_key = "label" if "label" in batch else "mask"
            lb = batch[lbl_key].to(self.device, non_blocking=self.non_blocking_transfer)
            logits = normalize_model_output(self.model(ia, ib))["change_logits"]
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

        # ── Dataset statistics ────────────────────────────────────────────────
        try:
            from data.dataset_builder import log_dataset_stats, save_dataset_manifest
            dc = self.cfg.get("dataset", {})
            train_ds = self.train_loader.dataset
            val_ds   = self.val_loader.dataset
            stats = log_dataset_stats(train_ds, val_ds, None, self.logger, dc)
            manifest_path = self.output_dir / "dataset_manifests" / "levircd_manifest.json"
            save_dataset_manifest(stats, dc, manifest_path)
        except Exception as _e:
            self.logger.warning(f"Dataset stats logging skipped: {_e}")

        if self.second_semantic_mode:
            csv_cols = [
                "iteration", "dataset", "OA", "mIoU", "SeK", "Fscd",
            ]
        else:
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
            step_stats = self._step(batch)

            if self.scheduler:
                self.scheduler.step()

            if (iteration + 1) % self.log_every == 0:
                lr  = self.optimizer.param_groups[0]["lr"]
                # Display nan/inf as "NaN" instead of crashing format string
                def _fmt(v):
                    return f"{v:.4f}" if isinstance(v, float) and not (v != v) else "NaN"
                loss_type = str(self.cfg.get("loss", {}).get("type", "bce_dice")).lower().replace("-", "_")
                if self.second_semantic_mode:
                    msg = (
                        f"[{iteration+1}/{self.max_iter}] "
                        f"loss={_fmt(step_stats['loss'])} "
                        f"change={_fmt(step_stats['change_loss'])} "
                        f"sem_ce={_fmt(step_stats['semantic_ce_loss'])} "
                        f"cons={_fmt(step_stats['consistency_loss'])} "
                        f"sek={_fmt(step_stats['sek_loss'])} lr={lr:.2e}"
                    )
                elif loss_type == "bce_dice":
                    msg = (
                        f"[{iteration+1}/{self.max_iter}] "
                        f"loss={_fmt(step_stats['loss'])} "
                        f"bce={_fmt(step_stats['bce_loss'])} "
                        f"dice={_fmt(step_stats['dice_loss'])} lr={lr:.2e}"
                    )
                else:
                    msg = (
                        f"[{iteration+1}/{self.max_iter}] "
                        f"loss={_fmt(step_stats['loss'])} "
                        f"dice={_fmt(step_stats['dice_loss'])} "
                        f"focal={_fmt(step_stats['focal_loss'])} "
                        f"sek={_fmt(step_stats['sek_loss'])} lr={lr:.2e}"
                    )
                if pbar:
                    pbar.set_description(msg)
                else:
                    self.logger.info(msg)
                if self.writer:
                    self.writer.add_scalar("train/loss", step_stats["loss"], iteration)
                    self.writer.add_scalar("train/dice_loss", step_stats["dice_loss"], iteration)
                    self.writer.add_scalar("train/focal_loss", step_stats["focal_loss"], iteration)
                    self.writer.add_scalar("train/sek_loss", step_stats["sek_loss"], iteration)
                    self.writer.add_scalar("train/soft_kappa", step_stats["soft_kappa"], iteration)
                    self.writer.add_scalar("train/bce", step_stats["bce_loss"], iteration)
                    if self.second_semantic_mode:
                        self.writer.add_scalar("train/change_loss", step_stats["change_loss"], iteration)
                        self.writer.add_scalar("train/semantic_ce_loss", step_stats["semantic_ce_loss"], iteration)
                        self.writer.add_scalar("train/consistency_loss", step_stats["consistency_loss"], iteration)
                    self.writer.add_scalar("train/lr", lr, iteration)

            if pbar:
                pbar.update(1)

            if (iteration + 1) % self.val_every == 0:
                vm  = self._validate(iteration + 1)
                self._print_val_block(vm, iteration + 1)
                log_table(self.logger, vm, title="")  # detailed table

                if self.writer:
                    for k, v in vm.items():
                        if isinstance(v, bool):
                            self.writer.add_scalar(f"val/{k}", float(v), iteration)
                        elif isinstance(v, (int, float)):
                            self.writer.add_scalar(f"val/{k}", v, iteration)
                        elif torch.is_tensor(v) and v.numel() == 1:
                            self.writer.add_scalar(f"val/{k}", float(v.item()), iteration)

                with open(self.val_csv, "a", newline="") as f:
                    row = [iteration + 1, vm.get("dataset", self.dataset_name)]
                    row += [vm.get(c, 0.0) for c in csv_cols[2:]]
                    csv.writer(f).writerow(row)

                monitor_val = vm.get(self.monitor, vm.get("f1", 0.0))
                if monitor_val is None:
                    monitor_val = float("-inf") if self.monitor_mode == "max" else float("inf")
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
        if self.second_semantic_mode:
            self.logger.info(f"  OA              : {vm.get('OA', 0.0):.4f}")
            self.logger.info(f"  mIoU            : {vm.get('mIoU', 0.0):.4f}")
            sek_val = vm.get('SeK')
            self.logger.info(f"  SeK             : {'N/A' if sek_val is None else f'{sek_val:.4f}'}")
            self.logger.info(f"  Fscd            : {vm.get('Fscd', 0.0):.4f}")
            self.logger.info(sep)
            return
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
