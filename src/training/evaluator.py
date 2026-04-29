"""Standalone evaluator for change detection models.

Combines StreamingMetrics (pixel-level) with BoundaryMetrics (edge-level)
for a complete evaluation pass over a DataLoader.

Features
--------
* Threshold sweep — tries a configurable list of thresholds and selects the
  one that maximises F1.  Saves ``validation/best_threshold.json``.
* TTA (Test-Time Augmentation) — averages logits over flips and rotation.
* Improved validation log with clear header/footer.
* Saves ``validation/val_metrics.csv`` and ``validation/tta_results.json``
  when ``save_dir`` is provided.

Usage
-----
    evaluator = Evaluator(cfg, device, save_dir=output_dir / "validation")
    results   = evaluator.evaluate(model, loader, dataset_name="DSIFN-CD")

Config keys
-----------
    evaluation:
      threshold: 0.5              # used when sweep is disabled
      threshold_sweep: false
      threshold_list: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
      use_tta: false
      tta_augmentations: [original, hflip, vflip, rot90]
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from training.metrics          import StreamingMetrics
from training.boundary_metrics import BoundaryMetrics
from training.logger           import log_table
from training.model_outputs    import normalize_model_output
from training.tta              import apply_tta, build_tta_augmentations


_LABELS = {
    "mf1":                 "mF1",
    "f1_1":                "F1_1 change",
    "f1_0":                "F1_0 no-change",
    "miou":                "mIoU",
    "iou_1":               "IoU_1 change",
    "iou_0":               "IoU_0 no-change",
    "precision_1":         "Precision_1",
    "recall_1":            "Recall_1",
    "oa":                  "OA",
    "boundary_f1":         "Boundary F1",
    "boundary_precision":  "Bnd Precision",
    "boundary_recall":     "Bnd Recall",
    "edge_iou":            "Edge IoU",
    "pred_positive_ratio": "Pred Positive Ratio",
    "gt_positive_ratio":   "GT Positive Ratio",
    "best_threshold":      "Best Threshold",
    # backward-compat aliases
    "f1":                  "F1_1 change",
    "iou":                 "IoU_1 change",
    "precision":           "Precision_1",
    "recall":              "Recall_1",
}


def _merged_eval_cfg(cfg: dict) -> dict:
    """Return evaluation config, accepting both legacy `evaluation` and new `eval`."""
    merged = dict(cfg.get("evaluation", {}) or {})
    merged.update(dict(cfg.get("eval", {}) or {}))
    return merged


class Evaluator:
    """Full evaluation pass combining pixel and boundary metrics.

    Args:
        cfg:      merged config dict.
        device:   torch device.
        logger:   optional logger.
        save_dir: if provided, saves threshold JSON and metric CSV here.
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.cfg      = cfg
        self.device   = device
        self.save_dir = Path(save_dir) if save_dir is not None else None

        ec = _merged_eval_cfg(cfg)
        self.threshold        = float(ec.get("threshold", 0.5))
        sweep_cfg = ec.get("threshold_sweep", False)
        if isinstance(sweep_cfg, dict):
            self.do_sweep = bool(sweep_cfg.get("enabled", False))
            sweep_values = sweep_cfg.get("values", ec.get("threshold_list", None))
        else:
            self.do_sweep = bool(sweep_cfg)
            sweep_values = ec.get("threshold_list", None)
        self.threshold_list: List[float] = [
            float(t) for t in (sweep_values or [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
        ]
        # Metric used to select the best threshold: "mF1" | "F1_1" | "IoU_1"
        self.threshold_select_metric: str = str(
            ec.get("threshold_select_metric", "mF1")
        )
        self.archived_metrics_enabled = False
        self.compute_sek = False
        self.sek_binary_fallback = False
        self.use_tta          = bool(ec.get("use_tta", False))
        self.tta_augmentations = build_tta_augmentations(cfg)
        self.inference_mode = str(ec.get("inference_mode", "patch")).lower()
        self.crop_size = int(ec.get("crop_size", cfg.get("dataset", {}).get("image_size", 256)))
        self.overlap = float(ec.get("overlap", 0.25))
        self.stride = max(1, int(round(self.crop_size * (1.0 - self.overlap))))
        self.log_mask_debug = bool(ec.get("log_mask_debug", True))
        self.save_debug_outputs = bool(ec.get("save_debug_outputs", False))
        self.save_predictions = bool(ec.get("save_predictions", cfg.get("output", {}).get("save_predictions", False)))
        self.save_visualizations = bool(ec.get("save_visualizations", cfg.get("output", {}).get("save_visualizations", True)))
        self.save_binary_head_change = bool(ec.get("save_binary_head_change", cfg.get("output", {}).get("save_binary_head_change", False)))
        self.debug_output_root = Path(ec.get("debug_output_root", "debug/eval"))
        self.debug_max_samples = int(ec.get("debug_max_samples", 20))
        self._debug_saved = 0

        dataset_cfg = cfg.get("dataset", {})
        model_cfg = cfg.get("model", {})
        self.output_mode = str(model_cfg.get("output_mode", "binary")).lower()
        self.dataset_mode = str(dataset_cfg.get("mode", "binary")).lower()
        self.archived_num_classes = int(dataset_cfg.get("num_classes", 1))
        self.archived_ignore_index = int(dataset_cfg.get("ignore_index", 255))

        bm_cfg           = cfg.get("boundary_metrics", {})
        self.bnd_width   = int(bm_cfg.get("boundary_width", 3))
        self.bnd_tol     = int(bm_cfg.get("tolerance", 2))
        self.bnd_enabled = bool(bm_cfg.get("enabled", True))

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("evaluator")
            if not self.logger.handlers:
                import sys
                h = logging.StreamHandler(sys.stdout)
                h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
                self.logger.addHandler(h)
                self.logger.setLevel(logging.INFO)

        self.logger.info(
            "Evaluation inference mode: %s | crop_size=%d | stride=%d | overlap=%.2f | logits averaged=%s",
            self.inference_mode,
            self.crop_size,
            self.stride,
            self.overlap,
            self.inference_mode == "sliding_window",
        )

    # ------------------------------------------------------------------
    # Internal: collect all logits + labels from one pass
    # ------------------------------------------------------------------
    @staticmethod
    def _squeeze_spatial(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            return tensor[:, 0]
        return tensor

    def _uses_archived_metrics(self, dataset_name: str) -> bool:
        return False

    @torch.no_grad()
    def _collect_logits(
        self,
        model: nn.Module,
        loader: DataLoader,
        amp: bool,
        dataset_name: str,
    ):
        """Run one forward pass over the loader and return stacked tensors.

        Returns stacked logits, stacked change labels, optional extras, and sample count."""
        all_logits = []
        all_labels = []
        extras = {
            "ignore_mask": [],
            "valid_mask": [],
            "label_a": [],
            "label_b": [],
            "change_mask": [],
            "pred_sem1": [],
            "pred_sem2": [],
            "sample_ids": [],
        }
        num_samples = 0

        desc = f"Eval [{dataset_name}]"
        bar  = tqdm(loader, desc=desc, leave=False, unit="batch") if _TQDM else loader

        for batch in bar:
            ia  = batch["image_a"].to(self.device, non_blocking=True)
            ib  = batch["image_b"].to(self.device, non_blocking=True)
            if "change_mask" in batch:
                lbl_key = "change_mask"
            else:
                lbl_key = "label" if "label" in batch else "mask"
            lb  = batch[lbl_key].to(self.device, non_blocking=True)
            batch_start = num_samples
            num_samples += ia.shape[0]

            if self.inference_mode == "sliding_window":
                logits = self._sliding_window_logits(model, ia, ib, amp)
                outputs = {"change_logits": logits}
            else:
                outputs = self._forward_outputs(model, ia, ib, amp)
                logits = torch.clamp(outputs["change_logits"], -20.0, 20.0)
            if logits.shape[-2:] != lb.shape[-2:]:
                logits = F.interpolate(logits, size=lb.shape[-2:], mode="bilinear", align_corners=False)

            if self.log_mask_debug and batch_start < 5:
                self._log_mask_debug(batch, lb, batch_start, getattr(loader, "dataset", None))

            if self.save_debug_outputs and self._debug_saved < self.debug_max_samples:
                self._save_binary_debug_batch(batch, logits, lb, dataset_name)

            all_logits.append(logits.cpu())
            all_labels.append(lb.cpu())
            for key in ("ignore_mask", "valid_mask", "label_a", "label_b", "change_mask"):
                if key in batch:
                    extras[key].append(batch[key].cpu())
            extras["sample_ids"].extend([str(x) for x in batch.get("name", batch.get("id", []))])

        all_logits = torch.cat(all_logits, dim=0)   # [N, 1, H, W]
        all_labels = torch.cat(all_labels, dim=0)
        stacked_extras = {
            key: (list(value) if key == "sample_ids" else (torch.cat(value, dim=0) if value else None))
            for key, value in extras.items()
        }
        return all_logits, all_labels, stacked_extras, num_samples

    def _forward_outputs(self, model: nn.Module, ia: torch.Tensor, ib: torch.Tensor, amp: bool) -> dict:
        if self.use_tta:
            return normalize_model_output(apply_tta(
                model, ia, ib,
                amp=amp,
                augmentations=self.tta_augmentations,
            ))
        with torch.amp.autocast("cuda", enabled=(amp and self.device.type == "cuda")):
            return normalize_model_output(model(ia, ib))

    @staticmethod
    def _window_positions(length: int, crop_size: int, stride: int) -> list[int]:
        if length <= crop_size:
            return [0]
        positions = list(range(0, length - crop_size + 1, stride))
        last = length - crop_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    def _sliding_window_logits(self, model: nn.Module, ia: torch.Tensor, ib: torch.Tensor, amp: bool) -> torch.Tensor:
        """Run tiled inference and average logits in overlapping regions."""
        bsz, _, h, w = ia.shape
        crop = self.crop_size
        stride = self.stride
        pad_h = max(0, crop - h)
        pad_w = max(0, crop - w)
        if pad_h or pad_w:
            ia = F.pad(ia, (0, pad_w, 0, pad_h), mode="replicate")
            ib = F.pad(ib, (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = ia.shape[-2:]
        ys = self._window_positions(padded_h, crop, stride)
        xs = self._window_positions(padded_w, crop, stride)

        outputs = []
        for n in range(bsz):
            accum = None
            count = ia.new_zeros((1, 1, padded_h, padded_w))
            for y in ys:
                for x in xs:
                    tile_a = ia[n : n + 1, :, y : y + crop, x : x + crop]
                    tile_b = ib[n : n + 1, :, y : y + crop, x : x + crop]
                    tile_out = self._forward_outputs(model, tile_a, tile_b, amp)
                    tile_logits = torch.clamp(tile_out["change_logits"], -20.0, 20.0)
                    if tile_logits.shape[-2:] != (crop, crop):
                        tile_logits = F.interpolate(tile_logits, size=(crop, crop), mode="bilinear", align_corners=False)
                    if accum is None:
                        accum = ia.new_zeros((1, tile_logits.shape[1], padded_h, padded_w))
                    accum[:, :, y : y + crop, x : x + crop] += tile_logits
                    count[:, :, y : y + crop, x : x + crop] += 1.0
            averaged = accum / count.clamp_min(1.0)
            outputs.append(averaged[:, :, :h, :w])
        return torch.cat(outputs, dim=0)

    def _log_mask_debug(self, batch: dict, labels: torch.Tensor, start_index: int, dataset=None) -> None:
        labels_cpu = labels.detach().cpu()
        batch_size = min(labels_cpu.shape[0], max(0, 5 - start_index))
        for i in range(batch_size):
            mask = labels_cpu[i]
            unique_after = sorted(float(x) for x in torch.unique(mask).tolist())
            positive_ratio = float((mask > 0.5).float().mean().item())
            raw_unique = "unavailable"
            raw_shape = tuple(mask.shape)
            if dataset is not None and hasattr(dataset, "raw_mask_stats"):
                try:
                    raw_stats = dataset.raw_mask_stats(start_index + i)
                    raw_unique = raw_stats.get("raw_unique", raw_unique)
                    raw_shape = tuple(raw_stats.get("shape", raw_shape))
                except Exception as exc:
                    raw_unique = f"unavailable ({exc})"
            sample_id = batch.get("id", batch.get("name", [f"sample_{start_index + i}"]))
            if isinstance(sample_id, (list, tuple)):
                sample_id = sample_id[i]
            self.logger.info(
                "Mask debug sample=%s | raw_unique=%s | converted_unique=%s | shape=%s | positive_ratio=%.6f",
                sample_id,
                raw_unique,
                unique_after,
                raw_shape,
                positive_ratio,
            )

    @staticmethod
    def _denorm_image(tensor: torch.Tensor) -> np.ndarray:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        img = (tensor.detach().cpu() * std + mean).clamp(0, 1)
        return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    @staticmethod
    def _safe_name(value: object) -> str:
        return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _save_binary_debug_batch(self, batch: dict, logits: torch.Tensor, labels: torch.Tensor, dataset_name: str) -> None:
        from PIL import Image

        split = str(_merged_eval_cfg(self.cfg).get("split", "eval"))
        root = self.debug_output_root / split
        dirs = {
            "image_t1": root / "image_t1",
            "image_t2": root / "image_t2",
            "gt": root / "gt",
            "pred": root / "pred",
            "prob": root / "prob",
            "error_map": root / "error_map",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        probs = torch.sigmoid(logits.detach().cpu())
        preds = (probs > self.threshold).to(torch.uint8)
        labels_cpu = (labels.detach().cpu() > 0.5).to(torch.uint8)
        batch_size = min(logits.shape[0], self.debug_max_samples - self._debug_saved)
        ids = batch.get("id", batch.get("name", [f"sample_{self._debug_saved + i}" for i in range(batch_size)]))
        for i in range(batch_size):
            sample_id = ids[i] if isinstance(ids, (list, tuple)) else f"sample_{self._debug_saved + i}"
            stem = f"{self._debug_saved:03d}_{self._safe_name(sample_id)}"
            gt = labels_cpu[i, 0] if labels_cpu[i].ndim == 3 else labels_cpu[i]
            pred = preds[i, 0] if preds[i].ndim == 3 else preds[i]
            prob = probs[i, 0] if probs[i].ndim == 3 else probs[i]

            Image.fromarray(self._denorm_image(batch["image_a"][i])).save(dirs["image_t1"] / f"{stem}.png")
            Image.fromarray(self._denorm_image(batch["image_b"][i])).save(dirs["image_t2"] / f"{stem}.png")
            Image.fromarray((gt.numpy() * 255).astype(np.uint8)).save(dirs["gt"] / f"{stem}.png")
            Image.fromarray((pred.numpy() * 255).astype(np.uint8)).save(dirs["pred"] / f"{stem}.png")
            Image.fromarray((prob.numpy() * 255.0).clip(0, 255).astype(np.uint8)).save(dirs["prob"] / f"{stem}.png")

            err = np.zeros((*gt.shape, 3), dtype=np.uint8)
            gt_np = gt.numpy().astype(bool)
            pred_np = pred.numpy().astype(bool)
            err[pred_np & gt_np] = (0, 180, 0)
            err[pred_np & ~gt_np] = (255, 80, 0)
            err[~pred_np & gt_np] = (0, 120, 255)
            Image.fromarray(err).save(dirs["error_map"] / f"{stem}.png")
            self._debug_saved += 1

    # ------------------------------------------------------------------
    # Internal: compute metrics at one threshold from pre-collected tensors
    # ------------------------------------------------------------------
    def _metrics_at_threshold(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
        threshold: float,
        compute_boundary: bool = True,
    ) -> dict:
        """Evaluate pre-collected logits at a specific threshold (batch-wise)."""
        bs = 8  # process in small batches to avoid OOM when running on CPU
        pix = StreamingMetrics(threshold=threshold)
        bnd = BoundaryMetrics(
            boundary_width=self.bnd_width,
            tolerance=self.bnd_tol,
            threshold=threshold,
        ) if (self.bnd_enabled and compute_boundary) else None

        n = all_logits.shape[0]
        for start in range(0, n, bs):
            lg = all_logits[start:start + bs].to(self.device)
            lb = all_labels[start:start + bs].to(self.device)
            pix.update(lg, lb)
            if bnd is not None:
                bnd.update(lg, lb)

        result = pix.compute()
        if bnd is not None:
            result.update(bnd.compute())
        else:
            result.setdefault("boundary_f1", 0.0)
            result.setdefault("edge_iou",    0.0)
        return result

    # ------------------------------------------------------------------
    # Public evaluate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        dataset_name: str = "unknown",
        amp: bool = False,
    ) -> dict:
        """Run evaluation (with optional threshold sweep and TTA).

        Returns:
            dict with all metrics + "dataset", "num_samples", "best_threshold".
        """
        model.eval()

        # ── 1. Collect all logits in one pass ────────────────────────────
        all_logits, all_labels, extras, num_samples = self._collect_logits(
            model, loader, amp, dataset_name
        )

        # ── 2. Threshold selection ────────────────────────────────────────
        if self.do_sweep:
            best_thr, best_f1, sweep_log = self._threshold_sweep(
                all_logits, all_labels, extras=extras
            )
        else:
            best_thr = self.threshold
            best_f1  = None
            sweep_log = None

        # ── 3. Full metrics at best threshold (including boundary) ────────
        result = self._metrics_at_threshold(
            all_logits,
            all_labels,
            best_thr,
            compute_boundary=bool(self.cfg.get("debug", {}).get("metrics", False)),
        )
        diagnostic_result = dict(result)
        if not bool(self.cfg.get("debug", {}).get("metrics", False)):
            result = {
                key: result[key]
                for key in ("precision", "recall", "f1", "iou", "oa")
                if key in result
            }
            for key in ("pred_positive_ratio", "gt_positive_ratio"):
                if key in diagnostic_result:
                    result[key] = diagnostic_result[key]
        probs = torch.sigmoid(torch.clamp(all_logits.float(), -20.0, 20.0))
        result["mean_sigmoid_probability"] = float(probs.mean().item())
        result["min_sigmoid_probability"] = float(probs.min().item())
        result["max_sigmoid_probability"] = float(probs.max().item())
        self.logger.info(
            "Prediction diagnostic | pred_positive_ratio=%.6f gt_positive_ratio=%.6f "
            "mean_prob=%.6f min_prob=%.6f max_prob=%.6f threshold=%.2f",
            float(result.get("pred_positive_ratio", 0.0)),
            float(result.get("gt_positive_ratio", 0.0)),
            result["mean_sigmoid_probability"],
            result["min_sigmoid_probability"],
            result["max_sigmoid_probability"],
            best_thr,
        )
        result["best_threshold"] = best_thr
        result["dataset"]        = dataset_name
        result["num_samples"]    = num_samples

        # ── 4. Save outputs ───────────────────────────────────────────────
        if self.save_dir is not None:
            self._save_outputs(result, sweep_log, dataset_name)

        return result

    # ------------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------------
    def _threshold_sweep(self, all_logits, all_labels, extras=None):
        """Try all thresholds in threshold_list.

        Selects the best threshold by ``self.threshold_select_metric``
        (one of: "mF1", "F1_1", "IoU_1").  Also records the best
        threshold for the other two metrics and saves
        ``best_thresholds.json`` when ``save_dir`` is set.

        Returns:
            (best_thr, best_score, sweep_log)
        """
        _metric_key_map = {
            "mF1":  "mf1",
            "F1_1": "f1_1",
            "IoU_1": "iou_1",
        }
        primary_key = _metric_key_map.get(self.threshold_select_metric, "mf1")

        best_by: dict = {"mf1": (-1.0, None), "f1_1": (-1.0, None), "iou_1": (-1.0, None)}
        sweep_log: dict = {}

        for thr in self.threshold_list:
            m = self._metrics_at_threshold(
                all_logits, all_labels, thr, compute_boundary=False
            )
            mf1_val  = float(m.get("mf1",  m.get("f1", 0.0)))
            f1_1_val = float(m.get("f1_1", m.get("f1", 0.0)))
            iou_1_val = float(m.get("iou_1", m.get("iou", 0.0)))
            sweep_log[f"{thr:.2f}"] = {
                "mf1":  round(mf1_val, 6),
                "f1_1": round(f1_1_val, 6),
                "iou_1": round(iou_1_val, 6),
            }
            for mk, val in [("mf1", mf1_val), ("f1_1", f1_1_val), ("iou_1", iou_1_val)]:
                if val > best_by[mk][0]:
                    best_by[mk] = (val, thr)

        best_thr   = best_by[primary_key][1]
        best_score = best_by[primary_key][0]

        self.logger.info(
            f"  Threshold sweep → best={best_thr:.2f}  "
            f"{self.threshold_select_metric}={best_score:.4f}"
        )

        # Save best_thresholds.json if save_dir is configured
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            thr_data = {
                "select_metric": self.threshold_select_metric,
                "best_thresholds": {
                    "mF1":  {"threshold": best_by["mf1"][1],  "value": round(best_by["mf1"][0],  6)},
                    "F1_1": {"threshold": best_by["f1_1"][1], "value": round(best_by["f1_1"][0], 6)},
                    "IoU_1": {"threshold": best_by["iou_1"][1], "value": round(best_by["iou_1"][0], 6)},
                },
                "sweep": sweep_log,
            }
            (self.save_dir / "best_thresholds.json").write_text(
                json.dumps(thr_data, indent=2)
            )

        return best_thr, best_score, sweep_log

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------
    def _save_outputs(self, result: dict, sweep_log, dataset_name: str) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # best_thresholds.json (written by _threshold_sweep when sweep is on;
        # write a simple single-threshold version when sweep is off)
        if not self.do_sweep:
            thr_data = {
                "select_metric": self.threshold_select_metric,
                "best_thresholds": {
                    "mF1":  {"threshold": result["best_threshold"], "value": round(result.get("mf1", result.get("f1", 0.0)), 6)},
                    "F1_1": {"threshold": result["best_threshold"], "value": round(result.get("f1_1", result.get("f1", 0.0)), 6)},
                    "IoU_1": {"threshold": result["best_threshold"], "value": round(result.get("iou_1", result.get("iou", 0.0)), 6)},
                },
            }
            thr_path = self.save_dir / "best_thresholds.json"
            thr_path.write_text(json.dumps(thr_data, indent=2))

        # val_metrics.csv  (append)
        csv_path = self.save_dir / "val_metrics.csv"
        write_header = not csv_path.exists()
        numeric_keys = [
            k for k, v in result.items()
            if isinstance(v, float) and k != "num_samples"
        ]
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["dataset"] + numeric_keys)
            w.writerow([dataset_name] + [result[k] for k in numeric_keys])

        metrics_payload = {}
        for key, value in result.items():
            if isinstance(value, float):
                metrics_payload[key] = round(value, 6)
            elif isinstance(value, (int, bool, str)):
                metrics_payload[key] = value
        try:
            from utils.ablation import config_fingerprint, module_flags
            meta = self.cfg.get("_meta", {}) if isinstance(self.cfg.get("_meta", {}), dict) else {}
            metrics_payload["config_fingerprint"] = meta.get("config_fingerprint", config_fingerprint(self.cfg))
            metrics_payload["module_flags"] = module_flags(self.cfg)
        except Exception:
            pass
        (self.save_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

        # tta_results.json  (if TTA was used)
        if self.use_tta:
            tta_path = self.save_dir / "tta_results.json"
            tta_data = {
                "dataset":        dataset_name,
                "augmentations":  self.tta_augmentations,
                "mf1":            round(result.get("mf1", result.get("f1", 0.0)), 6),
                "f1_1":           round(result.get("f1_1", result.get("f1", 0.0)), 6),
                "iou_1":          round(result.get("iou_1", result.get("iou", 0.0)), 6),
                "best_threshold": result["best_threshold"],
            }
            if sweep_log:
                tta_data["sweep"] = sweep_log
            tta_path.write_text(json.dumps(tta_data, indent=2))

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------
    def print_table(self, results: dict, title: str = "EVALUATION RESULTS") -> None:
        """Print a formatted validation results block."""
        sep = "=" * 40
        self.logger.info(sep)
        self.logger.info(title)
        self.logger.info(sep)

        # Ordered display list — canonical new keys preferred, fallback to aliases
        def _v(key: str, alias: str | None = None) -> float | None:
            if key in results and isinstance(results[key], (int, float)):
                return results[key]
            if alias and alias in results and isinstance(results[alias], (int, float)):
                return results[alias]
            return None

        rows = [
            ("mF1",                  _v("mf1")),
            ("F1_1 change",          _v("f1_1", "f1")),
            ("F1_0 no-change",       _v("f1_0")),
            ("mIoU",                 _v("miou")),
            ("IoU_1 change",         _v("iou_1", "iou")),
            ("IoU_0 no-change",      _v("iou_0")),
            ("Precision_1",          _v("precision_1", "precision")),
            ("Recall_1",             _v("recall_1", "recall")),
            ("OA",                   _v("oa")),
            ("Boundary F1",          _v("boundary_f1")),
            ("Edge IoU",             _v("edge_iou")),
            ("Pred Positive Ratio",  _v("pred_positive_ratio")),
            ("GT Positive Ratio",    _v("gt_positive_ratio")),
        ]
        for label, val in rows:
            if val is not None:
                self.logger.info(f"{label:<22}: {val:.4f}")

        self.logger.info(sep)
