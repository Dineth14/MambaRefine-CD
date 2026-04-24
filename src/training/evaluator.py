"""Standalone evaluator for change detection models.

Combines StreamingMetrics (pixel-level) with BoundaryMetrics (edge-level)
for a complete evaluation pass over a DataLoader.

Usage:
    evaluator = Evaluator(cfg, device)
    results   = evaluator.evaluate(model, loader, dataset_name="LEVIR-CD")

Results dict contains:
    f1, iou, miou, precision, recall, oa,
    boundary_f1, boundary_precision, boundary_recall, edge_iou,
    pred_positive_ratio, gt_positive_ratio,
    dataset, num_samples
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from training.metrics          import StreamingMetrics
from training.boundary_metrics import BoundaryMetrics
from training.logger           import log_table


class Evaluator:
    """Full evaluation pass combining pixel and boundary metrics.

    Args:
        cfg:    merged config dict (used to extract threshold, boundary config).
        device: torch device.
        logger: optional logger; if None a basic stdout logger is created.
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg    = cfg
        self.device = device

        self.threshold   = float(cfg.get("evaluation", {}).get("threshold", 0.5))
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

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        dataset_name: str = "unknown",
        amp: bool = False,
    ) -> dict:
        """Run one full evaluation pass and return the combined metric dict.

        Args:
            model:        model in eval or train mode (set to eval internally).
            loader:       DataLoader yielding {image_a, image_b, label/mask} batches.
            dataset_name: string identifier added to the results dict.
            amp:          enable automatic mixed precision.

        Returns:
            dict with all metrics + "dataset" and "num_samples" keys.
        """
        model.eval()
        pix_metrics = StreamingMetrics(threshold=self.threshold)
        bnd_metrics = BoundaryMetrics(
            boundary_width = self.bnd_width,
            tolerance      = self.bnd_tol,
            threshold      = self.threshold,
        ) if self.bnd_enabled else None

        num_samples = 0
        desc = f"Eval [{dataset_name}]"
        bar  = tqdm(loader, desc=desc, leave=False, unit="batch") if _TQDM else loader

        for batch in bar:
            ia  = batch["image_a"].to(self.device, non_blocking=True)
            ib  = batch["image_b"].to(self.device, non_blocking=True)
            lbl_key = "label" if "label" in batch else "mask"
            lb  = batch[lbl_key].to(self.device, non_blocking=True)
            num_samples += ia.shape[0]

            with torch.amp.autocast("cuda", enabled=(amp and self.device.type == "cuda")):
                logits, _ = model(ia, ib)

            pix_metrics.update(logits, lb)
            if bnd_metrics is not None:
                bnd_metrics.update(logits, lb)

            if _TQDM:
                m = pix_metrics.compute()
                bar.set_postfix(  # type: ignore[union-attr]
                    F1=f"{m['f1']:.3f}",
                    IoU=f"{m['iou']:.3f}",
                    BF1=f"{m.get('boundary_f1', 0):.3f}",
                )

        result = pix_metrics.compute()
        if bnd_metrics is not None:
            bm = bnd_metrics.compute()
            # Override boundary_f1 with the more accurate tolerance-aware version
            result.update(bm)
        else:
            result.setdefault("boundary_f1", 0.0)
            result.setdefault("edge_iou",    0.0)

        result["dataset"]     = dataset_name
        result["num_samples"] = num_samples
        return result

    def print_table(self, results: dict, title: str = "") -> None:
        """Print a formatted metric table using the existing logger infrastructure."""
        _LABELS = {
            "f1":                  "F1",
            "iou":                 "IoU-change",
            "miou":                "mIoU",
            "precision":           "Precision",
            "recall":              "Recall",
            "oa":                  "OA",
            "boundary_f1":         "Boundary F1",
            "boundary_precision":  "Bnd Precision",
            "boundary_recall":     "Bnd Recall",
            "edge_iou":            "Edge IoU",
            "pred_positive_ratio": "Pred Positive Ratio",
            "gt_positive_ratio":   "GT Positive Ratio",
        }
        rows = [
            (_LABELS.get(k, k), v)
            for k, v in results.items()
            if isinstance(v, (int, float)) and k not in ("num_samples",)
        ]
        if not rows:
            return
        if title:
            self.logger.info(title)
        w   = max(len(r[0]) for r in rows)
        sep = f"+-{'-' * w}-+-{'-' * 8}-+"
        self.logger.info(sep)
        self.logger.info(f"| {'Metric':<{w}} | {'Value':>8} |")
        self.logger.info(sep)
        for name, val in rows:
            self.logger.info(f"| {name:<{w}} | {val:>8.4f} |")
        self.logger.info(sep)
