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
    results   = evaluator.evaluate(model, loader, dataset_name="LEVIR-CD")

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
from training.model_outputs    import normalize_model_output
from training.second_metrics   import SECONDMetrics
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

        ec = cfg.get("evaluation", {})
        self.threshold        = float(ec.get("threshold", 0.5))
        self.do_sweep         = bool(ec.get("threshold_sweep", False))
        self.threshold_list: List[float] = [
            float(t) for t in ec.get(
                "threshold_list",
                [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            )
        ]
        # Metric used to select the best threshold: "mF1" | "F1_1" | "IoU_1"
        self.threshold_select_metric: str = str(
            ec.get("threshold_select_metric", "mF1")
        )
        self.second_metrics_enabled = bool(ec.get("second_metrics", False))
        self.compute_sek = bool(ec.get("compute_SeK", True))
        self.sek_binary_fallback = bool(ec.get("sek_binary_fallback", False))
        self.use_tta          = bool(ec.get("use_tta", False))
        self.tta_augmentations = build_tta_augmentations(cfg)

        dataset_cfg = cfg.get("dataset", {})
        model_cfg = cfg.get("model", {})
        self.output_mode = str(model_cfg.get("output_mode", "binary")).lower()
        self.dataset_mode = str(dataset_cfg.get("mode", "binary")).lower()
        self.second_num_classes = int(dataset_cfg.get("num_classes", model_cfg.get("semantic_num_classes", 7)))
        self.second_ignore_index = int(dataset_cfg.get("ignore_index", 255))

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

    def _uses_second_metrics(self, dataset_name: str) -> bool:
        return dataset_name.strip().upper() == "SECOND" and self.second_metrics_enabled

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
            num_samples += ia.shape[0]

            if self.use_tta:
                outputs = normalize_model_output(apply_tta(
                    model, ia, ib,
                    amp=amp,
                    augmentations=self.tta_augmentations,
                ))
            else:
                with torch.amp.autocast("cuda", enabled=(amp and self.device.type == "cuda")):
                    outputs = normalize_model_output(model(ia, ib))

            logits = torch.clamp(outputs["change_logits"], -20.0, 20.0)

            all_logits.append(logits.cpu())
            all_labels.append(lb.cpu())
            for key in ("ignore_mask", "valid_mask", "label_a", "label_b", "change_mask"):
                if key in batch:
                    extras[key].append(batch[key].cpu())
            extras["sample_ids"].extend([str(x) for x in batch.get("name", batch.get("id", []))])
            if outputs.get("sem_logits_t1") is not None:
                extras["pred_sem1"].append(torch.argmax(outputs["sem_logits_t1"], dim=1).cpu())
            if outputs.get("sem_logits_t2") is not None:
                extras["pred_sem2"].append(torch.argmax(outputs["sem_logits_t2"], dim=1).cpu())

        all_logits = torch.cat(all_logits, dim=0)   # [N, 1, H, W]
        all_labels = torch.cat(all_labels, dim=0)
        stacked_extras = {
            key: (list(value) if key == "sample_ids" else (torch.cat(value, dim=0) if value else None))
            for key, value in extras.items()
        }
        return all_logits, all_labels, stacked_extras, num_samples

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

    def _second_metrics_at_threshold(
        self,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor,
        extras: dict,
        threshold: float,
    ) -> dict:
        metrics = SECONDMetrics(
            num_classes=self.second_num_classes,
            ignore_index=self.second_ignore_index,
            compute_sek=self.compute_sek,
            sek_binary_fallback=self.sek_binary_fallback,
        )
        bs = 8
        n = all_logits.shape[0]
        for start in range(0, n, bs):
            end = start + bs
            logits = all_logits[start:end].float()
            change_gt = all_labels[start:end] > 0.5

            ignore_mask = self._squeeze_spatial(extras.get("ignore_mask")[start:end] if extras.get("ignore_mask") is not None else None)
            valid_mask = self._squeeze_spatial(extras.get("valid_mask")[start:end] if extras.get("valid_mask") is not None else None)
            if valid_mask is None and ignore_mask is not None:
                valid_mask = ~ignore_mask.bool()
            label_a = extras.get("label_a")
            label_b = extras.get("label_b")
            label_a = label_a[start:end] if label_a is not None else None
            label_b = label_b[start:end] if label_b is not None else None
            pred_sem1 = extras.get("pred_sem1")
            pred_sem2 = extras.get("pred_sem2")
            pred_sem1 = pred_sem1[start:end] if pred_sem1 is not None else None
            pred_sem2 = pred_sem2[start:end] if pred_sem2 is not None else None
            if start == 0 and pred_sem1 is not None and pred_sem2 is not None and label_a is not None and label_b is not None:
                valid_for_log = (label_a != self.second_ignore_index) & (label_b != self.second_ignore_index)
                self.logger.info(
                    "SECOND sanity | gt_t1=%s gt_t2=%s pred_t1=%s pred_t2=%s ignore=%d valid=%d change_ratio=%.6f",
                    sorted(torch.unique(label_a.detach().cpu()).tolist()),
                    sorted(torch.unique(label_b.detach().cpu()).tolist()),
                    sorted(torch.unique(pred_sem1.detach().cpu()).tolist()),
                    sorted(torch.unique(pred_sem2.detach().cpu()).tolist()),
                    int((~valid_for_log).sum().item()),
                    int(valid_for_log.sum().item()),
                    float((((label_a != label_b) & valid_for_log).sum().float() / valid_for_log.sum().clamp_min(1)).item()),
                )
            metrics.update(
                change_pred_binary=None,
                change_pred_semantic=None,
                change_gt=self._squeeze_spatial(change_gt),
                pred_sem1=pred_sem1,
                pred_sem2=pred_sem2,
                gt_sem1=label_a,
                gt_sem2=label_b,
                valid_mask=valid_mask,
            )

        return metrics.compute()

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
        use_second_metrics = self._uses_second_metrics(dataset_name)
        all_logits, all_labels, extras, num_samples = self._collect_logits(
            model, loader, amp, dataset_name
        )

        # ── 2. Threshold selection ────────────────────────────────────────
        if self.do_sweep:
            best_thr, best_f1, sweep_log = self._threshold_sweep(
                all_logits, all_labels, extras=extras, use_second_metrics=use_second_metrics
            )
        else:
            best_thr = self.threshold
            best_f1  = None
            sweep_log = None

        # ── 3. Full metrics at best threshold (including boundary) ────────
        if use_second_metrics:
            result = self._second_metrics_at_threshold(
                all_logits, all_labels, extras, best_thr
            )
            if self.save_dir is not None and bool(self.cfg.get("output", {}).get("save_predictions", False)):
                from utils.second_outputs import assert_second_prediction_dirs, save_second_prediction_batch

                pred_sem1 = extras.get("pred_sem1")
                pred_sem2 = extras.get("pred_sem2")
                sample_ids = extras.get("sample_ids") or [f"sample_{i}" for i in range(pred_sem1.shape[0] if pred_sem1 is not None else 0)]
                if pred_sem1 is not None and pred_sem2 is not None:
                    save_second_prediction_batch(
                        pred_t1=pred_sem1,
                        pred_t2=pred_sem2,
                        pred_change=pred_sem1 != pred_sem2,
                        sample_ids=sample_ids,
                        output_root=self.save_dir,
                        binary_head_logits=all_logits if bool(self.cfg.get("output", {}).get("save_binary_head_change", False)) else None,
                        save_visualizations=bool(self.cfg.get("output", {}).get("save_visualizations", True)),
                        save_binary_head_change=bool(self.cfg.get("output", {}).get("save_binary_head_change", False)),
                        threshold=best_thr,
                    )
                    assert_second_prediction_dirs(self.save_dir)
        else:
            result = self._metrics_at_threshold(
                all_logits, all_labels, best_thr, compute_boundary=True
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
    def _threshold_sweep(self, all_logits, all_labels, extras=None, use_second_metrics: bool = False):
        """Try all thresholds in threshold_list.

        Selects the best threshold by ``self.threshold_select_metric``
        (one of: "mF1", "F1_1", "IoU_1").  Also records the best
        threshold for the other two metrics and saves
        ``best_thresholds.json`` when ``save_dir`` is set.

        Returns:
            (best_thr, best_score, sweep_log)
        """
        if use_second_metrics:
            tracked_metrics = ["OA", "mIoU", "SeK", "Fscd"]
            primary_key = self.threshold_select_metric if self.threshold_select_metric in tracked_metrics else "Fscd"
            best_by = {key: (-1.0, None) for key in tracked_metrics}
            sweep_log: dict = {}
            for thr in self.threshold_list:
                m = self._second_metrics_at_threshold(all_logits, all_labels, extras or {}, thr)
                sweep_log[f"{thr:.2f}"] = {
                    key: round(float(m.get(key, 0.0) or 0.0), 6)
                    for key in tracked_metrics
                }
                for key in tracked_metrics:
                    value = float(m.get(key, 0.0) or 0.0)
                    if value > best_by[key][0]:
                        best_by[key] = (value, thr)
            best_thr = best_by[primary_key][1] if best_by[primary_key][1] is not None else self.threshold
            best_score = best_by[primary_key][0]
        else:
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
            if use_second_metrics:
                thr_data = {
                    "select_metric": self.threshold_select_metric,
                    "best_thresholds": {
                        key: {"threshold": best_by[key][1], "value": round(best_by[key][0], 6)}
                        for key in tracked_metrics
                    },
                    "sweep": sweep_log,
                }
                (self.save_dir / "best_thresholds_SECOND.json").write_text(
                    json.dumps(thr_data, indent=2)
                )
            else:
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

        if result.get("metric_family") == "second":
            self._save_second_outputs(result, sweep_log, dataset_name)
            return

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

    def _save_second_outputs(self, result: dict, sweep_log, dataset_name: str) -> None:
        if not self.do_sweep:
            thr_data = {
                "select_metric": self.threshold_select_metric,
                "best_thresholds": {
                    "Fscd": {
                        "threshold": result["best_threshold"],
                        "value": round(float(result.get("Fscd", 0.0)), 6),
                    },
                },
            }
            (self.save_dir / "best_thresholds_SECOND.json").write_text(json.dumps(thr_data, indent=2))

        second_payload = {
            "dataset": dataset_name,
            "evaluation_mode": result.get("second_eval_level", "semantic_change"),
            "output_mode": self.output_mode,
            "metrics": {
                "OA": result.get("OA"),
                "mIoU": result.get("mIoU"),
                "SeK": result.get("SeK"),
                "Fscd": result.get("Fscd"),
            },
        }
        if sweep_log:
            second_payload["sweep"] = sweep_log
        (self.save_dir / "second_metrics.json").write_text(json.dumps(second_payload, indent=2))
        if result.get("second_eval_level") == "semantic_change":
            (self.save_dir / "second_semantic_metrics.json").write_text(json.dumps(second_payload, indent=2))

        csv_path = self.save_dir / "second_metrics.csv"
        headers = ["dataset", "OA", "mIoU", "SeK", "Fscd"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow([
                dataset_name,
                result.get("OA", ""),
                result.get("mIoU", ""),
                result.get("SeK", ""),
                result.get("Fscd", ""),
            ])

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------
    def print_table(self, results: dict, title: str = "EVALUATION RESULTS") -> None:
        """Print a formatted validation results block."""
        sep = "=" * 40
        self.logger.info(sep)
        self.logger.info(title)
        self.logger.info(sep)

        if results.get("metric_family") == "second":
            rows = [
                ("OA", results.get("OA")),
                ("mIoU", results.get("mIoU")),
                ("SeK", results.get("SeK")),
                ("Fscd", results.get("Fscd")),
            ]
            for label, val in rows:
                if isinstance(val, (int, float)):
                    self.logger.info(f"{label:<22}: {val:.4f}")
                elif val is None and label in {"SeK"}:
                    self.logger.info(f"{label:<22}: N/A")
            self.logger.info(sep)
            return

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
