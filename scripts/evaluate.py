"""Standalone model evaluation script.

Uses only configs/global_config.yaml.
No CLI arguments needed.

Outputs:
    outputs/eval_runs/<name>/eval_metrics.json
    outputs/eval_runs/<name>/eval_metrics.csv
    (+ predictions if evaluation.save_predictions: true)

Run:
    conda activate mamba_new
    cd MambaRefine-CD
    python scripts/evaluate.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from utils.config             import GLOBAL_CONFIG_PATH, load_config
from utils.seed               import set_seed
from data.dataset_builder     import build_test_loader
from models.cd_model          import build_model
from training.evaluator       import Evaluator
from training.checkpoint      import peek as peek_ckpt, load as load_ckpt
from training.logger          import get_logger


def main() -> None:
    cfg  = load_config()
    exp  = cfg.experiment
    hw   = cfg.hardware

    set_seed(int(exp.seed))

    # ── Output directory ──────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / exp.output_root / f"eval_{ts}_{exp.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    logger = get_logger(exp.name, out_dir / "logs")
    logger.info(f"Evaluation  : {GLOBAL_CONFIG_PATH.relative_to(ROOT)}")
    logger.info(f"Output dir  : {out_dir}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(hw.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    logger.info(f"Device      : {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = cfg.dataset
    split  = cfg.evaluation.split
    logger.info(f"Dataset     : {ds_cfg.name}  split={split}")

    loader = build_test_loader(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    ckpt_path = cfg.checkpoint.path
    if not ckpt_path:
        raise ValueError(
            "checkpoint.path is null in eval config. "
            "Set it to a trained checkpoint, e.g.:\n"
            "  checkpoint:\n"
            "    path: outputs/benchmark_runs/run_XXX/checkpoints/best.pth"
        )
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_info = peek_ckpt(ckpt_path)
    logger.info(f"Checkpoint  : {ckpt_path}")
    logger.info(f"  Iteration : {ckpt_info.get('iteration', '?')}")
    logger.info(f"  Best F1   : {ckpt_info.get('best_metric', '?')}")

    model.load_state_dict(ckpt_info["model"], strict=True)
    model.eval()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator = Evaluator(cfg, device, logger=logger)
    amp       = bool(hw.mixed_precision)
    dataset_name = ds_cfg.name

    logger.info(f"\nRunning evaluation on {dataset_name} [{split}] ...")
    results = evaluator.evaluate(model, loader, dataset_name=dataset_name, amp=amp)

    # ── Print table ───────────────────────────────────────────────────────────
    evaluator.print_table(results, title="EVALUATION RESULTS")

    # ── Build flat metrics dict with canonical key names ─────────────────────
    threshold   = results.get("best_threshold", float(cfg.evaluation.threshold))
    tta_enabled = bool(cfg.evaluation.use_tta)
    temperature = float(getattr(getattr(cfg, "model", None) or type("_", (), {"temperature": 1.0})(), "temperature", 1.0))

    # Required CSV/JSON columns in order
    _CSV_KEYS = [
        "mF1", "F1_0", "F1_1", "mIoU", "IoU_0", "IoU_1",
        "precision_1", "recall_1", "OA",
        "boundary_f1", "edge_iou",
        "pred_positive_ratio", "gt_positive_ratio",
    ]
    _KEY_MAP = {
        "mF1":  ("mf1",),
        "F1_0": ("f1_0",),
        "F1_1": ("f1_1", "f1"),
        "mIoU": ("miou",),
        "IoU_0": ("iou_0",),
        "IoU_1": ("iou_1", "iou"),
        "precision_1": ("precision_1", "precision"),
        "recall_1":    ("recall_1",    "recall"),
        "OA":          ("oa",),
        "boundary_f1":         ("boundary_f1",),
        "edge_iou":            ("edge_iou",),
        "pred_positive_ratio": ("pred_positive_ratio",),
        "gt_positive_ratio":   ("gt_positive_ratio",),
    }

    def _get(col: str) -> float | str:
        for k in _KEY_MAP.get(col, (col,)):
            if k in results and isinstance(results[k], (int, float)):
                return results[k]
        return ""

    flat_metrics = {col: _get(col) for col in _CSV_KEYS}

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results_meta = {
        "config":      str(GLOBAL_CONFIG_PATH.relative_to(ROOT)),
        "checkpoint":  str(ckpt_path),
        "dataset":     dataset_name,
        "split":       split,
        "timestamp":   ts,
        "threshold":   threshold,
        "tta_enabled": tta_enabled,
        "temperature": temperature,
        "metrics":     flat_metrics,
    }
    json_path = out_dir / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_meta, f, indent=2)
    logger.info(f"\nSaved JSON  → {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "eval_metrics.csv"
    csv_header = (
        ["dataset"] + _CSV_KEYS +
        ["threshold", "tta_enabled", "temperature"]
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)
        w.writerow(
            [dataset_name]
            + [flat_metrics[col] for col in _CSV_KEYS]
            + [threshold, tta_enabled, temperature]
        )
    logger.info(f"Saved CSV   → {csv_path}")


if __name__ == "__main__":
    main()
