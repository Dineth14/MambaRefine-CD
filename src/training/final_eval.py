"""Post-training final test evaluation.

Called automatically at the end of training when
``post_training.run_test_eval: true`` is set in global_config.yaml.

Usage (internal — invoked by scripts/train.py):
    from training.final_eval import run_final_test_evaluation
    run_final_test_evaluation(cfg, model, output_dir, device, ema=trainer.ema)
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from data.dataset_builder  import build_test_loader
from training.evaluator    import Evaluator
from training.ema          import EMA


_SEP  = "=" * 40
_DASH = "-" * 40


def run_final_test_evaluation(
    cfg: dict,
    model: nn.Module,
    output_dir: Path,
    device: torch.device,
    ema: Optional[EMA] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """Run evaluation on the test split using the best saved checkpoint.

    Behaviour
    ---------
    * Skips silently if ``post_training.run_test_eval`` is False.
    * Warns and returns None if ``output_dir/checkpoints/best.pth`` is absent.
    * Applies EMA shadow weights for evaluation, then restores original weights.
    * Saves results to ``output_dir/test_results/``.
    * Copies best checkpoint to ``output_dir/best_model_final.pth``.

    Args:
        cfg:        Loaded global config dict (dot-accessible).
        model:      The model instance (weights will be temporarily swapped).
        output_dir: Run output directory (Path object).
        device:     Torch device.
        ema:        Optional EMA object from the trainer (may be None).
        logger:     Optional logger; falls back to a simple stdout logger.

    Returns:
        Metrics dict, or None if evaluation was skipped.
    """
    # ── Guard: check config flag ──────────────────────────────────────────────
    pt_cfg = cfg.get("post_training", {})
    if not bool(pt_cfg.get("run_test_eval", True)):
        if logger:
            logger.info("post_training.run_test_eval is False — skipping final test evaluation.")
        return None

    if logger is None:
        logger = _make_logger()

    logger.info(_DASH)
    logger.info("Running FINAL TEST evaluation...")
    logger.info("Using best checkpoint")
    logger.info(_DASH)

    # ── Locate best checkpoint ────────────────────────────────────────────────
    ckpt_path = Path(output_dir) / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        logger.warning(
            f"[final_eval] No best checkpoint found at {ckpt_path}. "
            "Skipping final test evaluation."
        )
        return None

    # ── Load best weights into model ──────────────────────────────────────────
    logger.info(f"Loading best checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    logger.info(
        f"  Checkpoint iteration : {ckpt.get('iteration', 'N/A')}"
        f"  |  Best val metric : {ckpt.get('best_metric', 'N/A')}"
    )

    # ── Apply EMA shadow weights if available ─────────────────────────────────
    ema_applied = False
    if ema is not None:
        ema.apply_shadow(model)
        ema_applied = True
        logger.info("  EMA shadow weights applied for test evaluation.")

    # ── Build test loader (force split=test) ──────────────────────────────────
    # Temporarily override evaluation.split to guarantee "test"
    ec = cfg.get("evaluation", {})
    original_split = ec.get("split", "test")
    ec["split"] = "test"
    cfg["evaluation"] = ec

    try:
        test_loader = build_test_loader(cfg)
    except Exception as exc:
        logger.warning(f"[final_eval] Could not build test loader: {exc}. Skipping.")
        if ema_applied:
            ema.restore(model)
        ec["split"] = original_split
        cfg["evaluation"] = ec
        return None
    finally:
        ec["split"] = original_split
        cfg["evaluation"] = ec

    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    num_test = len(test_loader.dataset)
    if num_test == 0:
        logger.warning("[final_eval] Test dataset is empty. Skipping final evaluation.")
        if ema_applied:
            ema.restore(model)
        return None
    logger.info(f"  Test samples : {num_test}  |  Dataset : {dataset_name}")

    # ── Output directory for test results ────────────────────────────────────
    test_results_dir = Path(output_dir) / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluation ────────────────────────────────────────────────────────
    amp = bool(cfg.get("hardware", {}).get("mixed_precision", True))
    evaluator = Evaluator(cfg, device, logger=logger, save_dir=test_results_dir)

    try:
        results = evaluator.evaluate(model, test_loader, dataset_name=dataset_name, amp=amp)
    finally:
        # Always restore original weights after EMA
        if ema_applied:
            ema.restore(model)

    # ── Augment results with metadata ────────────────────────────────────────
    results["tta"]          = bool(cfg.get("evaluation", {}).get("use_tta", False))
    results["threshold"]    = results.get("best_threshold", 0.5)
    results["checkpoint"]   = str(ckpt_path)
    results["ema_applied"]  = ema_applied

    # ── Save test_metrics.json ────────────────────────────────────────────────
    json_path = test_results_dir / "test_metrics.json"
    _save_json(results, json_path)
    logger.info(f"  Saved JSON  : {json_path}")

    # ── Save test_metrics.csv ─────────────────────────────────────────────────
    csv_path = test_results_dir / "test_metrics.csv"
    _save_csv(results, csv_path)
    logger.info(f"  Saved CSV   : {csv_path}")

    # ── Save test_summary.txt ─────────────────────────────────────────────────
    txt_path = test_results_dir / "test_summary.txt"
    summary_text = _build_summary(results)
    txt_path.write_text(summary_text)
    logger.info(f"  Saved TXT   : {txt_path}")

    # ── Optional: copy best checkpoint as best_model_final.pth ───────────────
    if bool(pt_cfg.get("export_best_model", True)):
        export_path = Path(output_dir) / "best_model_final.pth"
        shutil.copy2(ckpt_path, export_path)
        logger.info(f"  Exported    : {export_path}")

    # ── Print final summary ───────────────────────────────────────────────────
    print(summary_text)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_logger() -> logging.Logger:
    import sys
    log = logging.getLogger("final_eval")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def _save_json(results: dict, path: Path) -> None:
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, float):
            serialisable[k] = round(v, 6)
        elif isinstance(v, (int, bool, str)):
            serialisable[k] = v
        elif isinstance(v, torch.Tensor) and v.numel() == 1:
            serialisable[k] = round(v.item(), 6)
    path.write_text(json.dumps(serialisable, indent=2))


def _save_csv(results: dict, path: Path) -> None:
    numeric_keys = [
        k for k, v in results.items()
        if isinstance(v, (int, float)) and k not in ("num_samples",)
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(numeric_keys)
        w.writerow([round(results[k], 6) if isinstance(results[k], float) else results[k]
                    for k in numeric_keys])


def _build_summary(results: dict) -> str:
    lines = [
        _SEP,
        "FINAL TEST RESULTS",
        _SEP,
    ]

    def _fmt(key: str, label: str) -> Optional[str]:
        v = results.get(key)
        if v is None:
            return None
        if isinstance(v, bool):
            return f"{label:<16}: {v}"
        if isinstance(v, float):
            return f"{label:<16}: {v:.4f}"
        return f"{label:<16}: {v}"

    ordered = [
        ("f1",             "F1"),
        ("iou",            "IoU"),
        ("miou",           "mIoU"),
        ("precision",      "Precision"),
        ("recall",         "Recall"),
        ("oa",             "OA"),
        ("boundary_f1",    "Boundary F1"),
        ("edge_iou",       "Edge IoU"),
        ("pred_positive_ratio", "Pred Pos Ratio"),
        ("gt_positive_ratio",   "GT Pos Ratio"),
        ("threshold",      "Best Thresh"),
        ("tta",            "TTA Enabled"),
        ("ema_applied",    "EMA Applied"),
    ]
    for key, label in ordered:
        row = _fmt(key, label)
        if row:
            lines.append(row)

    lines.append(_SEP)
    return "\n".join(lines) + "\n"
