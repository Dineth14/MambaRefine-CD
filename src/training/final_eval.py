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
from training.checkpoint   import load_for_eval
from training.evaluator    import Evaluator
from training.ema          import EMA


_SEP  = "=" * 40
_DASH = "-" * 40


def _merged_eval_cfg(cfg: dict) -> dict:
    merged = dict(cfg.get("evaluation", {}) or {})
    merged.update(dict(cfg.get("eval", {}) or {}))
    return merged


def _paper_binary_metrics(raw: dict) -> dict:
    return {
        "Pre": round(float(raw.get("precision", raw.get("precision_1", 0.0))) * 100.0, 4),
        "Rec": round(float(raw.get("recall", raw.get("recall_1", 0.0))) * 100.0, 4),
        "F1": round(float(raw.get("f1", raw.get("f1_1", 0.0))) * 100.0, 4),
        "IoU": round(float(raw.get("iou", raw.get("iou_1", 0.0))) * 100.0, 4),
        "OA": round(float(raw.get("oa", 0.0)) * 100.0, 4),
    }


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
    eval_cfg = cfg.get("eval", cfg.get("evaluation", {}))
    use_ema_cfg = bool(eval_cfg.get("use_ema", cfg.get("training", {}).get("use_ema", ema is not None)))
    load_info = load_for_eval(
        ckpt_path,
        model,
        map_location=device,
        strict=bool(cfg.get("resume", {}).get("strict", True)),
        use_ema=use_ema_cfg,
    )
    model.to(device)
    model.eval()
    logger.info(
        f"  Checkpoint iteration : {load_info.get('iteration', 'N/A')}"
        f"  |  Best val metric : {load_info.get('best_metric', 'N/A')}"
    )
    logger.info(f"  Using EMA          : {str(load_info['ema_used']).lower()}")
    logger.info(f"  EMA weights found  : {str(load_info['ema_found']).lower()}")
    if use_ema_cfg and not load_info["ema_found"]:
        logger.warning("eval.use_ema requested but checkpoint has no EMA weights; using raw model weights.")
    logger.info(f"  Missing keys       : {load_info['missing_keys']}")
    logger.info(f"  Unexpected keys    : {load_info['unexpected_keys']}")
    ema_applied = bool(load_info["ema_used"])

    eval_cfg = _merged_eval_cfg(cfg)
    if load_info.get("best_threshold") is not None:
        threshold = float(load_info["best_threshold"])
        threshold_source = "checkpoint"
    else:
        threshold = float(eval_cfg.get("threshold", 0.5))
        threshold_source = "config"
    logger.info(f"  Using threshold    : {threshold:.4f}")
    logger.info(f"  Threshold source   : {threshold_source}")

    # ── Build test loader (force split=test) ──────────────────────────────────
    # Temporarily override evaluation settings to guarantee held-out test eval.
    # Test never sweeps thresholds; it uses the saved validation threshold when
    # available, otherwise the configured default.
    ec = cfg.get("evaluation", {})
    alias_ec = cfg.get("eval", {})
    original_split = ec.get("split", "test")
    original_threshold = ec.get("threshold", 0.5)
    original_sweep = ec.get("threshold_sweep", False)
    original_alias_split = alias_ec.get("split", original_split) if isinstance(alias_ec, dict) else original_split
    original_alias_threshold = alias_ec.get("threshold", original_threshold) if isinstance(alias_ec, dict) else original_threshold
    original_alias_sweep = alias_ec.get("threshold_sweep", original_sweep) if isinstance(alias_ec, dict) else original_sweep
    ec["split"] = "test"
    ec["threshold"] = threshold
    ec["threshold_sweep"] = False
    cfg["evaluation"] = ec
    if isinstance(alias_ec, dict):
        alias_ec["split"] = "test"
        alias_ec["threshold"] = threshold
        alias_ec["threshold_sweep"] = False
        cfg["eval"] = alias_ec

    try:
        test_loader = build_test_loader(cfg)
    except Exception as exc:
        logger.warning(f"[final_eval] Could not build test loader: {exc}. Skipping.")
        ec["split"] = original_split
        ec["threshold"] = original_threshold
        ec["threshold_sweep"] = original_sweep
        cfg["evaluation"] = ec
        if isinstance(alias_ec, dict):
            alias_ec["split"] = original_alias_split
            alias_ec["threshold"] = original_alias_threshold
            alias_ec["threshold_sweep"] = original_alias_sweep
            cfg["eval"] = alias_ec
        return None

    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    num_test = len(test_loader.dataset)
    if num_test == 0:
        logger.warning("[final_eval] Test dataset is empty. Skipping final evaluation.")
        ec["split"] = original_split
        ec["threshold"] = original_threshold
        ec["threshold_sweep"] = original_sweep
        cfg["evaluation"] = ec
        if isinstance(alias_ec, dict):
            alias_ec["split"] = original_alias_split
            alias_ec["threshold"] = original_alias_threshold
            alias_ec["threshold_sweep"] = original_alias_sweep
            cfg["eval"] = alias_ec
        return None
    logger.info(f"  Test samples : {num_test}  |  Dataset : {dataset_name}")

    # ── Output directory for test results ────────────────────────────────────
    test_results_dir = Path(output_dir) / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluation ────────────────────────────────────────────────────────
    amp = bool(cfg.get("hardware", {}).get("mixed_precision", True))
    evaluator = Evaluator(cfg, device, logger=logger, save_dir=test_results_dir)

    try:
        raw_results = evaluator.evaluate(model, test_loader, dataset_name=dataset_name, amp=amp)
    finally:
        ec["split"] = original_split
        ec["threshold"] = original_threshold
        ec["threshold_sweep"] = original_sweep
        cfg["evaluation"] = ec
        if isinstance(alias_ec, dict):
            alias_ec["split"] = original_alias_split
            alias_ec["threshold"] = original_alias_threshold
            alias_ec["threshold_sweep"] = original_alias_sweep
            cfg["eval"] = alias_ec

    # ── Augment results with metadata ────────────────────────────────────────
    results = _paper_binary_metrics(raw_results)
    results["threshold"] = threshold
    results["threshold_source"] = threshold_source
    results["checkpoint"] = str(ckpt_path)
    results["ema_used"] = ema_applied
    results["ema_found"] = bool(load_info["ema_found"])

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
    metric_order = ["Pre", "Rec", "F1", "IoU", "OA"]
    keys = [key for key in metric_order if key in results]
    keys.extend([key for key in ("threshold", "threshold_source", "ema_used", "ema_found", "checkpoint") if key in results])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerow([round(results[k], 6) if isinstance(results[k], float) else results[k] for k in keys])


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

    ordered = [("Pre", "Pre"), ("Rec", "Rec"), ("F1", "F1"), ("IoU", "IoU"), ("OA", "OA")]
    for key, label in ordered:
        row = _fmt(key, label)
        if row:
            lines.append(row)
    lines.append(_DASH)
    for key, label in [
        ("threshold", "Threshold"),
        ("threshold_source", "Threshold Src"),
        ("ema_used", "EMA Used"),
        ("ema_found", "EMA Found"),
    ]:
        row = _fmt(key, label)
        if row:
            lines.append(row)

    lines.append(_SEP)
    return "\n".join(lines) + "\n"
