"""Test script — evaluates on the held-out test split and saves predictions.

Usage:
    python scripts/test.py --config configs/ablations/levir/a4_full.yaml \\
                           --ckpt outputs/levir/a4_full/checkpoints/best.pth
    python scripts/test.py --config configs/ablations/second/a4_full.yaml \\
                           --ckpt outputs/second/a4_full/checkpoints/best.pth \\
                           --save_predictions
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

import torch

from utils.config import load_config
from utils.ablation import compare_checkpoint_config, config_fingerprint, log_parameter_breakdown, log_startup_config, module_flags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_BINARY_ALLOWED = {"Pre", "Rec", "F1", "IoU", "OA"}
_SECOND_ALLOWED = {"OA", "mIoU", "SeK", "Fscd"}


def _get_allowed_metrics(cfg: dict) -> list[str]:
    explicit = cfg.get("metrics", {}).get("allowed", None)
    if explicit is not None:
        return list(explicit)
    task = str(cfg.get("task", cfg.get("dataset", {}).get("task_type", "binary_cd"))).lower()
    if "SECOND" in str(cfg.get("dataset", {}).get("name", "")).upper() or task == "semantic_cd":
        return ["OA", "mIoU", "SeK", "Fscd"]
    return ["Pre", "Rec", "F1", "IoU", "OA"]


def _filter_metrics(raw: dict, allowed: list[str]) -> dict:
    return {k: v for k, v in raw.items() if k in allowed}


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


def _resolve_threshold(args, cfg: dict, ckpt: dict) -> tuple[float, str]:
    ec = _merged_eval_cfg(cfg)
    if args.threshold is not None:
        return float(args.threshold), "command-line"
    if ckpt.get("best_threshold") is not None:
        return float(ckpt["best_threshold"]), "checkpoint"
    return float(ec.get("threshold", 0.5)), "config"


def _resolve_use_ema(args, cfg: dict) -> bool:
    if args.use_ema is not None:
        return bool(args.use_ema)
    ec = _merged_eval_cfg(cfg)
    if "use_ema" in ec:
        return bool(ec.get("use_ema"))
    return bool(cfg.get("training", {}).get("use_ema", cfg.get("ema", {}).get("enabled", False)))


def _save_results(metrics: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(save_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    logger.info(f"Saved results to {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MambaRefineCD on held-out test split.")
    parser.add_argument("--config",          required=True)
    parser.add_argument("--ckpt",            required=True)
    parser.add_argument("--split",           default="test", choices=["test"])
    parser.add_argument("--threshold",       type=float, default=None)
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use_ema", dest="use_ema", action="store_true", default=None)
    ema_group.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--non_strict", dest="strict", action="store_false")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    cfg     = load_config(args.config)
    if args.num_workers is not None:
        cfg.setdefault("dataset", {})["num_workers"] = int(args.num_workers)
    allowed = _get_allowed_metrics(cfg)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_second = "SECOND" in str(cfg.get("dataset", {}).get("name", "")).upper()
    save_predictions = bool(args.save_predictions) if args.save_predictions is not None else bool(is_second)
    save_visualizations = bool(args.save_visualizations) if args.save_visualizations is not None else bool(is_second)

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Allowed metrics: {allowed}")
    log_startup_config(logger, cfg, args.config)

    from models.mambarefinecd import build_model
    from data.dataset_builder import build_test_loader
    from training.checkpoint import load_for_eval, peek as peek_ckpt
    from training.evaluator import Evaluator

    ckpt_meta = peek_ckpt(args.ckpt, map_location=device)
    logger.info("Config ablation name: %s", cfg.get("experiment", {}).get("name", "unknown"))
    compare_checkpoint_config(logger, cfg, ckpt_meta, strict=bool(args.strict))
    threshold, threshold_source = _resolve_threshold(args, cfg, ckpt_meta)
    use_ema_requested = _resolve_use_ema(args, cfg)

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["split"] = args.split
    cfg["evaluation"]["threshold"] = threshold
    cfg["evaluation"]["save_predictions"] = save_predictions
    cfg["evaluation"]["save_visualizations"] = save_visualizations
    cfg.setdefault("eval", {})
    cfg["eval"]["split"] = args.split
    cfg["eval"]["threshold"] = threshold
    cfg["eval"]["save_predictions"] = save_predictions
    cfg["eval"]["save_visualizations"] = save_visualizations
    cfg.setdefault("output", {})
    cfg["output"]["save_predictions"] = save_predictions
    cfg["output"]["save_visualizations"] = save_visualizations
    if args.save_debug:
        dataset_slug = str(cfg.get("dataset", {}).get("name", "dataset")).lower().replace("-cd", "").replace("-", "_")
        cfg["evaluation"]["save_debug_outputs"] = True
        cfg["evaluation"]["debug_output_root"] = f"debug/{dataset_slug}"
        cfg["evaluation"]["debug_max_samples"] = 50
        cfg["eval"]["save_debug_outputs"] = True
        cfg["eval"]["debug_output_root"] = f"debug/{dataset_slug}"
        cfg["eval"]["debug_max_samples"] = 50
    if args.split == "test" or args.threshold is not None or threshold_source == "checkpoint":
        cfg["evaluation"]["threshold_sweep"] = False
        cfg["eval"]["threshold_sweep"] = False

    loader = build_test_loader(cfg)

    model = build_model(cfg).to(device)
    log_parameter_breakdown(logger, model)
    load_info = load_for_eval(
        args.ckpt,
        model,
        map_location=device,
        strict=bool(args.strict),
        use_ema=use_ema_requested,
    )
    logger.info(f"Loaded checkpoint from {args.ckpt}")
    logger.info(f"Checkpoint iteration: {load_info['iteration']} | best_metric: {load_info['best_metric']}")
    logger.info(f"Checkpoint threshold stored: {load_info['best_threshold']}")
    logger.info(f"Using threshold: {threshold:.4f}")
    logger.info(f"Threshold source: {threshold_source}")
    logger.info(f"Using EMA: {str(load_info['ema_used']).lower()}")
    logger.info(f"EMA weights found: {str(load_info['ema_found']).lower()}")
    if use_ema_requested and not load_info["ema_found"]:
        logger.warning("eval.use_ema/--use_ema requested but checkpoint has no EMA weights; using raw model weights.")
    logger.info(f"Missing keys: {load_info['missing_keys']}")
    logger.info(f"Unexpected keys: {load_info['unexpected_keys']}")

    out_root = Path(cfg.get("experiment", {}).get("output_root", "outputs/test"))
    evaluator = Evaluator(cfg, device, logger=logger, save_dir=out_root)
    raw = evaluator.evaluate(
        model,
        loader,
        dataset_name=cfg.get("dataset", {}).get("name", "unknown"),
        amp=bool(cfg.get("hardware", {}).get("mixed_precision", True)),
    )

    effective_threshold = float(raw.get("best_threshold", threshold))
    effective_source = "validation-sweep" if effective_threshold != threshold and args.split == "val" else threshold_source
    results = _filter_metrics(raw, allowed) if is_second else _paper_binary_metrics(raw)
    results["threshold"] = effective_threshold
    results["threshold_source"] = effective_source
    results["ema_used"] = bool(load_info["ema_used"])
    results["ema_found"] = bool(load_info["ema_found"])
    results["config_fingerprint"] = cfg.get("_meta", {}).get("config_fingerprint", config_fingerprint(cfg))
    results["module_flags"] = module_flags(cfg)

    logger.info("=" * 50)
    logger.info("Test Results:")
    for k in allowed:
        v = results.get(k)
        if v is None:
            continue
        logger.info(f"  {k:16s}: {float(v):.4f}")
    logger.info("-" * 50)
    logger.info("Evaluation metadata:")
    for k in ("threshold", "threshold_source", "ema_used", "ema_found"):
        v = results.get(k)
        if v is None:
            continue
        if isinstance(v, bool):
            logger.info(f"  {k:16s}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            logger.info(f"  {k:16s}: {v:.4f}")
        else:
            logger.info(f"  {k:16s}: {v}")
    logger.info("=" * 50)

    _save_results(results, out_root)


if __name__ == "__main__":
    main()
