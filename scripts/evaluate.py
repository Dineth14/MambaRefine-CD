"""Evaluation script for MambaRefineCD.

Loads a checkpoint and evaluates on val or test split.
Saves metrics.json and metrics.csv.

Metric restriction:
  - LEVIR-CD / WHU-CD / DSIFN-CD: Pre, Rec, F1, IoU, OA
  - SECOND:                        OA, mIoU, SeK, Fscd

Usage:
    python scripts/evaluate.py --config configs/ablations/levir/a4_full.yaml \\
                                --ckpt outputs/levir/a4_full/checkpoints/best.pth
    python scripts/evaluate.py --config configs/ablations/second/a4_full.yaml \\
                                --ckpt outputs/second/a4_full/checkpoints/best.pth \\
                                --split val
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


def _save_results(metrics: dict, save_dir: Path, split: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / f"metrics_{split}.json"
    csv_path  = save_dir / f"metrics_{split}.csv"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    if split in {"val", "test"}:
        canonical_json = save_dir / "metrics.json"
        canonical_csv = save_dir / "metrics.csv"
        with open(canonical_json, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(canonical_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in metrics.items():
                w.writerow([k, v])
    logger.info(f"Saved metrics to {json_path}")


def evaluate_binary(model, loader, device, threshold: float = 0.5) -> dict:
    from metrics.binary_cd_metrics import BinaryMetrics
    from training.model_outputs import normalize_model_output

    m = BinaryMetrics(threshold=threshold)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img_a = batch["image_a"].to(device, non_blocking=True)
            img_b = batch["image_b"].to(device, non_blocking=True)
            mask  = batch.get("mask", batch.get("label")).to(device)
            out   = model(img_a, img_b)
            logits = normalize_model_output(out)["change_logits"]
            m.update(logits, mask)
    return m.compute()


def evaluate_second(model, loader, device, num_classes: int = 7,
                    ignore_index: int = 255, threshold: float = 0.5,
                    output_root: Path | None = None,
                    save_predictions: bool = False,
                    save_visualizations: bool = True,
                    save_binary_head_change: bool = False) -> dict:
    from metrics.second_scd_metrics import SECONDSCDMetrics
    from training.model_outputs import normalize_model_output
    from utils.second_outputs import assert_second_prediction_dirs, save_second_prediction_batch, second_semantic_predictions

    m = SECONDSCDMetrics(num_classes=num_classes, ignore_index=ignore_index, threshold=threshold)
    model.eval()
    sanity_logged = False
    with torch.no_grad():
        for batch in loader:
            img_a = batch["image_a"].to(device, non_blocking=True)
            img_b = batch["image_b"].to(device, non_blocking=True)
            gt_s1 = batch.get("label_t1", batch.get("sem_label_t1", batch.get("label_a"))).to(device)
            gt_s2 = batch.get("label_t2", batch.get("sem_label_t2", batch.get("label_b"))).to(device)
            out = model(img_a, img_b)
            if isinstance(out, dict):
                outputs = normalize_model_output(out)
                sem1, sem2, ch = second_semantic_predictions(outputs)
                binary_head = outputs.get("change_logits")
            else:
                raise RuntimeError("SECOND evaluation requires dict outputs with sem_logits_t1 and sem_logits_t2.")
            m.update(sem1, sem2, gt_s1, gt_s2)
            if not sanity_logged:
                valid = (gt_s1 != ignore_index) & (gt_s2 != ignore_index)
                logger.info(
                    "SECOND sanity | gt_t1=%s gt_t2=%s pred_t1=%s pred_t2=%s ignore=%d valid=%d change_ratio=%.6f",
                    sorted(torch.unique(gt_s1.detach().cpu()).tolist()),
                    sorted(torch.unique(gt_s2.detach().cpu()).tolist()),
                    sorted(torch.unique(sem1.detach().cpu()).tolist()),
                    sorted(torch.unique(sem2.detach().cpu()).tolist()),
                    int((~valid).sum().item()),
                    int(valid.sum().item()),
                    float((((gt_s1 != gt_s2) & valid).sum().float() / valid.sum().clamp_min(1)).item()),
                )
                sanity_logged = True
            if save_predictions and output_root is not None:
                save_second_prediction_batch(
                    pred_t1=sem1,
                    pred_t2=sem2,
                    pred_change=ch,
                    sample_ids=batch.get("name", batch.get("id", [f"sample_{i}" for i in range(sem1.shape[0])])),
                    output_root=output_root,
                    binary_head_logits=binary_head,
                    save_visualizations=save_visualizations,
                    save_binary_head_change=save_binary_head_change,
                    threshold=threshold,
                )
    if save_predictions and output_root is not None:
        assert_second_prediction_dirs(output_root)
    return m.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MambaRefineCD checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--split",  default="val", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use_ema", dest="use_ema", action="store_true", default=None)
    ema_group.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--non_strict", dest="strict", action="store_false")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    cfg   = load_config(args.config)
    if args.num_workers is not None:
        cfg.setdefault("dataset", {})["num_workers"] = int(args.num_workers)
    allowed = _get_allowed_metrics(cfg)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_second = "SECOND" in str(cfg.get("dataset", {}).get("name", "")).upper()

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Allowed metrics: {allowed}")

    from models.mambarefinecd import build_model
    from data.dataset_builder import build_test_loader
    from training.checkpoint import load_for_eval, peek as peek_ckpt
    from training.evaluator import Evaluator

    ckpt_meta = peek_ckpt(args.ckpt, map_location=device)
    threshold, threshold_source = _resolve_threshold(args, cfg, ckpt_meta)
    use_ema_requested = _resolve_use_ema(args, cfg)

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["split"] = args.split
    cfg["evaluation"]["threshold"] = threshold
    cfg["evaluation"]["save_predictions"] = bool(args.save_predictions)
    cfg.setdefault("eval", {})
    cfg["eval"]["split"] = args.split
    cfg["eval"]["threshold"] = threshold
    cfg["eval"]["save_predictions"] = bool(args.save_predictions)
    if args.save_debug:
        cfg["evaluation"]["save_debug_outputs"] = True
        cfg["evaluation"]["debug_output_root"] = "outputs/debug_levir_eval"
        cfg["evaluation"]["debug_max_samples"] = 20
        cfg["eval"]["save_debug_outputs"] = True
        cfg["eval"]["debug_output_root"] = "outputs/debug_levir_eval"
        cfg["eval"]["debug_max_samples"] = 20
    if args.split == "test" or args.threshold is not None or threshold_source == "checkpoint":
        cfg["evaluation"]["threshold_sweep"] = False
        cfg["eval"]["threshold_sweep"] = False

    loader = build_test_loader(cfg)

    model = build_model(cfg).to(device)
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
    logger.info(f"Missing keys: {load_info['missing_keys']}")
    logger.info(f"Unexpected keys: {load_info['unexpected_keys']}")

    out_dir = Path(cfg.get("experiment", {}).get("output_root", "outputs/eval"))
    evaluator = Evaluator(cfg, device, logger=logger, save_dir=out_dir)
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
    logger.info("=" * 50)
    logger.info(f"Results ({args.split}):")
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

    _save_results(results, out_dir, args.split)


if __name__ == "__main__":
    main()
