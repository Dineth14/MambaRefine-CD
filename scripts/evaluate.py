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
from torch.utils.data import DataLoader

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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    allowed = _get_allowed_metrics(cfg)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_second = "SECOND" in str(cfg.get("dataset", {}).get("name", "")).upper()

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Allowed metrics: {allowed}")

    from models.mambarefinecd import build_model
    from datasets import build_dataset
    from training.checkpoint import load as load_ckpt

    cfg["dataset"]["split"] = args.split
    ds = build_dataset(cfg, split=args.split)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("validation", {}).get("batch_size", 4)),
        num_workers=int(cfg.get("dataset", {}).get("num_workers", 4)),
        shuffle=False,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    load_ckpt(args.ckpt, model, map_location=device)
    logger.info(f"Loaded checkpoint from {args.ckpt}")

    out_dir = Path(cfg.get("experiment", {}).get("output_root", "outputs/eval"))
    output_cfg = cfg.get("output", {})
    eval_cfg = cfg.get("evaluation", {})
    save_predictions = bool(args.save_predictions or output_cfg.get("save_predictions", eval_cfg.get("save_predictions", False)))
    save_visualizations = bool(output_cfg.get("save_visualizations", True))
    save_binary_head_change = bool(output_cfg.get("save_binary_head_change", False))

    if is_second:
        raw = evaluate_second(
            model, loader, device,
            num_classes=int(cfg.get("dataset", {}).get("num_classes", 7)),
            ignore_index=int(cfg.get("dataset", {}).get("ignore_index", 255)),
            threshold=args.threshold,
            output_root=out_dir,
            save_predictions=save_predictions,
            save_visualizations=save_visualizations,
            save_binary_head_change=save_binary_head_change,
        )
    else:
        raw = evaluate_binary(model, loader, device, threshold=args.threshold)

    results = _filter_metrics(raw, allowed)
    logger.info("=" * 50)
    logger.info(f"Results ({args.split}):")
    for k, v in results.items():
        logger.info(f"  {k:8s}: {v:.4f}")
    logger.info("=" * 50)

    _save_results(results, out_dir, args.split)


if __name__ == "__main__":
    main()
