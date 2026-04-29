"""Dataset validation tool.

Checks each configured dataset returns correctly shaped tensors and expected keys.

Usage:
    python tools/validate_dataset.py --config configs/datasets/dsifn.yaml
    python tools/validate_dataset.py --all   # validate all known configs
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import torch
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_DATASET_CONFIGS = [
    "configs/datasets/whu.yaml",
    "configs/datasets/dsifn.yaml",
]

_BINARY_KEYS  = {"image_a", "image_b", "mask"}


def _validate_config(config_path: str) -> bool:
    from datasets import build_dataset

    try:
        cfg  = load_config(config_path)
        ds   = build_dataset(cfg, split="val")
    except Exception as e:
        logger.error(f"[FAIL] Could not build dataset from {config_path}: {e}")
        return False

    required  = _BINARY_KEYS

    try:
        sample = ds[0]
    except Exception as e:
        logger.error(f"[FAIL] dataset[0] raised: {e}")
        return False

    # Check keys
    missing = required - set(sample.keys())
    if missing:
        logger.error(f"[FAIL] {config_path}: missing keys {missing}. Got: {set(sample.keys())}")
        return False

    # Check image tensors
    img_a = sample["image_a"]
    img_b = sample["image_b"]
    if not (isinstance(img_a, torch.Tensor) and img_a.ndim == 3 and img_a.shape[0] == 3):
        logger.error(f"[FAIL] image_a has unexpected shape: {img_a.shape if isinstance(img_a, torch.Tensor) else type(img_a)}")
        return False
    if img_a.shape != img_b.shape:
        logger.error(f"[FAIL] image_a.shape {img_a.shape} != image_b.shape {img_b.shape}")
        return False

    # Check label(s)
    mask = sample["mask"]
    if not isinstance(mask, torch.Tensor):
        logger.error(f"[FAIL] mask should be Tensor, got {type(mask)}")
        return False

    img_size = int(cfg.get("dataset", {}).get("image_size", img_a.shape[-1]))
    if img_a.shape[-1] != img_size or img_a.shape[-2] != img_size:
        logger.warning(f"[WARN] Expected image size {img_size}, got {img_a.shape[-2:]}")

    n = len(ds)
    logger.info(f"[OK]   {config_path} | split=val | {n} samples | image_shape={list(img_a.shape)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset configurations.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        configs = [str(_REPO / c) for c in _DATASET_CONFIGS]
    elif args.config:
        configs = [args.config]
    else:
        parser.error("Provide --config or --all")
        return

    results = {c: _validate_config(c) for c in configs}

    n_pass = sum(results.values())
    n_fail = len(results) - n_pass
    logger.info("=" * 50)
    logger.info(f"Passed: {n_pass} / {len(results)}  |  Failed: {n_fail}")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
