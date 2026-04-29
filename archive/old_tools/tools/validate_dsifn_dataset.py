#!/usr/bin/env python3
"""Validate DSIFN-CD pairing, masks, and split statistics."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data.dataset_builder import build_dataset
from utils.config import load_config


def _stem(path: str) -> str:
    return Path(path).stem


def _raw_unique(path: str) -> list:
    raw = np.array(Image.open(path))
    values = np.unique(raw)
    return values[:32].tolist()


def _mask_tensor(item: dict) -> torch.Tensor:
    mask = item.get("mask", item.get("label"))
    if mask is None:
        raise KeyError("Dataset item has no mask/label")
    return mask.detach().cpu()


def _print_first_samples(ds, split: str, n: int) -> list[str]:
    errors: list[str] = []
    print(f"\n[{split}] first {min(n, len(ds))} samples")
    for idx in range(min(n, len(ds))):
        info = ds.sample_info(idx) if hasattr(ds, "sample_info") else {}
        if not info:
            item = ds[idx]
            info = {
                "sample_id": item.get("name", item.get("id", idx)),
                "image_t1_path": item.get("image_t1_path", "unavailable"),
                "image_t2_path": item.get("image_t2_path", "unavailable"),
                "mask_path": item.get("mask_path", "unavailable"),
            }
        t1 = info["image_t1_path"]
        t2 = info["image_t2_path"]
        mask = info["mask_path"]
        sample_id = info["sample_id"]
        print(f"{idx:04d} id={sample_id} t1={t1} t2={t2} mask={mask}")
        if t1 != "unavailable" and not (_stem(t1) == _stem(t2) == _stem(mask)):
            errors.append(f"{split}[{idx}] stem mismatch: {t1}, {t2}, {mask}")
    return errors


def _print_mask_debug(ds, split: str, n: int) -> list[str]:
    errors: list[str] = []
    print(f"\n[{split}] first {min(n, len(ds))} mask conversions")
    for idx in range(min(n, len(ds))):
        if hasattr(ds, "raw_mask_stats"):
            raw = ds.raw_mask_stats(idx)
            raw_unique = raw["raw_unique"]
            raw_dtype = raw["raw_dtype"]
            raw_shape = raw["raw_shape"]
        else:
            item = ds[idx]
            raw_unique = "unavailable"
            raw_dtype = "unavailable"
            raw_shape = "unavailable"
        item = ds[idx]
        mask = _mask_tensor(item)
        unique = sorted(float(v) for v in torch.unique(mask).tolist())
        ratio = float((mask > 0.5).float().mean().item())
        print(
            f"{idx:04d} raw_dtype={raw_dtype} raw_shape={raw_shape} raw_unique={raw_unique} "
            f"converted_dtype={mask.dtype} converted_unique={unique} positive_ratio={ratio:.6f}"
        )
        if set(unique) - {0.0, 1.0}:
            errors.append(f"{split}[{idx}] converted mask is not binary: {unique}")
    return errors


def _split_stats(ds, split: str, max_samples: int | None = None) -> dict:
    changed = 0
    total = 0
    all_zero = 0
    all_one = 0
    ratios = []
    n_items = len(ds) if max_samples is None or max_samples <= 0 else min(len(ds), max_samples)
    for idx in range(n_items):
        item = ds[idx]
        mask = (_mask_tensor(item) > 0.5)
        pos = int(mask.sum().item())
        count = int(mask.numel())
        changed += pos
        total += count
        ratio = pos / max(count, 1)
        ratios.append(ratio)
        all_zero += int(pos == 0)
        all_one += int(pos == count)
    ratio = changed / max(total, 1)
    stats = {
        "split": split,
        "samples": len(ds),
        "counted_samples": n_items,
        "changed_pixels": changed,
        "unchanged_pixels": total - changed,
        "gt_positive_ratio": ratio,
        "all_zero_masks": all_zero,
        "all_one_masks": all_one,
        "mean_positive_ratio": float(statistics.fmean(ratios)) if ratios else 0.0,
        "median_positive_ratio": float(statistics.median(ratios)) if ratios else 0.0,
    }
    return stats


def _print_stats(stats: dict) -> list[str]:
    print(f"\n[{stats['split']}] split statistics")
    for key in (
        "samples",
        "counted_samples",
        "changed_pixels",
        "unchanged_pixels",
        "gt_positive_ratio",
        "all_zero_masks",
        "all_one_masks",
        "mean_positive_ratio",
        "median_positive_ratio",
    ):
        value = stats[key]
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    errors = []
    if stats["gt_positive_ratio"] < 0.01 or stats["gt_positive_ratio"] > 0.90:
        errors.append(f"{stats['split']} GT positive ratio suspicious: {stats['gt_positive_ratio']:.6f}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DSIFN-CD dataset pairing and masks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--first_n", type=int, default=20)
    parser.add_argument("--mask_n", type=int, default=50)
    parser.add_argument("--max_stat_samples", type=int, default=0, help="Limit stats scan; 0 means full split.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"config_path: {args.config}")
    print(f"dataset_root: {cfg['dataset']['root']}")
    print(f"dataset_name: {cfg['dataset']['name']}")
    errors: list[str] = []

    for split in ("train", "val", "test"):
        ds = build_dataset(
            cfg["dataset"],
            split=split,
            augment=False,
            seed=int(cfg.get("experiment", {}).get("seed", 42)),
        )
        errors.extend(_print_first_samples(ds, split, args.first_n))
        errors.extend(_print_mask_debug(ds, split, args.mask_n))
        errors.extend(_print_stats(_split_stats(ds, split, args.max_stat_samples)))

    if errors:
        print("\nERRORS")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nDSIFN dataset validation passed.")


if __name__ == "__main__":
    main()
