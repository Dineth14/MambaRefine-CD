"""Multi-dataset builder for change detection benchmarks.

Usage:
    from data.dataset_builder import build_dataset, build_dataloaders

``build_dataset(dataset_cfg, split)`` dispatches on ``dataset_cfg["name"]``.
``build_dataloaders(cfg)`` reads the active ``cfg["dataset"]`` block from the
single global configuration.

Tile-based training (LEVIR-CD):
  Set ``dataset.train_mode: "tile"`` in global_config.yaml to use
  ``LEVIRCDTileDataset`` instead of the image-level ``LEVIRCDDataset``.
  val/test always use existing tiles (``val_mode/test_mode: "existing"``).

Supported dataset names (case-insensitive, flexible aliases):
  LEVIR-CD / levir / levircd
  WHU-CD   / whu   / whucd
  SYSU-CD  / sysu  / sysucd
  DSIFN-CD / dsifn / dsifncd
    SECOND   / second
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from torch.utils.data import DataLoader, Dataset, RandomSampler

from data.levircd  import LEVIRCDDataset, LEVIRCDTileDataset
from data.whucd    import WHUCDDataset
from data.sysucd   import SYSUCDDataset
from data.dsifncd  import DSIFNCDDataset
from data.second   import SECONDDataset

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY: dict = {
    "levir-cd":  LEVIRCDDataset,
    "levir":     LEVIRCDDataset,
    "levircd":   LEVIRCDDataset,
    "whu-cd":    WHUCDDataset,
    "whu":       WHUCDDataset,
    "whucd":     WHUCDDataset,
    "sysu-cd":   SYSUCDDataset,
    "sysu":      SYSUCDDataset,
    "sysucd":    SYSUCDDataset,
    "dsifn-cd":  DSIFNCDDataset,
    "dsifn":     DSIFNCDDataset,
    "dsifncd":   DSIFNCDDataset,
    "second":    SECONDDataset,
}

_LEVIR_NAMES = {"levir-cd", "levir", "levircd"}


def _get_class(name: str):
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        supported = sorted(set(_REGISTRY.values()), key=lambda c: c.__name__)
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: "
            + ", ".join(c.__name__ for c in supported)
        )
    return cls


def _is_levir(name: str) -> bool:
    return name.lower() in _LEVIR_NAMES


# ── Single-split builder ───────────────────────────────────────────────────────

def build_dataset(
    dataset_cfg: dict,
    split: str,
    augment: Optional[bool] = None,
    seed: int = 42,
) -> Dataset:
    """Instantiate a dataset from a dataset-config dict.

    For LEVIR-CD + ``train_mode="tile"``, training uses ``LEVIRCDTileDataset``.
    Val and test always use the tile-aware path (non-overlapping tiles).

    Args:
        dataset_cfg: the ``dataset:`` section (must include ``name`` and ``root``).
        split:       "train", "val", or "test".
        augment:     override augmentation flag; default is True for train only.
        seed:        random seed for reproducible val split.

    Returns:
        Dataset instance.
    """
    name      = dataset_cfg.get("name", "LEVIR-CD")
    root      = dataset_cfg.get("root")
    if root is None:
        raise ValueError(f"dataset config for '{name}' is missing 'root' key.")

    do_augment = augment if augment is not None else (split == "train")

    # ── LEVIR-CD tile-mode routing ────────────────────────────────────────────
    if _is_levir(name):
        train_mode = str(dataset_cfg.get("train_mode", "image")).lower()
        use_tiles  = (split == "train" and train_mode == "tile") or split in ("val", "test")
        if use_tiles:
            return _build_levircd_tile(dataset_cfg, root, split, do_augment, seed)

    # ── Default / image-level path ────────────────────────────────────────────
    cls = _get_class(name)
    kwargs = dict(
        root       = root,
        split      = split,
        image_size = int(dataset_cfg.get("image_size", 256)),
        val_ratio  = float(dataset_cfg.get("val_ratio", 0.2)),
        seed       = seed,
        augment    = do_augment,
    )
    if cls is LEVIRCDDataset:
        pass  # no extra kwargs
    elif cls is SECONDDataset:
        kwargs.update({
            "mode": str(dataset_cfg.get("mode", "binary")),
            "task_type": str(dataset_cfg.get("task_type", "semantic_change")),
            "ignore_index": int(dataset_cfg.get("ignore_index", 255)),
            "binary_from_semantic": bool(dataset_cfg.get("binary_from_semantic", True)),
            "num_classes": int(dataset_cfg.get("num_classes", 7)),
            "a_candidates": dataset_cfg.get("image_a_dir_candidates", []),
            "b_candidates": dataset_cfg.get("image_b_dir_candidates", []),
            "label_a_candidates": dataset_cfg.get("label_a_dir_candidates", []),
            "label_b_candidates": dataset_cfg.get("label_b_dir_candidates", []),
            "binary_label_candidates": dataset_cfg.get("binary_label_dir_candidates", []),
            "train_split": dataset_cfg.get("train_split"),
            "val_split": dataset_cfg.get("val_split"),
            "test_split": dataset_cfg.get("test_split"),
            "precompute_binary_masks": bool(dataset_cfg.get("precompute_second_binary_masks", False)),
            "second_binary_cache_dir": dataset_cfg.get("second_binary_cache_dir"),
            "cache_images_in_ram": bool(dataset_cfg.get("cache_images_in_ram", False)),
            "cache_masks_in_ram": bool(dataset_cfg.get("cache_masks_in_ram", False)),
            "profile_enabled": bool(dataset_cfg.get("profile_enabled", False)),
            "second_label_palette": dataset_cfg.get("second_label_palette"),
        })
    else:
        if "image_a_dir_candidates" in dataset_cfg:
            kwargs["a_candidates"] = dataset_cfg["image_a_dir_candidates"]
        if "image_b_dir_candidates" in dataset_cfg:
            kwargs["b_candidates"] = dataset_cfg["image_b_dir_candidates"]
        if "label_dir_candidates" in dataset_cfg:
            kwargs["label_candidates"] = dataset_cfg["label_dir_candidates"]

    return cls(**kwargs)


def _build_levircd_tile(
    dc: dict, root: str, split: str, do_augment: bool, seed: int
) -> LEVIRCDTileDataset:
    """Construct a ``LEVIRCDTileDataset`` from dataset config."""
    cache_dir = dc.get("tile_cache_dir", "outputs/dataset_indices")
    # Resolve relative cache path relative to repo root
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        cache_path = repo_root / cache_dir

    return LEVIRCDTileDataset(
        root                = root,
        split               = split,
        image_size          = int(dc.get("tile_size", dc.get("image_size", 256))),
        val_ratio           = float(dc.get("val_ratio", 0.2)),
        seed                = seed,
        augment             = do_augment,
        train_stride        = int(dc.get("train_stride", 128)),
        val_stride          = int(dc.get("val_stride", 256)),
        test_stride         = int(dc.get("test_stride", 256)),
        min_change_pixels   = int(dc.get("min_change_pixels", 1)),
        include_empty_ratio = float(dc.get("include_empty_ratio", 0.25)),
        use_cache           = bool(dc.get("use_tile_cache", True)),
        cache_dir           = cache_path,
    )


# ── Sampler helper ─────────────────────────────────────────────────────────────

def _make_train_sampler(train_ds: Dataset, dc: dict):
    """Return a ``BalancedChangeSampler`` if enabled, else ``None`` (uses shuffle)."""
    if not bool(dc.get("balance_change_tiles", False)):
        return None
    if not isinstance(train_ds, LEVIRCDTileDataset):
        return None
    try:
        from data.sampler import BalancedChangeSampler
        ratio = float(dc.get("target_change_tile_ratio", 0.5))
        return BalancedChangeSampler(train_ds, target_change_ratio=ratio)
    except Exception as e:
        logger.warning(f"BalancedChangeSampler unavailable: {e} — using random shuffle.")
        return None


# ── Dataset stats logging ──────────────────────────────────────────────────────

def log_dataset_stats(
    train_ds: Dataset,
    val_ds:   Dataset,
    test_ds:  Optional[Dataset],
    log: logging.Logger,
    dc: dict,
) -> dict:
    """Print and return dataset statistics for the training run."""
    stats: dict = {}

    train_mode = str(dc.get("train_mode", "image")).lower()
    val_mode   = str(dc.get("val_mode",   "existing")).lower()
    test_mode  = str(dc.get("test_mode",  "existing")).lower()

    sep = "-" * 56
    log.info(sep)
    log.info("DATASET STATISTICS")
    log.info(f"  Train mode : {train_mode}")
    log.info(f"  Val mode   : {val_mode}")
    log.info(f"  Test mode  : {test_mode}")
    log.info(sep)

    if isinstance(train_ds, LEVIRCDTileDataset):
        n_train     = len(train_ds)
        n_change    = train_ds.n_change()
        n_no_change = train_ds.n_no_change()
        avg_cr      = train_ds.mean_changed_pixel_ratio()
        cr_frac     = train_ds.change_ratio()
        log.info(f"  Train image pairs  : (from train/ split, 80%)")
        log.info(f"  Train tiles        : {n_train}")
        log.info(f"  Change tiles       : {n_change}")
        log.info(f"  No-change tiles    : {n_no_change}")
        log.info(f"  Change tile ratio  : {cr_frac:.1%}")
        log.info(f"  Avg changed pixels : {avg_cr:.2%}")
        stats.update({
            "train_tiles":      n_train,
            "train_change":     n_change,
            "train_no_change":  n_no_change,
            "train_change_ratio": round(cr_frac, 4),
            "train_avg_pixel_ratio": round(avg_cr, 4),
        })
    else:
        log.info(f"  Train samples : {len(train_ds)} (image-level)")
        stats["train_samples"] = len(train_ds)

    n_val = len(val_ds)
    val_pos_ratio = _estimate_positive_ratio(val_ds)
    log.info(f"  Val tiles          : {n_val}")
    log.info(f"  Val GT pos ratio   : {val_pos_ratio:.1%}")
    stats.update({"val_tiles": n_val, "val_pos_ratio": round(val_pos_ratio, 4)})

    if test_ds is not None:
        n_test = len(test_ds)
        test_pos_ratio = _estimate_positive_ratio(test_ds)
        log.info(f"  Test tiles         : {n_test}")
        log.info(f"  Test GT pos ratio  : {test_pos_ratio:.1%}")
        stats.update({"test_tiles": n_test, "test_pos_ratio": round(test_pos_ratio, 4)})

    log.info(sep)
    return stats


def _estimate_positive_ratio(ds: Dataset, n: int = 200) -> float:
    """Quick positive-pixel ratio estimate from first n samples."""
    import numpy as np
    import torch
    total, positive = 0, 0
    for i in range(min(n, len(ds))):
        try:
            item = ds[i]
            m = item.get("mask", item.get("change_mask", item.get("label")))
            if m is not None:
                if isinstance(m, torch.Tensor):
                    m = m.numpy()
                positive += int((m > 0.5).sum())
                total    += m.size
        except Exception:
            pass
    return float(positive / total) if total else 0.0


# ── Dataset manifest ──────────────────────────────────────────────────────────

def save_dataset_manifest(
    stats: dict,
    dc: dict,
    out_path: Path,
    leakage_status: str = "not_checked",
) -> None:
    """Write a JSON dataset manifest for reproducibility."""
    manifest = {
        "root":               dc.get("root"),
        "tile_size":          dc.get("tile_size", dc.get("image_size", 256)),
        "train_stride":       dc.get("train_stride", 128),
        "val_stride":         dc.get("val_stride", 256),
        "include_empty_ratio": dc.get("include_empty_ratio", 0.25),
        "balance_change_tiles": dc.get("balance_change_tiles", True),
        "target_change_tile_ratio": dc.get("target_change_tile_ratio", 0.5),
        "leakage_check_status": leakage_status,
        **stats,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Dataset manifest saved to {out_path}")


# ── DataLoader builders ────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict, dataset_cfg: Optional[dict] = None):
    """Build (train_loader, val_loader) from a merged experiment config.

    When ``dataset.train_mode="tile"`` and ``dataset.balance_change_tiles=true``,
    a ``BalancedChangeSampler`` replaces the default random shuffle.

    Returns:
        (train_loader, val_loader)
    """
    dc   = dataset_cfg if dataset_cfg is not None else cfg.get("dataset", {})
    tc   = cfg["training"]
    hw   = cfg.get("hardware", {})
    vc   = cfg.get("validation", {})
    exp  = cfg.get("experiment", {})
    seed = int(exp.get("seed", 42))

    train_ds = build_dataset(dc, "train", augment=True,  seed=seed)
    val_ds   = build_dataset(dc, "val",   augment=False, seed=seed)

    nw = int(dc.get("num_workers", 8))
    pin = bool(dc.get("pin_memory", str(hw.get("device", "cuda")).startswith("cuda")))
    persistent_workers = bool(dc.get("persistent_workers", True)) and nw > 0
    prefetch_factor = int(dc.get("prefetch_factor", 2)) if nw > 0 else None

    sampler = _make_train_sampler(train_ds, dc)
    use_shuffle = sampler is None

    train_loader_kwargs = {
        "dataset": train_ds,
        "batch_size": int(tc["batch_size"]),
        "shuffle": use_shuffle,
        "sampler": sampler,
        "num_workers": nw,
        "pin_memory": pin,
        "drop_last": True,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor

    val_loader_kwargs = {
        "dataset": val_ds,
        "batch_size": int(vc.get("batch_size", tc["batch_size"])),
        "shuffle": False,
        "num_workers": nw,
        "pin_memory": pin,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)
    return train_loader, val_loader


def build_test_loader(cfg: dict, dataset_cfg: Optional[dict] = None) -> DataLoader:
    """Build a test DataLoader from config."""
    dc   = dataset_cfg if dataset_cfg is not None else cfg.get("dataset", {})
    tc   = cfg.get("training", {})
    hw   = cfg.get("hardware", {})
    vc   = cfg.get("validation", {})
    exp  = cfg.get("experiment", {})
    seed = int(exp.get("seed", 42))

    split   = str(cfg.get("evaluation", {}).get("split", "test"))
    test_ds = build_dataset(dc, split, augment=False, seed=seed)

    nw = int(dc.get("num_workers", 8))
    pin = bool(dc.get("pin_memory", str(hw.get("device", "cuda")).startswith("cuda")))
    persistent_workers = bool(dc.get("persistent_workers", True)) and nw > 0
    prefetch_factor = int(dc.get("prefetch_factor", 2)) if nw > 0 else None
    loader_kwargs = {
        "dataset": test_ds,
        "batch_size": int(vc.get("batch_size", tc.get("batch_size", 8))),
        "shuffle": False,
        "num_workers": nw,
        "pin_memory": pin,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**loader_kwargs)
    return loader
