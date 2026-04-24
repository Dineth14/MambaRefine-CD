"""Multi-dataset builder for change detection benchmarks.

Usage:
    from data.dataset_builder import build_dataset, build_dataloaders

``build_dataset(dataset_cfg, split)`` dispatches on ``dataset_cfg["name"]``.
``build_dataloaders(cfg)`` loads the dataset config referenced by
``cfg["dataset_config"]`` (or falls back to ``cfg["dataset"]`` directly),
then builds train/val DataLoaders.

Supported dataset names (case-insensitive, flexible aliases):
  LEVIR-CD / levir / levircd
  WHU-CD   / whu   / whucd
  SYSU-CD  / sysu  / sysucd
  DSIFN-CD / dsifn / dsifncd
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from torch.utils.data import DataLoader, Dataset

from data.levircd  import LEVIRCDDataset
from data.whucd    import WHUCDDataset
from data.sysucd   import SYSUCDDataset
from data.dsifncd  import DSIFNCDDataset

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
}


def _get_class(name: str):
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        supported = sorted(set(_REGISTRY.values()), key=lambda c: c.__name__)
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: "
            + ", ".join(c.__name__ for c in supported)
        )
    return cls


def build_dataset(
    dataset_cfg: dict,
    split: str,
    augment: Optional[bool] = None,
    seed: int = 42,
) -> Dataset:
    """Instantiate a dataset from a dataset-config dict.

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

    cls = _get_class(name)

    do_augment = augment if augment is not None else (split == "train")

    kwargs = dict(
        root         = root,
        split        = split,
        image_size   = int(dataset_cfg.get("image_size", 256)),
        val_ratio    = float(dataset_cfg.get("val_ratio", 0.2)),
        seed         = seed,
        augment      = do_augment,
    )

    # Pass directory candidates if provided in dataset config
    if "image_a_dir_candidates" in dataset_cfg:
        kwargs["a_candidates"] = dataset_cfg["image_a_dir_candidates"]
    if "image_b_dir_candidates" in dataset_cfg:
        kwargs["b_candidates"] = dataset_cfg["image_b_dir_candidates"]
    if "label_dir_candidates" in dataset_cfg:
        kwargs["label_candidates"] = dataset_cfg["label_dir_candidates"]

    # LEVIRCDDataset uses slightly different kwarg names — handle gracefully
    if cls is LEVIRCDDataset:
        kwargs.pop("a_candidates", None)
        kwargs.pop("b_candidates", None)
        kwargs.pop("label_candidates", None)

    return cls(**kwargs)


def build_dataloaders(cfg: dict, dataset_cfg: Optional[dict] = None):
    """Build (train_loader, val_loader) from a merged experiment config.

    Reads training, hardware, validation, and dataset sections.
    ``dataset_cfg`` may be passed directly; otherwise falls back to
    ``cfg["dataset"]``.

    Returns:
        (train_loader, val_loader)
    """
    dc  = dataset_cfg if dataset_cfg is not None else cfg.get("dataset", {})
    tc  = cfg["training"]
    hw  = cfg.get("hardware", {})
    vc  = cfg.get("validation", {})
    exp = cfg.get("experiment", {})
    seed = int(exp.get("seed", 42))

    train_ds = build_dataset(dc, "train", augment=True,  seed=seed)
    val_ds   = build_dataset(dc, "val",   augment=False, seed=seed)

    nw  = int(dc.get("num_workers", 8))
    pin = str(hw.get("device", "cuda")).startswith("cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size       = int(tc["batch_size"]),
        shuffle          = True,
        num_workers      = nw,
        pin_memory       = pin,
        drop_last        = True,
        persistent_workers = nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size       = int(vc.get("batch_size", tc["batch_size"])),
        shuffle          = False,
        num_workers      = nw,
        pin_memory       = pin,
        persistent_workers = nw > 0,
    )
    return train_loader, val_loader


def build_test_loader(cfg: dict, dataset_cfg: Optional[dict] = None) -> DataLoader:
    """Build a test DataLoader from config."""
    dc  = dataset_cfg if dataset_cfg is not None else cfg.get("dataset", {})
    tc  = cfg.get("training", {})
    hw  = cfg.get("hardware", {})
    vc  = cfg.get("validation", {})
    exp = cfg.get("experiment", {})
    seed = int(exp.get("seed", 42))

    test_ds = build_dataset(dc, "test", augment=False, seed=seed)

    nw  = int(dc.get("num_workers", 4))
    pin = str(hw.get("device", "cuda")).startswith("cuda")

    return DataLoader(
        test_ds,
        batch_size       = int(vc.get("batch_size", tc.get("batch_size", 8))),
        shuffle          = False,
        num_workers      = nw,
        pin_memory       = pin,
        persistent_workers = nw > 0,
    )
