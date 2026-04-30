"""Dataset and DataLoader builders for active binary CD experiments."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from data.dsifncd import DSIFNCDDataset
from data.whucd import WHUCDDataset

logger = logging.getLogger(__name__)

_REGISTRY = {
    "dsifn-cd": DSIFNCDDataset,
    "dsifn": DSIFNCDDataset,
    "dsifncd": DSIFNCDDataset,
    "whu-cd": WHUCDDataset,
    "whu": WHUCDDataset,
    "whucd": WHUCDDataset,
}


def _dataset_key(name: str) -> str:
    return str(name).lower().strip().replace("_", "-")


def _get_class(name: str):
    key = _dataset_key(name)
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown dataset {name!r}. Active datasets are DSIFN-CD and WHU-CD."
        )
    return cls


def build_dataset(
    dataset_cfg: dict,
    split: str,
    augment: Optional[bool] = None,
    seed: int = 42,
) -> Dataset:
    """Instantiate an active binary change-detection dataset."""
    name = dataset_cfg.get("name")
    root = dataset_cfg.get("root")
    if not name:
        raise ValueError("dataset.name is required. Active options: DSIFN-CD, WHU-CD.")
    if root is None:
        raise ValueError(f"dataset config for {name!r} is missing 'root'.")

    cls = _get_class(str(name))
    do_augment = augment if augment is not None else split == "train"
    kwargs = {
        "root": root,
        "split": split,
        "image_size": int(dataset_cfg.get("image_size", 256)),
        "val_ratio": float(dataset_cfg.get("val_ratio", 0.2)),
        "seed": seed,
        "augment": do_augment,
    }
    if "augmentation_ops" in dataset_cfg and cls is DSIFNCDDataset:
        kwargs["augmentation_ops"] = dataset_cfg["augmentation_ops"]
    if "image_a_dir_candidates" in dataset_cfg:
        kwargs["a_candidates"] = dataset_cfg["image_a_dir_candidates"]
    if "image_b_dir_candidates" in dataset_cfg:
        kwargs["b_candidates"] = dataset_cfg["image_b_dir_candidates"]
    if "label_dir_candidates" in dataset_cfg:
        kwargs["label_candidates"] = dataset_cfg["label_dir_candidates"]
    return cls(**kwargs)


def _extract_mask(item: dict):
    return item.get("mask", item.get("change_mask", item.get("label")))


def _estimate_positive_ratio(ds: Dataset, n: int = 200) -> float:
    total, positive = 0, 0
    for i in range(min(n, len(ds))):
        try:
            mask = _extract_mask(ds[i])
            if mask is None:
                continue
            if isinstance(mask, torch.Tensor):
                mask_arr = mask.detach().cpu()
                positive += int((mask_arr > 0.5).sum().item())
                total += int(mask_arr.numel())
            else:
                positive += int((mask > 0.5).sum())
                total += int(mask.size)
        except Exception:
            continue
    return float(positive / total) if total else 0.0


def log_dataset_stats(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Optional[Dataset],
    log: logging.Logger,
    dc: dict,
) -> dict:
    """Log compact binary CD dataset statistics for reproducibility."""
    stats = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "val_pos_ratio": round(_estimate_positive_ratio(val_ds), 6),
    }
    log.info("-" * 56)
    log.info("DATASET STATISTICS")
    log.info("  Dataset       : %s", dc.get("name"))
    log.info("  Train samples : %d", stats["train_samples"])
    log.info("  Val samples   : %d", stats["val_samples"])
    log.info("  Val pos ratio : %.4f", stats["val_pos_ratio"])
    if test_ds is not None:
        stats["test_samples"] = len(test_ds)
        stats["test_pos_ratio"] = round(_estimate_positive_ratio(test_ds), 6)
        log.info("  Test samples  : %d", stats["test_samples"])
        log.info("  Test pos ratio: %.4f", stats["test_pos_ratio"])
    log.info("-" * 56)
    return stats


def save_dataset_manifest(
    stats: dict,
    dc: dict,
    out_path: Path,
    leakage_status: str = "not_checked",
) -> None:
    """Write a JSON dataset manifest for reproducibility."""
    manifest = {
        "name": dc.get("name"),
        "root": dc.get("root"),
        "image_size": dc.get("image_size", 256),
        "val_ratio": dc.get("val_ratio", 0.2),
        "leakage_check_status": leakage_status,
        **stats,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Dataset manifest saved to %s", out_path)


def build_dataloaders(cfg: dict, dataset_cfg: Optional[dict] = None):
    """Build train and validation loaders."""
    dc = dataset_cfg if dataset_cfg is not None else cfg.get("dataset", {})
    dlc = cfg.get("dataloader", {})
    tc = cfg["training"]
    hw = cfg.get("hardware", {})
    vc = cfg.get("validation", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))

    train_ds = build_dataset(dc, "train", augment=True, seed=seed)
    val_ds = build_dataset(dc, "val", augment=False, seed=seed)

    num_workers = int(dlc.get("num_workers", dc.get("num_workers", 8)))
    pin_memory = bool(dlc.get("pin_memory", dc.get("pin_memory", str(hw.get("device", "cuda")).startswith("cuda"))))
    persistent_workers = bool(dlc.get("persistent_workers", dc.get("persistent_workers", True))) and num_workers > 0
    prefetch_factor = int(dlc.get("prefetch_factor", dc.get("prefetch_factor", 2))) if num_workers > 0 else None
    drop_last = bool(dlc.get("drop_last", True))

    train_kwargs = {
        "dataset": train_ds,
        "batch_size": int(tc["batch_size"]),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
    }
    val_kwargs = {
        "dataset": val_ds,
        "batch_size": int(vc.get("batch_size", tc["batch_size"])),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        train_kwargs["prefetch_factor"] = prefetch_factor
        val_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**train_kwargs), DataLoader(**val_kwargs)


def build_test_loader(cfg: dict, dataset_cfg: Optional[dict] = None) -> DataLoader:
    """Build a test loader."""
    dc = dataset_cfg if dataset_cfg is not None else cfg.get("dataset", {})
    dlc = cfg.get("dataloader", {})
    tc = cfg.get("training", {})
    hw = cfg.get("hardware", {})
    vc = cfg.get("validation", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    split = str(cfg.get("evaluation", {}).get("split", "test"))
    test_ds = build_dataset(dc, split, augment=False, seed=seed)

    num_workers = int(dlc.get("num_workers", dc.get("num_workers", 8)))
    pin_memory = bool(dlc.get("pin_memory", dc.get("pin_memory", str(hw.get("device", "cuda")).startswith("cuda"))))
    persistent_workers = bool(dlc.get("persistent_workers", dc.get("persistent_workers", True))) and num_workers > 0
    prefetch_factor = int(dlc.get("prefetch_factor", dc.get("prefetch_factor", 2))) if num_workers > 0 else None
    kwargs = {
        "dataset": test_ds,
        "batch_size": int(vc.get("batch_size", tc.get("batch_size", 8))),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)
