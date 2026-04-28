"""SECOND semantic change detection dataset wrapper.

Delegates to src/data/second.py (SECONDDataset).

Expected config keys (dataset section)
---------------------------------------
    dataset.root          : path to SECOND root
    dataset.split         : train | val | test
    dataset.image_size    : 256
    dataset.val_ratio     : 0.2
    dataset.augment       : true/false
    dataset.num_classes   : 7
    dataset.ignore_index  : 255
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.second import SECONDDataset  # noqa: E402


def build_second_dataset(cfg: dict[str, Any], split: str) -> Any:
    """Build a SECOND SCD dataset from config."""
    ds_cfg = cfg.get("dataset", {})
    return SECONDDataset(
        root=ds_cfg["root"],
        split=split,
        image_size=int(ds_cfg.get("image_size", 256)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.2)),
        augment=bool(ds_cfg.get("augment", True)),
        mode=str(ds_cfg.get("mode", "semantic" if str(ds_cfg.get("task_type", "")).lower() == "semantic_change" else "binary")),
        task_type=str(ds_cfg.get("task_type", "semantic_change")),
        num_classes=int(ds_cfg.get("num_classes", 7)),
        ignore_index=int(ds_cfg.get("ignore_index", 255)),
        binary_from_semantic=bool(ds_cfg.get("binary_from_semantic", True)),
        a_candidates=ds_cfg.get("image_a_dir_candidates"),
        b_candidates=ds_cfg.get("image_b_dir_candidates"),
        label_a_candidates=ds_cfg.get("label_a_dir_candidates"),
        label_b_candidates=ds_cfg.get("label_b_dir_candidates"),
        binary_label_candidates=ds_cfg.get("binary_label_dir_candidates"),
        train_split=ds_cfg.get("train_split"),
        val_split=ds_cfg.get("val_split"),
        test_split=ds_cfg.get("test_split"),
        precompute_binary_masks=bool(ds_cfg.get("precompute_second_binary_masks", False)),
        second_binary_cache_dir=ds_cfg.get("second_binary_cache_dir"),
        cache_images_in_ram=bool(ds_cfg.get("cache_images_in_ram", False)),
        cache_masks_in_ram=bool(ds_cfg.get("cache_masks_in_ram", False)),
        second_label_palette=ds_cfg.get("second_label_palette"),
    )
