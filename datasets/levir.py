"""LEVIR-CD dataset wrapper.

Delegates to src/data/levircd.py (LEVIRCDDataset / LEVIRCDTileDataset).
This file provides a clean, config-driven interface for the scripts layer.

Expected config keys (dataset section)
---------------------------------------
    dataset.root          : path to LEVIR-CD root
    dataset.split         : train | val | test
    dataset.image_size    : 256
    dataset.val_ratio     : 0.2
    dataset.augment       : true/false
    dataset.balance.enabled              : true/false
    dataset.balance.min_change_ratio     : 0.001
    dataset.balance.max_nochange_fraction: 0.5
    dataset.balance.oversample_change    : true
    dataset.balance.changed_patch_weight : 2.0
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on path when this module is used standalone
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.levircd import LEVIRCDDataset, LEVIRCDTileDataset  # noqa: E402


def build_levir_dataset(cfg: dict[str, Any], split: str) -> Any:
    """Build a LEVIR-CD dataset from config.

    Args:
        cfg:   Full config dict (uses cfg['dataset']).
        split: 'train' | 'val' | 'test'.

    Returns:
        A torch.utils.data.Dataset instance.
    """
    ds_cfg = cfg.get("dataset", {})
    root       = ds_cfg["root"]
    image_size = int(ds_cfg.get("image_size", 256))
    val_ratio  = float(ds_cfg.get("val_ratio", 0.2))
    augment    = bool(ds_cfg.get("augment", True))

    # Prefer tiled dataset which supports balanced sampling
    return LEVIRCDTileDataset(
        root=root,
        split=split,
        image_size=image_size,
        val_ratio=val_ratio,
        augment=augment,
    )
