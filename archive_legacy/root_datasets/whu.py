"""WHU-CD dataset wrapper.

Delegates to src/data/whucd.py (WHUCDDataset).

Expected config keys (dataset section)
---------------------------------------
    dataset.root       : path to WHU-CD root
    dataset.split      : train | val | test
    dataset.image_size : 256
    dataset.val_ratio  : 0.2
    dataset.augment    : true/false
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.whucd import WHUCDDataset  # noqa: E402


def build_whu_dataset(cfg: dict[str, Any], split: str) -> Any:
    """Build a WHU-CD dataset from config."""
    ds_cfg = cfg.get("dataset", {})
    return WHUCDDataset(
        root=ds_cfg["root"],
        split=split,
        image_size=int(ds_cfg.get("image_size", 256)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.2)),
        augment=bool(ds_cfg.get("augment", True)),
    )
