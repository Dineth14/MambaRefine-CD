"""Legacy dataset factory for active binary CD datasets."""
from __future__ import annotations

from typing import Any


def build_dataset(cfg: dict[str, Any], split: str):
    """Build a dataset from config.

    Args:
        cfg:   Full config dict.
        split: 'train' | 'val' | 'test'.

    Returns:
        torch.utils.data.Dataset
    """
    name = cfg.get("dataset", {}).get("name", "").upper().replace(" ", "").replace("-", "")

    if name in ("WHUCD", "WHU"):
        from datasets.whu import build_whu_dataset
        return build_whu_dataset(cfg, split)
    elif name in ("DSIFNCD", "DSIFN"):
        from datasets.dsifn import build_dsifn_dataset
        return build_dsifn_dataset(cfg, split)
    else:
        raise ValueError(
            f"Unknown dataset name {cfg.get('dataset', {}).get('name')!r}. "
            "Expected one of: WHU-CD, DSIFN-CD."
        )
