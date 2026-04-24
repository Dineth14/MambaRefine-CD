"""Dataset and DataLoader factory.

Delegates to ``data.dataset_builder`` for multi-dataset support.
Falls back to LEVIR-CD if no ``dataset.name`` is specified (backward
compatibility with existing single-dataset configs).
"""
from __future__ import annotations

from data.dataset_builder import build_dataloaders as _build_dataloaders


def build_dataloaders(cfg: dict):
    """Build (train_loader, val_loader) from config.

    Supports LEVIR-CD, WHU-CD, SYSU-CD, DSIFN-CD via ``dataset.name``.
    Defaults to LEVIR-CD when ``dataset.name`` is absent.
    """
    # Inject default dataset name for backward compatibility
    dc = cfg.setdefault("dataset", {})
    if "name" not in dc:
        dc["name"] = "LEVIR-CD"
    return _build_dataloaders(cfg)
