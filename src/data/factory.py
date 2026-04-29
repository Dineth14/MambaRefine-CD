"""Dataset and DataLoader factory.

Delegates to ``data.dataset_builder`` for multi-dataset support using the
single global configuration object.
"""
from __future__ import annotations

from data.dataset_builder import build_dataloaders as _build_dataloaders


def build_dataloaders(cfg: dict):
    """Build (train_loader, val_loader) from config.

    Supports DSIFN-CD and WHU-CD via ``dataset.name``.
    """
    # Inject default dataset name for backward compatibility
    dc = cfg.setdefault("dataset", {})
    if "name" not in dc:
        dc["name"] = "DSIFN-CD"
    return _build_dataloaders(cfg)
