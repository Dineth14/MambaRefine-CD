"""Legacy compatibility wrapper.

The repo now uses a single global config file:
    configs/global_config.yaml

Any legacy import of ``utils.config_loader.load_config`` is redirected to the
new global loader so fragmented config files are no longer part of the runtime
path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.config import load_config as _load_global_config


def load_config(path: str | Path | None = None) -> Any:
    """Return the runtime config.

    When *path* is provided, it is passed through to the shared config loader,
    which merges the selected config on top of the global base config.
    """
    return _load_global_config(path)
