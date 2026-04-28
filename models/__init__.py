"""Models package — thin wrappers over src/models."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_MODELS = _REPO_ROOT / "src" / "models"
if str(_SRC_MODELS) not in __path__:
    __path__.append(str(_SRC_MODELS))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.cd_model import build_model, DRBISiameseMambaCD  # noqa: F401

__all__ = ["build_model", "DRBISiameseMambaCD"]
