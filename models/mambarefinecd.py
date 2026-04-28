"""MambaRefineCD model entry point.

This module provides the build_model() factory as a top-level interface.
All model logic lives in src/models/cd_model.py.

Example:
    from models.mambarefinecd import build_model
    model = build_model(cfg)
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.cd_model import build_model, DRBISiameseMambaCD  # noqa: F401

__all__ = ["build_model", "DRBISiameseMambaCD"]
