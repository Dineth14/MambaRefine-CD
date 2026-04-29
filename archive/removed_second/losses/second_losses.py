"""SECOND semantic change detection losses.

The implementation lives in ``src/training/second_loss.py`` so the training
pipeline and legacy top-level imports use the same loss code.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.second_loss import SecondSemanticChangeLoss  # noqa: E402,F401

__all__ = ["SecondSemanticChangeLoss"]
