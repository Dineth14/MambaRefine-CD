"""SeK loss — re-exports the implementation from src/training/sek_loss.py."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.sek_loss import (  # noqa: F401
    SeKLoss,
    binary_soft_kappa_loss,
    SoftKappaResult,
)

__all__ = ["SeKLoss", "binary_soft_kappa_loss", "SoftKappaResult"]
