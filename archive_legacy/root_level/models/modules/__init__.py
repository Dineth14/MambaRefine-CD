"""Model modules — re-exports from src/models/modules/."""
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_MODULES = _REPO_ROOT / "src" / "models" / "modules"
if str(_SRC_MODULES) not in __path__:
    __path__.append(str(_SRC_MODULES))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.modules.differential_region_boundary import DifferentialRegionBoundaryInteraction  # noqa: F401
from models.modules.cram_lite import CRAMLite, CRAMLiteBank  # noqa: F401

__all__ = ["DifferentialRegionBoundaryInteraction", "CRAMLite", "CRAMLiteBank"]
