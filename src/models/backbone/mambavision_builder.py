"""MambaVision backbone builder.

Maps human-readable variant aliases to the MambaVision registry names and
returns a MambaVisionFeatureExtractor ready for Siamese use.

Variant table
-------------
alias   | registry name    | dim  | channels
--------|------------------|------|------------------------------
tiny    | mamba_vision_T   | 80   | [80,  160,  320,  640]
tiny2   | mamba_vision_T2  | 80   | [80,  160,  320,  640]
small   | mamba_vision_S   | 96   | [96,  192,  384,  768]
base    | mamba_vision_B   | 128  | [128, 256,  512,  1024]
large   | mamba_vision_L   | 196  | [196, 392,  784,  1568]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List

# ── Ensure MambaVision dependencies are importable ───────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]        # MambaRefine-CD/
_MV_EXP_SRC = _REPO_ROOT.parent / "MambaVision_experiments" / "src"
_MV_REPO = _REPO_ROOT.parent / "MambaVisionCD"

for _p in (_MV_EXP_SRC, _MV_REPO):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from mvcd.model import MambaVisionFeatureExtractor  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - import-time environment guard
    raise ModuleNotFoundError(
        "Could not import mvcd.model.MambaVisionFeatureExtractor. "
        f"Expected sibling dependency at {_MV_EXP_SRC}. "
        "Check that MambaVision_experiments/src and MambaVisionCD are present."
    ) from exc

# ── Variant maps ──────────────────────────────────────────────────────────────
_ALIAS_MAP: dict[str, str] = {
    "tiny":  "mamba_vision_T",
    "tiny2": "mamba_vision_T2",
    "small": "mamba_vision_S",
    "base":  "mamba_vision_B",
    "large": "mamba_vision_L",
    # Single-letter shortcuts
    "t":  "mamba_vision_T",
    "t2": "mamba_vision_T2",
    "s":  "mamba_vision_S",
    "b":  "mamba_vision_B",
    "l":  "mamba_vision_L",
}

_KNOWN: set[str] = {
    "mamba_vision_T", "mamba_vision_T2",
    "mamba_vision_S", "mamba_vision_B", "mamba_vision_L",
}

_INFO: dict[str, dict] = {
    "mamba_vision_T":  {"dim": 80,  "channels": [80,  160,  320,  640]},
    "mamba_vision_T2": {"dim": 80,  "channels": [80,  160,  320,  640]},
    "mamba_vision_S":  {"dim": 96,  "channels": [96,  192,  384,  768]},
    "mamba_vision_B":  {"dim": 128, "channels": [128, 256,  512,  1024]},
    "mamba_vision_L":  {"dim": 196, "channels": [196, 392,  784,  1568]},
}


def resolve_name(variant: str) -> str:
    """Return the canonical ``mamba_vision_*`` registry name for *variant*."""
    if variant.startswith("mamba_vision_"):
        if variant not in _KNOWN:
            warnings.warn(f"Unknown MambaVision name {variant!r}. Proceeding anyway.")
        return variant
    name = _ALIAS_MAP.get(variant.lower().strip())
    if name is None:
        raise ValueError(
            f"Unknown variant {variant!r}. "
            f"Choose from: {sorted(_ALIAS_MAP)} or full names like 'mamba_vision_T'."
        )
    return name


def build(variant: str, pretrained: bool = True) -> MambaVisionFeatureExtractor:
    """Construct a MambaVisionFeatureExtractor for *variant*.

    Args:
        variant:   Short alias ('tiny', 'small', …) or full registry name.
        pretrained: Whether to load pretrained ImageNet weights.

    Returns:
        A MambaVisionFeatureExtractor with ``.channels`` auto-detected.
    """
    return MambaVisionFeatureExtractor(
        model_name=resolve_name(variant),
        pretrained=pretrained,
    )


def approx_channels(variant: str) -> List[int]:
    """Return approximate encoder output channels for *variant*."""
    return list(_INFO[resolve_name(variant)]["channels"])
