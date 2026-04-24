"""Test-Time Augmentation (TTA) for change detection.

Supported augmentations (all applied in logit space):
    - original
    - horizontal flip
    - vertical flip
    - 90-degree rotation

For each augmentation the transform is applied to both images, the model
is run, and the prediction is inverse-transformed before averaging.
Logits (not probabilities) are averaged to preserve numerical precision.

Usage
-----
    from training.tta import apply_tta

    logits = apply_tta(model, image_a, image_b, amp=True)

Config keys
-----------
    evaluation:
      use_tta: true
      tta_augmentations: [original, hflip, vflip, rot90]  # optional subset
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Transform helpers ─────────────────────────────────────────────────────────

def _hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def _vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def _rot90(x: torch.Tensor) -> torch.Tensor:
    """90-degree counter-clockwise rotation."""
    return torch.rot90(x, k=1, dims=[-2, -1])


def _irot90(x: torch.Tensor) -> torch.Tensor:
    """Inverse of _rot90 (270-degree CCW = 90-degree CW)."""
    return torch.rot90(x, k=3, dims=[-2, -1])


# ── Augmentation registry ────────────────────────────────────────────────────

# Each entry: (fwd_fn, inv_fn)
_AUGS = {
    "original": (lambda x: x,  lambda x: x),
    "hflip":    (_hflip,        _hflip),      # hflip is its own inverse
    "vflip":    (_vflip,        _vflip),      # same
    "rot90":    (_rot90,        _irot90),
}

_DEFAULT_AUGS: List[str] = ["original", "hflip", "vflip", "rot90"]


# ── Public API ────────────────────────────────────────────────────────────────

@torch.no_grad()
def apply_tta(
    model: nn.Module,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    amp: bool = False,
    augmentations: Optional[List[str]] = None,
) -> torch.Tensor:
    """Run inference with TTA and return averaged logits.

    Args:
        model:         model in eval mode; called as ``model(ia, ib)``.
        image_a:       [B, C, H, W] pre-image tensor on the correct device.
        image_b:       [B, C, H, W] post-image tensor on the correct device.
        amp:           use torch.amp.autocast for inference.
        augmentations: list of augmentation names to apply.  Defaults to all
                       four (original + hflip + vflip + rot90).

    Returns:
        averaged_logits: [B, 1, H, W]
    """
    aug_names = augmentations if augmentations else _DEFAULT_AUGS

    # Validate names early
    for name in aug_names:
        if name not in _AUGS:
            raise ValueError(
                f"Unknown TTA augmentation '{name}'. "
                f"Valid options: {list(_AUGS.keys())}"
            )

    logit_sum: Optional[torch.Tensor] = None

    device = image_a.device
    autocast_ctx = torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda"))

    for name in aug_names:
        fwd, inv = _AUGS[name]

        aug_a = fwd(image_a)
        aug_b = fwd(image_b)

        with autocast_ctx:
            logits, _ = model(aug_a, aug_b)     # [B, 1, H, W]

        # Inverse-transform the prediction back to canonical orientation
        logits_inv = inv(logits)

        if logit_sum is None:
            logit_sum = logits_inv
        else:
            logit_sum = logit_sum + logits_inv

    averaged = logit_sum / len(aug_names)   # type: ignore[operator]
    return averaged


def build_tta_augmentations(cfg: dict) -> List[str]:
    """Read augmentation list from config, defaulting to all four."""
    eval_cfg = cfg.get("evaluation", {})
    return list(eval_cfg.get("tta_augmentations", _DEFAULT_AUGS))
