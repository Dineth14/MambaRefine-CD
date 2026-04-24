"""Shared augmentation transforms for change-detection dataset pairs.

Provides:
  build_train_transforms(image_size)  → albumentations Compose or None
  build_eval_transforms()             → albumentations Compose or None
  norm_tensor(arr)                    → torch.Tensor (HWC uint8 → CHW float, fallback)
"""
from __future__ import annotations

import numpy as np
import torch

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALB = True
except ImportError:
    _ALB = False


def build_train_transforms(image_size: int = 256):
    """Return albumentations Compose for training, or None if unavailable."""
    if not _ALB:
        return None
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
            ToTensorV2(),
        ],
        additional_targets={"image_b": "image"},
    )


def build_eval_transforms():
    """Return albumentations Compose for eval (normalisation only), or None."""
    if not _ALB:
        return None
    return A.Compose(
        [
            A.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
            ToTensorV2(),
        ],
        additional_targets={"image_b": "image"},
    )


def norm_tensor(arr: np.ndarray) -> torch.Tensor:
    """Normalise HWC uint8 numpy array → CHW float32 tensor (no augmentation)."""
    x = arr.astype(np.float32) / 255.0
    return torch.from_numpy(((x - _MEAN) / _STD).transpose(2, 0, 1))
