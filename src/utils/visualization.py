"""Prediction grid visualisation.

Saves a PNG grid of rows: [pre-change | post-change | GT | prediction].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_rgb(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().float().permute(1, 2, 0).numpy()
    x = np.clip(x * _STD + _MEAN, 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def _mask_rgb(t: torch.Tensor) -> np.ndarray:
    m = t.detach().cpu().float()
    if m.dim() == 3:
        m = m.squeeze(0)
    if m.min() < 0 or m.max() > 1:
        m = torch.sigmoid(m)
    m = (m > 0.5).numpy().astype(np.uint8) * 255
    return np.stack([m, m, m], axis=-1)


def save_prediction_grid(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    label: torch.Tensor,
    pred: torch.Tensor,
    save_path: Path,
    count: int = 4,
) -> None:
    if not _PIL:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(count, img_a.shape[0])
    rows = [
        np.concatenate([_to_rgb(img_a[i]), _to_rgb(img_b[i]),
                        _mask_rgb(label[i]), _mask_rgb(pred[i])], axis=1)
        for i in range(n)
    ]
    Image.fromarray(np.concatenate(rows, axis=0)).save(save_path)
