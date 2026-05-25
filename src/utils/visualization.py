from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _denorm(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    arr = (x.float() * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).astype(np.uint8)


def _gray(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().squeeze().cpu().numpy()
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def save_side_by_side(image_a, image_b, mask, pred, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = _denorm(image_a)
    b = _denorm(image_b)
    gt = np.repeat(_gray(mask)[:, :, None], 3, axis=2)
    pr = np.repeat(_gray(pred)[:, :, None], 3, axis=2)
    Image.fromarray(np.concatenate([a, b, gt, pr], axis=1)).save(path)
