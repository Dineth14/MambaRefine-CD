"""Folder inference helpers."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


def load_image(path: Path, image_size: int, device):
    img = Image.open(path).convert("RGB")
    tensor = TF.normalize(TF.to_tensor(TF.resize(img, [image_size, image_size])), IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0).to(device)


@torch.no_grad()
def predict_pair(model, path_a: Path, path_b: Path, cfg, device, threshold: float):
    image_a = load_image(path_a, int(cfg.data.image_size), device)
    image_b = load_image(path_b, int(cfg.data.image_size), device)
    logits = model(image_a, image_b)["logits"]
    return (torch.sigmoid(logits)[0, 0] >= threshold).float().cpu()
