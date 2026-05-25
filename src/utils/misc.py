from __future__ import annotations

from pathlib import Path

import torch


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_metrics(metrics: dict) -> str:
    keys = ["F1", "IoU", "Precision", "Recall", "OA"]
    return " ".join(f"{k}={float(metrics[k]):.4f}" for k in keys if k in metrics)
