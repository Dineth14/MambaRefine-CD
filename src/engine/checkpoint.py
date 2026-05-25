"""Best-only checkpoint helpers."""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state: dict, ckpt_dir, iteration: int, metric_value: float, metric_name: str = "F1") -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for old in ckpt_dir.glob(f"best_iter_*_{metric_name}_*.pth"):
        old.unlink(missing_ok=True)
    path = ckpt_dir / f"best_iter_{iteration:06d}_{metric_name}_{metric_value:.4f}.pth"
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, cfg=None):
    if path is None:
        raise RuntimeError("Checkpoint path is null.")
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint {path}: {exc}") from exc


def find_latest_best(output_root) -> Path | None:
    root = Path(output_root)
    if not root.exists():
        return None
    candidates = sorted(root.glob("run_*/checkpoints/best_iter_*_F1_*.pth"))
    return candidates[-1] if candidates else None
