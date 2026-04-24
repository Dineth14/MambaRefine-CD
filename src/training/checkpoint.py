"""Checkpoint save / load / peek utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch


def save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    iteration: int,
    best_metric: float,
    cfg: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    variant = cfg.get("model", {}).get("variant", "unknown")
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "iteration":   iteration,
            "best_metric": best_metric,
            "variant":     variant,
            "config":      cfg,
        },
        path,
    )


def load(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Tuple[int, float]:
    """Load checkpoint, return (iteration, best_metric)."""
    ckpt = peek(path, map_location)
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("iteration", 0), ckpt.get("best_metric", 0.0)


def peek(path: Path | str, map_location: str = "cpu") -> dict:
    """Return the raw checkpoint dict without applying it."""
    return torch.load(str(path), map_location=map_location, weights_only=False)


def find_latest(outputs_root: Path) -> Optional[Path]:
    """Scan outputs_root for the newest run_*/checkpoints/best.pth."""
    if not outputs_root.exists():
        return None
    runs = sorted(
        [d for d in outputs_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )
    for run in reversed(runs):
        ckpt = run / "checkpoints" / "best.pth"
        if ckpt.exists():
            return ckpt
    return None
