"""Small CUDA memory helpers for run metadata.

These helpers intentionally use PyTorch peak-memory counters rather than
external tools.  They return 0 on CPU so callers can keep one code path.
"""
from __future__ import annotations

from pathlib import Path
import json

import torch


def reset_peak_memory(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device | str) -> float:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 4)


def params_m(model: torch.nn.Module) -> float:
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 4)


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
