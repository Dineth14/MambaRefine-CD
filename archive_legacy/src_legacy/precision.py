"""Precision and memory-format helpers."""
from __future__ import annotations

import torch


def amp_dtype_from_config(cfg: dict, device: torch.device) -> torch.dtype:
    eff = cfg.get("efficiency", {})
    raw = str(eff.get("amp_dtype", "fp16")).lower()
    if raw in {"bf16", "bfloat16"}:
        if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        return torch.float16
    return torch.float16


def channels_last_enabled(cfg: dict) -> bool:
    return bool(cfg.get("efficiency", {}).get("channels_last", False))


def maybe_channels_last_image(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor
