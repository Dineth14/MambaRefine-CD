from __future__ import annotations

import torch


def reset_peak_memory(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device) -> float:
    device = torch.device(device)
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
