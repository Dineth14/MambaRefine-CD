from __future__ import annotations

import time

import torch


def measure_fps(model: torch.nn.Module, image_size: int, device, warmup: int = 5, repeats: int = 30) -> float:
    model.eval()
    a = torch.zeros(1, 3, image_size, image_size, device=device)
    b = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(a, b)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(a, b)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    return repeats / max(time.perf_counter() - start, 1e-9)
