from __future__ import annotations

import torch


def measure_flops(model: torch.nn.Module, image_size: int, device) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:
        print(f"WARNING: FLOPs skipped because fvcore is unavailable: {exc}")
        return None
    model.eval()
    a = torch.zeros(1, 3, image_size, image_size, device=device)
    b = torch.zeros(1, 3, image_size, image_size, device=device)
    try:
        return float(FlopCountAnalysis(model, (a, b)).total() / 1e9)
    except Exception as exc:
        print(f"WARNING: FLOPs measurement failed: {exc}")
        return None
