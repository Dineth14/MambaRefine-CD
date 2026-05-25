from __future__ import annotations

import torch


def get_device(cfg) -> torch.device:
    requested = str(cfg.train.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: CUDA is not available; using CPU instead of {requested}.")
        return torch.device("cpu")
    return torch.device(requested)
