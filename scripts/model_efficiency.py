#!/usr/bin/env python3
"""Model efficiency profiler.

Reports:
  - Total parameters
  - Trainable parameters
  - Estimated FLOPs (via ptflops or fvcore if installed)
  - Peak GPU memory for one forward+backward pass
  - Throughput (images/sec)

Usage:
    cd /storage2/ChangeDetection/MV/MambaRefine-CD
    conda run -n mamba_new python scripts/model_efficiency.py

No CLI arguments required — reads configs/global_config.yaml.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

# Add project src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg():
    from utils.config import load_config
    cfg_path = ROOT / "configs" / "global_config.yaml"
    cfg = load_config(str(cfg_path))
    # Override for efficiency test: no pretrained download needed, small batch
    cfg["model"]["pretrained"] = False
    return cfg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _param_counts(model: torch.nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _human(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} G"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.2f} K"
    return str(n)


def _try_ptflops(model, image_size: int):
    """Returns (macs_string, params_string) or None."""
    try:
        from ptflops import get_model_complexity_info
        # ptflops needs a single-input model; wrap our siamese model
        class _Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x, x)

        macs, params = get_model_complexity_info(
            _Wrapper(model),
            (3, image_size, image_size),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        return macs, params
    except ImportError:
        return None
    except Exception as e:
        return f"ptflops error: {e}", None


def _try_fvcore(model, image_size: int, device):
    """Returns FlopCountAnalysis result or None."""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_str
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        flops = FlopCountAnalysis(model, (dummy, dummy))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        return total, flop_count_str(flops)
    except ImportError:
        return None
    except Exception as e:
        return None


def _peak_memory(model, image_size: int, device, batch_size: int = 2):
    """Returns peak GPU memory in MB (forward + backward)."""
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    ia = torch.randn(batch_size, 3, image_size, image_size, device=device)
    ib = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.cuda.amp.autocast():
        out, _ = model(ia, ib)
        loss = out.mean()
    loss.backward()
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    model.zero_grad(set_to_none=True)
    del ia, ib, out, loss
    torch.cuda.empty_cache()
    gc.collect()
    return peak_mb


def _throughput(model, image_size: int, device, batch_size: int = 4, runs: int = 50):
    """Returns images/sec (inference-only, with AMP if CUDA)."""
    model.eval()
    ia = torch.randn(batch_size, 3, image_size, image_size, device=device)
    ib = torch.randn(batch_size, 3, image_size, image_size, device=device)

    use_amp = device.type == "cuda"
    ctx = torch.cuda.amp.autocast() if use_amp else torch.amp.autocast("cpu", enabled=False)

    # Warmup
    with torch.no_grad(), ctx:
        for _ in range(5):
            model(ia, ib)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad(), ctx:
        for _ in range(runs):
            model(ia, ib)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    model.train()
    return (runs * batch_size) / elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = _load_cfg()
    device_str = cfg.get("hardware", {}).get("device", "cuda")
    device     = torch.device(device_str if torch.cuda.is_available() else "cpu")
    image_size = int(cfg.get("dataset", {}).get("image_size", 256))
    batch_size = int(cfg.get("training", {}).get("batch_size", 4))

    print(f"\n{'='*60}")
    print("  Model Efficiency Report")
    print(f"{'='*60}")
    print(f"  Config  : {ROOT / 'configs' / 'global_config.yaml'}")
    print(f"  Device  : {device}")
    print(f"  Image   : {image_size}×{image_size}")
    print(f"  Mode    : {cfg.get('model', {}).get('mode', 'dual')}")
    print(f"  Decoder : {cfg.get('model', {}).get('decoder', '?')}")
    print(f"  D-RBI   : {cfg.get('difference', {}).get('enabled', False)}")
    print(f"{'='*60}\n")

    from models.cd_model import build_model
    model = build_model(cfg).to(device)

    # ── Params ────────────────────────────────────────────────────────
    total, trainable = _param_counts(model)
    frozen = total - trainable
    print(f"  Parameters")
    print(f"    Total      : {_human(total)}")
    print(f"    Trainable  : {_human(trainable)}")
    print(f"    Frozen     : {_human(frozen)}")

    # ── FLOPs ─────────────────────────────────────────────────────────
    print(f"\n  FLOPs (image_size={image_size})")
    pt = _try_ptflops(model.cpu() if device.type == "cuda" else model, image_size)
    if pt and pt[0] is not None:
        print(f"    ptflops MACs : {pt[0]}")
    else:
        fv = _try_fvcore(model.to(device), image_size, device)
        if fv:
            gflops = fv[0] / 1e9
            print(f"    fvcore FLOPs : {gflops:.2f} G")
        else:
            print("    (ptflops and fvcore not available — skip)")
    model = model.to(device)

    # ── Peak memory ───────────────────────────────────────────────────
    if device.type == "cuda":
        print(f"\n  Peak GPU Memory (forward+backward, batch={batch_size})")
        try:
            peak = _peak_memory(model, image_size, device, batch_size)
            print(f"    {peak:.1f} MB")
        except Exception as e:
            print(f"    Error: {e}")

    # ── Throughput ────────────────────────────────────────────────────
    print(f"\n  Throughput (inference, batch={batch_size}, {image_size}px)")
    try:
        fps = _throughput(model, image_size, device, batch_size)
        print(f"    {fps:.1f} images/sec")
    except Exception as e:
        print(f"    Error: {e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
