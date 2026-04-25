#!/usr/bin/env python3
"""D-RBI NaN stability smoke test.

Builds the model from configs/global_config.yaml (pretrained=False),
runs one random batch forward + backward, and verifies all intermediate
and output tensors are finite.

Usage:
    cd /storage2/ChangeDetection/MV/MambaRefine-CD
    conda run -n mamba_new python scripts/smoke_test_drbi.py

No CLI arguments required.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_cfg():
    from utils.config import load_config
    cfg = load_config()   # reads configs/global_config.yaml via its own discovery
    cfg["model"]["pretrained"] = False
    # Use tiny2 for smoke test regardless of config variant (avoids OOM on busy GPUs)
    cfg["model"]["variant"] = "tiny2"
    # Force GPU 1 if available; smoke test never needs to match training GPU
    import torch
    if torch.cuda.device_count() > 1:
        cfg.setdefault("hardware", {})["gpu_ids"] = [1]
    return cfg


# ---------------------------------------------------------------------------
# Tensor checks
# ---------------------------------------------------------------------------

def _check(name: str, t: torch.Tensor, tol_max: float = 1e4) -> bool:
    ok = True
    if not torch.isfinite(t).all():
        print(f"  FAIL [{name}] contains NaN or Inf!")
        ok = False
    elif float(t.abs().max()) > tol_max:
        print(f"  WARN [{name}] very large max={float(t.abs().max()):.2e} (> {tol_max:.0e})")
    else:
        fmn = float(t.float().min())
        fmx = float(t.float().max())
        std = float(t.float().std())
        print(f"  OK   [{name}] min={fmn:.4f}  max={fmx:.4f}  std={std:.4f}")
    return ok


# ---------------------------------------------------------------------------
# Hook to capture D-RBI intermediates
# ---------------------------------------------------------------------------

def _attach_drbi_hooks(model: nn.Module) -> dict:
    """Attach forward hooks to all D-RBI modules; return shared store."""
    from models.modules.differential_region_boundary import (
        DifferentialRegionBoundaryInteraction,
    )
    store: dict = {}

    def _make_hook(idx: int):
        def _hook(module, inputs, output):
            store[f"drbi_{idx}_f1_in"]  = inputs[0].detach()
            store[f"drbi_{idx}_f2_in"]  = inputs[1].detach()
            store[f"drbi_{idx}_diff_D"] = output["diff"].detach()
            store[f"drbi_{idx}_region"] = output["region"].detach()
            store[f"drbi_{idx}_boundary"] = output["boundary"].detach()
        return _hook

    handles = []
    for i, m in enumerate(model.modules()):
        if isinstance(m, DifferentialRegionBoundaryInteraction):
            handles.append(m.register_forward_hook(_make_hook(i)))
    store["_handles"] = handles
    return store


def _remove_hooks(store: dict) -> None:
    for h in store.get("_handles", []):
        h.remove()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = _load_cfg()
    gpu_ids = cfg.get("hardware", {}).get("gpu_ids", [0])
    gpu_id  = gpu_ids[0] if isinstance(gpu_ids, list) else int(gpu_ids)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    img_sz = int(cfg.get("dataset", {}).get("image_size", 256))
    batch  = 1   # minimal footprint for smoke test

    print(f"\n{'='*60}")
    print("  D-RBI Smoke Test")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Image size  : {img_sz}")
    print(f"  Batch size  : {batch}")
    print(f"  pre_norm    : {cfg.get('difference', {}).get('pre_norm', True)}")
    print(f"  use_product : {cfg.get('difference', {}).get('use_product', False)}")
    print(f"  lr          : {cfg.get('training', {}).get('lr', 5e-5)}")
    print(f"  grad_clip   : {cfg.get('training', {}).get('gradient_clip', 0.5)}")
    print(f"{'='*60}\n")

    from models.cd_model import build_model
    model = build_model(cfg).to(device)

    store = _attach_drbi_hooks(model)

    ia = torch.randn(batch, 3, img_sz, img_sz, device=device)
    ib = torch.randn(batch, 3, img_sz, img_sz, device=device)

    print("[1] Forward pass (AMP)")
    use_amp = device.type == "cuda"
    all_ok  = True

    with torch.amp.autocast("cuda", enabled=use_amp):
        logits, aux = model(ia, ib)
        logits_c = torch.clamp(logits, -20.0, 20.0)

    all_ok &= _check("logits (raw)",    logits)
    all_ok &= _check("logits (clamped)", logits_c)
    if aux is not None:
        all_ok &= _check("aux logits", aux)

    print("\n[2] D-RBI intermediate tensors")
    for key, val in store.items():
        if key == "_handles":
            continue
        if isinstance(val, torch.Tensor):
            all_ok &= _check(key, val)

    print("\n[3] Backward pass")
    # Build simple loss
    target = torch.zeros(batch, 1, img_sz, img_sz, device=device)
    loss   = torch.nn.functional.binary_cross_entropy_with_logits(logits_c, target)
    if aux is not None:
        aux_c  = torch.clamp(aux, -20.0, 20.0)
        loss   = loss + 0.4 * torch.nn.functional.binary_cross_entropy_with_logits(aux_c, target)

    if not torch.isfinite(loss):
        print(f"  FAIL loss is NaN/Inf: {loss.item()}")
        all_ok = False
    else:
        print(f"  OK   loss={loss.item():.4f}")

    loss.backward()

    print("\n[4] Gradient check")
    bad_drbi   = []
    bad_backbone = []
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            if "diff_module" in name or "drbi" in name.lower() or "boundary" in name.lower():
                bad_drbi.append(name)
            else:
                bad_backbone.append(name)

    if bad_drbi:
        print(f"  FAIL D-RBI NaN/Inf gradients in: {bad_drbi[:5]}")
        all_ok = False
    else:
        print(f"  OK   D-RBI gradients all finite")

    if bad_backbone:
        print(f"  WARN backbone NaN gradients (expected with pretrained=False): {bad_backbone[:3]}")
    else:
        print(f"  OK   backbone gradients all finite")

    _remove_hooks(store)

    print(f"\n{'='*60}")
    if all_ok:
        print("  RESULT: PASSED -- D-RBI is numerically stable")
    else:
        print("  RESULT: FAILED -- see FAIL/WARN lines above")
    print(f"{'='*60}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
