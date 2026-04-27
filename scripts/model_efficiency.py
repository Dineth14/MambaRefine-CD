#!/usr/bin/env python3
"""Profile local model efficiency for the website.

Reads configs/global_config.yaml and writes normalized JSON outputs.
No CLI args.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from utils.config import load_config
from models.cd_model import build_model

WEBSITE_DATA = ROOT / "website" / "assets" / "data"
OUTPUT_DIR = ROOT / "outputs" / "model_efficiency"


def _count_params(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters())


def _format_millions(count: int | None) -> float | None:
    if count is None:
        return None
    return round(count / 1e6, 4)


def _device_from_cfg(cfg: dict[str, Any]) -> torch.device:
    raw = str(cfg.get("hardware", {}).get("device", "cuda"))
    if torch.cuda.is_available():
        return torch.device(raw)
    return torch.device("cpu")


def _load_cfg() -> dict[str, Any]:
    cfg = load_config()
    cfg["model"]["pretrained"] = False
    if str(cfg.get("model", {}).get("output_mode", "binary")).lower() == "semantic":
        cfg["model"]["output_mode"] = "binary"
        cfg.setdefault("_notes", []).append("output_mode forced to binary for unsupported semantic head path")
    return cfg


def _try_ptflops(model: torch.nn.Module, image_size: int) -> tuple[str | None, str | None]:
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return None, "ptflops not installed"

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> Any:
            return self.inner(x, x)

    try:
        macs, _ = get_model_complexity_info(
            _Wrapper(model.cpu()),
            (3, image_size, image_size),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        return macs, None
    except Exception as exc:  # pragma: no cover
        return None, f"ptflops failed: {exc}"


def _try_fvcore(model: torch.nn.Module, image_size: int, device: torch.device) -> tuple[float | None, str | None]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None, "fvcore not installed"
    try:
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        analysis = FlopCountAnalysis(model, (dummy, dummy))
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = analysis.total()
        return float(total) / 1e9, None
    except Exception as exc:  # pragma: no cover
        return None, f"fvcore failed: {exc}"


def _peak_memory(model: torch.nn.Module, image_size: int, device: torch.device, train_step: bool) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    image_a = torch.randn(1, 3, image_size, image_size, device=device)
    image_b = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        with torch.amp.autocast("cuda", enabled=True):
            logits, _ = model(image_a, image_b)
            if train_step:
                loss = logits.mean()
                loss.backward()
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    finally:
        model.zero_grad(set_to_none=True)
        del image_a, image_b
        if "logits" in locals():
            del logits
        if "loss" in locals():
            del loss
        torch.cuda.empty_cache()
        gc.collect()
    return round(float(peak), 3)


def _throughput(model: torch.nn.Module, image_size: int, device: torch.device, runs: int = 10) -> float | None:
    image_a = torch.randn(1, 3, image_size, image_size, device=device)
    image_b = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval()
    try:
        for _ in range(3):
            with torch.no_grad():
                model(image_a, image_b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                model(image_a, image_b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        model.train()
        del image_a, image_b
    if elapsed <= 0:
        return None
    return round(runs / elapsed, 4)


def main() -> None:
    WEBSITE_DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg()
    device = _device_from_cfg(cfg)
    image_size = int(cfg.get("dataset", {}).get("image_size", 256))
    model = build_model(cfg).to(device)

    total_params = _count_params(model)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    backbone_params = _count_params(getattr(model, "encoder", None))
    drbi_params = _count_params(getattr(model, "diff_modules", None))
    decoder_params = _count_params(getattr(model, "decoder", None))

    flops_g = None
    flops_reason = None
    ptflops_text, ptflops_reason = _try_ptflops(model, image_size)
    if ptflops_text is not None:
        flops_reason = None
    else:
        flops_g, flops_reason = _try_fvcore(model, image_size, device)
    model = model.to(device)

    runtime_notes: list[str] = []
    try:
        peak_forward_memory_mb = _peak_memory(model, image_size, device, train_step=False)
    except Exception as exc:  # pragma: no cover
        peak_forward_memory_mb = None
        runtime_notes.append(f"forward memory unavailable: {exc}")
    try:
        peak_train_step_memory_mb = _peak_memory(model, image_size, device, train_step=True)
    except Exception as exc:  # pragma: no cover
        peak_train_step_memory_mb = None
        runtime_notes.append(f"train-step memory unavailable: {exc}")
    try:
        fps = _throughput(model, image_size, device)
    except Exception as exc:  # pragma: no cover
        fps = None
        runtime_notes.append(f"throughput unavailable: {exc}")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str((ROOT / "configs" / "global_config.yaml").resolve()),
        "notes": cfg.get("_notes", []) + runtime_notes,
        "metrics": {
            "device": str(device),
            "image_size": image_size,
            "variant": cfg.get("model", {}).get("variant"),
            "decoder": cfg.get("model", {}).get("decoder"),
            "difference_enabled": bool(cfg.get("difference", {}).get("enabled", True)),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "backbone_params": backbone_params,
            "drbi_params": drbi_params,
            "decoder_params": decoder_params,
            "total_params_millions": _format_millions(total_params),
            "trainable_params_millions": _format_millions(trainable_params),
            "backbone_params_millions": _format_millions(backbone_params),
            "drbi_params_millions": _format_millions(drbi_params),
            "decoder_params_millions": _format_millions(decoder_params),
            "ptflops_macs": ptflops_text if ptflops_text is not None else "TBD",
            "flops_gmac": flops_g if flops_g is not None else "TBD",
            "flops_reason": flops_reason or ptflops_reason,
            "peak_forward_memory_mb": peak_forward_memory_mb if peak_forward_memory_mb is not None else "TBD",
            "peak_train_step_memory_mb": peak_train_step_memory_mb if peak_train_step_memory_mb is not None else "TBD",
            "fps": fps if fps is not None else "TBD",
        },
    }

    for target in [WEBSITE_DATA / "ours_efficiency.json", OUTPUT_DIR / "latest_efficiency.json"]:
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
