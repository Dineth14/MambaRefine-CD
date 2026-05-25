"""Builds model and runs a dummy forward pass.

Reads: configs/active.yaml
Usage: python tools/check_model.py
"""
from __future__ import annotations

import traceback
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.build import build_model
from src.models.mambarefine_cd import MambaRefineCD
from src.models.modules.temporal_difference import TemporalDifference
from src.utils.config import load_config
from src.utils.flops import measure_flops
from src.utils.misc import count_parameters


class DummyEncoder(nn.Module):
    out_channels = [32, 64, 128, 256]

    def __init__(self) -> None:
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in self.out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


def _build_for_check(cfg):
    try:
        return build_model(cfg), True
    except Exception as exc:
        print(f"Actual encoder check unavailable: {exc}")
        print("Running structural check with DummyEncoder; training still requires the configured encoder dependency.")
        return MambaRefineCD(DummyEncoder(), cfg), False


def main() -> None:
    cfg = load_config()
    model, actual_encoder = _build_for_check(cfg)
    model = model.to("cpu")
    model.eval()
    image_size = int(cfg.data.image_size)
    image_a = torch.randn(1, 3, image_size, image_size)
    image_b = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        outputs = model(image_a, image_b)

    total, trainable = count_parameters(model)
    print(f"Temporal mode: {cfg.ablation.temporal_input_mode}")
    print(f"Actual encoder: {actual_encoder}")
    print(f"Encoder out_channels: {model.encoder.out_channels}")
    print(f"D-RBI input channels: {model.drbi_input_channels}")
    print(f"Output keys: {list(outputs.keys())}")
    for key, value in outputs.items():
        print(f"{key}: {tuple(value.shape)}")
    print(f"Params: total={total:,}, trainable={trainable:,}")
    flops = measure_flops(model, image_size=image_size, device="cpu")
    if flops:
        print(f"FLOPs_G: {flops:.3f}")

    print("Temporal modes:")
    for mode in TemporalDifference.VALID_MODES:
        try:
            cfg.ablation.temporal_input_mode = mode
            m, _ = _build_for_check(cfg)
            m = m.to("cpu").eval()
            with torch.no_grad():
                out = m(image_a, image_b)
            if out["logits"].shape[-2:] != (image_size, image_size):
                raise RuntimeError(f"Output spatial shape mismatch: {tuple(out['logits'].shape)}")
            print(f"  {mode}: PASS")
        except Exception:
            print(f"  {mode}: FAIL")
            traceback.print_exc()
            raise

    print("PASS")


if __name__ == "__main__":
    main()
