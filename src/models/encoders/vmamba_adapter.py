"""VMamba encoder adapter."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn


class VMambaAdapter(nn.Module):
    def __init__(self, variant: str = "small", pretrained: bool = True) -> None:
        super().__init__()
        root = Path(__file__).resolve().parents[3]
        vmamba_dir = root / "third_party" / "VMamba"
        candidates = [vmamba_dir / "classification", vmamba_dir]
        for candidate in candidates:
            if (candidate / "models").is_dir():
                sys.path.insert(0, str(candidate))
                sys.path.insert(0, str(candidate / "models"))
                break
        else:
            raise ImportError(
                "VMamba is not installed. Place the official VMamba repository under "
                "third_party/VMamba, then run python tools/setup_vmamba.py."
            )

        module = None
        for name in ("models.vmamba", "models.vmamba_v2", "vmamba"):
            try:
                module = importlib.import_module(name)
                break
            except ImportError:
                continue
        if module is None:
            raise ImportError("Could not import VMamba VSSM modules from third_party/VMamba.")

        factories = {
            "tiny": "vmamba_tiny_s1l8",
            "small": "vmamba_small_s2l15",
            "base": "vmamba_base_s2l15",
        }
        factory = getattr(module, factories.get(str(variant).lower(), "vmamba_small_s2l15"), None)
        if factory is None:
            raise ImportError(f"VMamba factory for variant={variant!r} was not found.")
        self.backbone = factory()
        self.model_name = f"vmamba_{variant}"
        self.out_channels = self._detect_channels()
        if pretrained:
            print("WARNING: VMamba pretrained loading is not automatic in the clean adapter; load through checkpoints when needed.")

    @staticmethod
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] > x.shape[-2] and x.shape[-1] > x.shape[1]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def _forward_backbone(self, x: torch.Tensor) -> list[torch.Tensor]:
        bb = self.backbone
        if not (hasattr(bb, "patch_embed") and hasattr(bb, "layers")):
            raise RuntimeError("Expected VMamba VSSM with patch_embed and layers.")
        x = bb.patch_embed(x)
        if hasattr(bb, "pos_drop"):
            x = bb.pos_drop(x)
        feats = []
        for layer in bb.layers:
            if hasattr(layer, "blocks") and hasattr(layer, "downsample"):
                for block in layer.blocks:
                    x = block(x)
                feats.append(self._to_bchw(x))
                if layer.downsample is not None:
                    x = layer.downsample(x)
            else:
                x = layer(x)
                feats.append(self._to_bchw(x))
        return feats

    def _detect_channels(self) -> list[int]:
        was_training = self.training
        self.eval()
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            feats = self._forward_backbone(torch.zeros(1, 3, 64, 64, device=device))
        self.train(was_training)
        return [int(f.shape[1]) for f in feats]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self._forward_backbone(x)
