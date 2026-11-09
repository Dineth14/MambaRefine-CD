"""Four-stage feature adapter for the official NVIDIA MambaVision package."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from src.utils.torchvision_compat import patch_register_fake


class MambaVisionAdapter(nn.Module):
    def __init__(
        self,
        variant: str = "small",
        pretrained: bool = True,
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        patch_register_fake()
        root = Path(__file__).resolve().parents[3]
        try:
            from mambavision import create_model
        except ImportError as exc:
            raise ImportError(
                "The official NVIDIA mambavision package is required. Install "
                "mambavision==1.2.0 in the training environment."
            ) from exc

        self.model_name = self._resolve_name(variant)
        default_path = root / "pretrained_weights" / f"mambavision_{variant}_1k.pth.tar"
        weight_path = Path(pretrained_path).expanduser() if pretrained_path else default_path
        if not weight_path.is_absolute():
            weight_path = root / weight_path
        if pretrained:
            weight_path.parent.mkdir(parents=True, exist_ok=True)
        self.pretrained_path = weight_path
        self.backbone = create_model(
            self.model_name,
            pretrained=pretrained,
            model_path=str(weight_path),
            num_classes=0,
        )
        channel_map = {
            "mamba_vision_T": [80, 160, 320, 640],
            "mamba_vision_T2": [80, 160, 320, 640],
            "mamba_vision_S": [96, 192, 384, 768],
            "mamba_vision_B": [128, 256, 512, 1024],
            "mamba_vision_L": [196, 392, 784, 1568],
        }
        if self.model_name not in channel_map:
            raise ValueError(f"Unsupported MambaVision feature variant: {self.model_name}")
        self.out_channels = channel_map[self.model_name]

    @staticmethod
    def _resolve_name(variant: str) -> str:
        aliases = {
            "tiny": "mamba_vision_T",
            "tiny2": "mamba_vision_T2",
            "small": "mamba_vision_S",
            "base": "mamba_vision_B",
            "large": "mamba_vision_L",
            "t": "mamba_vision_T",
            "s": "mamba_vision_S",
            "b": "mamba_vision_B",
            "l": "mamba_vision_L",
        }
        return aliases.get(str(variant).lower(), variant)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.backbone.patch_embed(x)
        features = []
        for level in self.backbone.levels:
            if level.downsample is None:
                x = level(x)
                features.append(x)
                continue
            captured = []

            def capture_before_downsample(_module, inputs):
                captured.append(inputs[0])

            handle = level.downsample.register_forward_pre_hook(capture_before_downsample)
            try:
                x = level(x)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError("Failed to capture a MambaVision stage feature")
            features.append(captured[0])
        if len(features) != 4:
            raise RuntimeError(f"Expected four MambaVision stages, got {len(features)}")
        return features
