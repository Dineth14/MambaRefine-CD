"""MambaVision encoder adapter."""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from src.utils.torchvision_compat import patch_register_fake


class MambaVisionAdapter(nn.Module):
    def __init__(self, variant: str = "small", pretrained: bool = True) -> None:
        super().__init__()
        patch_register_fake()
        root = Path(__file__).resolve().parents[3]
        mv_src_candidates = (
            root.parent / "MambaVision_experiments" / "src",
            root.parent / "MambaVisionCD" / "MambaVision_experiments" / "src",
        )
        mv_repo = root.parent / "MambaVisionCD"
        for path in (*mv_src_candidates, mv_repo):
            if path.exists() and str(path) not in sys.path:
                sys.path.append(str(path))
        try:
            from mvcd.model import MambaVisionFeatureExtractor
        except Exception:
            model_path = next(
                (path / "mvcd" / "model.py" for path in mv_src_candidates if (path / "mvcd" / "model.py").exists()),
                mv_src_candidates[0] / "mvcd" / "model.py",
            )
            if not model_path.exists():
                raise ImportError(
                    "Could not import MambaVisionFeatureExtractor. "
                    "Expected MambaVision_experiments/src or MambaVisionCD/MambaVision_experiments/src."
                )
            spec = importlib.util.spec_from_file_location("mvcd_model_direct", model_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load MambaVision module from {model_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("mvcd_model_direct", module)
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                missing = exc.name or "required dependency"
                raise ImportError(
                    f"MambaVision dependency is missing: {missing}. "
                    "Install the MambaVision/Mamba SSM dependencies before training with encoder_family=mambavision."
                ) from exc
            MambaVisionFeatureExtractor = module.MambaVisionFeatureExtractor
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Could not import MambaVisionFeatureExtractor. "
                "Expected sibling dependency MambaVision_experiments/src."
            ) from exc

        self.model_name = self._resolve_name(variant)
        self.backbone = MambaVisionFeatureExtractor(model_name=self.model_name, pretrained=pretrained)
        self.out_channels = list(getattr(self.backbone, "channels"))

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
        return self.backbone(x)
