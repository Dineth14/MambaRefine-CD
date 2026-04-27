from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HALF_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
HALF_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


class AdapterUnavailableError(RuntimeError):
    """Raised when an adapter cannot be used in the current environment."""


class BaseAdapter:
    model_name = "BaseModel"
    expected_normalization = "imagenet"
    expected_image_order = "A,B"
    expected_output_type = "binary_logits"

    def __init__(self, cfg: dict, model_cfg: dict, dataset_name: str, checkpoint_path: str | None, device: torch.device):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.dataset_name = dataset_name
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.repo_dir = self._resolve_path(model_cfg.get("repo_dir"))
        self.weights_dir = self._resolve_path(model_cfg.get("weights_dir"))
        self.last_normalization_used = self.expected_normalization

    def _resolve_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        path = Path(value)
        return path if path.is_absolute() else (root / path)

    def is_available(self) -> tuple[bool, str]:
        if self.repo_dir is not None and not self.repo_dir.exists():
            return False, f"repo missing: {self.repo_dir}"
        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            return False, f"checkpoint missing: {self.checkpoint_path}"
        return True, "available"

    def add_repo_to_path(self) -> None:
        if self.repo_dir is None:
            return
        repo = str(self.repo_dir.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def import_module_candidates(self, module_names: list[str], file_candidates: list[str] | None = None):
        self.add_repo_to_path()
        errors: list[str] = []
        for name in module_names:
            try:
                return importlib.import_module(name)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if self.repo_dir is not None and file_candidates:
            for rel in file_candidates:
                path = self.repo_dir / rel
                if not path.exists():
                    continue
                try:
                    spec = importlib.util.spec_from_file_location(path.stem, path)
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
        raise AdapterUnavailableError("; ".join(errors) or "no import candidates succeeded")

    def import_external_models_module(self, module_name: str = "models.networks"):
        self.add_repo_to_path()
        for key in [name for name in list(sys.modules) if name == "models" or name.startswith("models.")]:
            del sys.modules[key]
        return importlib.import_module(module_name)

    def namespace(self, **kwargs):
        return Namespace(**kwargs)

    def load_torch_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path:
            raise AdapterUnavailableError("checkpoint path is not set")
        return torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

    def extract_state_dict(self, checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict", "model_G_state_dict", "net_G", "network", "net"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
        raise AdapterUnavailableError("could not locate a state dict in checkpoint")

    def strip_prefixes(self, state_dict: dict[str, Any], prefixes: tuple[str, ...] = ("module.", "model.")) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            out[new_key] = value
        return out

    def maybe_re_normalize(self, tensor: torch.Tensor, target_norm: str) -> torch.Tensor:
        if target_norm == "imagenet":
            self.last_normalization_used = "imagenet"
            return tensor
        mean = IMAGENET_MEAN.to(tensor.device, dtype=tensor.dtype)
        std = IMAGENET_STD.to(tensor.device, dtype=tensor.dtype)
        raw = tensor * std + mean
        if target_norm == "half":
            self.last_normalization_used = "half"
            half_mean = HALF_MEAN.to(tensor.device, dtype=tensor.dtype)
            half_std = HALF_STD.to(tensor.device, dtype=tensor.dtype)
            return (raw - half_mean) / half_std
        if target_norm == "none":
            self.last_normalization_used = "none"
            return raw
        self.last_normalization_used = target_norm
        return tensor

    def find_class(self, module: Any, class_names: list[str]):
        for name in class_names:
            candidate = getattr(module, name, None)
            if candidate is not None and inspect.isclass(candidate):
                return candidate
        raise AdapterUnavailableError(f"none of the requested classes exist: {class_names}")

    def build_model(self):
        raise NotImplementedError

    def load_checkpoint(self, model):
        raise NotImplementedError

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def forward(self, model, batch: dict[str, Any]):
        raise NotImplementedError

    def logits_to_change_prob(self, output) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise AdapterUnavailableError("adapter output is not a torch tensor")
        if output.ndim == 3:
            output = output.unsqueeze(1)
        if output.ndim != 4:
            raise AdapterUnavailableError(f"unexpected output shape: {tuple(output.shape)}")
        if self.expected_output_type == "two_class_logits" and output.shape[1] >= 2:
            return torch.softmax(output, dim=1)[:, 1:2]
        if self.expected_output_type == "probability":
            if output.shape[1] > 1:
                return output[:, 1:2]
            return output
        if self.expected_output_type == "class_map":
            if output.shape[1] == 1:
                return output.float()
            return output[:, 1:2].float()
        if output.shape[1] > 1:
            return torch.softmax(output, dim=1)[:, 1:2]
        return torch.sigmoid(output)

    def output_to_prediction(self, output, threshold: float) -> torch.Tensor:
        prob = self.logits_to_change_prob(output)
        return (prob > threshold).float()
