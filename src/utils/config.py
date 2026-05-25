"""
src/utils/config.py

Simple config loader. Reads configs/active.yaml and returns an EasyDict.
Validates required top-level keys. Saves a copy to the run output folder.
"""

import os
import shutil
from pathlib import Path

import yaml
try:
    from easydict import EasyDict
except ImportError:
    class EasyDict(dict):
        def __init__(self, mapping=None, **kwargs):
            super().__init__()
            mapping = {} if mapping is None else dict(mapping)
            mapping.update(kwargs)
            for key, value in mapping.items():
                self[key] = self._convert(value)

        @classmethod
        def _convert(cls, value):
            if isinstance(value, dict) and not isinstance(value, EasyDict):
                return cls(value)
            if isinstance(value, list):
                return [cls._convert(v) for v in value]
            return value

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = self._convert(value)

REQUIRED_KEYS = ["project", "data", "model", "ablation", "train", "loss", "eval",
                 "checkpoint", "resume", "logging"]

CONFIG_PATH = "configs/active.yaml"


def load_config(path=CONFIG_PATH) -> EasyDict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"{path} is empty")
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise ValueError(f"configs/active.yaml is missing required key: '{key}'")
    return EasyDict(raw)


def save_config(cfg: EasyDict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, "config.yaml")
    shutil.copy(CONFIG_PATH, dst)
    print(f"Config saved to: {dst}")


def to_plain_dict(value):
    if isinstance(value, dict):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(v) for v in value]
    return value


def find_latest_run(output_root: str | Path = "outputs") -> Path | None:
    root = Path(output_root)
    if not root.exists():
        return None
    runs = sorted(p for p in root.glob("run_*") if p.is_dir())
    return runs[-1] if runs else None


def find_latest_best_checkpoint(output_root: str | Path = "outputs") -> Path | None:
    root = Path(output_root)
    if not root.exists():
        return None
    candidates = sorted(root.glob("run_*/checkpoints/best_iter_*_F1_*.pth"))
    return candidates[-1] if candidates else None
