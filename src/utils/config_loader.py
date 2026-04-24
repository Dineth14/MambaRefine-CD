"""YAML config loader with base-config inheritance and dataset_config support.

Usage:
    cfg = load_config("configs/experiments/train_levir_refinement.yaml")

Inheritance chain:
1. ``base: ../base.yaml``       → deep-merged (base first, child overrides)
2. ``dataset_config: ../datasets/levircd.yaml``
       → loads dataset + metrics sections and merges into cfg
       → experiment overrides win over dataset_config values
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*. Returns a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_path(current_dir: Path, ref: str) -> Path:
    """Resolve a config-relative or repo-absolute path reference."""
    p = Path(ref)
    if p.is_absolute():
        return p
    # Try relative to current file's directory first
    candidate = (current_dir / ref).resolve()
    if candidate.exists():
        return candidate
    # Try relative to repo root (two levels up from configs/)
    repo_root = current_dir
    for _ in range(5):   # walk up at most 5 levels
        if (repo_root / "src").is_dir():
            break
        repo_root = repo_root.parent
    candidate2 = (repo_root / ref).resolve()
    if candidate2.exists():
        return candidate2
    # Fall back to the first attempt (will raise a clear error later)
    return candidate


def load_config(path: str | Path) -> dict:
    """Load a YAML config file, resolving ``base:`` and ``dataset_config:``."""
    path = Path(path)
    with open(path, "r") as f:
        cfg: dict = yaml.safe_load(f) or {}

    # 1. Handle base inheritance
    base_rel = cfg.pop("base", None)
    if base_rel is not None:
        base_path = _resolve_path(path.parent, base_rel)
        base_cfg  = load_config(base_path)      # recurse for chained bases
        cfg       = _deep_merge(base_cfg, cfg)

    # 2. Handle dataset_config reference
    ds_cfg_ref = cfg.pop("dataset_config", None)
    if ds_cfg_ref is not None:
        ds_cfg_path = _resolve_path(path.parent, ds_cfg_ref)
        ds_cfg      = load_config(ds_cfg_path)  # recurse (may have its own base)
        # Merge: dataset_cfg is the base; current cfg (experiment) overrides
        cfg = _deep_merge(ds_cfg, cfg)

    return cfg
