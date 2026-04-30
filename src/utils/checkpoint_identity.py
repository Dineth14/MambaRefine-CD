"""Checkpoint identity helpers for ablation/evaluation provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of a checkpoint file without loading tensors."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_identity(path: str | Path, ckpt_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return stable metadata that proves exactly which checkpoint was evaluated."""
    p = Path(path).resolve()
    ckpt_meta = ckpt_meta or {}
    size_mb = p.stat().st_size / (1024.0 * 1024.0)
    best_metric = ckpt_meta.get("best_metric")
    if best_metric is None:
        for key in ("best_F1", "best_f1", "best_IoU", "best_iou"):
            if key in ckpt_meta:
                best_metric = ckpt_meta[key]
                break
    return {
        "checkpoint_path": str(p),
        "checkpoint_file_size_MB": round(size_mb, 4),
        "checkpoint_sha256": sha256_file(p),
        "checkpoint_epoch_or_iter": ckpt_meta.get("iteration", ckpt_meta.get("iter", ckpt_meta.get("epoch"))),
        "checkpoint_best_metric_if_available": best_metric,
        "checkpoint_experiment_name": ckpt_meta.get("experiment_name", ckpt_meta.get("config", {}).get("experiment", {}).get("name")),
        "checkpoint_config_path": ckpt_meta.get("config_path"),
        "checkpoint_config_fingerprint": ckpt_meta.get("config_fingerprint"),
    }
