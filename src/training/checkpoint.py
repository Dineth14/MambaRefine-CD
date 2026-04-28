"""Checkpoint save / load / peek utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch


def save(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    iteration: int,
    best_metric: float,
    cfg: dict,
    *,
    ema_state: Optional[dict] = None,
    best_threshold: Optional[float] = None,
    val_metrics: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    variant = cfg.get("model", {}).get("variant", "unknown")
    payload = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "iteration":   iteration,
        "best_metric": best_metric,
        "variant":     variant,
        "config":      cfg,
    }
    if ema_state is not None:
        payload["ema"] = ema_state
    if best_threshold is not None:
        payload["best_threshold"] = float(best_threshold)
    if val_metrics is not None:
        payload["val_metrics"] = _checkpoint_safe(val_metrics)
    torch.save(payload, path)


def _checkpoint_safe(value):
    if isinstance(value, dict):
        return {str(k): _checkpoint_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def state_dict_for_eval(ckpt: dict, *, use_ema: bool) -> tuple[dict, bool]:
    """Return model weights for evaluation and whether EMA weights were used."""
    model_state = dict(ckpt["model"])
    ema = ckpt.get("ema")
    ema_shadow = ema.get("shadow") if isinstance(ema, dict) else None
    if use_ema and ema_shadow:
        model_state.update(ema_shadow)
        return model_state, True
    return model_state, False


def load(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
    strict: bool = True,
    use_ema: bool = False,
) -> Tuple[int, float]:
    """Load checkpoint, return (iteration, best_metric)."""
    ckpt = peek(path, map_location)
    model_state, _ = state_dict_for_eval(ckpt, use_ema=use_ema)
    model.load_state_dict(model_state, strict=strict)
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("iteration", 0), ckpt.get("best_metric", 0.0)


def load_for_eval(
    path: Path | str,
    model: torch.nn.Module,
    *,
    map_location: str = "cpu",
    strict: bool = True,
    use_ema: bool = False,
) -> dict:
    """Load model weights for evaluation and return checkpoint metadata."""
    ckpt = peek(path, map_location)
    model_state, ema_used = state_dict_for_eval(ckpt, use_ema=use_ema)
    incompatible = model.load_state_dict(model_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint key mismatch with strict loading enabled: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "checkpoint": ckpt,
        "iteration": ckpt.get("iteration", 0),
        "best_metric": ckpt.get("best_metric", 0.0),
        "best_threshold": ckpt.get("best_threshold"),
        "ema_found": bool(ckpt.get("ema")),
        "ema_used": ema_used,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def peek(path: Path | str, map_location: str = "cpu") -> dict:
    """Return the raw checkpoint dict without applying it."""
    return torch.load(str(path), map_location=map_location, weights_only=False)


def find_latest(outputs_root: Path) -> Optional[Path]:
    """Scan outputs_root for the newest run_*/checkpoints/best.pth."""
    if not outputs_root.exists():
        return None
    runs = sorted(
        [d for d in outputs_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )
    for run in reversed(runs):
        ckpt = run / "checkpoints" / "best.pth"
        if ckpt.exists():
            return ckpt
    return None
