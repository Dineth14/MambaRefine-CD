"""Evaluation and threshold sweep."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.engine.metrics import BinaryMetricAccumulator


def _thresholds(cfg) -> list[float]:
    values = []
    t = float(cfg.eval.threshold_min)
    while t <= float(cfg.eval.threshold_max) + 1e-9:
        values.append(round(t, 2))
        t += float(cfg.eval.threshold_step)
    return values


@torch.no_grad()
def _run_once(model, dataloader, cfg, threshold: float, device, save_dir: Path | None = None) -> dict:
    acc = BinaryMetricAccumulator()
    model.eval()
    for batch in dataloader:
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=bool(cfg.train.amp) and torch.device(device).type == "cuda"):
            outputs = model(image_a, image_b)
        logits = outputs["logits"]
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        acc.update(logits, mask, threshold)
        if save_dir is not None:
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            save_dir.mkdir(parents=True, exist_ok=True)
            for idx, name in enumerate(batch["name"]):
                arr = (preds[idx, 0].detach().cpu().numpy() * 255).astype("uint8")
                Image.fromarray(arr).save(save_dir / f"{name}.png")
    metrics = acc.compute()
    metrics["threshold"] = float(threshold)
    return metrics


def evaluate(model, dataloader, cfg, split: str, threshold=None, sweep_thresholds: bool = False, device=None, save_dir=None):
    if device is None:
        device = next(model.parameters()).device
    if sweep_thresholds:
        best = None
        sweep = {}
        for thr in _thresholds(cfg):
            metrics = _run_once(model, dataloader, cfg, thr, device)
            sweep[f"{thr:.2f}"] = metrics
            if best is None or metrics["F1"] > best["F1"]:
                best = metrics
        best = dict(best)
        best["best_threshold"] = best["threshold"]
        best["sweep"] = sweep
        return best
    thr = float(cfg.eval.threshold if threshold is None else threshold)
    pred_dir = Path(save_dir) / "predictions" / split if save_dir is not None and bool(cfg.eval.save_predictions) else None
    return _run_once(model, dataloader, cfg, thr, device, save_dir=pred_dir)
