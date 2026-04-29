#!/usr/bin/env python3
"""Tiny DSIFN overfit diagnostic.

Trains on 2-4 samples and evaluates on those same samples. If this fails,
the issue is in data/masks/loss/model-output wiring rather than generalization.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from data.dataset_builder import build_dataset
from training.losses import build_loss
from training.metrics import StreamingMetrics
from training.model_outputs import normalize_model_output
from utils.config import load_config
from utils.seed import set_seed
from models.mambarefinecd import build_model


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().norm().item() ** 2)
    return math.sqrt(total)


def _denorm_image(tensor: torch.Tensor):
    import numpy as np

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    img = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _save_debug(batch: dict, logits: torch.Tensor, out_dir: Path, threshold: float) -> None:
    from PIL import Image
    import numpy as np

    out_dirs = {name: out_dir / name for name in ("image_t1", "image_t2", "gt", "pred", "prob", "error_map")}
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(logits.detach().cpu())
    preds = (probs > threshold).to(torch.uint8)
    labels = (batch["label"].detach().cpu() > 0.5).to(torch.uint8)
    ids = batch.get("name", batch.get("id", [f"sample_{i}" for i in range(logits.shape[0])]))
    for i in range(logits.shape[0]):
        sample_id = ids[i] if isinstance(ids, (list, tuple)) else f"sample_{i}"
        stem = str(sample_id).replace("/", "_")
        gt = labels[i, 0] if labels[i].ndim == 3 else labels[i]
        pred = preds[i, 0] if preds[i].ndim == 3 else preds[i]
        prob = probs[i, 0] if probs[i].ndim == 3 else probs[i]
        Image.fromarray(_denorm_image(batch["image_a"][i])).save(out_dirs["image_t1"] / f"{stem}.png")
        Image.fromarray(_denorm_image(batch["image_b"][i])).save(out_dirs["image_t2"] / f"{stem}.png")
        Image.fromarray((gt.numpy() * 255).astype(np.uint8)).save(out_dirs["gt"] / f"{stem}.png")
        Image.fromarray((pred.numpy() * 255).astype(np.uint8)).save(out_dirs["pred"] / f"{stem}.png")
        Image.fromarray((prob.numpy() * 255.0).clip(0, 255).astype(np.uint8)).save(out_dirs["prob"] / f"{stem}.png")
        err = np.zeros((*gt.shape, 3), dtype=np.uint8)
        gt_np = gt.numpy().astype(bool)
        pred_np = pred.numpy().astype(bool)
        err[pred_np & gt_np] = (0, 180, 0)
        err[pred_np & ~gt_np] = (255, 80, 0)
        err[~pred_np & gt_np] = (0, 120, 255)
        Image.fromarray(err).save(out_dirs["error_map"] / f"{stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit a tiny DSIFN subset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu_simple_cnn", action="store_true", help="Use simple CNN when CUDA is unavailable.")
    parser.add_argument("--debug_dir", default="debug/dsifn_tiny_overfit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    device = torch.device(cfg.get("hardware", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and str(cfg.get("model", {}).get("backbone", "")).lower() == "mambavision":
        if not args.cpu_simple_cnn:
            raise RuntimeError("CUDA is unavailable and MambaVision forward requires CUDA. Re-run on CUDA or pass --cpu_simple_cnn for data/loss diagnosis.")
        cfg["model"]["backbone"] = "simple_cnn"
        cfg["model"]["baseline_channels"] = [8, 16, 32, 64]
        cfg["model"]["variant"] = "baseline"
        cfg["model"]["pretrained"] = False
        cfg["model"]["cram_lite"]["enabled"] = False
        cfg["difference"]["enabled"] = False
        cfg["model"]["decoder"] = "baseline"
        cfg["decoder"]["type"] = "baseline"
        cfg["decoder"]["use_boundary_residual"] = False
        cfg["loss"]["boundary"]["enabled"] = False
        cfg["loss"]["boundary_weight"] = 0.0
        cfg["decoder"]["channels"] = 64
        print("CUDA unavailable: using tiny simple_cnn override for data/loss overfit diagnosis.", flush=True)

    ds = build_dataset(cfg["dataset"], "train", augment=False, seed=int(cfg["experiment"]["seed"]))
    subset = Subset(ds, list(range(min(args.samples, len(ds)))))
    loader = DataLoader(subset, batch_size=min(args.samples, len(subset)), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("sample_ids:", list(batch.get("name", batch.get("id", []))), flush=True)
    print("mask_unique:", sorted(float(v) for v in torch.unique(batch["label"]).tolist()), flush=True)
    print("mask_positive_ratio:", float((batch["label"] > 0.5).float().mean().item()), flush=True)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    threshold = float(cfg.get("evaluation", cfg.get("eval", {})).get("threshold", 0.5))

    ia = batch["image_a"].to(device)
    ib = batch["image_b"].to(device)
    lb = batch["label"].to(device)

    first_loss = None
    last_loss = None
    for iteration in range(1, args.iters + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = normalize_model_output(model(ia, ib))
        logits = outputs["change_logits"]
        if logits.shape[-2:] != lb.shape[-2:]:
            logits = F.interpolate(logits, size=lb.shape[-2:], mode="bilinear", align_corners=False)
        loss, _, _ = loss_fn(logits, lb)
        loss.backward()
        grad = _grad_norm(model)
        optimizer.step()
        last_loss = float(loss.detach().item())
        if first_loss is None:
            first_loss = last_loss

        if iteration <= 5:
            stats = getattr(loss_fn, "latest_stats", {})
            print(
                f"iter={iteration} output_keys={list(outputs.keys())} "
                f"final_logits_shape={tuple(logits.shape)} coarse_logits_shape={tuple(outputs['aux_logits'].shape) if outputs.get('aux_logits') is not None else None} "
                f"mask_shape={tuple(lb.shape)} logits_min={float(logits.min()):.4f} logits_max={float(logits.max()):.4f} logits_mean={float(logits.mean()):.4f} "
                f"loss={last_loss:.6f} bce={float(stats.get('bce_loss', 0.0)):.6f} dice={float(stats.get('dice_loss', 0.0)):.6f} "
                f"boundary={float(stats.get('boundary_loss', 0.0)):.6f} grad_norm={grad:.6f}"
            , flush=True)
        if iteration % 50 == 0 or iteration == args.iters:
            metrics = StreamingMetrics(threshold=threshold)
            metrics.update(logits.detach(), lb.detach())
            result = metrics.compute()
            probs = torch.sigmoid(logits.detach())
            print(
                f"iter={iteration} loss={last_loss:.6f} f1={result['f1']:.4f} iou={result['iou']:.4f} "
                f"pred_pos={result['pred_positive_ratio']:.4f} gt_pos={result['gt_positive_ratio']:.4f} "
                f"mean_prob={float(probs.mean()):.4f} min_prob={float(probs.min()):.4f} max_prob={float(probs.max()):.4f}"
            , flush=True)

    _save_debug(batch, logits.detach(), REPO / args.debug_dir, threshold)
    print(f"first_loss={first_loss:.6f} last_loss={last_loss:.6f}", flush=True)
    print(f"Saved debug predictions to {REPO / args.debug_dir}", flush=True)
    if last_loss is not None and first_loss is not None and last_loss >= first_loss * 0.5:
        raise SystemExit("Tiny overfit did not reduce loss by at least 50%.")


if __name__ == "__main__":
    main()
