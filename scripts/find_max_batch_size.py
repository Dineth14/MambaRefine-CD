#!/usr/bin/env python3
"""Find a memory-safe synthetic training batch size for one config."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch

from models.mambarefinecd import build_model
from training.losses import build_loss
from training.model_outputs import normalize_model_output
from utils.config import load_config
from utils.memory import peak_memory_gb, reset_peak_memory
from utils.precision import amp_dtype_from_config, channels_last_enabled, maybe_channels_last_image


FIELDS = ["batch_size", "image_size", "success", "peak_mem_GB", "notes"]


def _is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _try_batch(cfg: dict, batch_size: int, image_size: int, device: torch.device) -> dict:
    row = {"batch_size": batch_size, "image_size": image_size, "success": "false", "peak_mem_GB": "", "notes": ""}
    try:
        reset_peak_memory(device)
        model = build_model(cfg).to(device).train()
        if channels_last_enabled(cfg):
            model = model.to(memory_format=torch.channels_last)
        loss_fn = build_loss(cfg)
        optimizer = getattr(torch.optim, cfg.get("training", {}).get("optimizer", "AdamW"))(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(cfg.get("training", {}).get("lr", 5e-5)),
            weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.01)),
        )
        amp = bool(cfg.get("hardware", {}).get("mixed_precision", True)) and device.type == "cuda"
        amp_dtype = amp_dtype_from_config(cfg, device)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        x1 = torch.randn(batch_size, 3, image_size, image_size, device=device)
        x2 = torch.randn(batch_size, 3, image_size, image_size, device=device)
        x1 = maybe_channels_last_image(x1, channels_last_enabled(cfg))
        x2 = maybe_channels_last_image(x2, channels_last_enabled(cfg))
        y = torch.randint(0, 2, (batch_size, 1, image_size, image_size), device=device).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
            outputs = normalize_model_output(model(x1, x2))
            logits = torch.clamp(outputs["change_logits"], -20.0, 20.0)
            loss, _, _ = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row["success"] = "true"
        row["peak_mem_GB"] = f"{peak_memory_gb(device):.4f}"
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not _is_oom(exc):
            raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Find max successful synthetic batch size.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start_batch", type=int, default=2)
    parser.add_argument("--max_batch", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--out", default="outputs/max_batch_size_report.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("model", {})["pretrained"] = False
    cfg.setdefault("post_training", {})["run_test_eval"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    max_ok = None
    for batch_size in range(int(args.start_batch), int(args.max_batch) + 1):
        row = _try_batch(cfg, batch_size, int(args.image_size), device)
        rows.append(row)
        print(f"batch={batch_size}: success={row['success']} peak_mem_GB={row['peak_mem_GB']} notes={row['notes']}")
        if row["success"] == "true":
            max_ok = batch_size
        else:
            break
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Max successful batch size: {max_ok}")
    print(f"Saved report CSV: {out}")


if __name__ == "__main__":
    main()
