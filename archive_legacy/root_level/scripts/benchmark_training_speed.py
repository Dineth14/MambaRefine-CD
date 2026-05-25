#!/usr/bin/env python3
"""Short controlled training-speed benchmark for configs."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch

from data.factory import build_dataloaders
from models.mambarefinecd import build_model
from training.losses import build_loss
from training.model_outputs import normalize_model_output
from utils.config import load_config
from utils.memory import params_m, peak_memory_gb, reset_peak_memory
from utils.precision import amp_dtype_from_config, channels_last_enabled, maybe_channels_last_image


FIELDS = [
    "config",
    "batch_size",
    "image_size",
    "amp",
    "channels_last",
    "checkpointing",
    "compile",
    "fast_mode",
    "avg_iter_time_ms",
    "samples_per_sec",
    "peak_mem_GB",
    "params_M",
    "status",
    "notes",
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_model(cfg: dict, device: torch.device):
    model = build_model(cfg).to(device)
    if channels_last_enabled(cfg):
        model = model.to(memory_format=torch.channels_last)
    if bool(cfg.get("efficiency", {}).get("compile", False)) and hasattr(torch, "compile"):
        mode = str(cfg.get("efficiency", {}).get("compile_mode", "reduce-overhead"))
        model = torch.compile(model, mode=mode)
    return model


def _run_one(config_path: Path, args) -> dict:
    cfg = load_config(config_path)
    cfg.setdefault("model", {})["pretrained"] = False
    if args.batch_size_override is not None:
        cfg.setdefault("training", {})["batch_size"] = int(args.batch_size_override)
    cfg.setdefault("checkpoint", {})["save_latest"] = False
    cfg["checkpoint"]["save_best"] = False
    cfg["checkpoint"]["save_last"] = False
    cfg["checkpoint"]["save_every"] = None
    cfg.setdefault("post_training", {})["run_test_eval"] = False
    cfg.setdefault("profiling", {})["enabled"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row = {
        "config": str(config_path),
        "batch_size": cfg.get("training", {}).get("batch_size", ""),
        "image_size": cfg.get("dataset", {}).get("image_size", ""),
        "amp": cfg.get("efficiency", {}).get("amp", cfg.get("hardware", {}).get("mixed_precision", "")),
        "channels_last": cfg.get("efficiency", {}).get("channels_last", False),
        "checkpointing": cfg.get("efficiency", {}).get("gradient_checkpointing", False),
        "compile": cfg.get("efficiency", {}).get("compile", False),
        "fast_mode": cfg.get("efficiency", {}).get("fast_mode", False),
        "avg_iter_time_ms": "",
        "samples_per_sec": "",
        "peak_mem_GB": "",
        "params_M": "",
        "status": "FAIL",
        "notes": "",
    }
    try:
        train_loader, _ = build_dataloaders(cfg)
        loader_iter = iter(train_loader)
        model = _prepare_model(cfg, device).train()
        row["params_M"] = f"{params_m(model):.4f}"
        loss_fn = build_loss(cfg)
        optimizer = getattr(torch.optim, cfg.get("training", {}).get("optimizer", "AdamW"))(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(cfg.get("training", {}).get("lr", 5e-5)),
            weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.01)),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("hardware", {}).get("mixed_precision", True)) and device.type == "cuda")
        amp_dtype = amp_dtype_from_config(cfg, device)
        channels_last = channels_last_enabled(cfg)
        reset_peak_memory(device)
        times: list[float] = []
        total = int(args.warmup) + int(args.iters)
        for idx in range(total):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            t0 = time.perf_counter()
            ia = maybe_channels_last_image(batch["image_a"].to(device, non_blocking=True), channels_last)
            ib = maybe_channels_last_image(batch["image_b"].to(device, non_blocking=True), channels_last)
            lb = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg.get("hardware", {}).get("mixed_precision", True)) and device.type == "cuda", dtype=amp_dtype):
                outputs = normalize_model_output(model(ia, ib))
                logits = torch.clamp(outputs["change_logits"], -20.0, 20.0)
                loss, _, _ = loss_fn(logits, lb)
                aux = outputs.get("aux_logits")
                if aux is not None:
                    aux_loss, _, _ = loss_fn(torch.clamp(aux, -20.0, 20.0), lb)
                    loss = loss + float(cfg.get("decoder", {}).get("aux_weight", 0.4)) * aux_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at iter {idx + 1}: {float(loss.detach().item())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_clip = float(cfg.get("training", {}).get("gradient_clip", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            _sync(device)
            if idx >= int(args.warmup):
                times.append((time.perf_counter() - t0) * 1000.0)
        avg_ms = sum(times) / max(len(times), 1)
        batch_size = int(cfg.get("training", {}).get("batch_size", 1))
        row["avg_iter_time_ms"] = f"{avg_ms:.4f}"
        row["samples_per_sec"] = f"{batch_size / max(avg_ms / 1000.0, 1e-9):.4f}"
        row["peak_mem_GB"] = f"{peak_memory_gb(device):.4f}"
        row["status"] = "ok"
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark short training speed for configs.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch_size_override", type=int, default=None)
    parser.add_argument("--out", default="outputs/training_speed_benchmark.csv")
    args = parser.parse_args()

    rows = [_run_one((REPO / p).resolve() if not Path(p).is_absolute() else Path(p), args) for p in args.configs]
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['config']}: {row['status']} avg_ms={row['avg_iter_time_ms']} mem={row['peak_mem_GB']} notes={row['notes']}")
    print(f"Saved benchmark CSV: {out}")


if __name__ == "__main__":
    main()
