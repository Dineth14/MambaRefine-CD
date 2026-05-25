#!/usr/bin/env python3
"""Profile SECOND data/compute balance without running full training."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.factory import build_dataloaders
from data.second import query_nvidia_smi_utilization
from models.cd_model import build_model
from training.losses import build_loss
from training.model_outputs import normalize_model_output
from utils.config import load_config

WARMUP_ITERS = 3
PROFILE_ITERS = 12
PROFILE_DIR = ROOT / "outputs" / "profiling"
PROFILE_JSON = PROFILE_DIR / "second_speed_profile.json"
PROFILE_CSV = PROFILE_DIR / "second_speed_profile.csv"


def _as_dict(cfg: Any) -> dict[str, Any]:
    return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)


def _parse_device(device_name: str) -> tuple[torch.device, int]:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_name)
        gpu_index = device.index if device.index is not None else 0
        torch.cuda.set_device(gpu_index)
        return device, gpu_index
    return torch.device("cpu"), 0


def _make_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    training_cfg = cfg.get("training", {})
    lr = float(training_cfg.get("lr", 5e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    return torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def _extract_batch_stat(batch: dict[str, Any], key: str, reduce: str = "max") -> float:
    value = batch.get(key)
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        tensor = value.detach().float().cpu()
        if tensor.numel() == 0:
            return 0.0
        if reduce == "mean":
            return float(tensor.mean().item())
        return float(tensor.max().item())
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _move_batch(batch: dict[str, Any], device: torch.device, non_blocking: bool) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def main() -> None:
    cfg = _as_dict(load_config())
    dataset_cfg = cfg.setdefault("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "")).upper()
    if dataset_name != "SECOND":
        raise ValueError(f"configs/global_config.yaml must target SECOND for this profiler, got {dataset_name!r}.")

    dataset_cfg["profile_enabled"] = True

    device, gpu_index = _parse_device(str(cfg.get("hardware", {}).get("device", "cpu")))
    amp_enabled = bool(cfg.get("hardware", {}).get("mixed_precision", True)) and device.type == "cuda"
    non_blocking = bool(cfg.get("training", {}).get("non_blocking_transfer", True))

    train_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = _make_optimizer(cfg, model)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    rows: list[dict[str, Any]] = []
    loader_iter = iter(train_loader)
    total_iters = WARMUP_ITERS + PROFILE_ITERS
    previous_step_end = time.perf_counter()

    for iteration in range(total_iters):
        batch_fetch_start = previous_step_end
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        batch_loading_time = time.perf_counter() - batch_fetch_start

        cpu_load_time = _extract_batch_stat(batch, "profile_cpu_load_time_ms", reduce="max") / 1000.0
        cpu_mask_time = _extract_batch_stat(batch, "profile_cpu_mask_time_ms", reduce="max") / 1000.0
        cpu_transform_time = _extract_batch_stat(batch, "profile_cpu_transform_time_ms", reduce="max") / 1000.0
        cpu_total_time = _extract_batch_stat(batch, "profile_cpu_total_time_ms", reduce="max") / 1000.0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        transfer_start = time.perf_counter()
        batch = _move_batch(batch, device, non_blocking=non_blocking)
        _sync_if_needed(device)
        gpu_transfer_time = time.perf_counter() - transfer_start

        image_a = batch["image_a"]
        image_b = batch["image_b"]
        label = batch["label"]
        ignore_mask = batch.get("ignore_mask")
        valid_mask = None
        if torch.is_tensor(ignore_mask):
            valid_mask = (ignore_mask <= 0.5).float()

        forward_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = normalize_model_output(model(image_a, image_b))
            logits = torch.clamp(outputs["change_logits"], -20.0, 20.0)
            outputs["change_logits"] = logits
            aux = outputs.get("aux_logits")
            if aux is not None:
                outputs["aux_logits"] = torch.clamp(aux, -20.0, 20.0)

            if str(cfg.get("loss", {}).get("type", "")).lower().replace("-", "_") == "second_semantic_cd":
                semantic_batch = {
                    "change_mask": batch["change_mask"],
                    "label_a": batch["label_a"],
                    "label_b": batch["label_b"],
                    "valid_mask": batch.get("valid_mask", valid_mask),
                }
                total_loss = loss_fn(outputs, semantic_batch)
            else:
                total_loss, _, _ = loss_fn(logits, label, valid_mask=valid_mask)
                if aux is not None:
                    aux_loss, _, _ = loss_fn(outputs["aux_logits"], label, valid_mask=valid_mask)
                    total_loss = total_loss + float(cfg.get("decoder", {}).get("aux_weight", 0.4)) * aux_loss
        _sync_if_needed(device)
        forward_time = time.perf_counter() - forward_start

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss encountered during profiling at iteration {iteration}.")

        backward_start = time.perf_counter()
        scaler.scale(total_loss).backward()
        _sync_if_needed(device)
        backward_time = time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        _sync_if_needed(device)
        optimizer_step_time = time.perf_counter() - optimizer_start

        peak_memory_mb = None
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

        gpu_sample = query_nvidia_smi_utilization(gpu_index) if device.type == "cuda" else None
        compute_time = forward_time + backward_time + optimizer_step_time
        bottleneck_ratio = batch_loading_time / compute_time if compute_time > 0 else math.inf

        if iteration >= WARMUP_ITERS:
            rows.append(
                {
                    "iteration": iteration - WARMUP_ITERS + 1,
                    "batch_loading_time_sec": batch_loading_time,
                    "cpu_load_time_sec": cpu_load_time,
                    "cpu_mask_time_sec": cpu_mask_time,
                    "cpu_transform_time_sec": cpu_transform_time,
                    "cpu_total_time_sec": cpu_total_time,
                    "gpu_transfer_time_sec": gpu_transfer_time,
                    "forward_time_sec": forward_time,
                    "backward_time_sec": backward_time,
                    "optimizer_step_time_sec": optimizer_step_time,
                    "gpu_compute_time_sec": compute_time,
                    "data_bottleneck_ratio": bottleneck_ratio,
                    "gpu_memory_mb": peak_memory_mb,
                    "gpu_utilization_pct": None if gpu_sample is None else gpu_sample.get("gpu_utilization"),
                    "gpu_memory_used_nvidia_smi_mb": None if gpu_sample is None else gpu_sample.get("memory_used_mb"),
                }
            )

        previous_step_end = time.perf_counter()

    average_data_loading = _mean(rows, "batch_loading_time_sec")
    average_compute = _mean(rows, "gpu_compute_time_sec")
    average_transfer = _mean(rows, "gpu_transfer_time_sec")
    average_forward = _mean(rows, "forward_time_sec")
    average_backward = _mean(rows, "backward_time_sec")
    average_optimizer = _mean(rows, "optimizer_step_time_sec")
    average_cpu_transform = _mean(rows, "cpu_transform_time_sec")
    average_cpu_load = _mean(rows, "cpu_load_time_sec")
    average_cpu_mask = _mean(rows, "cpu_mask_time_sec")
    average_cpu_total = _mean(rows, "cpu_total_time_sec")
    average_bottleneck_ratio = _mean(rows, "data_bottleneck_ratio")
    average_gpu_mem = mean(row["gpu_memory_mb"] for row in rows if row["gpu_memory_mb"] is not None) if any(row["gpu_memory_mb"] is not None for row in rows) else None
    average_gpu_util = mean(row["gpu_utilization_pct"] for row in rows if row["gpu_utilization_pct"] is not None) if any(row["gpu_utilization_pct"] is not None for row in rows) else None

    warning_message = (
        "GPU is waiting on CPU dataloader. Increase num_workers, enable mask precompute, enable persistent workers, or cache masks."
        if average_data_loading > average_compute
        else "Model compute is bottleneck."
    )

    summary = {
        "dataset": "SECOND",
        "profile_iterations": PROFILE_ITERS,
        "warmup_iterations": WARMUP_ITERS,
        "device": str(device),
        "config_snapshot": {
            "batch_size": int(cfg.get("training", {}).get("batch_size", 0)),
            "num_workers": int(dataset_cfg.get("num_workers", 0)),
            "pin_memory": bool(dataset_cfg.get("pin_memory", False)),
            "persistent_workers": bool(dataset_cfg.get("persistent_workers", False)),
            "prefetch_factor": int(dataset_cfg.get("prefetch_factor", 0)) if int(dataset_cfg.get("num_workers", 0)) > 0 else None,
            "non_blocking_transfer": non_blocking,
            "precompute_second_binary_masks": bool(dataset_cfg.get("precompute_second_binary_masks", False)),
            "cache_images_in_ram": bool(dataset_cfg.get("cache_images_in_ram", False)),
            "cache_masks_in_ram": bool(dataset_cfg.get("cache_masks_in_ram", False)),
        },
        "averages": {
            "data_loading_time_sec_per_iteration": _round_or_none(average_data_loading),
            "cpu_load_time_sec_per_iteration": _round_or_none(average_cpu_load),
            "cpu_mask_time_sec_per_iteration": _round_or_none(average_cpu_mask),
            "cpu_transform_time_sec_per_iteration": _round_or_none(average_cpu_transform),
            "cpu_total_time_sec_per_iteration": _round_or_none(average_cpu_total),
            "gpu_transfer_time_sec_per_iteration": _round_or_none(average_transfer),
            "forward_time_sec_per_iteration": _round_or_none(average_forward),
            "backward_time_sec_per_iteration": _round_or_none(average_backward),
            "optimizer_step_time_sec_per_iteration": _round_or_none(average_optimizer),
            "gpu_compute_time_sec_per_iteration": _round_or_none(average_compute),
            "data_bottleneck_ratio": _round_or_none(average_bottleneck_ratio),
            "gpu_memory_mb_peak_per_iteration": _round_or_none(average_gpu_mem, 3),
            "gpu_utilization_pct": _round_or_none(average_gpu_util, 3),
        },
        "warning": warning_message,
        "iterations": rows,
    }

    with PROFILE_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_columns = [
        "iteration",
        "batch_loading_time_sec",
        "cpu_load_time_sec",
        "cpu_mask_time_sec",
        "cpu_transform_time_sec",
        "cpu_total_time_sec",
        "gpu_transfer_time_sec",
        "forward_time_sec",
        "backward_time_sec",
        "optimizer_step_time_sec",
        "gpu_compute_time_sec",
        "data_bottleneck_ratio",
        "gpu_memory_mb",
        "gpu_utilization_pct",
        "gpu_memory_used_nvidia_smi_mb",
    ]
    with PROFILE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print("SECOND speed profile")
    print(f"Data loading time / iteration : {average_data_loading:.4f} sec")
    print(f"CPU transform time / iteration: {average_cpu_transform:.4f} sec")
    print(f"GPU transfer time / iteration : {average_transfer:.4f} sec")
    print(f"Forward time / iteration      : {average_forward:.4f} sec")
    print(f"Backward time / iteration     : {average_backward:.4f} sec")
    print(f"Optimizer step / iteration    : {average_optimizer:.4f} sec")
    print(f"GPU compute time / iteration  : {average_compute:.4f} sec")
    print(f"Data bottleneck ratio         : {average_bottleneck_ratio:.3f}")
    if average_gpu_mem is not None:
        print(f"GPU memory                    : {average_gpu_mem:.1f} MB peak")
    else:
        print("GPU memory                    : N/A")
    if average_gpu_util is not None:
        print(f"GPU utilization               : {average_gpu_util:.1f}%")
    else:
        print("GPU utilization               : N/A")
    print(warning_message)
    print(f"JSON saved to                 : {PROFILE_JSON}")
    print(f"CSV saved to                  : {PROFILE_CSV}")


if __name__ == "__main__":
    main()
