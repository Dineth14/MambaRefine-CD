"""Runs one forward/backward/optimizer step.

Reads: configs/active.yaml
Usage: python tools/check_training_step.py
"""
from __future__ import annotations

import argparse
import traceback
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.cd_dataset import ChangeDetectionDataset
from src.datasets.transforms import get_train_transform
from src.engine.losses import build_loss
from src.models.build import build_model
from src.utils.config import load_config
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/active.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        device = get_device(cfg)
        ds = ChangeDetectionDataset(cfg.data.root, cfg.data.train_dir, cfg, transform=get_train_transform(cfg))
        dl = DataLoader(ds, batch_size=min(args.batch_size, int(cfg.train.batch_size)), shuffle=True, num_workers=0)
        model = build_model(cfg).to(device).train()
        criterion = build_loss(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
        amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        loader_iter = iter(dl)
        for step in range(1, args.steps + 1):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dl)
                batch = next(loader_iter)
            image_a = batch["image_a"].to(device)
            image_b = batch["image_b"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(image_a, image_b)
                loss, loss_dict = criterion(outputs, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at smoke-test step {step}: {loss_dict}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        print("PASS")
        print(f"steps={args.steps}")
        print({k: round(v, 6) for k, v in loss_dict.items()})
    except Exception:
        print("FAIL")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
