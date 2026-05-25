"""Runs one forward/backward/optimizer step.

Reads: configs/active.yaml
Usage: python tools/check_training_step.py
"""
from __future__ import annotations

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
    cfg = load_config()
    try:
        device = get_device(cfg)
        ds = ChangeDetectionDataset(cfg.data.root, cfg.data.train_dir, cfg, transform=get_train_transform(cfg))
        dl = DataLoader(ds, batch_size=min(2, int(cfg.train.batch_size)), shuffle=True, num_workers=0)
        batch = next(iter(dl))
        model = build_model(cfg).to(device).train()
        criterion = build_loss(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
        image_a = batch["image_a"].to(device)
        image_b = batch["image_b"].to(device)
        mask = batch["mask"].to(device)
        outputs = model(image_a, image_b)
        loss, loss_dict = criterion(outputs, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print("PASS")
        print({k: round(v, 6) for k, v in loss_dict.items()})
    except Exception:
        print("FAIL")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
