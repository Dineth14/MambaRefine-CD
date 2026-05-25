"""Entry point for validation.

Reads: configs/active.yaml
Usage: python val.py
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.cd_dataset import ChangeDetectionDataset
from src.datasets.transforms import get_val_transform
from src.engine.checkpoint import find_latest_best, load_checkpoint
from src.engine.evaluator import evaluate
from src.models.build import build_model
from src.utils.config import load_config
from src.utils.device import get_device


if __name__ == "__main__":
    cfg = load_config()
    device = get_device(cfg)
    model = build_model(cfg).to(device)
    ckpt_path = cfg.checkpoint.path or find_latest_best(cfg.project.output_root)
    ckpt = load_checkpoint(ckpt_path, model)
    ds = ChangeDetectionDataset(cfg.data.root, cfg.data.val_dir, cfg, transform=get_val_transform(cfg))
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    metrics = evaluate(
        model,
        dl,
        cfg,
        split="val",
        sweep_thresholds=cfg.eval.sweep_thresholds_on_val,
        device=device,
    )
    out_dir = Path(ckpt_path).parents[1] if ckpt_path else Path(cfg.project.output_root)
    (out_dir / "val.log").write_text(str(metrics) + "\n", encoding="utf-8")
    print(metrics)
