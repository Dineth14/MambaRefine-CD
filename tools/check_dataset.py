"""Verifies dataset and prints sample counts.

Reads: configs/active.yaml
Usage: python tools/check_dataset.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.cd_dataset import ChangeDetectionDataset
from src.datasets.transforms import get_val_transform
from src.datasets.verify import verify_dataset
from src.utils.config import load_config


def main() -> None:
    cfg = load_config()
    verify_dataset(cfg)
    print("Dataset verification: PASS")
    for split in (cfg.data.train_dir, cfg.data.val_dir, cfg.data.test_dir):
        ds = ChangeDetectionDataset(cfg.data.root, split, cfg, transform=get_val_transform(cfg))
        names = [sample[3] for sample in ds.samples]
        sample_names = random.sample(names, min(5, len(names))) if names else []
        print(f"{split}: {len(ds)} samples")
        print(f"  examples: {sample_names}")


if __name__ == "__main__":
    main()
