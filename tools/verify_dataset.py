"""Runs full dataset verification and prints report.

Reads: configs/active.yaml
Usage: python tools/verify_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.verify import verify_dataset
from src.utils.config import load_config


def main() -> None:
    cfg = load_config()
    verify_dataset(cfg)
    report = getattr(cfg, "_dataset_verification", {})
    print("Dataset verification: PASS")
    for split, info in report.items():
        print(f"{split}: {info['count']} samples")


if __name__ == "__main__":
    main()
