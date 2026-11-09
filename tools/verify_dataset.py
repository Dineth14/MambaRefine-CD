"""Runs full dataset verification and prints report.

Reads: configs/active.yaml
Usage: python tools/verify_dataset.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.verify import verify_dataset
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/active.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    verify_dataset(cfg)
    report = getattr(cfg, "_dataset_verification", [])
    print("Dataset verification: PASS")
    for info in report:
        print(
            f"{info['logical_split']} ({info['split']}): {info['count']} samples, "
            f"positive ratio={info['gt_positive_ratio']:.6f}"
        )


if __name__ == "__main__":
    main()
