"""Training entry point.

By default this loads configs/global_config.yaml. For day-to-day training use
one of the dataset-specific configs under configs/train via --dataset or
provide a custom config path with --config.

Usage:
    conda activate mamba_new
    cd MambaRefine-CD
    python scripts/train.py --dataset levir
    python scripts/train.py --dataset second
    python scripts/train.py --config configs/train/whu_cd.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.pipeline import run_training_pipeline
from utils.config import TRAIN_CONFIG_DIR, load_config


TRAIN_CONFIG_MAP = {
    "levir": TRAIN_CONFIG_DIR / "levir_cd.yaml",
    "levir-cd": TRAIN_CONFIG_DIR / "levir_cd.yaml",
    "whu": TRAIN_CONFIG_DIR / "whu_cd.yaml",
    "whu-cd": TRAIN_CONFIG_DIR / "whu_cd.yaml",
    "dsifn": TRAIN_CONFIG_DIR / "dsifn_cd.yaml",
    "dsifn-cd": TRAIN_CONFIG_DIR / "dsifn_cd.yaml",
    "second": TRAIN_CONFIG_DIR / "second_semantic.yaml",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MambaRefine-CD with dataset-specific config selection.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset preset to train: levir, whu, dsifn, or second.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a training config override YAML. Merged on top of configs/global_config.yaml.",
    )
    args = parser.parse_args()
    if args.dataset and args.config:
        parser.error("Use either --dataset or --config, not both.")
    return args


def _resolve_train_config(args: argparse.Namespace) -> Path | None:
    if args.config:
        return Path(args.config)
    if args.dataset:
        key = args.dataset.strip().lower()
        if key not in TRAIN_CONFIG_MAP:
            valid = ", ".join(sorted({k for k in TRAIN_CONFIG_MAP if "-" not in k}))
            raise ValueError(f"Unknown --dataset value {args.dataset!r}. Valid options: {valid}.")
        return TRAIN_CONFIG_MAP[key]
    return None

def main() -> None:
    args = _parse_args()
    config_path = _resolve_train_config(args)
    cfg = load_config(config_path)
    resolved_source = None
    if config_path is not None:
        resolved_source = config_path if config_path.is_absolute() else (ROOT / config_path).resolve()
    run_training_pipeline(cfg, config_source_path=resolved_source)


if __name__ == "__main__":
    main()
