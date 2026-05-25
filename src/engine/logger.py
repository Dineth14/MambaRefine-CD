"""Simple console/file/TensorBoard logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, run_dir: Path, filename: str = "train.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    file_handler = logging.FileHandler(Path(run_dir) / filename)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def make_writer(cfg, run_dir: Path):
    if not bool(cfg.logging.tensorboard):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(Path(run_dir) / "tensorboard"))
    except Exception as exc:
        print(f"WARNING: TensorBoard disabled: {exc}")
        return None
