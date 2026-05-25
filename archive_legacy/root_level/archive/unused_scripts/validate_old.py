"""Standalone validation / inference script.

Runs a full validation pass from configs/global_config.yaml.
No CLI arguments.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from utils.config import load_config
from data.factory        import build_dataloaders
from data.dataset_builder import build_test_loader
from models.cd_model     import build_model
from training.evaluator  import Evaluator
from training.logger     import get_logger, log_table
from training.checkpoint import peek as peek_ckpt


def main() -> None:
    cfg    = load_config()
    logger = get_logger("validate", ROOT / "outputs" / "validate_logs")

    device_str = cfg.hardware.device
    device     = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    ckpt_path = Path(cfg.checkpoint.path)
    if not ckpt_path.is_absolute():
        ckpt_path = (ROOT / ckpt_path).resolve()

    ckpt = peek_ckpt(ckpt_path)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    logger.info(f"Checkpoint iter={ckpt.get('iteration')}  best={ckpt.get('best_metric', 0):.4f}")

    split = str(cfg.validation.split)
    loader = build_test_loader(cfg) if split == "test" else val_loader
    evaluator = Evaluator(cfg, device, logger=logger)
    results = evaluator.evaluate(model, loader, dataset_name=str(cfg.dataset.name), amp=bool(cfg.hardware.mixed_precision))
    evaluator.print_table(results, title="VALIDATION RESULTS")
    log_table(logger, results, title="── Final Validation Results ──")


if __name__ == "__main__":
    main()
