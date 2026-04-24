"""Standalone validation / inference script.

Runs a full validation pass on a saved checkpoint and prints a metric table.

Usage:
    python scripts/validate.py \
        --config configs/refinement_decoder.yaml \
        --checkpoint outputs/run_XXXX/checkpoints/best.pth
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import torch

from utils.config_loader import load_config
from data.factory        import build_dataloaders
from models.cd_model     import build_model
from training.metrics    import StreamingMetrics
from training.logger     import get_logger, log_table
from training.checkpoint import peek as peek_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split",      default="val", choices=["val", "test"])
    args = ap.parse_args()

    cfg    = load_config(ROOT / args.config)
    logger = get_logger("validate", ROOT / "outputs" / "validate_logs")

    device_str = cfg.get("hardware", {}).get("device", "cuda")
    device     = torch.device(device_str if torch.cuda.is_available() else "cpu")

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (ROOT / ckpt_path).resolve()

    ckpt = peek_ckpt(ckpt_path)
    model.load_state_dict(ckpt["model"])
    logger.info(f"Loaded checkpoint: {ckpt_path}")
    logger.info(f"Checkpoint iter={ckpt.get('iteration')}  best={ckpt.get('best_metric', 0):.4f}")

    model.eval()
    metrics = StreamingMetrics()
    with torch.no_grad():
        try:
            from tqdm import tqdm
            loader = tqdm(val_loader, desc=f"Validate ({args.split})")
        except ImportError:
            loader = val_loader

        for batch in loader:
            ia = batch["image_a"].to(device)
            ib = batch["image_b"].to(device)
            lb = batch["label"].to(device)
            logits, _ = model(ia, ib)
            metrics.update(logits, lb)

    log_table(logger, metrics.compute(), title="── Final Validation Results ──")


if __name__ == "__main__":
    main()
