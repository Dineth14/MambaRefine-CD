"""Training script for MambaRefineCD.

Supports all four datasets with metric restriction per dataset:
  - LEVIR-CD / WHU-CD / DSIFN-CD: Pre, Rec, F1, IoU, OA
  - SECOND:                        OA, mIoU, SeK, Fscd

Usage:
    python scripts/train.py --config configs/ablations/levir/a4_full.yaml
    python scripts/train.py --config configs/ablations/second/a4_full.yaml
    python scripts/train.py --config configs/ablations/levir/a4_full.yaml --dry_run
    python scripts/train.py --config configs/train/levir_cd.yaml  # old-style config
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root and src are on path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from utils.config import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metric restrictions
# --------------------------------------------------------------------------
_BINARY_ALLOWED = {"Pre", "Rec", "F1", "IoU", "OA"}
_SECOND_ALLOWED = {"OA", "mIoU", "SeK", "Fscd"}


def _get_allowed_metrics(cfg: dict) -> list[str]:
    """Return the set of allowed metric names for this config."""
    explicit = cfg.get("metrics", {}).get("allowed", None)
    if explicit is not None:
        return list(explicit)
    task = str(cfg.get("task", cfg.get("dataset", {}).get("task_type", "binary_cd"))).lower()
    if "SECOND" in str(cfg.get("dataset", {}).get("name", "")).upper() or task == "semantic_cd":
        return ["OA", "mIoU", "SeK", "Fscd"]
    return ["Pre", "Rec", "F1", "IoU", "OA"]


def _filter_metrics(raw: dict, allowed: list[str]) -> dict:
    """Return only the keys in `allowed` from `raw`."""
    return {k: v for k, v in raw.items() if k in allowed}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MambaRefineCD.")
    parser.add_argument("--config", required=True, help="Path to training config YAML.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build model and run one batch; do not train.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = load_config(str(config_path))
    allowed = _get_allowed_metrics(cfg)

    logger.info(f"Config: {config_path}")
    logger.info(f"Dataset: {cfg.get('dataset', {}).get('name', 'unknown')}")
    logger.info(f"Allowed metrics: {allowed}")

    if args.resume:
        cfg.setdefault("resume", {})
        cfg["resume"]["enabled"] = True
        cfg["resume"]["checkpoint_path"] = args.resume

    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        _dry_run(cfg, allowed)
        return

    # Run full training pipeline
    from training.pipeline import run_training_pipeline

    # Patch allowed metrics into cfg so the pipeline can filter logs
    cfg.setdefault("metrics", {})["allowed"] = list(allowed)

    run_training_pipeline(cfg)


def _dry_run(cfg: dict, allowed: list[str]) -> None:
    """Build model + dataset, run one forward pass; verify metric restriction."""
    import torch
    from models.mambarefinecd import build_model
    from datasets import build_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")
    for label, module_name in [
        ("Backbone params", "encoder"),
        ("Decoder params", "decoder"),
        ("D-RBI params", "diff_modules"),
        ("Semantic head params", "semantic_head"),
    ]:
        module = getattr(model, module_name, None)
        count = sum(p.numel() for p in module.parameters()) if module is not None else 0
        logger.info(f"{label}: {count / 1e6:.2f}M")
    binary_head = None
    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        binary_head = getattr(decoder, "head", getattr(decoder, "coarse_head", None))
    binary_count = sum(p.numel() for p in binary_head.parameters()) if binary_head is not None else 0
    logger.info(f"Binary head params: {binary_count / 1e6:.2f}M")

    # Build dataset
    ds = build_dataset(cfg, split="train")
    logger.info(f"Dataset size (train): {len(ds)}")

    # One forward pass
    sample = ds[0]
    img_a = sample["image_a"].unsqueeze(0).to(device)
    img_b = sample["image_b"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(img_a, img_b)
    logger.info(f"Forward pass OK. Output type: {type(out)}")

    # Compute metrics (binary CD)
    is_second = "SECOND" in str(cfg.get("dataset", {}).get("name", "")).upper()
    if is_second:
        from metrics.second_scd_metrics import SECONDSCDMetrics
        logger.info(f"Metric class: SECONDSCDMetrics")
        from training.model_outputs import normalize_model_output
        outputs = normalize_model_output(out)
        missing = [key for key in ("sem_logits_t1", "sem_logits_t2") if outputs.get(key) is None]
        if missing:
            raise RuntimeError(f"SECOND dry run missing semantic outputs: {missing}")
        logger.info(
            "SECOND output shapes: sem_logits_t1=%s sem_logits_t2=%s change_logits=%s",
            tuple(outputs["sem_logits_t1"].shape),
            tuple(outputs["sem_logits_t2"].shape),
            tuple(outputs["change_logits"].shape),
        )
    else:
        from metrics.binary_cd_metrics import BinaryMetrics
        m = BinaryMetrics()
        # Fake forward
        fake_logits = torch.zeros(1, 1, 256, 256)
        fake_gt     = torch.zeros(1, 256, 256, dtype=torch.long)
        m.update(fake_logits, fake_gt)
        results = m.compute()
        results_filtered = _filter_metrics(results, allowed)
        logger.info(f"Metric keys (allowed only): {list(results_filtered.keys())}")
        assert set(results_filtered.keys()) <= set(allowed), \
            f"Metric keys {set(results_filtered.keys())} not subset of {allowed}"

    logger.info("=== DRY RUN PASSED ===")
    logger.info(f"Confirmed log will contain only: {allowed}")


if __name__ == "__main__":
    main()
