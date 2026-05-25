"""Count trainable parameters and FLOPs for MambaRefineCD.

Requires `fvcore` (pip install fvcore).
Falls back to a manual FLOPs estimate if fvcore is unavailable.

Usage:
    python scripts/count_params_flops.py --config configs/experiments/dsifn_full.yaml
    python scripts/count_params_flops.py --config configs/models/mambarefinecd_full.yaml \\
                                          --image_size 256
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import torch
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _count_with_fvcore(model: torch.nn.Module, img_size: int) -> dict[str, float]:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    dummy_a = torch.zeros(1, 3, img_size, img_size)
    dummy_b = torch.zeros(1, 3, img_size, img_size)
    flops   = FlopCountAnalysis(model, (dummy_a, dummy_b))
    flops.unsupported_ops_settings().run_on_unsupported = True
    flops.uncalled_modules_settings().run_on_uncalled   = True
    flops_count = flops.total()
    params      = parameter_count(model)[""]
    return {"params_M": params / 1e6, "flops_G": flops_count / 1e9}


def _count_params_only(model: torch.nn.Module) -> dict[str, float]:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_M": params / 1e6, "flops_G": float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Count params and FLOPs.")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--save",       type=str, default=None,
                        help="Optional path to save params_flops.json")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    image_size = int(cfg.get("dataset", {}).get("image_size", args.image_size))

    from models.mambarefinecd import build_model
    model = build_model(cfg).cpu().eval()

    try:
        results = _count_with_fvcore(model, image_size)
        method  = "fvcore"
    except ImportError:
        logger.warning("fvcore not installed — counting params only (no FLOPs estimate).")
        results = _count_params_only(model)
        method  = "manual"

    results["image_size"] = image_size
    results["method"]     = method

    logger.info("=" * 50)
    logger.info(f"Params : {results['params_M']:.2f} M")
    if not (isinstance(results["flops_G"], float) and results["flops_G"] != results["flops_G"]):
        logger.info(f"FLOPs  : {results['flops_G']:.2f} G")
    logger.info("=" * 50)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved to {save_path}")
    else:
        out_dir = Path(cfg.get("experiment", {}).get("output_root", "outputs/profile"))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "params_flops.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved to {out_dir / 'params_flops.json'}")


if __name__ == "__main__":
    main()
