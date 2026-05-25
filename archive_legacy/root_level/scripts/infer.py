"""Inference script — run change detection on an image pair or a folder.

No metrics are computed if ground truth is not provided.
Outputs change probability maps (PNG) and optionally binary masks.

Usage:
    # Single pair
    python scripts/infer.py --config configs/experiments/dsifn_full.yaml \\
                             --ckpt <checkpoint> \\
                             --img_a /path/to/t1.png --img_b /path/to/t2.png

    # Folder of pairs (expects subfolders A/ and B/ with matching filenames)
    python scripts/infer.py --config configs/experiments/whu_full.yaml \\
                             --ckpt <checkpoint> \\
                             --folder /path/to/pairs/ --out /path/to/output/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image as PILImage

from utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_IMG_MEAN = [0.485, 0.456, 0.406]
_IMG_STD  = [0.229, 0.224, 0.225]


def _load_image(path: str | Path) -> torch.Tensor:
    """Load RGB image → normalised float tensor [1, 3, H, W]."""
    img = PILImage.open(path).convert("RGB")
    t   = TF.to_tensor(img)
    t   = TF.normalize(t, _IMG_MEAN, _IMG_STD)
    return t.unsqueeze(0)


def _run_pair(model: torch.nn.Module, path_a: Path, path_b: Path,
              device: torch.device, threshold: float):
    """Run one pair."""
    from training.model_outputs import normalize_model_output

    with torch.no_grad():
        img_a = _load_image(path_a).to(device)
        img_b = _load_image(path_b).to(device)
        out   = model(img_a, img_b)
        outputs = normalize_model_output(out)
        logits = outputs["change_logits"]
        prob = torch.sigmoid(logits[0, 0]).cpu().numpy()

    prob_u8   = (prob * 255).astype(np.uint8)
    binary_u8 = ((prob > threshold) * 255).astype(np.uint8)
    return prob_u8, binary_u8


def main() -> None:
    parser = argparse.ArgumentParser(description="MambaRefineCD inference.")
    parser.add_argument("--config",    required=True)
    parser.add_argument("--ckpt",      required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    # Single pair
    parser.add_argument("--img_a", type=str, default=None)
    parser.add_argument("--img_b", type=str, default=None)
    # Folder mode
    parser.add_argument("--folder", type=str, default=None,
                        help="Root folder with subdirs A/ and B/ containing matching files.")
    parser.add_argument("--out",    type=str, default=None,
                        help="Output directory for predictions.")
    args = parser.parse_args()

    if args.img_a is None and args.folder is None:
        parser.error("Provide either --img_a / --img_b or --folder.")

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.mambarefinecd import build_model
    from training.checkpoint import load as load_ckpt

    model = build_model(cfg).to(device)
    load_ckpt(args.ckpt, model, map_location=device)
    model.eval()
    logger.info(f"Loaded model from {args.ckpt}")

    out_root = Path(args.out) if args.out else _REPO / "outputs" / "infer"

    if args.img_a is not None:
        # Single pair
        result = _run_pair(model, Path(args.img_a), Path(args.img_b), device, args.threshold)
        out_root.mkdir(parents=True, exist_ok=True)
        stem = Path(args.img_a).stem
        prob, binary = result
        PILImage.fromarray(prob).save(out_root / f"{stem}_prob.png")
        PILImage.fromarray(binary).save(out_root / f"{stem}_binary.png")
        logger.info(f"Saved to {out_root}")
    else:
        # Folder mode
        folder = Path(args.folder)
        a_dir  = folder / "A"
        b_dir  = folder / "B"
        if not a_dir.is_dir() or not b_dir.is_dir():
            logger.error(f"Expected subfolders A/ and B/ under {folder}")
            sys.exit(1)

        prob_dir   = out_root / "prob"
        binary_dir = out_root / "binary"
        prob_dir.mkdir(parents=True, exist_ok=True)
        binary_dir.mkdir(parents=True, exist_ok=True)

        pairs = sorted(a_dir.glob("*"))
        logger.info(f"Found {len(pairs)} pairs")

        for path_a in pairs:
            path_b = b_dir / path_a.name
            if not path_b.exists():
                logger.warning(f"No matching B for {path_a.name}, skipping.")
                continue
            result = _run_pair(model, path_a, path_b, device, args.threshold)
            prob, binary = result
            PILImage.fromarray(prob).save(prob_dir / path_a.name)
            PILImage.fromarray(binary).save(binary_dir / path_a.name)

        logger.info(f"Inference complete. Results in {out_root}")


if __name__ == "__main__":
    main()
