from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from datasets.transforms import ChangeTransforms
from engine.checkpoint import load_checkpoint
from models import build_model
from utils.config import apply_overrides, load_config
from utils.misc import resolve_device
from utils.visualizer import save_binary_mask, save_label_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image-a", required=True)
    parser.add_argument("--image-b", required=True)
    parser.add_argument("--out-dir", default="outputs/inference")
    args, overrides = parser.parse_known_args()
    cfg = apply_overrides(load_config(args.config), overrides)
    device = resolve_device(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(args.ckpt, model, device=device)
    tfm = ChangeTransforms(cfg["dataset"].get("image_size", 256), train=False)
    sample = tfm({
        "image_a": Image.open(args.image_a).convert("RGB"),
        "image_b": Image.open(args.image_b).convert("RGB"),
        "mask": Image.new("L", Image.open(args.image_a).size),
        "filename": Path(args.image_a).name,
    })
    with torch.no_grad():
        out = model(sample["image_a"].unsqueeze(0).to(device), sample["image_b"].unsqueeze(0).to(device))
    out_dir = Path(args.out_dir)
    if cfg["model"]["name"] == "mercon_second":
        save_binary_mask(out["binary_change_logits"][0], out_dir / "binary_change.png", cfg["eval"].get("threshold", 0.5))
        save_label_map(torch.argmax(out["semantic_t1_logits"][0], dim=0), out_dir / "semantic_t1.png")
        save_label_map(torch.argmax(out["semantic_t2_logits"][0], dim=0), out_dir / "semantic_t2.png")
        save_label_map(torch.argmax(out["semantic_change_logits"][0], dim=0), out_dir / "semantic_change.png")
    else:
        save_binary_mask(out["logits"][0], out_dir / "change_mask.png", cfg["eval"].get("threshold", 0.5))


if __name__ == "__main__":
    main()
