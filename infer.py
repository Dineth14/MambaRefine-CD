"""Entry point for inference on paired image folders.

Reads: configs/active.yaml
Usage: python infer.py
"""
from pathlib import Path

from PIL import Image

from src.engine.checkpoint import find_latest_best, load_checkpoint
from src.engine.inference import predict_pair
from src.models.build import build_model
from src.utils.config import load_config
from src.utils.device import get_device


if __name__ == "__main__":
    cfg = load_config()
    device = get_device(cfg)
    model = build_model(cfg).to(device)
    ckpt_path = cfg.checkpoint.path or find_latest_best(cfg.project.output_root)
    ckpt = load_checkpoint(ckpt_path, model)
    threshold = float(ckpt.get("best_threshold", cfg.eval.threshold))
    root = Path(cfg.data.root) / cfg.data.test_dir
    a_dir = root / cfg.data.a_folder
    b_dir = root / cfg.data.b_folder
    out_dir = (Path(ckpt_path).parents[1] if ckpt_path else Path(cfg.project.output_root)) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    b_by_stem = {p.stem: p for p in b_dir.iterdir() if p.is_file()}
    for a_path in sorted(p for p in a_dir.iterdir() if p.is_file()):
        b_path = b_by_stem.get(a_path.stem)
        if b_path is None:
            continue
        pred = predict_pair(model, a_path, b_path, cfg, device, threshold)
        Image.fromarray((pred.numpy() * 255).astype("uint8")).save(out_dir / f"{a_path.stem}.png")
    print(f"Predictions saved to: {out_dir}")
