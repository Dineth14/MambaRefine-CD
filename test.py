"""Entry point for testing.

Reads: configs/active.yaml
Usage: python test.py
"""
import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.cd_dataset import ChangeDetectionDataset
from src.datasets.transforms import get_val_transform
from src.engine.checkpoint import find_latest_best, load_checkpoint
from src.engine.evaluator import evaluate
from src.models.build import build_model
from src.utils.config import load_config
from src.utils.device import get_device


if __name__ == "__main__":
    cfg = load_config()
    device = get_device(cfg)
    model = build_model(cfg).to(device)
    ckpt_path = cfg.checkpoint.path or find_latest_best(cfg.project.output_root)
    ckpt = load_checkpoint(ckpt_path, model)
    threshold = float(ckpt.get("best_threshold", cfg.eval.threshold)) if cfg.eval.use_val_threshold_for_test else float(cfg.eval.threshold)
    ds = ChangeDetectionDataset(cfg.data.root, cfg.data.test_dir, cfg, transform=get_val_transform(cfg))
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    out_dir = Path(ckpt_path).parents[1] if ckpt_path else Path(cfg.project.output_root)
    metrics = evaluate(model, dl, cfg, split="test", threshold=threshold, device=device, save_dir=out_dir)
    metrics["best_val_threshold"] = float(ckpt.get("best_threshold", threshold))
    metrics["test_threshold_used"] = threshold
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "test.log").write_text(str(metrics) + "\n", encoding="utf-8")
    print(metrics)
