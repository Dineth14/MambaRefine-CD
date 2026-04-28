"""Validate SECOND dataset decoding and save visual sanity samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from data.dataset_builder import build_dataset
from utils.config import load_config
from utils.second_outputs import colorize_second


def _denorm(tensor):
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    img = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _squeeze(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _split_stats(dataset, sample_limit: int) -> dict:
    if hasattr(dataset, "debug_stats_summary"):
        return dataset.debug_stats_summary(sample_limit=sample_limit)
    raise TypeError("Configured dataset does not expose SECOND debug stats.")


def _save_samples(dataset, split: str, out_root: Path, limit: int = 10) -> None:
    dirs = {
        "image_t1": out_root / split / "image_t1",
        "image_t2": out_root / split / "image_t2",
        "label_t1_color": out_root / split / "label_t1_color",
        "label_t2_color": out_root / split / "label_t2_color",
        "change_mask": out_root / split / "change_mask",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    for idx in range(min(limit, len(dataset))):
        item = dataset[idx]
        sample_id = Path(str(item.get("sample_id", item.get("id", idx)))).stem
        Image.fromarray(_denorm(item["image_t1"])).save(dirs["image_t1"] / f"{idx:03d}_{sample_id}.png")
        Image.fromarray(_denorm(item["image_t2"])).save(dirs["image_t2"] / f"{idx:03d}_{sample_id}.png")
        Image.fromarray(colorize_second(_squeeze(item["label_t1"]))).save(dirs["label_t1_color"] / f"{idx:03d}_{sample_id}.png")
        Image.fromarray(colorize_second(_squeeze(item["label_t2"]))).save(dirs["label_t2_color"] / f"{idx:03d}_{sample_id}.png")
        change = (_squeeze(item["change_mask"]) > 0).astype(np.uint8) * 255
        Image.fromarray(change).save(dirs["change_mask"] / f"{idx:03d}_{sample_id}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SECOND semantic dataset decoding.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="outputs/second_dataset_debug")
    parser.add_argument("--sample_limit", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("dataset", {})["mode"] = "semantic"
    cfg["dataset"]["debug_stats"] = False
    out_root = Path(args.out_dir)
    manifest = {"config": args.config, "splits": {}}

    for split in ("train", "val", "test"):
        ds = build_dataset(cfg.get("dataset", {}), split=split, augment=False, seed=int(cfg.get("experiment", {}).get("seed", 42)))
        stats = _split_stats(ds, args.sample_limit)
        manifest["splits"][split] = stats
        print(json.dumps(stats, indent=2))
        _save_samples(ds, split, out_root, limit=10)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "second_dataset_stats.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved SECOND dataset debug samples to: {out_root}")


if __name__ == "__main__":
    main()
