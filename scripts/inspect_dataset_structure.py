#!/usr/bin/env python3
"""Dataset structure inspector for LEVIR-CD (and other CD datasets).

Reads dataset.root from configs/global_config.yaml.
Prints a summary and saves:
    outputs/dataset_inspection/inspection_report.json

No CLI arguments required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
import numpy as np

# ── Load config ──────────────────────────────────────────────────────────────
from utils.config import load_config
from data.second import inspect_second_dataset
cfg = load_config()
dc = cfg.get("dataset", {})
data_root = Path(str(dc.get("root", "")))
dataset_name = str(dc.get("name", "unknown"))

out_dir = ROOT / "outputs" / "dataset_inspection"
out_dir.mkdir(parents=True, exist_ok=True)

if dataset_name.upper() == "SECOND":
    manifest = inspect_second_dataset(dc)
    report_path = out_dir / "inspection_report.json"
    manifest_path = ROOT / "outputs" / "dataset_manifests" / "SECOND_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDataset root : {data_root}")
    print(f"Dataset      : {dataset_name}")
    print(f"Exists       : {data_root.exists()}\n")
    for split, info in manifest.get("splits", {}).items():
        print(f"[{split.upper()}]")
        print(f"  image A dir      : {info.get('detected_image_a_dir')}")
        print(f"  image B dir      : {info.get('detected_image_b_dir')}")
        print(f"  label A dir      : {info.get('detected_label_a_dir')}")
        print(f"  label B dir      : {info.get('detected_label_b_dir')}")
        print(f"  binary mask dir  : {info.get('detected_binary_mask_dir')}")
        print(f"  change ratio     : {info.get('change_pixel_ratio')}")
        print(f"  ignore ratio     : {info.get('ignore_pixel_ratio')}")
        print(f"  class IDs A      : {info.get('class_ids_label_a')}")
        print(f"  class IDs B      : {info.get('class_ids_label_b')}")
        if info.get("error"):
            print(f"  error            : {info.get('error')}")
        print()
    print(f"Report saved to: {report_path}")
    print(f"Manifest saved to: {manifest_path}\n")
    raise SystemExit(0)

# ── Helpers ───────────────────────────────────────────────────────────────────
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def find_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)

def sample_sizes(imgs: list[Path], n: int = 5) -> list[dict]:
    sizes = []
    for p in imgs[:n]:
        try:
            w, h = Image.open(p).size
            sizes.append({"file": p.name, "width": w, "height": h})
        except Exception as e:
            sizes.append({"file": p.name, "error": str(e)})
    return sizes

def is_tiled(sizes: list[dict], tile_size: int = 256, tol: int = 4) -> bool:
    """True if every sampled image is within tol pixels of tile_size."""
    if not sizes:
        return False
    for s in sizes:
        if "error" in s:
            continue
        if abs(s["width"] - tile_size) > tol or abs(s["height"] - tile_size) > tol:
            return False
    return True

def change_pixel_ratio(mask_paths: list[Path], n_sample: int = 20) -> float:
    """Estimate fraction of changed pixels from a random sample of masks."""
    sample = mask_paths[:n_sample]
    totals, positives = 0, 0
    for p in sample:
        try:
            arr = np.array(Image.open(p).convert("L"))
            positives += int((arr > 127).sum())
            totals    += arr.size
        except Exception:
            pass
    return float(positives / totals) if totals else 0.0

# ── Inspect ───────────────────────────────────────────────────────────────────
print(f"\nDataset root : {data_root}")
print(f"Exists       : {data_root.exists()}\n")

tile_size = int(dc.get("tile_size", 256))
report = {
    "root": str(data_root),
    "root_exists": data_root.exists(),
    "tile_size_expected": tile_size,
    "splits": {},
}

subdirs = sorted(data_root.iterdir()) if data_root.exists() else []
found_splits = [d.name for d in subdirs if d.is_dir()]
print(f"Splits found : {found_splits}")
report["found_splits"] = found_splits

for split_name in found_splits:
    split_dir = data_root / split_name
    split_info: dict = {"path": str(split_dir)}

    for sub in ["A", "B", "label"]:
        sub_dir = split_dir / sub
        imgs = find_images(sub_dir)
        sizes = sample_sizes(imgs)
        tiled = is_tiled(sizes, tile_size)
        split_info[sub] = {
            "exists": sub_dir.exists(),
            "count": len(imgs),
            "sample_sizes": sizes,
            "appears_tiled": tiled,
        }

    # Change pixel ratio from label dir
    label_imgs = find_images(split_dir / "label")
    split_info["change_pixel_ratio"] = change_pixel_ratio(label_imgs)

    # Summary
    n_a = split_info.get("A", {}).get("count", 0)
    n_b = split_info.get("B", {}).get("count", 0)
    n_l = split_info.get("label", {}).get("count", 0)
    t_a = split_info.get("A", {}).get("appears_tiled", False)
    t_l = split_info.get("label", {}).get("appears_tiled", False)

    # Estimate tiles if large images
    if split_info.get("A", {}).get("sample_sizes"):
        s0 = split_info["A"]["sample_sizes"][0]
        if "error" not in s0:
            img_w, img_h = s0["width"], s0["height"]
            stride = int(dc.get("train_stride" if split_name == "train" else "val_stride", 256))
            if not t_a:
                n_per_w = max(1, (img_w - tile_size) // stride + 1)
                n_per_h = max(1, (img_h - tile_size) // stride + 1)
                # border tiles
                if img_w % stride != 0:
                    n_per_w += 1
                if img_h % stride != 0:
                    n_per_h += 1
                est_tiles = n_a * min(n_per_w, (img_w // tile_size) + 1) * min(n_per_h, (img_h // tile_size) + 1)
                split_info["estimated_tiles"] = est_tiles
                split_info["image_dims"] = {"width": img_w, "height": img_h}

    report["splits"][split_name] = split_info

    ratio_str = f"{split_info['change_pixel_ratio']:.2%}"
    print(f"\n  [{split_name.upper()}]")
    print(f"    A images      : {n_a}")
    print(f"    B images      : {n_b}")
    print(f"    Label images  : {n_l}")
    print(f"    Appears tiled : {t_a}")
    print(f"    Change ratio  : {ratio_str}")
    if "estimated_tiles" in split_info:
        print(f"    Est. tiles    : {split_info['estimated_tiles']}  (stride={stride})")
    if split_info.get("A", {}).get("sample_sizes"):
        for s in split_info["A"]["sample_sizes"][:3]:
            if "error" not in s:
                print(f"    Sample: {s['file']}  {s['width']}x{s['height']}")

# ── Current dataset mode from config ─────────────────────────────────────────
print("\n  [CONFIG SUMMARY]")
print(f"    train_mode : {dc.get('train_mode', '(not set — image-level crop)')}")
print(f"    val_mode   : {dc.get('val_mode',   '(not set)')}")
print(f"    test_mode  : {dc.get('test_mode',  '(not set)')}")
print(f"    tile_size  : {dc.get('tile_size',  256)}")
print(f"    stride     : train={dc.get('train_stride', '(not set)')}  val/test={dc.get('val_stride', 256)}")
print(f"    image_size : {dc.get('image_size', 256)}")

# ── Save report ───────────────────────────────────────────────────────────────
report_path = out_dir / "inspection_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved to: {report_path}\n")
