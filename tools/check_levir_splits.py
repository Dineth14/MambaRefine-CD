#!/usr/bin/env python3
"""Diagnose LEVIR split integrity and val/test distribution."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from utils.config import load_config

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def discover(split_dir: Path) -> list[str]:
    return sorted(p.name for p in (split_dir / "A").iterdir() if p.suffix.lower() in EXTS)


def split_train_val(names: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def resolve_repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def names_from_split_file(all_names: list[str], path: str | Path | None) -> list[str] | None:
    resolved = resolve_repo_path(path)
    if resolved is None:
        return None
    if not resolved.exists():
        raise FileNotFoundError(f"Split file not found: {resolved}")
    lookup = {name: name for name in all_names}
    lookup.update({Path(name).stem: name for name in all_names})
    selected = []
    missing = []
    with open(resolved) as f:
        for line in f:
            sample_id = line.strip()
            if not sample_id or sample_id.startswith("#"):
                continue
            key = Path(sample_id).name
            matched = lookup.get(key) or lookup.get(Path(key).stem)
            if matched is None:
                missing.append(sample_id)
            elif matched not in selected:
                selected.append(matched)
    if missing:
        raise ValueError(f"{len(missing)} split ids are missing from train folder. Examples: {missing[:10]}")
    return sorted(selected)


def infer_original_id(name: str) -> str:
    stem = Path(str(name)).stem
    # Dataset sample ids may append tile x/y: train_103.png_256_0.
    stem = re.sub(r"\.(png|jpg|jpeg|tif|tiff|bmp)_\d+_\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_\d+_\d+$", "", stem)
    return stem


def sliding_coords(w: int, h: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    if w == tile_size and h == tile_size:
        return [(0, 0)]
    xs = list(range(0, w - tile_size + 1, stride))
    ys = list(range(0, h - tile_size + 1, stride))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    return [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]


def triplets(root: Path, disk_split: str, names: Iterable[str]) -> list[dict]:
    base = root / disk_split
    rows = []
    for name in names:
        rows.append({
            "name": name,
            "id": infer_original_id(name),
            "a": base / "A" / name,
            "b": base / "B" / name,
            "mask": base / "label" / name,
        })
    return rows


def build_tiles(rows: list[dict], tile_size: int, stride: int) -> list[dict]:
    tiles = []
    for row in rows:
        w, h = Image.open(row["a"]).size
        for x, y in sliding_coords(w, h, tile_size, stride):
            item = dict(row)
            item.update({"x": x, "y": y, "tile_size": tile_size})
            tiles.append(item)
    return tiles


def crop_mask(tile: dict) -> np.ndarray:
    s, x, y = tile["tile_size"], tile["x"], tile["y"]
    return np.array(Image.open(tile["mask"]).convert("L"))[y:y + s, x:x + s]


def crop_image(path: Path, tile: dict) -> np.ndarray:
    s, x, y = tile["tile_size"], tile["x"], tile["y"]
    return np.array(Image.open(path).convert("RGB"))[y:y + s, x:x + s]


def connected_components(mask: np.ndarray) -> int:
    binary = mask.astype(bool)
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    count = 0
    for yy in range(h):
        for xx in range(w):
            if not binary[yy, xx] or seen[yy, xx]:
                continue
            count += 1
            q = deque([(yy, xx)])
            seen[yy, xx] = True
            while q:
                cy, cx = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
    return count


def summarize_tiles(tiles: list[dict], max_components: int = 50) -> dict:
    changed = 0
    total = 0
    ratios = []
    areas = []
    comps = []
    for i, tile in enumerate(tiles):
        raw = crop_mask(tile)
        conv = raw > 127
        c = int(conv.sum())
        n = int(conv.size)
        changed += c
        total += n
        ratios.append(c / max(1, n))
        areas.append(c)
        if i < max_components:
            comps.append(connected_components(conv))
    return {
        "samples": len(tiles),
        "changed_pixels": changed,
        "no_change_pixels": total - changed,
        "gt_positive_ratio": changed / max(1, total),
        "avg_positive_ratio": mean(ratios) if ratios else 0.0,
        "mean_change_area_per_sample": mean(areas) if areas else 0.0,
        "avg_connected_components": mean(comps) if comps else 0.0,
        "components_sampled": len(comps),
    }


def image_tensor_stats(tiles: list[dict], max_samples: int = 200) -> dict:
    vals = []
    for tile in tiles[:max_samples]:
        for key in ("a", "b"):
            img = crop_image(tile[key], tile).astype(np.float32) / 255.0
            norm = (img - MEAN) / STD
            vals.append(norm.reshape(-1, 3))
    if not vals:
        return {}
    arr = np.concatenate(vals, axis=0)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "samples": min(len(tiles), max_samples),
    }


def mask_debug(tiles: list[dict], n: int = 20) -> list[dict]:
    rows = []
    for tile in tiles[:n]:
        raw = crop_mask(tile)
        converted = (raw > 127).astype(np.uint8)
        rows.append({
            "sample": f"{tile['name']}_{tile['x']}_{tile['y']}",
            "raw_unique": sorted(int(v) for v in np.unique(raw).tolist()),
            "converted_unique": sorted(int(v) for v in np.unique(converted).tolist()),
            "shape": list(raw.shape),
            "positive_ratio": float(converted.mean()),
        })
    return rows


def print_triplets(label: str, rows: list[dict], n: int = 20) -> None:
    print(f"\nFirst {min(n, len(rows))} {label} sample triplets:")
    for row in rows[:n]:
        print(f"  id={row['id']}")
        print(f"    image_t1: {row['a']}")
        print(f"    image_t2: {row['b']}")
        print(f"    mask    : {row['mask']}")


def validate_pairing(rows: list[dict]) -> list[str]:
    errors = []
    for row in rows:
        if not row["a"].exists():
            errors.append(f"missing A: {row['a']}")
        if not row["b"].exists():
            errors.append(f"missing B: {row['b']}")
        if not row["mask"].exists():
            errors.append(f"missing mask: {row['mask']}")
        if row["a"].name != row["b"].name or row["a"].name != row["mask"].name:
            errors.append(f"name mismatch: {row['a'].name}, {row['b'].name}, {row['mask'].name}")
    return errors


def write_csv(path: Path, tiles: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_t1", "image_t2", "mask", "x", "y", "tile_size"])
        writer.writeheader()
        for t in tiles:
            writer.writerow({
                "sample_id": t["id"],
                "image_t1": t["a"],
                "image_t2": t["b"],
                "mask": t["mask"],
                "x": t["x"],
                "y": t["y"],
                "tile_size": t["tile_size"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LEVIR split leakage, pairing, masks, and distribution.")
    parser.add_argument("--config", default="configs/ablations/levir/a6_full.yaml")
    parser.add_argument("--write_metadata", action="store_true")
    parser.add_argument("--metadata_dir", default="outputs/levir_split_metadata")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["dataset"]
    root = Path(dc["root"])
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    val_ratio = float(dc.get("val_ratio", 0.2))
    tile_size = int(dc.get("tile_size", dc.get("image_size", 256)))
    train_stride = int(dc.get("train_stride", 128))
    val_stride = int(dc.get("val_stride", 256))
    test_stride = int(dc.get("test_stride", 256))
    train_mode = str(dc.get("train_mode", "image")).lower()

    all_train_names = discover(root / "train")
    test_dir = str(dc.get("test_dir", "test"))
    test_names = discover(root / test_dir)
    split_files = dc.get("split_files", {}) or {}
    train_names = names_from_split_file(all_train_names, split_files.get("train"))
    val_names = names_from_split_file(all_train_names, split_files.get("val"))
    split_source = "split_files" if train_names is not None and val_names is not None else f"val_ratio={val_ratio}, seed={seed}"
    if train_names is None or val_names is None:
        train_names, val_names = split_train_val(all_train_names, val_ratio, seed)

    train_rows = triplets(root, "train", train_names)
    val_rows = triplets(root, "train", val_names)
    test_rows = triplets(root, test_dir, test_names)

    train_tiles = build_tiles(train_rows, tile_size, train_stride if train_mode == "tile" else tile_size)
    val_tiles = build_tiles(val_rows, tile_size, val_stride)
    test_tiles = build_tiles(test_rows, tile_size, test_stride)

    split_ids = {
        "train": {r["id"] for r in train_rows},
        "val": {r["id"] for r in val_rows},
        "test": {r["id"] for r in test_rows},
    }
    name_sets = {
        "train": {r["name"] for r in train_rows},
        "val": {r["name"] for r in val_rows},
        "test": {r["name"] for r in test_rows},
    }

    print("LEVIR split integrity")
    print(f"Config: {args.config}")
    print(f"Root: {root}")
    print(f"Official-looking folders present: train={bool((root/'train').exists())}, val={bool((root/'val').exists())}, test={bool((root/test_dir).exists())}")
    print(f"Validation source: train/ using {split_source}")
    print(f"Train mode: {'tile' if train_mode == 'tile' else 'full_image/random_crop'}")
    print(f"Val mode: tile")
    print(f"Test mode: tile")
    print(f"Train images: {len(train_rows)} | train tile/index samples: {len(train_tiles)}")
    print(f"Val images: {len(val_rows)} | val tile samples: {len(val_tiles)}")
    print(f"Test images: {len(test_rows)} | test tile samples: {len(test_tiles)}")
    print(f"Sample train filenames: {[r['name'] for r in train_rows[:10]]}")
    print(f"Sample val filenames: {[r['name'] for r in val_rows[:10]]}")
    print(f"Sample test filenames: {[r['name'] for r in test_rows[:10]]}")

    overlaps = {
        "train_val_names": sorted(name_sets["train"] & name_sets["val"]),
        "train_test_names": sorted(name_sets["train"] & name_sets["test"]),
        "val_test_names": sorted(name_sets["val"] & name_sets["test"]),
        "train_val_ids": sorted(split_ids["train"] & split_ids["val"]),
        "train_test_ids": sorted(split_ids["train"] & split_ids["test"]),
        "val_test_ids": sorted(split_ids["val"] & split_ids["test"]),
    }
    print("\nOverlap checks:")
    for key, values in overlaps.items():
        print(f"  {key}: {len(values)}")
        if values[:20]:
            print(f"    first: {values[:20]}")
    if overlaps["train_val_ids"]:
        print("WARNING: train and val share original image IDs; validation may be inflated.")
    else:
        print("No train/val original image ID overlap detected.")

    pairing_errors = {
        "train": validate_pairing(train_rows),
        "val": validate_pairing(val_rows),
        "test": validate_pairing(test_rows),
    }
    print("\nPairing checks:")
    for split, errors in pairing_errors.items():
        print(f"  {split}: {'OK' if not errors else str(len(errors)) + ' errors'}")
        for err in errors[:20]:
            print(f"    {err}")
    print_triplets("test", test_rows)

    print("\nMask conversion debug: val")
    for row in mask_debug(val_tiles, 20):
        print(json.dumps(row))
    print("\nMask conversion debug: test")
    for row in mask_debug(test_tiles, 20):
        print(json.dumps(row))

    summaries = {
        "train": summarize_tiles(train_tiles),
        "val": summarize_tiles(val_tiles),
        "test": summarize_tiles(test_tiles),
    }
    image_stats = {
        "val": image_tensor_stats(val_tiles),
        "test": image_tensor_stats(test_tiles),
    }
    print("\nDistribution summary:")
    for split, stats in summaries.items():
        print(f"  {split}: {json.dumps(stats, sort_keys=True)}")
    print("\nImage tensor stats after ImageNet normalization:")
    for split, stats in image_stats.items():
        print(f"  {split}: {json.dumps(stats, sort_keys=True)}")

    if args.write_metadata:
        out = REPO / args.metadata_dir
        write_csv(out / "train_tiles.csv", train_tiles)
        write_csv(out / "val_tiles.csv", val_tiles)
        write_csv(out / "test_tiles.csv", test_tiles)
        print(f"\nWrote metadata CSVs to {out}")

    report = {
        "config": args.config,
        "root": str(root),
        "counts": {
            "train_images": len(train_rows),
            "val_images": len(val_rows),
            "test_images": len(test_rows),
            "train_samples": len(train_tiles),
            "val_samples": len(val_tiles),
            "test_samples": len(test_tiles),
        },
        "overlaps": {k: len(v) for k, v in overlaps.items()},
        "pairing_errors": {k: len(v) for k, v in pairing_errors.items()},
        "summaries": summaries,
        "image_tensor_stats": image_stats,
        "conclusion": {
            "train_val_original_id_overlap": bool(overlaps["train_val_ids"]),
            "val_source": "image-level split file from train/" if split_source == "split_files" else "random image-level subset of train/",
            "test_source": "test/",
            "official_val_folder_used": (root / "val").exists(),
        },
    }
    print("\nJSON summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
