#!/usr/bin/env python3
"""Create deterministic non-overlapping DSIFN-CD split files."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_EXT_PRIORITY = {".png": 0, ".tif": 1, ".tiff": 2, ".jpg": 3, ".jpeg": 4}


def list_images(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in EXTS], key=lambda p: p.name)


def build_lookup(path: Path, prefer_mask: bool = False) -> dict[str, Path]:
    files = list_images(path)
    if prefer_mask:
        files = sorted(files, key=lambda p: (p.stem, MASK_EXT_PRIORITY.get(p.suffix.lower(), 99), p.name))
    lookup: dict[str, Path] = {}
    for item in files:
        lookup[item.name] = item
        lookup.setdefault(item.stem, item)
    return lookup


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_names(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def verify_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must all be positive.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic DSIFN-CD train/val/test split files.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    verify_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    t1_dir = root / "t1"
    t2_dir = root / "t2"
    mask_dir = root / "mask"
    for path in (t1_dir, t2_dir, mask_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing DSIFN directory: {path}")

    t1_files = list_images(t1_dir)
    t2_lookup = build_lookup(t2_dir)
    mask_lookup = build_lookup(mask_dir, prefer_mask=True)
    names = [p.name for p in t1_files]
    missing = []
    for name in names:
        stem = Path(name).stem
        if name not in t2_lookup and stem not in t2_lookup:
            missing.append(f"t2:{name}")
        if name not in mask_lookup and stem not in mask_lookup:
            missing.append(f"mask:{name}")
    if missing:
        raise FileNotFoundError(f"{len(missing)} DSIFN samples have missing pairs. Examples: {missing[:20]}")

    shuffled = list(names)
    random.Random(args.seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * args.train_ratio)
    val_count = int(total * args.val_ratio)
    test_count = total - train_count - val_count
    train_names = sorted(shuffled[:train_count])
    val_names = sorted(shuffled[train_count:train_count + val_count])
    test_names = sorted(shuffled[train_count + val_count:])
    sets = {"train": set(train_names), "val": set(val_names), "test": set(test_names)}
    overlaps = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Unexpected split overlap: { {k: sorted(v)[:10] for k, v in overlaps.items() if v} }")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": out_dir / "train.txt",
        "val": out_dir / "val.txt",
        "test": out_dir / "test.txt",
    }
    write_names(split_paths["train"], train_names)
    write_names(split_paths["val"], val_names)
    write_names(split_paths["test"], test_names)

    metadata = {
        "dataset_root": str(root.resolve()),
        "total_image_count": total,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "test_count": len(test_names),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "split_file_train": str(split_paths["train"].resolve()),
        "split_file_val": str(split_paths["val"].resolve()),
        "split_file_test": str(split_paths["test"].resolve()),
        "split_hash_train": sha256_file(split_paths["train"]),
        "split_hash_val": sha256_file(split_paths["val"]),
        "split_hash_test": sha256_file(split_paths["test"]),
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Old leaked DSIFN scores are invalid as held-out test results.",
        "patch_policy": "Split original image IDs first; train crops only from train IDs; val/test tiles only from their split IDs.",
    }
    (out_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {split_paths['train']} ({len(train_names)} images)")
    print(f"Wrote {split_paths['val']} ({len(val_names)} images)")
    print(f"Wrote {split_paths['test']} ({len(test_names)} images)")
    print(f"Wrote {out_dir / 'split_metadata.json'}")


if __name__ == "__main__":
    main()
