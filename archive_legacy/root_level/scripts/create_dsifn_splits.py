#!/usr/bin/env python3
"""Create deterministic non-overlapping DSIFN-CD split files.

Modes:
  clean      (default) — ratio-based split from flat image directory
  lit-patch             — literature 14400/1360/192 patch split extracted from
                          train.zip / val.zip / test.zip in the source root
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_EXT_PRIORITY = {".png": 0, ".tif": 1, ".tiff": 2, ".jpg": 3, ".jpeg": 4}
PATCH_SIZE = 256
LIT_EXPECTED = {"train": 14400, "val": 1360, "test": 192}
LIT_PREFIXES = {"train": "tr", "val": "va", "test": "te"}


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


# ---------------------------------------------------------------------------
# Literature patch split (14400/1360/192) from source zip files
# ---------------------------------------------------------------------------

def create_lit_patch_split(source_root: Path, out_root: Path) -> None:
    """Extract train/val/test.zip, tile 512x256 -> 256x256 patches, write split files.

    Output directory layout:
        out_root/
          t1/     (15952 256x256 JPEG patches with unique prefixed names)
          t2/
          mask/   (256x256 PNG patches, binary 0/1)
          splits/
            train.txt  (14400 lines)
            val.txt    (1360 lines)
            test.txt   (192 lines)
            split_metadata.json
    """
    import zipfile
    from io import BytesIO
    import numpy as np
    from PIL import Image

    zips = {s: source_root / f"{s}.zip" for s in ("train", "val", "test")}
    for s, zp in zips.items():
        if not zp.exists():
            raise FileNotFoundError(f"Missing source zip: {zp}")

    t1_dir = out_root / "t1"
    t2_dir = out_root / "t2"
    mask_dir = out_root / "mask"
    splits_dir = out_root / "splits"
    for d in (t1_dir, t2_dir, mask_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    split_names: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        prefix = LIT_PREFIXES[split]
        print(f"  Processing {split}.zip ...", flush=True)
        with zipfile.ZipFile(zips[split]) as z:
            all_entries = set(z.namelist())
            t1_entries = sorted(
                n for n in all_entries
                if (n.startswith("t1/") or "/t1/" in n)
                and not n.endswith("/")
                and Path(n).suffix.lower() in EXTS
            )
            # Build lookup: stem -> (t2_entry, mask_entry)
            t2_map: dict[str, str] = {}
            mask_map: dict[str, str] = {}
            for n in all_entries:
                if (n.startswith("t2/") or "/t2/" in n) and not n.endswith("/"):
                    t2_map[Path(n).stem] = n
                if (n.startswith("mask/") or "/mask/" in n) and not n.endswith("/"):
                    stem = Path(n).stem
                    # keep highest priority extension (png < tif)
                    if stem not in mask_map or MASK_EXT_PRIORITY.get(Path(n).suffix.lower(), 99) < MASK_EXT_PRIORITY.get(Path(mask_map[stem]).suffix.lower(), 99):
                        mask_map[stem] = n

            skipped = 0
            for entry in t1_entries:
                stem = Path(entry).stem
                t2_entry = t2_map.get(stem)
                mask_entry = mask_map.get(stem)
                if t2_entry is None or mask_entry is None:
                    skipped += 1
                    continue

                img_a = Image.open(BytesIO(z.read(entry))).convert("RGB")
                img_b = Image.open(BytesIO(z.read(t2_entry))).convert("RGB")
                raw_mask = Image.open(BytesIO(z.read(mask_entry)))
                # Normalise mask to uint8 binary (0/1)
                arr_m = np.array(raw_mask)
                if arr_m.ndim == 3:
                    arr_m = arr_m[..., 0]
                if arr_m.dtype != np.uint8 or arr_m.max() > 1:
                    threshold = 127 if arr_m.max() > 1 else 0
                    arr_m = (arr_m > threshold).astype(np.uint8)
                mask_pil = Image.fromarray(arr_m, mode="L")

                W, H = img_a.size
                rows_offsets = list(range(0, H - PATCH_SIZE + 1, PATCH_SIZE))
                cols_offsets = list(range(0, W - PATCH_SIZE + 1, PATCH_SIZE))
                if not rows_offsets or not cols_offsets:
                    # image smaller than patch size — use single crop at (0,0)
                    rows_offsets = [0]
                    cols_offsets = [0]

                for r in rows_offsets:
                    for c in cols_offsets:
                        patch_id = f"{prefix}_{stem}_r{r}_c{c}"
                        box = (c, r, c + PATCH_SIZE, r + PATCH_SIZE)
                        patch_a = img_a.crop(box)
                        patch_b = img_b.crop(box)
                        patch_m = mask_pil.crop(box)
                        name_jpg = f"{patch_id}.jpg"
                        name_png = f"{patch_id}.png"
                        patch_a.save(t1_dir / name_jpg, quality=95)
                        patch_b.save(t2_dir / name_jpg, quality=95)
                        patch_m.save(mask_dir / name_png)
                        split_names[split].append(name_jpg)

            if skipped:
                print(f"    WARNING: skipped {skipped} entries (missing t2 or mask)", flush=True)
            print(f"    {split}: {len(split_names[split])} patches from {len(t1_entries) - skipped} source images", flush=True)

    # Verify expected counts
    all_ok = True
    for split, expected in LIT_EXPECTED.items():
        actual = len(split_names[split])
        if actual != expected:
            print(f"  WARNING: {split} expected {expected} patches, got {actual}", flush=True)
            all_ok = False
    if all_ok:
        print("  Patch counts verified: 14400 / 1360 / 192 ✓", flush=True)

    # Verify no cross-split overlap
    id_sets = {s: {Path(n).stem for n in names} for s, names in split_names.items()}
    for label, (s1, s2) in [("train/val", ("train", "val")), ("train/test", ("train", "test")), ("val/test", ("val", "test"))]:
        overlap = id_sets[s1] & id_sets[s2]
        if overlap:
            raise RuntimeError(f"DATA LEAKAGE: {label} overlap ({len(overlap)} samples): {sorted(overlap)[:10]}")
    print("  Cross-split overlap check: PASS ✓", flush=True)

    # Write split text files (sorted for determinism)
    split_paths: dict[str, Path] = {}
    for split, names in split_names.items():
        p = splits_dir / f"{split}.txt"
        write_names(p, sorted(names))
        split_paths[split] = p

    metadata = {
        "protocol": "lit_patch_split",
        "description": "DSIFN-CD literature 14400/1360/192 non-overlapping 256x256 patches",
        "source_root": str(source_root.resolve()),
        "output_root": str(out_root.resolve()),
        "patch_size": PATCH_SIZE,
        "train_count": len(split_names["train"]),
        "val_count": len(split_names["val"]),
        "test_count": len(split_names["test"]),
        "split_hash_train": sha256_file(split_paths["train"]),
        "split_hash_val": sha256_file(split_paths["val"]),
        "split_hash_test": sha256_file(split_paths["test"]),
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Train images are already 256x256; random crop during training is a no-op. "
            "Val/test images are 256x256 so _build_tiles() produces exactly 1 tile per image. "
            "Total val dataset tiles = 1360, total test dataset tiles = 192."
        ),
    }
    (splits_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n  Literature patch split written to: {out_root}")
    print(f"    splits/train.txt : {len(split_names['train'])} entries")
    print(f"    splits/val.txt   : {len(split_names['val'])} entries")
    print(f"    splits/test.txt  : {len(split_names['test'])} entries")
    print(f"    splits/split_metadata.json written")


# ---------------------------------------------------------------------------
# Original clean split from flat image directory
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic DSIFN-CD train/val/test split files.")
    parser.add_argument("--mode", choices=["clean", "lit-patch"], default="clean",
                        help="'clean': ratio-based split from flat images; "
                             "'lit-patch': literature 14400/1360/192 patch split from zip files.")
    parser.add_argument("--root", required=True,
                        help="DSIFN source root (contains t1/t2/mask for clean mode, "
                             "or train.zip/val.zip/test.zip for lit-patch mode).")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for split files (clean) or full patch dataset (lit-patch).")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "lit-patch":
        print("Creating DSIFN-CD literature patch split (14400/1360/192) ...")
        create_lit_patch_split(Path(args.root), Path(args.out_dir))
        return

    # --- clean mode (original behaviour) ---
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
