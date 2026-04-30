#!/usr/bin/env python3
"""Import and validate external DSIFN-CD split files."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_EXT_PRIORITY = {".png": 0, ".tif": 1, ".tiff": 2, ".jpg": 3, ".jpeg": 4}


def read_names(path: Path) -> list[str]:
    return [Path(line.strip()).name for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_lookup(path: Path, prefer_mask: bool = False) -> dict[str, Path]:
    files = [p for p in path.iterdir() if p.suffix.lower() in EXTS]
    if prefer_mask:
        files = sorted(files, key=lambda p: (p.stem, MASK_EXT_PRIORITY.get(p.suffix.lower(), 99), p.name))
    else:
        files = sorted(files, key=lambda p: p.name)
    lookup: dict[str, Path] = {}
    for item in files:
        lookup[item.name] = item
        lookup.setdefault(item.stem, item)
    return lookup


def normalise_names(names: list[str], t1_lookup: dict[str, Path]) -> list[str]:
    resolved = []
    missing = []
    for name in names:
        stem = Path(name).stem
        path = t1_lookup.get(name) or t1_lookup.get(stem)
        if path is None:
            missing.append(name)
        else:
            resolved.append(path.name)
    if missing:
        raise FileNotFoundError(f"{len(missing)} split entries are missing from t1/. Examples: {missing[:20]}")
    return sorted(resolved)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official or external DSIFN-CD split files.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--test_txt", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    t1_lookup = build_lookup(root / "t1")
    t2_lookup = build_lookup(root / "t2")
    mask_lookup = build_lookup(root / "mask", prefer_mask=True)
    splits = {
        "train": normalise_names(read_names(Path(args.train_txt)), t1_lookup),
        "val": normalise_names(read_names(Path(args.val_txt)), t1_lookup),
        "test": normalise_names(read_names(Path(args.test_txt)), t1_lookup),
    }
    for split, names in splits.items():
        missing = []
        for name in names:
            stem = Path(name).stem
            if name not in t2_lookup and stem not in t2_lookup:
                missing.append(f"t2:{name}")
            if name not in mask_lookup and stem not in mask_lookup:
                missing.append(f"mask:{name}")
        if missing:
            raise FileNotFoundError(f"{split}: missing paired files. Examples: {missing[:20]}")

    id_sets = {split: {Path(name).stem for name in names} for split, names in splits.items()}
    overlaps = {
        "train_val": id_sets["train"] & id_sets["val"],
        "train_test": id_sets["train"] & id_sets["test"],
        "val_test": id_sets["val"] & id_sets["test"],
    }
    bad = {key: sorted(values)[:20] for key, values in overlaps.items() if values}
    if bad:
        raise RuntimeError(f"DATA LEAKAGE FOUND: refusing to import overlapping DSIFN split files: {bad}")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {split: out_dir / f"{split}.txt" for split in splits}
    for split, names in splits.items():
        split_paths[split].write_text("\n".join(names) + "\n", encoding="utf-8")
    metadata = {
        "dataset_root": str(root.resolve()),
        "source_train_txt": str(Path(args.train_txt).resolve()),
        "source_val_txt": str(Path(args.val_txt).resolve()),
        "source_test_txt": str(Path(args.test_txt).resolve()),
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "split_file_train": str(split_paths["train"].resolve()),
        "split_file_val": str(split_paths["val"].resolve()),
        "split_file_test": str(split_paths["test"].resolve()),
        "split_hash_train": sha256_file(split_paths["train"]),
        "split_hash_val": sha256_file(split_paths["val"]),
        "split_hash_test": sha256_file(split_paths["test"]),
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Old leaked DSIFN scores are invalid as held-out test results.",
    }
    (out_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for split, path in split_paths.items():
        print(f"Wrote {path} ({len(splits[split])} images)")
    print(f"Wrote {out_dir / 'split_metadata.json'}")


if __name__ == "__main__":
    main()
