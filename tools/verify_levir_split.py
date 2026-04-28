#!/usr/bin/env python3
"""Verify LEVIR-CD image-level split integrity."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_dataset_cfg(config_path: str | Path) -> dict:
    path = _resolve_repo_path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("dataset", cfg)


def _read_ids(path: str | Path) -> list[str]:
    resolved = _resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Split file not found: {resolved}")
    ids = []
    with open(resolved) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(Path(line).name)
    return ids


def _image_names(split_dir: Path) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(p.name for p in (split_dir / "A").iterdir() if p.suffix.lower() in exts)


def infer_original_id(name: str) -> str:
    stem = Path(name).stem
    patterns = [
        r"(.+?)[_-](?:x)?\d+[_-](?:y)?\d+$",
        r"(.+?)[_-]\d{3,5}[_-]\d{3,5}$",
        r"(.+?)[_-]row\d+[_-]col\d+$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return stem


def _id_set(names: list[str]) -> set[str]:
    return {infer_original_id(name) for name in names}


def _print_overlap(label: str, a: set[str], b: set[str]) -> int:
    overlap = sorted(a & b)
    print(f"{label}: {len(overlap)}")
    if overlap:
        print("  examples:", ", ".join(overlap[:20]))
    return len(overlap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LEVIR-CD train/val/test split integrity.")
    parser.add_argument("--config", default="configs/datasets/levir.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--val_split", default=None)
    args = parser.parse_args()

    ds_cfg = _load_dataset_cfg(args.config)
    root = _resolve_repo_path(args.root or ds_cfg.get("root", ""))
    split_files = ds_cfg.get("split_files", {}) or {}
    train_split = args.train_split or split_files.get("train", "splits/levir_train.txt")
    val_split = args.val_split or split_files.get("val", "splits/levir_val.txt")
    test_dir_name = str(ds_cfg.get("test_dir", "test"))

    train_ids = _read_ids(train_split)
    val_ids = _read_ids(val_split)
    test_ids = _image_names(root / test_dir_name)

    train_file_set = set(train_ids)
    val_file_set = set(val_ids)
    test_file_set = set(test_ids)
    train_orig = _id_set(train_ids)
    val_orig = _id_set(val_ids)
    test_orig = _id_set(test_ids)

    print("LEVIR split summary")
    print(f"Root: {root}")
    print(f"Train split file: {_resolve_repo_path(train_split)}")
    print(f"Val split file: {_resolve_repo_path(val_split)}")
    print(f"Test dir: {root / test_dir_name}")
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")
    print(f"Test images: {len(test_ids)}")
    print(f"Train original IDs: {len(train_orig)}")
    print(f"Val original IDs: {len(val_orig)}")
    print(f"Test original IDs: {len(test_orig)}")
    print("Sample train filenames:", ", ".join(train_ids[:10]))
    print("Sample val filenames:", ", ".join(val_ids[:10]))
    print("Sample test filenames:", ", ".join(test_ids[:10]))
    print("Filename overlap counts")
    errors = 0
    errors += _print_overlap("train/val", train_file_set, val_file_set)
    errors += _print_overlap("train/test", train_file_set, test_file_set)
    errors += _print_overlap("val/test", val_file_set, test_file_set)
    print("Original-id overlap counts")
    errors += _print_overlap("train/val", train_orig, val_orig)
    errors += _print_overlap("train/test", train_orig, test_orig)
    errors += _print_overlap("val/test", val_orig, test_orig)

    train_folder = set(_image_names(root / "train"))
    missing_train = sorted((train_file_set | val_file_set) - train_folder)
    if missing_train:
        print(f"Missing train/val files from train folder: {len(missing_train)}")
        print("  examples:", ", ".join(missing_train[:20]))
        errors += len(missing_train)

    if errors:
        raise SystemExit(f"LEVIR split verification failed with {errors} overlap/missing-file errors.")
    print("No leakage detected.")


if __name__ == "__main__":
    main()
