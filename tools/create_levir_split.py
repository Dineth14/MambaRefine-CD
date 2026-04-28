#!/usr/bin/env python3
"""Create deterministic image-level train/val split files for LEVIR-CD."""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dataset_cfg(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("dataset", cfg)


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _image_names(split_dir: Path) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    a_dir = split_dir / "A"
    if not a_dir.exists():
        raise FileNotFoundError(f"Missing LEVIR A directory: {a_dir}")
    names = sorted(p.name for p in a_dir.iterdir() if p.suffix.lower() in exts)
    missing = []
    for name in names:
        if not (split_dir / "B" / name).exists() or not (split_dir / "label" / name).exists():
            missing.append(name)
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"{len(missing)} LEVIR samples are missing B or label files. Examples: {preview}")
    return names


def infer_original_id(name: str) -> str:
    """Infer an image-level id when filenames encode tile coordinates."""
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


def _group_by_original_id(names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[infer_original_id(name)].append(name)
    return {k: sorted(v) for k, v in groups.items()}


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample_id in ids:
            f.write(f"{sample_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LEVIR-CD image-level train/val split files.")
    parser.add_argument("--config", default="configs/datasets/levir.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_out", default="splits/levir_train.txt")
    parser.add_argument("--val_out", default="splits/levir_val.txt")
    args = parser.parse_args()

    ds_cfg = _load_dataset_cfg(args.config)
    root = _resolve_repo_path(args.root or ds_cfg.get("root", ""))
    if not root.exists():
        raise FileNotFoundError(f"LEVIR root does not exist: {root}")

    names = _image_names(root / "train")
    groups = _group_by_original_id(names)
    group_ids = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)

    n_train_groups = int(round(len(group_ids) * args.train_ratio))
    n_train_groups = min(max(1, n_train_groups), max(1, len(group_ids) - 1))
    train_group_ids = set(group_ids[:n_train_groups])
    val_group_ids = set(group_ids[n_train_groups:])

    train_names = sorted(name for gid in train_group_ids for name in groups[gid])
    val_names = sorted(name for gid in val_group_ids for name in groups[gid])

    train_out = _resolve_repo_path(args.train_out)
    val_out = _resolve_repo_path(args.val_out)
    _write_ids(train_out, train_names)
    _write_ids(val_out, val_names)

    print("LEVIR split files created")
    print(f"Root: {root}")
    print(f"Seed: {args.seed}")
    print(f"Train groups: {len(train_group_ids)}")
    print(f"Val groups: {len(val_group_ids)}")
    print(f"Train images: {len(train_names)}")
    print(f"Val images: {len(val_names)}")
    print(f"Train split: {train_out}")
    print(f"Val split: {val_out}")
    print(f"Train/val original-id overlap: {len(train_group_ids & val_group_ids)}")
    if train_group_ids & val_group_ids:
        sys.exit(1)


if __name__ == "__main__":
    main()
