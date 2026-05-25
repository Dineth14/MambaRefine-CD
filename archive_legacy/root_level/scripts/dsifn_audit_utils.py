#!/usr/bin/env python3
"""Shared utilities for DSIFN-CD split integrity audits."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_EXT_PRIORITY = {".png": 0, ".tif": 1, ".tiff": 2, ".jpg": 3, ".jpeg": 4}
A_CANDS = ["t1", "T1", "A", "imageA", "before", "img1", "A_256"]
B_CANDS = ["t2", "T2", "B", "imageB", "after", "img2", "B_256"]
LABEL_CANDS = ["GT", "label", "labels", "mask", "OUT", "change_map"]
SPLIT_ALIASES = {
    "train": ["trainset", "train", "Train", "training"],
    "val": ["valset", "val", "Val", "valid", "validation"],
    "test": ["testset", "test", "Test", "testing"],
}
MANIFEST_FIELDS = [
    "split",
    "sample_index",
    "pre_image_path",
    "post_image_path",
    "mask_path",
    "pre_image_name",
    "post_image_name",
    "mask_name",
    "pre_stem",
    "post_stem",
    "mask_stem",
    "original_scene_id",
    "patch_id",
    "crop_x",
    "crop_y",
    "width",
    "height",
    "pre_sha256",
    "post_sha256",
    "mask_sha256",
    "pair_key",
    "mask_key",
]


@dataclass
class SplitResolution:
    split: str
    layout: str
    source: str
    base_dir: Path
    a_dir: Path
    b_dir: Path
    mask_dir: Path
    names: list[str]
    samples_are_tiles: bool


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml_config(config_path: str | Path) -> dict:
    path = repo_path(config_path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_dataset_config(config_path: str | Path) -> tuple[dict, dict]:
    cfg = load_yaml_config(config_path)
    return cfg, cfg.get("dataset", {})


def detect_dir(parent: Path, candidates: Iterable[str]) -> Optional[Path]:
    for candidate in candidates:
        p = parent / str(candidate)
        if p.is_dir():
            return p
    return None


def list_images(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.suffix.lower() in EXTS)


def build_file_lookup(directory: Path, prefer_mask_ext: bool = False) -> dict[str, Path]:
    paths = [p for p in directory.iterdir() if p.suffix.lower() in EXTS]
    if prefer_mask_ext:
        paths = sorted(paths, key=lambda p: (p.stem, MASK_EXT_PRIORITY.get(p.suffix.lower(), 99), p.name))
    else:
        paths = sorted(paths, key=lambda p: p.name)
    lookup: dict[str, Path] = {}
    for p in paths:
        lookup[p.name] = p
        lookup.setdefault(p.stem, p)
    return lookup


def read_split_file(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [Path(line.strip()).name for line in f if line.strip() and not line.lstrip().startswith("#")]


def normalise_id(name: str) -> str:
    return Path(str(name)).stem


def resolve_split_file(root: Path, split: str, split_dir: str | Path | None = None) -> Optional[Path]:
    candidates: list[Path] = []
    if split_dir:
        candidates.append(repo_path(split_dir))
    candidates.append(root / "splits")
    for directory in candidates:
        path = directory / f"{split}.txt"
        if path.exists():
            return path
    root_file = root / f"{split}.txt"
    return root_file if root_file.exists() else None


def split_file_sha256(path: str | Path) -> str:
    return sha256_file(path)


def validate_explicit_splits(ds_cfg: dict) -> dict:
    root = repo_path(ds_cfg["root"])
    split_dir = ds_cfg.get("split_dir")
    files = {split: resolve_split_file(root, split, split_dir) for split in ("train", "val", "test")}
    missing = [split for split, path in files.items() if path is None]
    if missing:
        return {
            "verdict": "FAIL",
            "reason": (
                "DSIFN flat layout requires explicit non-overlapping split files. "
                "Refusing to use all images as test because this causes train/test leakage."
            ),
            "missing": missing,
        }
    names_by_split = {split: read_split_file(path) for split, path in files.items() if path is not None}
    id_lists = {split: [normalise_id(name) for name in names] for split, names in names_by_split.items()}
    duplicates = {
        split: sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
        for split, ids in id_lists.items()
    }
    duplicate_bad = {split: values[:50] for split, values in duplicates.items() if values}
    if duplicate_bad:
        return {
            "verdict": "FAIL",
            "reason": "DATA LEAKAGE FOUND: refusing to train/evaluate.",
            "duplicates": duplicate_bad,
            "duplicate_counts": {split: len(values) for split, values in duplicates.items() if values},
        }
    ids = {split: set(values) for split, values in id_lists.items()}
    overlaps = {
        "train_val": sorted(ids["train"] & ids["val"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "val_test": sorted(ids["val"] & ids["test"]),
    }
    bad = {key: values for key, values in overlaps.items() if values}
    if bad:
        return {
            "verdict": "FAIL",
            "reason": "DATA LEAKAGE FOUND: refusing to train/evaluate.",
            "overlaps": {key: values[:50] for key, values in bad.items()},
            "overlap_counts": {key: len(values) for key, values in bad.items()},
        }
    flat_a_dir = detect_dir(root, A_CANDS)
    if flat_a_dir is not None:
        all_flat_ids = {normalise_id(name) for name in list_images(flat_a_dir)}
        if all_flat_ids and ids["test"] == all_flat_ids:
            return {
                "verdict": "FAIL",
                "reason": (
                    "DATA LEAKAGE FOUND: refusing to train/evaluate. "
                    "DSIFN test split contains all flat-layout images."
                ),
                "total_flat_images": len(all_flat_ids),
                "test_images": len(ids["test"]),
            }
    split_parent = files["train"].parent if files["train"] is not None else repo_path(split_dir or root / "splits")
    return {
        "verdict": "PASS",
        "split_dir": str(split_parent),
        "split_file_train": str(files["train"]),
        "split_file_val": str(files["val"]),
        "split_file_test": str(files["test"]),
        "split_metadata_json": str(split_parent / "split_metadata.json"),
        "split_hash_train": split_file_sha256(files["train"]),
        "split_hash_val": split_file_sha256(files["val"]),
        "split_hash_test": split_file_sha256(files["test"]),
        "counts": {split: len(ids[split]) for split in ("train", "val", "test")},
    }


def find_split_dir(root: Path, split: str) -> Optional[Path]:
    for alias in SPLIT_ALIASES.get(split, [split]):
        p = root / alias
        if p.is_dir():
            return p
    return None


def resolve_path(lookup: dict[str, Path], name: str, kind: str, split: str) -> Path:
    path = lookup.get(name) or lookup.get(Path(name).stem)
    if path is None:
        raise FileNotFoundError(f"Could not resolve {kind} for DSIFN split={split} sample={name}")
    return path


def resolve_split(ds_cfg: dict, split: str, seed: Optional[int] = None) -> SplitResolution:
    root = repo_path(ds_cfg["root"])
    image_size = int(ds_cfg.get("image_size", 256))
    seed = int(seed if seed is not None else ds_cfg.get("seed", 42))
    a_cands = ds_cfg.get("image_a_dir_candidates", A_CANDS)
    b_cands = ds_cfg.get("image_b_dir_candidates", B_CANDS)
    l_cands = ds_cfg.get("label_dir_candidates", LABEL_CANDS)

    split_file = resolve_split_file(root, split, ds_cfg.get("split_dir"))
    if split_file is not None and split_file.exists():
        a_dir = detect_dir(root, a_cands)
        b_dir = detect_dir(root, b_cands)
        mask_dir = detect_dir(root, l_cands)
        if a_dir is None or b_dir is None or mask_dir is None:
            raise FileNotFoundError(f"DSIFN split={split}: missing flat t1/t2/mask directories under {root}")
        return SplitResolution(
            split=split,
            layout="flat_split_files",
            source=str(split_file),
            base_dir=root,
            a_dir=a_dir,
            b_dir=b_dir,
            mask_dir=mask_dir,
            names=read_split_file(split_file),
            samples_are_tiles=(split != "train"),
        )

    split_dir = find_split_dir(root, split)
    if split_dir is not None:
        a_dir = detect_dir(split_dir, a_cands)
        b_dir = detect_dir(split_dir, b_cands)
        mask_dir = detect_dir(split_dir, l_cands)
        if a_dir is None or b_dir is None or mask_dir is None:
            raise FileNotFoundError(f"DSIFN split={split}: missing t1/t2/mask directories under {split_dir}")
        return SplitResolution(
            split=split,
            layout="split_directories",
            source=str(split_dir),
            base_dir=split_dir,
            a_dir=a_dir,
            b_dir=b_dir,
            mask_dir=mask_dir,
            names=list_images(a_dir),
            samples_are_tiles=(split != "train"),
        )

    if detect_dir(root, a_cands) is not None:
        raise RuntimeError(
            "DSIFN flat layout requires explicit non-overlapping split files. "
            "Refusing to use all images as test because this causes train/test leakage."
        )
    raise FileNotFoundError(f"DSIFN split={split}: no split file or split directory found under {root}")


def sliding_coords(width: int, height: int, tile_size: int) -> list[tuple[int, int]]:
    if width == tile_size and height == tile_size:
        return [(0, 0)]
    xs = list(range(0, width - tile_size + 1, tile_size))
    ys = list(range(0, height - tile_size + 1, tile_size))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    return [(x, y) for y in sorted(set(ys)) for x in sorted(set(xs))]


def infer_original_scene_id(stem: str) -> str:
    value = Path(stem).stem
    patterns = [
        r"(.+?)[_-](?:x)?\d+[_-](?:y)?\d+$",
        r"(.+?)[_-]\d{3,5}[_-]\d{3,5}$",
        r"(.+?)[_-]row\d+[_-]col\d+$",
        r"(.+?)[_-]r\d+[_-]c\d+$",
    ]
    for pattern in patterns:
        match = re.match(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return value


def infer_patch_id(stem: str) -> str:
    scene = infer_original_scene_id(stem)
    return stem[len(scene):].strip("_-") if stem != scene else ""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def manifest_rows(ds_cfg: dict, split: str, seed: int) -> list[dict]:
    resolution = resolve_split(ds_cfg, split, seed=seed)
    tile_size = int(ds_cfg.get("image_size", 256))
    a_lookup = build_file_lookup(resolution.a_dir)
    b_lookup = build_file_lookup(resolution.b_dir)
    m_lookup = build_file_lookup(resolution.mask_dir, prefer_mask_ext=True)
    rows: list[dict] = []
    sample_index = 0
    for name in resolution.names:
        pre_path = resolve_path(a_lookup, name, "pre image", split)
        post_path = resolve_path(b_lookup, name, "post image", split)
        mask_path = resolve_path(m_lookup, name, "mask", split)
        width, height = image_size(pre_path)
        coords = sliding_coords(width, height, tile_size) if resolution.samples_are_tiles else [(None, None)]
        for crop_x, crop_y in coords:
            pre_stem = pre_path.stem
            post_stem = post_path.stem
            mask_stem = mask_path.stem
            scene_id = infer_original_scene_id(pre_stem)
            patch_id = infer_patch_id(pre_stem)
            pair_key = f"{pre_stem}__{post_stem}"
            if crop_x is not None and crop_y is not None:
                pair_key = f"{pair_key}__x{crop_x}_y{crop_y}"
            mask_key = mask_stem
            if crop_x is not None and crop_y is not None:
                mask_key = f"{mask_key}__x{crop_x}_y{crop_y}"
            rows.append({
                "split": split,
                "sample_index": sample_index,
                "pre_image_path": str(pre_path.resolve()),
                "post_image_path": str(post_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "pre_image_name": pre_path.name,
                "post_image_name": post_path.name,
                "mask_name": mask_path.name,
                "pre_stem": pre_stem,
                "post_stem": post_stem,
                "mask_stem": mask_stem,
                "original_scene_id": scene_id,
                "patch_id": patch_id,
                "crop_x": "" if crop_x is None else crop_x,
                "crop_y": "" if crop_y is None else crop_y,
                "width": width,
                "height": height,
                "pre_sha256": sha256_file(pre_path),
                "post_sha256": sha256_file(post_path),
                "mask_sha256": sha256_file(mask_path),
                "pair_key": pair_key,
                "mask_key": mask_key,
            })
            sample_index += 1
    return rows


def write_csv(path: str | Path, rows: list[dict], fields: Optional[list[str]] = None) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_manifest(path: str | Path) -> list[dict]:
    with open(repo_path(path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: str | Path, data: dict) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def avg_hash(path: str | Path, hash_size: int = 8) -> str:
    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size, hash_size))
        pixels = list(img.getdata())
    mean_val = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= mean_val else "0" for p in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    width = max(len(a), len(b))
    return (int(a, 16) ^ int(b, 16)).bit_count() if a and b else width * 4


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)
