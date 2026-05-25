"""Dataset verification for the clean A/B/Mask layout."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.cd_dataset import SUPPORTED_EXTS


def _files_by_stem(path: Path) -> dict[str, Path]:
    return {p.stem: p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_dirs(cfg, split: str) -> tuple[Path, Path, Path]:
    root = Path(cfg.data.root) / split
    return root / cfg.data.a_folder, root / cfg.data.b_folder, root / cfg.data.mask_folder


def _verify_split(cfg, split: str) -> dict:
    a_dir, b_dir, mask_dir = _split_dirs(cfg, split)
    for directory in (a_dir, b_dir, mask_dir):
        if not directory.exists():
            raise ValueError(f"Missing required directory: {directory}")

    a = _files_by_stem(a_dir)
    b = _files_by_stem(b_dir)
    m = _files_by_stem(mask_dir)
    if set(a) != set(b) or set(a) != set(m):
        raise ValueError(
            f"Filename stem mismatch in split '{split}': "
            f"A={len(a)}, B={len(b)}, Mask={len(m)}"
        )
    if not a:
        raise ValueError(f"Split '{split}' contains no samples")

    positives = 0
    total = 0
    hashes = set()
    for stem in sorted(a):
        with Image.open(a[stem]) as ia, Image.open(b[stem]) as ib, Image.open(m[stem]) as im:
            if ia.size != ib.size or ia.size != im.size:
                raise ValueError(f"Size mismatch for {split}/{stem}: A={ia.size}, B={ib.size}, Mask={im.size}")
            mask = np.array(im.convert("L"))
            unique = set(np.unique(mask).tolist())
            if len(unique) > 2 and not all(v in {0, 1, 255} for v in unique):
                raise ValueError(f"Mask is not binary-like for {split}/{stem}: values={sorted(unique)[:10]}")
            bin_mask = mask > int(cfg.data.binary_threshold)
            positives += int(bin_mask.sum())
            total += int(bin_mask.size)
            if bool(cfg.data.get("check_hash_overlap", True)):
                hashes.add(_sha256(a[stem]))
    return {
        "split": split,
        "count": len(a),
        "stems": set(a),
        "hashes": hashes,
        "gt_positive_ratio": positives / max(total, 1),
    }


def verify_dataset(cfg) -> bool:
    summaries = [_verify_split(cfg, split) for split in (cfg.data.train_dir, cfg.data.val_dir, cfg.data.test_dir)]
    if bool(cfg.data.get("check_split_overlap", True)):
        names = [(item["split"], item["stems"]) for item in summaries]
        for i, (a_name, a_stems) in enumerate(names):
            for b_name, b_stems in names[i + 1:]:
                overlap = a_stems & b_stems
                if overlap:
                    raise ValueError(f"Filename overlap between {a_name} and {b_name}: {sorted(overlap)[:5]}")
    if bool(cfg.data.get("check_hash_overlap", True)):
        names = [(item["split"], item["hashes"]) for item in summaries]
        for i, (a_name, a_hashes) in enumerate(names):
            for b_name, b_hashes in names[i + 1:]:
                overlap = a_hashes & b_hashes
                if overlap:
                    raise ValueError(f"Image hash overlap between {a_name} and {b_name}: {len(overlap)} duplicate image(s)")
    cfg._dataset_verification = [{k: v for k, v in item.items() if k not in {"stems", "hashes"}} for item in summaries]
    return True


def dataset_report(cfg) -> list[dict]:
    verify_dataset(cfg)
    return list(cfg._dataset_verification)
