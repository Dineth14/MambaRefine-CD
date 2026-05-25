"""Tile index builder for large-image change detection datasets.

Generates a CSV index where each row is one 256×256 (or custom-size) tile.
Results are cached: if the cache file already exists AND the cache key matches
the current config, the cached index is returned without re-scanning.

Usage::

    from data.tile_index import build_tile_index
    df = build_tile_index(
        image_paths_a       = [...],
        image_paths_b       = [...],
        mask_paths          = [...],
        tile_size           = 256,
        stride              = 128,
        min_change_pixels   = 1,
        include_empty_ratio = 0.25,
        cache_path          = Path("outputs/dataset_indices/train_256_128.csv"),
    )
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Public API ────────────────────────────────────────────────────────────────

def build_tile_index(
    image_paths_a: List[Path],
    image_paths_b: List[Path],
    mask_paths: List[Path],
    tile_size: int = 256,
    stride: int = 256,
    min_change_pixels: int = 1,
    include_empty_ratio: float = 0.25,
    cache_path: Optional[Path] = None,
) -> List[dict]:
    """Build (or load) a tile index.

    Each entry is a dict with keys:
        image_a_path, image_b_path, mask_path,
        x, y, tile_size,
        change_pixel_count, change_ratio, has_change

    Parameters
    ----------
    image_paths_a / b / mask_paths
        Parallel lists — must be the same length and order.
    tile_size
        Tile height = width in pixels.
    stride
        Sliding-window step. stride < tile_size produces overlap.
    min_change_pixels
        A tile is considered "has_change=True" only if its mask contains
        at least this many changed pixels.
    include_empty_ratio
        Fraction of no-change tiles to keep (0 = keep none, 1 = keep all).
        Applied after sampling so the final index retains diversity.
    cache_path
        If provided, save/load from this CSV file.  A sidecar .meta.json
        stores the cache key (hash of build parameters); if it mismatches,
        the index is rebuilt.
    """
    assert len(image_paths_a) == len(image_paths_b) == len(mask_paths), \
        "image_paths_a, image_paths_b, mask_paths must be the same length."

    # ── Cache logic ───────────────────────────────────────────────────────────
    cache_key = _make_cache_key(
        image_paths_a, tile_size, stride, min_change_pixels, include_empty_ratio
    )
    if cache_path is not None:
        cached = _load_cache(cache_path, cache_key)
        if cached is not None:
            logger.info(f"Tile index loaded from cache ({len(cached)} tiles): {cache_path}")
            return cached

    # ── Build from scratch ────────────────────────────────────────────────────
    logger.info(
        f"Building tile index: {len(image_paths_a)} images, "
        f"tile={tile_size}, stride={stride}"
    )

    change_tiles:   List[dict] = []
    no_change_tiles: List[dict] = []

    rng = np.random.default_rng(42)

    for a_path, b_path, m_path in zip(image_paths_a, image_paths_b, mask_paths):
        a_path = Path(a_path)
        b_path = Path(b_path)
        m_path = Path(m_path)

        try:
            img_w, img_h = Image.open(a_path).size
        except Exception as e:
            logger.warning(f"Cannot open {a_path}: {e} — skipping.")
            continue

        # Special case: image is already exactly tile_size × tile_size
        if img_w == tile_size and img_h == tile_size:
            tile_coords = [(0, 0)]
        else:
            tile_coords = _sliding_coords(img_w, img_h, tile_size, stride)

        # Load mask once per image (faster than opening per tile)
        try:
            mask_full = np.array(Image.open(m_path).convert("L"))
        except Exception as e:
            logger.warning(f"Cannot open mask {m_path}: {e} — using zero mask.")
            mask_full = np.zeros((img_h, img_w), dtype=np.uint8)

        for (x, y) in tile_coords:
            tile_mask = mask_full[y : y + tile_size, x : x + tile_size]
            n_changed = int((tile_mask > 127).sum())
            n_pixels  = tile_mask.size
            has_chg   = n_changed >= min_change_pixels

            entry = {
                "image_a_path":       str(a_path),
                "image_b_path":       str(b_path),
                "mask_path":          str(m_path),
                "x":                  x,
                "y":                  y,
                "tile_size":          tile_size,
                "change_pixel_count": n_changed,
                "change_ratio":       round(n_changed / max(1, n_pixels), 6),
                "has_change":         has_chg,
            }
            if has_chg:
                change_tiles.append(entry)
            else:
                no_change_tiles.append(entry)

    # ── Balance empty/change tiles ─────────────────────────────────────────────
    n_keep_empty = int(len(no_change_tiles) * include_empty_ratio)
    if n_keep_empty < len(no_change_tiles):
        idx = rng.choice(len(no_change_tiles), size=n_keep_empty, replace=False)
        no_change_tiles = [no_change_tiles[i] for i in sorted(idx)]

    index = change_tiles + no_change_tiles
    logger.info(
        f"Tile index built: {len(change_tiles)} change tiles + "
        f"{len(no_change_tiles)} no-change tiles = {len(index)} total"
    )

    # ── Save cache ────────────────────────────────────────────────────────────
    if cache_path is not None:
        _save_cache(index, cache_path, cache_key)

    return index


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sliding_coords(
    img_w: int, img_h: int, tile_size: int, stride: int
) -> List[tuple[int, int]]:
    """Return (x, y) top-left corners for all tiles covering an image.

    The last tile in each dimension is snapped so it does not exceed
    the image boundary (may overlap the previous tile if needed).
    """
    xs = list(range(0, img_w - tile_size + 1, stride))
    ys = list(range(0, img_h - tile_size + 1, stride))

    # Ensure the rightmost/bottom tile always reaches the border
    if not xs or xs[-1] + tile_size < img_w:
        xs.append(max(0, img_w - tile_size))
    if not ys or ys[-1] + tile_size < img_h:
        ys.append(max(0, img_h - tile_size))

    # Deduplicate while preserving order
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    return [(x, y) for y in ys for x in xs]


def _make_cache_key(
    paths: List[Path],
    tile_size: int,
    stride: int,
    min_change_pixels: int,
    include_empty_ratio: float,
) -> str:
    """Stable hash of parameters that affect the tile index content."""
    # Sort paths to make the key order-independent for same file-set
    sorted_names = sorted(str(p) for p in paths)
    blob = json.dumps({
        "n_images":           len(sorted_names),
        "first":              sorted_names[:3],
        "last":               sorted_names[-3:],
        "tile_size":          tile_size,
        "stride":             stride,
        "min_change_pixels":  min_change_pixels,
        "include_empty_ratio": include_empty_ratio,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


_CSV_FIELDS = [
    "image_a_path", "image_b_path", "mask_path",
    "x", "y", "tile_size",
    "change_pixel_count", "change_ratio", "has_change",
]


def _load_cache(cache_path: Path, expected_key: str) -> Optional[List[dict]]:
    """Return cached index if cache_path exists and key matches."""
    meta_path = cache_path.with_suffix(".meta.json")
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("cache_key") != expected_key:
            logger.info("Tile cache key mismatch — rebuilding.")
            return None
        rows = []
        with open(cache_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "image_a_path":       row["image_a_path"],
                    "image_b_path":       row["image_b_path"],
                    "mask_path":          row["mask_path"],
                    "x":                  int(row["x"]),
                    "y":                  int(row["y"]),
                    "tile_size":          int(row["tile_size"]),
                    "change_pixel_count": int(row["change_pixel_count"]),
                    "change_ratio":       float(row["change_ratio"]),
                    "has_change":         row["has_change"].lower() in ("true", "1"),
                })
        return rows
    except Exception as e:
        logger.warning(f"Failed to load tile cache: {e} — rebuilding.")
        return None


def _save_cache(index: List[dict], cache_path: Path, cache_key: str) -> None:
    """Write tile index to CSV and sidecar meta JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(".meta.json")
    with open(cache_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(index)
    with open(meta_path, "w") as f:
        json.dump({"cache_key": cache_key, "n_tiles": len(index)}, f)
    logger.info(f"Tile index saved to {cache_path} ({len(index)} tiles)")
