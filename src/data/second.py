"""SECOND semantic change detection dataset.

Optimized for training throughput:

* split index caching under outputs/dataset_indices/
* optional precomputed binary mask cache under outputs/second_binary_masks/
* optional RAM caching for images and masks
* cv2-backed image decoding when available
* no directory scanning or filename matching inside __getitem__
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import build_train_transforms, norm_tensor

try:
    import cv2  # type: ignore
    _CV2 = True
except ImportError:
    cv2 = None
    _CV2 = False

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

_A_CANDS = ["A", "im1", "img1", "imageA", "T1", "time1"]
_B_CANDS = ["B", "im2", "img2", "imageB", "T2", "time2"]
_LABEL_A_CANDS = ["label1", "labelA", "label_t1", "semantic1", "sem1"]
_LABEL_B_CANDS = ["label2", "labelB", "label_t2", "semantic2", "sem2"]
_BINARY_LABEL_CANDS = ["label", "change", "change_label", "mask", "binary"]
_SECOND_DEFAULT_COLOR_MAP = {
    (0, 0, 0): 0,
    (255, 255, 255): 0,
    (0, 128, 0): 1,
    (128, 128, 128): 2,
    (0, 255, 0): 3,
    (0, 0, 255): 4,
    (128, 0, 0): 5,
    (255, 0, 0): 6,
}

_SPLIT_ALIASES = {
    "train": ["train", "trainset", "training", "Train"],
    "val": ["val", "valset", "valid", "validation", "Val"],
    "test": ["test", "testset", "testing", "Test"],
}

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_CACHE_DIR = _REPO_ROOT / "outputs" / "dataset_indices"
_DEFAULT_BINARY_CACHE_DIR = _REPO_ROOT / "outputs" / "second_binary_masks"


def _resolve_repo_path(path_value: str | Path | None, default: Path) -> Path:
    if path_value in (None, ""):
        return default
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _detect_dir(parent: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        path = parent / candidate
        if path.is_dir():
            return path
    return None


def _read_split_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _find_split_dir(root: Path, split: str, configured_name: Optional[str]) -> Optional[Path]:
    tried: list[str] = []
    if configured_name:
        tried.append(configured_name)
    tried.extend(_SPLIT_ALIASES.get(split, [split]))
    seen = set()
    for name in tried:
        if name in seen:
            continue
        seen.add(name)
        path = root / name
        if path.is_dir():
            return path
    return None


def _build_file_lookup(directory: Optional[Path]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    if directory is None:
        return lookup
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in _EXTS:
            continue
        for token in (path.name, path.stem, path.name.lower(), path.stem.lower()):
            lookup.setdefault(token, path)
    return lookup


def _resolve_lookup_path(lookup: dict[str, Path], sample_id: str, kind: str, split: str) -> Path:
    candidates = [
        sample_id,
        Path(sample_id).name,
        Path(sample_id).stem,
        sample_id.lower(),
        Path(sample_id).name.lower(),
        Path(sample_id).stem.lower(),
    ]
    for candidate in candidates:
        path = lookup.get(candidate)
        if path is not None:
            return path
    raise FileNotFoundError(
        f"SECONDDataset [{split}]: could not resolve {kind} for sample '{sample_id}'."
    )


def _list_image_ids(directory: Optional[Path]) -> list[str]:
    if directory is None:
        return []
    return sorted(path.stem for path in directory.iterdir() if path.suffix.lower() in _EXTS)


def _load_image_rgb(path: Path) -> np.ndarray:
    if _CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(path).convert("RGB"))


def _load_mask_raw(path: Path) -> np.ndarray:
    if _CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(path)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr
    return np.array(Image.open(path))


def _load_binary_mask(path: Path) -> np.ndarray:
    if _CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(path)
        return arr
    return np.array(Image.open(path).convert("L"))


def _read_image_size(path: Path) -> tuple[int, int]:
    if _CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(path)
        height, width = arr.shape[:2]
        return int(width), int(height)
    with Image.open(path) as img:
        width, height = img.size
    return int(width), int(height)


def _discover_semantic_color_map(
    directories: Sequence[Optional[Path]],
    num_classes: int,
    sample_limit_per_dir: int = 64,
) -> dict[tuple[int, int, int], int]:
    colors: set[tuple[int, int, int]] = set()
    seen_dirs: set[Path] = set()
    for directory in directories:
        if directory is None or not directory.is_dir() or directory in seen_dirs:
            continue
        seen_dirs.add(directory)
        scanned = 0
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in _EXTS:
                continue
            arr = _load_mask_raw(path)
            if arr.ndim != 3 or arr.shape[2] < 3:
                continue
            unique = np.unique(arr[..., :3].reshape(-1, 3), axis=0)
            colors.update(tuple(int(v) for v in color) for color in unique.tolist())
            scanned += 1
            if len(colors) >= num_classes or scanned >= sample_limit_per_dir:
                break
        if len(colors) >= num_classes:
            break
    return {color: idx for idx, color in enumerate(sorted(colors)[:num_classes])}


def _build_semantic_lookup_arrays(
    color_map: dict[tuple[int, int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    if not color_map:
        return (
            np.empty((0,), dtype=np.uint32),
            np.empty((0,), dtype=np.int64),
        )
    items = sorted(
        (((r << 16) | (g << 8) | b), cls_id)
        for (r, g, b), cls_id in color_map.items()
    )
    keys = np.asarray([item[0] for item in items], dtype=np.uint32)
    values = np.asarray([item[1] for item in items], dtype=np.int64)
    return keys, values


def _normalize_palette(palette: Optional[dict[Any, Any] | list[Any]]) -> dict[tuple[int, int, int], int]:
    if not palette:
        return {}
    normalized: dict[tuple[int, int, int], int] = {}
    items = palette.items() if isinstance(palette, dict) else enumerate(palette)
    for class_id, color in items:
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            continue
        normalized[tuple(int(v) for v in color[:3])] = int(class_id)
    return normalized


def _decode_semantic_label(
    arr: np.ndarray,
    color_map: dict[tuple[int, int, int], int],
    ignore_index: int,
    lut_keys: Optional[np.ndarray] = None,
    lut_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.int64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported SECOND semantic label shape: {arr.shape}")
    rgb = arr[..., :3].astype(np.uint8)
    if lut_keys is None or lut_values is None:
        lut_keys, lut_values = _build_semantic_lookup_arrays(color_map)
    if lut_keys.size == 0:
        return np.full(rgb.shape[:2], ignore_index, dtype=np.int64)

    flat_keys = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    ).reshape(-1)
    indices = np.searchsorted(lut_keys, flat_keys)
    decoded = np.full(flat_keys.shape[0], ignore_index, dtype=np.int64)
    valid = indices < lut_keys.size
    if np.any(valid):
        valid_indices = indices[valid]
        matches = lut_keys[valid_indices] == flat_keys[valid]
        if np.any(matches):
            flat_valid = np.flatnonzero(valid)
            decoded_positions = flat_valid[matches]
            decoded[decoded_positions] = lut_values[valid_indices[matches]]
    return decoded.reshape(rgb.shape[:2])


def _config_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _estimate_bytes(entries: Sequence[dict[str, Any]], include_images: bool, include_masks: bool) -> int:
    total = 0
    for entry in entries:
        width, height = entry["image_size"]
        if include_images:
            total += width * height * 3 * 2
        if include_masks:
            total += width * height * 3
    return total


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def _build_sample_entries(
    *,
    split: str,
    base: Path,
    sample_ids: Sequence[str],
    a_lookup: dict[str, Path],
    b_lookup: dict[str, Path],
    label_a_lookup: dict[str, Path],
    label_b_lookup: dict[str, Path],
    binary_lookup: dict[str, Path],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        image_a_path = _resolve_lookup_path(a_lookup, sample_id, "image_a", split)
        image_b_path = _resolve_lookup_path(b_lookup, sample_id, "image_b", split)
        label_a_path = _resolve_lookup_path(label_a_lookup, sample_id, "label_a", split) if label_a_lookup else None
        label_b_path = _resolve_lookup_path(label_b_lookup, sample_id, "label_b", split) if label_b_lookup else None
        binary_path = _resolve_lookup_path(binary_lookup, sample_id, "binary_label", split) if binary_lookup else None
        width, height = _read_image_size(image_a_path)
        valid = image_b_path.exists() and (binary_path is not None or (label_a_path is not None and label_b_path is not None))
        entries.append({
            "sample_id": sample_id,
            "split": split,
            "base_dir": str(base),
            "image_a_path": str(image_a_path),
            "image_b_path": str(image_b_path),
            "label_a_path": str(label_a_path) if label_a_path else None,
            "label_b_path": str(label_b_path) if label_b_path else None,
            "binary_mask_path": str(binary_path) if binary_path else None,
            "image_size": [width, height],
            "valid": bool(valid),
        })
    return entries


def _cache_payload_ok(cache_path: Path, signature: str) -> Optional[list[dict[str, Any]]]:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("config_hash") != signature:
        return None
    entries = payload.get("entries")
    return entries if isinstance(entries, list) else None


def _write_index_cache(cache_path: Path, signature: str, entries: Sequence[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"config_hash": signature, "entries": list(entries)}, indent=2),
        encoding="utf-8",
    )


def query_nvidia_smi_utilization(device_index: int) -> Optional[dict[str, float]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        utilization, memory_used, memory_total = [float(v.strip()) for v in proc.stdout.strip().split(",")[:3]]
        return {
            "gpu_utilization": utilization,
            "memory_used_mb": memory_used,
            "memory_total_mb": memory_total,
        }
    except Exception:
        return None


class SECONDDataset(Dataset):
    """SECOND dataset with binary and semantic modes."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        val_ratio: float = 0.2,
        seed: int = 42,
        augment: bool = True,
        mode: str = "binary",
        task_type: str = "semantic_change",
        ignore_index: int = 255,
        binary_from_semantic: bool = True,
        num_classes: int = 7,
        a_candidates: Optional[list[str]] = None,
        b_candidates: Optional[list[str]] = None,
        label_a_candidates: Optional[list[str]] = None,
        label_b_candidates: Optional[list[str]] = None,
        binary_label_candidates: Optional[list[str]] = None,
        train_split: Optional[str] = "train",
        val_split: Optional[str] = "val",
        test_split: Optional[str] = "test",
        precompute_binary_masks: bool = False,
        second_binary_cache_dir: str | Path | None = None,
        cache_images_in_ram: bool = False,
        cache_masks_in_ram: bool = False,
        profile_enabled: bool = False,
        second_label_palette: Optional[dict[Any, Any]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.size = int(image_size)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.mode = str(mode).lower()
        self.task_type = str(task_type)
        self.ignore_index = int(ignore_index)
        self.binary_from_semantic = bool(binary_from_semantic)
        self.num_classes = int(num_classes)
        self.do_augment = bool(augment and split == "train")
        self.precompute_binary_masks = bool(precompute_binary_masks)
        self.second_binary_cache_dir = _resolve_repo_path(second_binary_cache_dir, _DEFAULT_BINARY_CACHE_DIR)
        self.cache_images_in_ram = bool(cache_images_in_ram)
        self.cache_masks_in_ram = bool(cache_masks_in_ram)
        self.profile_enabled = bool(profile_enabled)
        self.second_label_palette = _normalize_palette(second_label_palette)
        self.profile_stats = {
            "getitem_calls": 0,
            "io_time": 0.0,
            "decode_time": 0.0,
            "mask_time": 0.0,
            "transform_time": 0.0,
        }
        self._array_cache: dict[str, np.ndarray] = {}
        self._semantic_label_cache: dict[str, np.ndarray] = {}
        self._sample_size_summary: dict[str, Any] = {}

        if self.mode not in {"binary", "semantic"}:
            raise ValueError(f"SECONDDataset mode must be 'binary' or 'semantic', got {mode!r}")
        if not self.root.is_dir():
            raise FileNotFoundError(f"SECONDDataset [{split}]: root directory does not exist: {self.root}")

        self.split_names = {"train": train_split, "val": val_split, "test": test_split}
        self.a_candidates = a_candidates or _A_CANDS
        self.b_candidates = b_candidates or _B_CANDS
        self.label_a_candidates = label_a_candidates or _LABEL_A_CANDS
        self.label_b_candidates = label_b_candidates or _LABEL_B_CANDS
        self.binary_label_candidates = binary_label_candidates or _BINARY_LABEL_CANDS

        self.split_file_name = self.split_names.get(split) or split
        self.split_file = self.root / f"{self.split_file_name}.txt"
        self.split_dir = _find_split_dir(self.root, split, self.split_names.get(split))
        self.base_dir = self._resolve_base_dir()

        self.a_dir = _detect_dir(self.base_dir, self.a_candidates)
        self.b_dir = _detect_dir(self.base_dir, self.b_candidates)
        self.label_a_dir = _detect_dir(self.base_dir, self.label_a_candidates)
        self.label_b_dir = _detect_dir(self.base_dir, self.label_b_candidates)
        self.binary_label_dir = _detect_dir(self.base_dir, self.binary_label_candidates)

        self._check_structure(self.base_dir)
        self.semantic_color_map = self._build_semantic_color_map()
        self.semantic_lut_keys, self.semantic_lut_values = _build_semantic_lookup_arrays(self.semantic_color_map)
        self.entries = self._load_or_build_entries()
        if self.mode == "binary" and self.precompute_binary_masks:
            self._ensure_binary_mask_cache()

        transform_targets = {"image_b": "image"}
        if self.mode == "semantic":
            transform_targets.update({
                "label_a": "mask",
                "label_b": "mask",
                "change_mask": "mask",
                "ignore_mask": "mask",
            })
        self.transform = build_train_transforms(self.size, additional_targets=transform_targets) if self.do_augment else None
        self.tiles = None if split == "train" else self._build_tiles()
        self._log_size_summary()
        self._prime_ram_cache()
        self._warn_unexpected_class_ids()
        self._log_class_histogram()

    def _resolve_base_dir(self) -> Path:
        base = self.split_dir if self.split_dir is not None else self.root
        if self.split_dir is None:
            root_a_dir = _detect_dir(self.root, self.a_candidates)
            root_b_dir = _detect_dir(self.root, self.b_candidates)
            train_dir = _find_split_dir(self.root, "train", self.split_names.get("train"))
            if root_a_dir is None and root_b_dir is None and train_dir is not None:
                base = train_dir
        return base

    def _index_signature_payload(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "split": self.split,
            "split_name": self.split_file_name,
            "base_dir": str(self.base_dir),
            "val_ratio": self.val_ratio,
            "seed": self.seed,
            "mode": self.mode,
            "ignore_index": self.ignore_index,
            "num_classes": self.num_classes,
            "second_label_palette": {
                str(class_id): list(color)
                for color, class_id in self.semantic_color_map.items()
            },
            "a_candidates": list(self.a_candidates),
            "b_candidates": list(self.b_candidates),
            "label_a_candidates": list(self.label_a_candidates),
            "label_b_candidates": list(self.label_b_candidates),
            "binary_candidates": list(self.binary_label_candidates),
        }

    def _index_cache_path(self) -> Path:
        return _INDEX_CACHE_DIR / f"SECOND_index_{self.split}.json"

    def _build_semantic_color_map(self) -> dict[tuple[int, int, int], int]:
        if self.second_label_palette:
            return self.second_label_palette
        return dict(_SECOND_DEFAULT_COLOR_MAP)

    def _load_or_build_entries(self) -> list[dict[str, Any]]:
        signature = _config_signature(self._index_signature_payload())
        cache_path = self._index_cache_path()
        cached = _cache_payload_ok(cache_path, signature)
        if cached is not None:
            return cached

        a_lookup = _build_file_lookup(self.a_dir)
        b_lookup = _build_file_lookup(self.b_dir)
        label_a_lookup = _build_file_lookup(self.label_a_dir)
        label_b_lookup = _build_file_lookup(self.label_b_dir)
        binary_lookup = _build_file_lookup(self.binary_label_dir)

        if self.split_file.exists():
            sample_ids = _read_split_file(self.split_file)
        else:
            sample_ids = self._discover_sample_ids(label_a_lookup, label_b_lookup, binary_lookup)
            if self.split_dir is None:
                rng = random.Random(self.seed)
                shuffled = list(sample_ids)
                rng.shuffle(shuffled)
                n_val = max(1, int(len(shuffled) * self.val_ratio)) if shuffled else 0
                if self.split == "train":
                    sample_ids = sorted(shuffled[n_val:])
                elif self.split == "val":
                    sample_ids = sorted(shuffled[:n_val])
                else:
                    sample_ids = sorted(shuffled)

        entries = _build_sample_entries(
            split=self.split,
            base=self.base_dir,
            sample_ids=sample_ids,
            a_lookup=a_lookup,
            b_lookup=b_lookup,
            label_a_lookup=label_a_lookup,
            label_b_lookup=label_b_lookup,
            binary_lookup=binary_lookup,
        )
        _write_index_cache(cache_path, signature, entries)
        return entries

    def _discover_sample_ids(
        self,
        label_a_lookup: dict[str, Path],
        label_b_lookup: dict[str, Path],
        binary_lookup: dict[str, Path],
    ) -> list[str]:
        ids = set(_list_image_ids(self.a_dir)) & set(_list_image_ids(self.b_dir))
        if binary_lookup:
            ids &= set(_list_image_ids(self.binary_label_dir))
        elif label_a_lookup and label_b_lookup:
            ids &= set(_list_image_ids(self.label_a_dir)) & set(_list_image_ids(self.label_b_dir))
        return sorted(ids)

    def _check_structure(self, base: Path) -> None:
        found_dirs = sorted(path.name for path in base.iterdir() if path.is_dir())
        if self.a_dir is None or self.b_dir is None:
            raise FileNotFoundError(
                f"SECONDDataset [{self.split}]: could not detect image folders under {base}. "
                f"Expected image A candidates {list(self.a_candidates)} and image B candidates {list(self.b_candidates)}. "
                f"Found folders: {found_dirs}"
            )
        has_binary = self.binary_label_dir is not None
        has_semantic = self.label_a_dir is not None and self.label_b_dir is not None
        if self.mode == "binary":
            if not has_binary and not has_semantic:
                raise FileNotFoundError(
                    f"SECONDDataset [{self.split}]: missing labels under {base}. Expected binary candidates {list(self.binary_label_candidates)} "
                    f"or semantic candidates {list(self.label_a_candidates)} / {list(self.label_b_candidates)}. Found folders: {found_dirs}"
                )
            if not has_binary and not self.binary_from_semantic:
                raise FileNotFoundError(
                    f"SECONDDataset [{self.split}]: binary labels not found and binary_from_semantic is disabled. "
                    f"Expected binary candidates {list(self.binary_label_candidates)}. Found folders: {found_dirs}"
                )
        elif not has_semantic:
            raise FileNotFoundError(
                f"SECONDDataset [{self.split}]: semantic mode requires label A/B folders under {base}. "
                f"Expected candidates {list(self.label_a_candidates)} and {list(self.label_b_candidates)}. Found folders: {found_dirs}"
            )

    def _ensure_binary_mask_cache(self) -> None:
        if any(entry.get("binary_mask_path") for entry in self.entries):
            return
        cache_dir = self.second_binary_cache_dir / self.split
        cache_dir.mkdir(parents=True, exist_ok=True)
        total_px = 0
        changed_px = 0
        ignored_px = 0
        created = 0
        for entry in self.entries:
            if not entry.get("valid", False):
                continue
            target_path = cache_dir / f"{entry['sample_id']}.png"
            entry["binary_mask_path"] = str(target_path)
            if target_path.exists():
                mask = _load_binary_mask(target_path)
                total_px += int(mask.size)
                changed_px += int((mask > 0).sum())
                continue
            label_a = self._load_semantic_label(Path(entry["label_a_path"])) if entry.get("label_a_path") else None
            label_b = self._load_semantic_label(Path(entry["label_b_path"])) if entry.get("label_b_path") else None
            mask, ignore_mask = self._derive_masks(label_a, label_b, None)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(target_path)
            created += 1
            total_px += int(mask.size)
            changed_px += int(mask.sum())
            ignored_px += int(ignore_mask.sum())
        if created:
            logger.info(
                "SECOND [%s] precomputed %s binary masks | change_ratio=%.4f | ignore_ratio=%.4f",
                self.split,
                created,
                float(changed_px / total_px) if total_px else 0.0,
                float(ignored_px / total_px) if total_px else 0.0,
            )
        _write_index_cache(self._index_cache_path(), _config_signature(self._index_signature_payload()), self.entries)

    def _log_size_summary(self) -> None:
        sizes = Counter(tuple(entry["image_size"]) for entry in self.entries if entry.get("valid", False))
        unique_sizes = sorted(sizes)
        self._sample_size_summary = {
            "unique_image_sizes": unique_sizes,
            "uniform_image_size": len(unique_sizes) <= 1,
            "requires_padding": any(width < self.size or height < self.size for width, height in unique_sizes),
            "target_image_size": self.size,
        }
        logger.info("SECOND [%s] sample image sizes: %s", self.split, unique_sizes[:6])
        if len(unique_sizes) > 1:
            logger.warning(
                "SECOND [%s] has %s unique source image sizes. Random crop/pad remains on the CPU path.",
                self.split,
                len(unique_sizes),
            )
        if self._sample_size_summary["requires_padding"]:
            logger.warning(
                "SECOND [%s] includes images smaller than crop size %s; zero-padding will occur before cropping.",
                self.split,
                self.size,
            )
        elif len(unique_sizes) == 1 and unique_sizes[0] == (self.size, self.size):
            logger.info("SECOND [%s] matches target crop size exactly; no resize/pad overhead on the hot path.", self.split)
        elif len(unique_sizes) == 1:
            width, height = unique_sizes[0]
            logger.info(
                "SECOND [%s] source images are %sx%s while target tiles are %sx%s. "
                "The loader uses direct crop/tile extraction and avoids per-batch resize.",
                self.split,
                width,
                height,
                self.size,
                self.size,
            )

    def _prime_ram_cache(self) -> None:
        if not self.cache_images_in_ram and not self.cache_masks_in_ram:
            return
        estimate = _estimate_bytes(self.entries, self.cache_images_in_ram, self.cache_masks_in_ram)
        logger.info(
            "SECOND [%s] RAM cache requested | images=%s masks=%s | estimated=%s",
            self.split,
            self.cache_images_in_ram,
            self.cache_masks_in_ram,
            _format_gb(estimate),
        )
        for entry in self.entries:
            if not entry.get("valid", False):
                continue
            if self.cache_images_in_ram:
                self._array_cache.setdefault(entry["image_a_path"], _load_image_rgb(Path(entry["image_a_path"])))
                self._array_cache.setdefault(entry["image_b_path"], _load_image_rgb(Path(entry["image_b_path"])))
            if self.cache_masks_in_ram:
                if entry.get("binary_mask_path"):
                    self._array_cache.setdefault(entry["binary_mask_path"], _load_binary_mask(Path(entry["binary_mask_path"])))
                needs_semantic = self.mode == "semantic" or not entry.get("binary_mask_path")
                if needs_semantic and entry.get("label_a_path"):
                    self._semantic_label_cache.setdefault(
                        entry["label_a_path"],
                        self._load_semantic_label(Path(entry["label_a_path"])),
                    )
                if needs_semantic and entry.get("label_b_path"):
                    self._semantic_label_cache.setdefault(
                        entry["label_b_path"],
                        self._load_semantic_label(Path(entry["label_b_path"])),
                    )

    def reset_profile_stats(self) -> None:
        for key in self.profile_stats:
            self.profile_stats[key] = 0 if key == "getitem_calls" else 0.0

    def get_profile_stats(self) -> dict[str, float]:
        return dict(self.profile_stats)

    def _build_tiles(self) -> list[tuple[int, int, int]]:
        tiles: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for entry_index, entry in enumerate(self.entries):
            width, height = entry["image_size"]
            rows = list(range(0, max(height - self.size + 1, 1), self.size))
            cols = list(range(0, max(width - self.size + 1, 1), self.size))
            if not rows or rows[-1] != max(height - self.size, 0):
                rows.append(max(height - self.size, 0))
            if not cols or cols[-1] != max(width - self.size, 0):
                cols.append(max(width - self.size, 0))
            for row in rows:
                for col in cols:
                    key = (entry_index, row, col)
                    if key not in seen:
                        seen.add(key)
                        tiles.append(key)
        return tiles

    def _warn_unexpected_class_ids(self) -> None:
        expected = set(range(self.num_classes)) | {self.ignore_index}
        observed: set[int] = set()
        for entry in self.entries[: min(8, len(self.entries))]:
            if not entry.get("label_a_path") or not entry.get("label_b_path"):
                continue
            try:
                label_a = self._load_semantic_label(Path(entry["label_a_path"]))
                label_b = self._load_semantic_label(Path(entry["label_b_path"]))
            except Exception:
                continue
            observed.update(int(v) for v in np.unique(label_a))
            observed.update(int(v) for v in np.unique(label_b))
        unexpected = sorted(v for v in observed if v not in expected)
        if unexpected:
            warnings.warn(f"SECONDDataset [{self.split}] observed unexpected class IDs: {unexpected}", stacklevel=2)

    def _log_class_histogram(self) -> None:
        if self.label_a_dir is None or self.label_b_dir is None:
            return
        hist_a = Counter()
        hist_b = Counter()
        valid_pixels = 0
        changed_pixels = 0
        for entry in self.entries:
            if not entry.get("valid", False) or not entry.get("label_a_path") or not entry.get("label_b_path"):
                continue
            label_a = self._load_semantic_label(Path(entry["label_a_path"]))
            label_b = self._load_semantic_label(Path(entry["label_b_path"]))
            valid = (label_a != self.ignore_index) & (label_b != self.ignore_index)
            changed = valid & (label_a != label_b)
            valid_pixels += int(valid.sum())
            changed_pixels += int(changed.sum())
            if valid.any():
                classes_a, counts_a = np.unique(label_a[valid], return_counts=True)
                classes_b, counts_b = np.unique(label_b[valid], return_counts=True)
                hist_a.update({int(cls): int(cnt) for cls, cnt in zip(classes_a.tolist(), counts_a.tolist())})
                hist_b.update({int(cls): int(cnt) for cls, cnt in zip(classes_b.tolist(), counts_b.tolist())})
        if valid_pixels == 0:
            logger.warning("SECOND [%s] class histogram skipped because no valid semantic pixels were found.", self.split)
            return
        logger.info(
            "SECOND [%s] semantic histogram | valid_pixels=%s | changed_ratio=%.4f | t1=%s | t2=%s",
            self.split,
            valid_pixels,
            float(changed_pixels / valid_pixels),
            dict(sorted(hist_a.items())),
            dict(sorted(hist_b.items())),
        )

    def __len__(self) -> int:
        return len(self.entries) if self.split == "train" else len(self.tiles or [])

    def _maybe_cached(self, path: Optional[str], loader) -> Optional[np.ndarray]:
        if path is None:
            return None
        if path in self._array_cache:
            return self._array_cache[path]
        arr = loader(Path(path))
        if self.cache_images_in_ram or self.cache_masks_in_ram:
            self._array_cache.setdefault(path, arr)
        return arr

    def _load_semantic_label(self, path: Path) -> np.ndarray:
        cache_key = str(path)
        cached = self._semantic_label_cache.get(cache_key)
        if cached is not None:
            return cached
        raw = self._maybe_cached(cache_key, _load_mask_raw)
        assert raw is not None
        decoded = (
            raw.astype(np.int64)
            if raw.ndim == 2
            else _decode_semantic_label(
                raw,
                self.semantic_color_map,
                self.ignore_index,
                self.semantic_lut_keys,
                self.semantic_lut_values,
            )
        )
        if self.cache_masks_in_ram:
            self._semantic_label_cache.setdefault(cache_key, decoded)
        return decoded

    def _load_entry_arrays(self, entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        image_a = self._maybe_cached(entry["image_a_path"], _load_image_rgb)
        image_b = self._maybe_cached(entry["image_b_path"], _load_image_rgb)
        assert image_a is not None and image_b is not None
        binary_label = self._maybe_cached(entry.get("binary_mask_path"), _load_binary_mask) if entry.get("binary_mask_path") else None
        needs_semantic = self.mode == "semantic" or binary_label is None
        label_a = self._load_semantic_label(Path(entry["label_a_path"])) if needs_semantic and entry.get("label_a_path") else None
        label_b = self._load_semantic_label(Path(entry["label_b_path"])) if needs_semantic and entry.get("label_b_path") else None
        return image_a, image_b, label_a, label_b, binary_label

    def _pad_to_min_size(self, *arrays: np.ndarray | None) -> list[np.ndarray | None]:
        padded: list[np.ndarray | None] = []
        heights = [arr.shape[0] for arr in arrays if arr is not None]
        widths = [arr.shape[1] for arr in arrays if arr is not None]
        target_h = max(max(heights, default=self.size), self.size)
        target_w = max(max(widths, default=self.size), self.size)
        for arr in arrays:
            if arr is None:
                padded.append(None)
                continue
            pad_h = max(0, target_h - arr.shape[0])
            pad_w = max(0, target_w - arr.shape[1])
            if pad_h == 0 and pad_w == 0:
                padded.append(arr)
                continue
            if arr.ndim == 3:
                padded.append(np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0))
            else:
                fill_value = self.ignore_index if arr.dtype.kind in {"u", "i"} else 0
                padded.append(np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=fill_value))
        return padded

    def _crop(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        label_a: Optional[np.ndarray],
        label_b: Optional[np.ndarray],
        binary_label: Optional[np.ndarray],
        row: int,
        col: int,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        end_row = row + self.size
        end_col = col + self.size
        return (
            image_a[row:end_row, col:end_col],
            image_b[row:end_row, col:end_col],
            None if label_a is None else label_a[row:end_row, col:end_col],
            None if label_b is None else label_b[row:end_row, col:end_col],
            None if binary_label is None else binary_label[row:end_row, col:end_col],
        )

    def _derive_masks(
        self,
        label_a: Optional[np.ndarray],
        label_b: Optional[np.ndarray],
        binary_label: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if binary_label is not None:
            mask = (binary_label > 0).astype(np.uint8)
            ignore_mask = np.zeros_like(mask, dtype=np.uint8)
            return mask, ignore_mask
        if label_a is None or label_b is None:
            raise FileNotFoundError(f"SECONDDataset [{self.split}]: neither binary change labels nor semantic labels are available.")
        valid = (label_a != self.ignore_index) & (label_b != self.ignore_index)
        mask = ((label_a != label_b) & valid).astype(np.uint8)
        ignore_mask = (~valid).astype(np.uint8)
        return mask, ignore_mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t0 = time.perf_counter() if self.profile_enabled else 0.0
        load_start = time.perf_counter() if self.profile_enabled else 0.0
        if self.split == "train":
            entry = self.entries[idx]
            image_a, image_b, label_a, label_b, binary_label = self._load_entry_arrays(entry)
            image_a, image_b, label_a, label_b, binary_label = self._pad_to_min_size(image_a, image_b, label_a, label_b, binary_label)
            assert image_a is not None and image_b is not None
            height, width = image_a.shape[:2]
            row = random.randint(0, max(height - self.size, 0))
            col = random.randint(0, max(width - self.size, 0))
        else:
            entry_index, row, col = (self.tiles or [])[idx]
            entry = self.entries[entry_index]
            image_a, image_b, label_a, label_b, binary_label = self._load_entry_arrays(entry)
            image_a, image_b, label_a, label_b, binary_label = self._pad_to_min_size(image_a, image_b, label_a, label_b, binary_label)
            assert image_a is not None and image_b is not None
        load_elapsed = time.perf_counter() - load_start if self.profile_enabled else 0.0
        if self.profile_enabled:
            self.profile_stats["decode_time"] += load_elapsed

        image_a, image_b, label_a, label_b, binary_label = self._crop(image_a, image_b, label_a, label_b, binary_label, row, col)
        t_mask = time.perf_counter() if self.profile_enabled else 0.0
        change_mask, ignore_mask = self._derive_masks(label_a, label_b, binary_label)
        mask_elapsed = time.perf_counter() - t_mask if self.profile_enabled else 0.0
        if self.profile_enabled:
            self.profile_stats["mask_time"] += mask_elapsed

        t_tf = time.perf_counter() if self.profile_enabled else 0.0
        if self.do_augment and self.transform is not None:
            if self.mode == "semantic":
                aug = self.transform(
                    image=image_a,
                    image_b=image_b,
                    label_a=label_a.astype(np.int64),
                    label_b=label_b.astype(np.int64),
                    change_mask=change_mask,
                    ignore_mask=ignore_mask,
                )
                tensor_a = aug["image"].float()
                tensor_b = aug["image_b"].float()
                tensor_label_a = aug["label_a"].long()
                tensor_label_b = aug["label_b"].long()
                tensor_change = aug["change_mask"].unsqueeze(0).float()
                tensor_ignore = aug["ignore_mask"].unsqueeze(0).float()
            else:
                aug = self.transform(image=image_a, image_b=image_b, mask=change_mask)
                tensor_a = aug["image"].float()
                tensor_b = aug["image_b"].float()
                tensor_change = aug["mask"].unsqueeze(0).float()
                tensor_ignore = torch.from_numpy(ignore_mask).unsqueeze(0).float()
        else:
            tensor_a = norm_tensor(image_a)
            tensor_b = norm_tensor(image_b)
            tensor_change = torch.from_numpy(change_mask).unsqueeze(0).float()
            tensor_ignore = torch.from_numpy(ignore_mask).unsqueeze(0).float()
            if self.mode == "semantic":
                tensor_label_a = torch.from_numpy(label_a.astype(np.int64)).long() if label_a is not None else None
                tensor_label_b = torch.from_numpy(label_b.astype(np.int64)).long() if label_b is not None else None
        transform_elapsed = time.perf_counter() - t_tf if self.profile_enabled else 0.0
        total_elapsed = time.perf_counter() - t0 if self.profile_enabled else 0.0
        if self.profile_enabled:
            self.profile_stats["transform_time"] += transform_elapsed
            self.profile_stats["io_time"] += total_elapsed
            self.profile_stats["getitem_calls"] += 1

        sample: dict[str, Any] = {
            "image_a": tensor_a,
            "image_b": tensor_b,
            "id": entry["sample_id"],
            "name": entry["sample_id"],
            "ignore_mask": tensor_ignore,
            "valid_mask": (1.0 - tensor_ignore),
        }
        if self.mode == "semantic":
            sample.update({
                "label_a": tensor_label_a,
                "label_b": tensor_label_b,
                "label_t1": tensor_label_a,
                "label_t2": tensor_label_b,
                "change_mask": tensor_change,
                "mask": tensor_change,
                "label": tensor_change,
            })
        else:
            sample.update({"mask": tensor_change, "label": tensor_change})
        if self.profile_enabled:
            sample.update({
                "profile_cpu_load_time_ms": float(load_elapsed * 1000.0),
                "profile_cpu_mask_time_ms": float(mask_elapsed * 1000.0),
                "profile_cpu_transform_time_ms": float(transform_elapsed * 1000.0),
                "profile_cpu_total_time_ms": float(total_elapsed * 1000.0),
            })
        return sample


def precompute_second_binary_masks(dataset_cfg: dict, force: bool = False) -> dict[str, Any]:
    root = Path(dataset_cfg.get("root", ""))
    if not root.is_dir():
        raise FileNotFoundError(f"SECOND root does not exist: {root}")

    summary: dict[str, Any] = {"cache_root": str(_resolve_repo_path(dataset_cfg.get("second_binary_cache_dir"), _DEFAULT_BINARY_CACHE_DIR)), "splits": {}}
    for split in ("train", "val", "test"):
        ds = SECONDDataset(
            root=root,
            split=split,
            image_size=int(dataset_cfg.get("image_size", 256)),
            val_ratio=float(dataset_cfg.get("val_ratio", 0.2)),
            seed=42,
            augment=False,
            mode="binary",
            task_type=str(dataset_cfg.get("task_type", "semantic_change")),
            ignore_index=int(dataset_cfg.get("ignore_index", 255)),
            binary_from_semantic=bool(dataset_cfg.get("binary_from_semantic", True)),
            num_classes=int(dataset_cfg.get("num_classes", 7)),
            a_candidates=dataset_cfg.get("image_a_dir_candidates", _A_CANDS),
            b_candidates=dataset_cfg.get("image_b_dir_candidates", _B_CANDS),
            label_a_candidates=dataset_cfg.get("label_a_dir_candidates", _LABEL_A_CANDS),
            label_b_candidates=dataset_cfg.get("label_b_dir_candidates", _LABEL_B_CANDS),
            binary_label_candidates=dataset_cfg.get("binary_label_dir_candidates", _BINARY_LABEL_CANDS),
            train_split=dataset_cfg.get("train_split"),
            val_split=dataset_cfg.get("val_split"),
            test_split=dataset_cfg.get("test_split"),
            precompute_binary_masks=False,
            second_binary_cache_dir=dataset_cfg.get("second_binary_cache_dir"),
        )
        split_dir = ds.second_binary_cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        created = 0
        total_px = 0
        changed_px = 0
        ignored_px = 0
        for entry in ds.entries:
            if not entry.get("valid", False):
                continue
            target_path = split_dir / f"{entry['sample_id']}.png"
            binary_label = _load_binary_mask(target_path) if target_path.exists() and not force else None
            label_a = ds._load_semantic_label(Path(entry["label_a_path"])) if entry.get("label_a_path") else None
            label_b = ds._load_semantic_label(Path(entry["label_b_path"])) if entry.get("label_b_path") else None
            mask, ignore_mask = ds._derive_masks(label_a, label_b, binary_label)
            if binary_label is None:
                Image.fromarray((mask.astype(np.uint8) * 255)).save(target_path)
                created += 1
            total_px += int(mask.size)
            changed_px += int(mask.sum())
            ignored_px += int(ignore_mask.sum())
        summary["splits"][split] = {
            "created_masks": created,
            "total_entries": len(ds.entries),
            "changed_pixel_ratio": round(float(changed_px / total_px), 6) if total_px else 0.0,
            "ignored_pixel_ratio": round(float(ignored_px / total_px), 6) if total_px else 0.0,
        }
    return summary


def inspect_second_dataset(dataset_cfg: dict, sample_limit: int = 24) -> dict:
    root = Path(dataset_cfg.get("root", ""))
    ignore_index = int(dataset_cfg.get("ignore_index", 255))
    mode = str(dataset_cfg.get("mode", "binary")).lower()
    manifest: dict[str, Any] = {
        "dataset_name": "SECOND",
        "root": str(root),
        "root_exists": root.is_dir(),
        "task_type": dataset_cfg.get("task_type", "semantic_change"),
        "mode": mode,
        "binary_from_semantic": bool(dataset_cfg.get("binary_from_semantic", True)),
        "num_classes": int(dataset_cfg.get("num_classes", 7)),
        "ignore_index": ignore_index,
        "splits": {},
        "warnings": [],
    }
    if not root.is_dir():
        manifest["error"] = f"Root directory does not exist: {root}"
        return manifest

    for split in ("train", "val", "test"):
        try:
            dataset = SECONDDataset(
                root=root,
                split=split,
                image_size=int(dataset_cfg.get("image_size", 256)),
                val_ratio=float(dataset_cfg.get("val_ratio", 0.2)),
                seed=42,
                augment=False,
                mode=mode,
                task_type=str(dataset_cfg.get("task_type", "semantic_change")),
                ignore_index=ignore_index,
                binary_from_semantic=bool(dataset_cfg.get("binary_from_semantic", True)),
                num_classes=int(dataset_cfg.get("num_classes", 7)),
                a_candidates=dataset_cfg.get("image_a_dir_candidates", _A_CANDS),
                b_candidates=dataset_cfg.get("image_b_dir_candidates", _B_CANDS),
                label_a_candidates=dataset_cfg.get("label_a_dir_candidates", _LABEL_A_CANDS),
                label_b_candidates=dataset_cfg.get("label_b_dir_candidates", _LABEL_B_CANDS),
                binary_label_candidates=dataset_cfg.get("binary_label_dir_candidates", _BINARY_LABEL_CANDS),
                train_split=dataset_cfg.get("train_split"),
                val_split=dataset_cfg.get("val_split"),
                test_split=dataset_cfg.get("test_split"),
                precompute_binary_masks=bool(dataset_cfg.get("precompute_second_binary_masks", False)),
                second_binary_cache_dir=dataset_cfg.get("second_binary_cache_dir"),
            )
        except Exception as exc:
            manifest["splits"][split] = {"error": str(exc)}
            continue

        class_hist_a: Counter[int] = Counter()
        class_hist_b: Counter[int] = Counter()
        total_px = 0
        changed_px = 0
        ignored_px = 0
        sample_sizes: list[dict[str, Any]] = []
        limit = min(sample_limit, len(dataset))
        for idx in range(limit):
            item = dataset[idx]
            image_a = item["image_a"]
            mask = item.get("mask", item.get("change_mask"))
            sample_sizes.append({"id": str(item.get("id")), "height": int(image_a.shape[-2]), "width": int(image_a.shape[-1])})
            if mask is not None:
                total_px += int(mask.numel())
                changed_px += int((mask > 0.5).sum().item())
            ignore_mask = item.get("ignore_mask")
            if ignore_mask is not None:
                ignored_px += int((ignore_mask > 0.5).sum().item())
            label_a = item.get("label_a")
            label_b = item.get("label_b")
            if label_a is not None:
                for key, value in Counter(label_a.detach().cpu().numpy().reshape(-1).tolist()).items():
                    class_hist_a[int(key)] += int(value)
            if label_b is not None:
                for key, value in Counter(label_b.detach().cpu().numpy().reshape(-1).tolist()).items():
                    class_hist_b[int(key)] += int(value)

        split_info = {
            "base_dir": str(dataset.base_dir),
            "split_dir": str(dataset.split_dir) if dataset.split_dir else None,
            "split_file": str(dataset.split_file) if dataset.split_file.exists() else None,
            "detected_image_a_dir": str(dataset.a_dir) if dataset.a_dir else None,
            "detected_image_b_dir": str(dataset.b_dir) if dataset.b_dir else None,
            "detected_label_a_dir": str(dataset.label_a_dir) if dataset.label_a_dir else None,
            "detected_label_b_dir": str(dataset.label_b_dir) if dataset.label_b_dir else None,
            "detected_binary_mask_dir": str(dataset.binary_label_dir) if dataset.binary_label_dir else None,
            "index_cache": str(dataset._index_cache_path()),
            "image_a_count": len(dataset.entries),
            "image_b_count": len(dataset.entries),
            "label_a_count": sum(1 for entry in dataset.entries if entry.get("label_a_path")),
            "label_b_count": sum(1 for entry in dataset.entries if entry.get("label_b_path")),
            "binary_label_count": sum(1 for entry in dataset.entries if entry.get("binary_mask_path")),
            "class_ids_label_a": sorted(class_hist_a),
            "class_ids_label_b": sorted(class_hist_b),
            "class_histogram_label_a": {str(k): int(v) for k, v in sorted(class_hist_a.items())},
            "class_histogram_label_b": {str(k): int(v) for k, v in sorted(class_hist_b.items())},
            "change_pixel_ratio": round(changed_px / max(total_px, 1), 6) if total_px else None,
            "ignore_pixel_ratio": round(ignored_px / max(total_px, 1), 6) if total_px else None,
            "sample_image_sizes": sample_sizes,
            "size_summary": dataset._sample_size_summary,
            "generated_binary_from_semantic": bool(not dataset.binary_label_dir and dataset.label_a_dir and dataset.label_b_dir),
        }
        expected_ids = set(range(int(dataset_cfg.get("num_classes", 7)))) | {ignore_index}
        observed_ids = set(class_hist_a) | set(class_hist_b)
        unexpected = sorted(v for v in observed_ids if v not in expected_ids)
        if unexpected:
            warning = f"SECOND [{split}] observed unexpected class IDs: {unexpected}"
            split_info.setdefault("warnings", []).append(warning)
            manifest["warnings"].append(warning)
        manifest["splits"][split] = split_info
    return manifest
