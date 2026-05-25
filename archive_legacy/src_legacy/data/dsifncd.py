"""DSIFN-CD change detection dataset.

DSIFN-CD was proposed in "A Deeply Supervised Image Fusion Network for Change
Detection in High-Resolution Bi-Temporal Remote-Sensing Images" (TGRS 2021).

Supported layout variants (auto-detected):

A) Standard layout:
   root/
     trainset/ (or train/)
       t1/  t2/  GT/
     testset/  (or test/)
       t1/  t2/  GT/

B) Flat with explicit split text files:
   root/
     t1/  t2/  GT/
     splits/train.txt  splits/val.txt  splits/test.txt

Folder name candidates are configurable via dataset config.

Each sample returns:
  {image_a, image_b, label, mask, id, name}
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import build_train_transforms, norm_tensor

_EXTS = {".png", ".jpg", ".tif", ".tiff", ".jpeg"}
_MASK_EXT_PRIORITY = {".png": 0, ".tif": 1, ".tiff": 2, ".jpg": 3, ".jpeg": 4}

_A_CANDS     = ["t1", "T1", "A", "imageA", "before", "img1", "A_256"]
_B_CANDS     = ["t2", "T2", "B", "imageB", "after",  "img2", "B_256"]
_LABEL_CANDS = ["GT", "label", "labels", "mask", "OUT", "change_map"]

_SPLIT_ALIASES = {
    "train": ["trainset", "train", "Train", "training"],
    "val":   ["valset",   "val",   "Val",   "valid", "validation"],
    "test":  ["testset",  "test",  "Test",  "testing"],
}


def _detect_dir(parent: Path, candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = parent / c
        if p.is_dir():
            return p
    return None


def _list_images(d: Path) -> List[str]:
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in _EXTS)


def _build_file_lookup(d: Path, *, prefer_mask_ext: bool = False) -> dict[str, Path]:
    """Map both exact filenames and filename stems to file paths.

    DSIFN layouts sometimes store RGB images as .jpg while masks use .png/.tif.
    Matching by stem keeps paired samples aligned even when extensions differ.
    """
    paths = [p for p in d.iterdir() if p.suffix.lower() in _EXTS]
    if prefer_mask_ext:
        paths = sorted(paths, key=lambda p: (p.stem, _MASK_EXT_PRIORITY.get(p.suffix.lower(), 99), p.name))
    else:
        paths = sorted(paths, key=lambda p: p.name)
    lookup: dict[str, Path] = {}
    for p in paths:
        if p.suffix.lower() not in _EXTS:
            continue
        lookup[p.name] = p
        lookup.setdefault(p.stem, p)
    return lookup


def _read_split_file(f: Path) -> List[str]:
    with open(f) as fh:
        return [Path(ln.strip()).name for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def _find_split_dir(root: Path, split: str) -> Optional[Path]:
    for c in _SPLIT_ALIASES.get(split, [split]):
        p = root / c
        if p.is_dir():
            return p
    return None


def _normalise_id(name: str) -> str:
    return Path(str(name)).stem


def _resolve_split_file(root: Path, split: str, split_dir: str | Path | None) -> Optional[Path]:
    candidates = []
    if split_dir is not None:
        candidates.append(Path(split_dir) / f"{split}.txt")
    candidates.append(root / "splits" / f"{split}.txt")
    candidates.append(root / f"{split}.txt")
    for path in candidates:
        if path.exists():
            return path
    return None


def _assert_disjoint_split_files(root: Path, split_dir: str | Path | None) -> dict[str, Path]:
    files: dict[str, Path] = {}
    ids_by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = _resolve_split_file(root, split, split_dir)
        if path is None:
            raise FileNotFoundError(
                "DSIFN flat layout requires explicit non-overlapping split files. "
                "Refusing to use all images as test because this causes train/test leakage. "
                f"Missing {split}.txt under split_dir/root/splits/root."
            )
        files[split] = path
        ids = [_normalise_id(name) for name in _read_split_file(path)]
        duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
        if duplicates:
            raise RuntimeError(
                "DATA LEAKAGE FOUND: refusing to train/evaluate. "
                f"DSIFN {split}.txt contains duplicate image IDs: {duplicates[:10]}"
            )
        ids_by_split[split] = set(ids)

    overlaps = {
        "train_val": ids_by_split["train"] & ids_by_split["val"],
        "train_test": ids_by_split["train"] & ids_by_split["test"],
        "val_test": ids_by_split["val"] & ids_by_split["test"],
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        preview = {k: v[:10] for k, v in bad.items()}
        raise RuntimeError(f"DATA LEAKAGE FOUND: refusing to train/evaluate. DSIFN split overlaps: {preview}")
    flat_a_dir = _detect_dir(root, _A_CANDS)
    if flat_a_dir is not None:
        all_flat_ids = {_normalise_id(name) for name in _list_images(flat_a_dir)}
        if all_flat_ids and ids_by_split["test"] == all_flat_ids:
            raise RuntimeError(
                "DATA LEAKAGE FOUND: refusing to train/evaluate. "
                "DSIFN test split contains all flat-layout images."
            )
    return files


def validate_dsifn_split_files(root: str | Path, split_dir: str | Path | None = None) -> dict:
    """Validate explicit DSIFN split files and return reproducibility metadata."""
    root = Path(root)
    files = _assert_disjoint_split_files(root, split_dir)
    ids_by_split = {split: [_normalise_id(name) for name in _read_split_file(path)] for split, path in files.items()}
    return {
        "split_dir": str(Path(split_dir) if split_dir is not None else files["train"].parent),
        "split_file_train": str(files["train"]),
        "split_file_val": str(files["val"]),
        "split_file_test": str(files["test"]),
        "split_metadata_json": str(files["train"].parent / "split_metadata.json"),
        "split_hash_train": _sha256_file(files["train"]),
        "split_hash_val": _sha256_file(files["val"]),
        "split_hash_test": _sha256_file(files["test"]),
        "split_integrity_verdict": "PASS",
        "num_train_images": len(ids_by_split["train"]),
        "num_val_images": len(ids_by_split["val"]),
        "num_test_images": len(ids_by_split["test"]),
        "old_leakage_protocol_used": False,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dsifn_result_split_metadata(dataset_cfg: dict, num_tiles: int | None = None) -> dict:
    """Metadata fields to attach to DSIFN result files."""
    name = str(dataset_cfg.get("name", "")).lower().replace("_", "-")
    if name not in {"dsifn-cd", "dsifn", "dsifncd"}:
        return {}
    root = dataset_cfg.get("root")
    if root is None:
        return {
            "split_integrity_verdict": "INCONCLUSIVE",
            "old_leakage_protocol_used": True,
        }
    try:
        metadata = validate_dsifn_split_files(root, dataset_cfg.get("split_dir"))
    except Exception as exc:
        return {
            "split_dir": str(dataset_cfg.get("split_dir", "")),
            "split_integrity_verdict": f"FAIL: {exc}",
            "old_leakage_protocol_used": True,
        }
    if num_tiles is not None:
        metadata["num_test_tiles"] = int(num_tiles)
    return metadata


class DSIFNCDDataset(Dataset):
    """DSIFN-CD binary change detection dataset.

    Returns dict: image_a, image_b, label (=mask), mask, id, name.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        val_ratio: float = 0.2,
        seed: int = 42,
        augment: bool = True,
        augmentation_ops: Optional[List[str]] = None,
        a_candidates: Optional[List[str]] = None,
        b_candidates: Optional[List[str]] = None,
        label_candidates: Optional[List[str]] = None,
        split_dir: str | Path | None = None,
        require_explicit_splits: bool = True,
    ) -> None:
        self.root       = Path(root)
        self.split      = split
        self.size       = image_size
        self.do_augment = augment and split == "train"

        a_cands = a_candidates or _A_CANDS
        b_cands = b_candidates or _B_CANDS
        l_cands = label_candidates or _LABEL_CANDS

        split_files: dict[str, Path] = {}
        try:
            split_files = _assert_disjoint_split_files(self.root, split_dir)
        except FileNotFoundError:
            if require_explicit_splits and _detect_dir(self.root, a_cands) is not None:
                raise
        split_file = split_files.get(split)

        # --- Layout B: flat + explicit split text files ---
        if split_file is not None:
            self.a_dir   = _detect_dir(self.root, a_cands)
            self.b_dir   = _detect_dir(self.root, b_cands)
            self.lbl_dir = _detect_dir(self.root, l_cands)
            self._check_dirs(self.root)
            self.names = _read_split_file(split_file)
            self.split_file = split_file
        else:
            # --- Layout A: split sub-folders ---
            split_root = _find_split_dir(self.root, split)
            if split_root is not None:
                self.a_dir   = _detect_dir(split_root, a_cands)
                self.b_dir   = _detect_dir(split_root, b_cands)
                self.lbl_dir = _detect_dir(split_root, l_cands)
                self._check_dirs(split_root)
                self.names = _list_images(self.a_dir)  # type: ignore
            else:
                if require_explicit_splits:
                    raise RuntimeError(
                        "DSIFN flat layout requires explicit non-overlapping split files. "
                        "Refusing to use all images as test because this causes train/test leakage."
                    )
                raise RuntimeError("DSIFN unsafe flat manual split fallback is disabled.")

        self.transform = build_train_transforms(image_size, augmentation_ops=augmentation_ops) if self.do_augment else None
        self.a_lookup  = _build_file_lookup(self.a_dir)    # type: ignore[arg-type]
        self.b_lookup  = _build_file_lookup(self.b_dir)    # type: ignore[arg-type]
        self.l_lookup  = _build_file_lookup(self.lbl_dir, prefer_mask_ext=True)  # type: ignore[arg-type]
        self._verify_names_exist()
        self.tiles     = None if split == "train" else self._build_tiles()

    def _check_dirs(self, parent: Path) -> None:
        for attr, label in [("a_dir", "A/t1"), ("b_dir", "B/t2"), ("lbl_dir", "GT/label")]:
            if getattr(self, attr) is None:
                raise FileNotFoundError(
                    f"DSIFNCDDataset [{self.split}]: could not find {label} dir under {parent}."
                )

    def _verify_names_exist(self) -> None:
        missing = []
        for name in self.names:
            try:
                self._resolve_path(self.a_lookup, name, "image_a")
                self._resolve_path(self.b_lookup, name, "image_b")
                self._resolve_path(self.l_lookup, name, "label")
            except FileNotFoundError as exc:
                missing.append(str(exc))
        if missing:
            raise FileNotFoundError(
                f"DSIFNCDDataset [{self.split}]: {len(missing)} split entries could not be resolved. "
                f"Examples: {missing[:5]}"
            )

    def _build_tiles(self) -> List[Tuple[str, int, int]]:
        s, tiles, seen = self.size, [], set()
        for name in self.names:
            try:
                W, H = Image.open(self._resolve_path(self.a_lookup, name, "image_a")).size
            except Exception:
                continue
            rows = list(range(0, H - s + 1, s)) + ([H - s] if H % s else [])
            cols = list(range(0, W - s + 1, s)) + ([W - s] if W % s else [])
            for r in rows:
                for c in cols:
                    key = (name, r, c)
                    if key not in seen:
                        seen.add(key)
                        tiles.append(key)
        return tiles

    def __len__(self) -> int:
        return len(self.names) if self.split == "train" else len(self.tiles)  # type: ignore

    def _resolve_path(self, lookup: dict[str, Path], name: str, kind: str) -> Path:
        path = lookup.get(name)
        if path is None:
            path = lookup.get(Path(name).stem)
        if path is None:
            raise FileNotFoundError(
                f"DSIFNCDDataset [{self.split}]: could not resolve {kind} for sample '{name}'."
            )
        return path

    @staticmethod
    def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask[..., 0]
        threshold = 0 if int(mask.max()) <= 1 else 127
        return (mask > threshold).astype(np.uint8)

    def sample_info(self, idx: int) -> dict:
        if self.split == "train":
            name = self.names[idx]
            row = None
            col = None
        else:
            name, row, col = self.tiles[idx]  # type: ignore[index]
        a_path = self._resolve_path(self.a_lookup, name, "image_a")
        b_path = self._resolve_path(self.b_lookup, name, "image_b")
        m_path = self._resolve_path(self.l_lookup, name, "label")
        tile_suffix = "" if row is None else f"_r{row}_c{col}"
        return {
            "name": name,
            "sample_id": f"{Path(name).stem}{tile_suffix}",
            "image_t1_path": str(a_path),
            "image_t2_path": str(b_path),
            "mask_path": str(m_path),
            "row": row,
            "col": col,
        }

    def raw_mask_stats(self, idx: int) -> dict:
        info = self.sample_info(idx)
        raw = np.array(Image.open(info["mask_path"]))
        converted = self._to_binary_mask(raw)
        return {
            **info,
            "raw_dtype": str(raw.dtype),
            "raw_shape": list(raw.shape),
            "raw_unique": sorted(np.unique(raw).tolist())[:32],
            "converted_dtype": str(converted.dtype),
            "converted_unique": sorted(np.unique(converted).tolist()),
            "positive_ratio": float(converted.mean()),
        }

    def __getitem__(self, idx: int) -> dict:
        s = self.size
        if self.split == "train":
            info = self.sample_info(idx)
            name   = info["name"]
            a_full = np.array(Image.open(self._resolve_path(self.a_lookup, name, "image_a")).convert("RGB"))
            b_full = np.array(Image.open(self._resolve_path(self.b_lookup, name, "image_b")).convert("RGB"))
            l_full = np.array(Image.open(self._resolve_path(self.l_lookup, name, "label")))
            l_full = self._to_binary_mask(l_full)
            H, W   = a_full.shape[:2]
            if H < s or W < s:
                ph, pw = max(0, s - H), max(0, s - W)
                a_full = np.pad(a_full, ((0, ph), (0, pw), (0, 0)))
                b_full = np.pad(b_full, ((0, ph), (0, pw), (0, 0)))
                l_full = np.pad(l_full, ((0, ph), (0, pw)))
                H, W   = a_full.shape[:2]
            r = random.randint(0, H - s)
            c = random.randint(0, W - s)
            img_a = a_full[r:r+s, c:c+s]
            img_b = b_full[r:r+s, c:c+s]
            lbl   = l_full[r:r+s, c:c+s]
        else:
            info = self.sample_info(idx)
            name, r, c = info["name"], info["row"], info["col"]
            img_a = np.array(Image.open(self._resolve_path(self.a_lookup, name, "image_a")).convert("RGB"))[r:r+s, c:c+s]
            img_b = np.array(Image.open(self._resolve_path(self.b_lookup, name, "image_b")).convert("RGB"))[r:r+s, c:c+s]
            lbl_full = self._to_binary_mask(np.array(Image.open(self._resolve_path(self.l_lookup, name, "label"))))
            lbl = lbl_full[r:r+s, c:c+s]

        lbl_bin = lbl.astype(np.uint8)

        if self.do_augment and self.transform is not None:
            aug = self.transform(image=img_a, image_b=img_b, mask=lbl_bin)
            ta  = aug["image"].float()
            tb  = aug["image_b"].float()
            tm  = aug["mask"].unsqueeze(0).float()
        else:
            ta = norm_tensor(img_a)
            tb = norm_tensor(img_b)
            tm = torch.from_numpy(lbl_bin).unsqueeze(0).float()

        return {
            "image_a": ta,
            "image_b": tb,
            "label":   tm,
            "mask":    tm,
            "id":      name,
            "name":    info["sample_id"],
            "image_t1_path": info["image_t1_path"],
            "image_t2_path": info["image_t2_path"],
            "mask_path": info["mask_path"],
        }
