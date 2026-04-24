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

B) Flat with split text files:
   root/
     t1/  t2/  GT/
     train.txt  val.txt  test.txt

C) Flat manual split:
   root/
     t1/  t2/  GT/

Folder name candidates are configurable via dataset config.

Each sample returns:
  {image_a, image_b, label, mask, id, name}
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import build_train_transforms, norm_tensor

_EXTS = {".png", ".jpg", ".tif", ".tiff", ".jpeg"}

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


def _read_split_file(f: Path) -> List[str]:
    with open(f) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _find_split_dir(root: Path, split: str) -> Optional[Path]:
    for c in _SPLIT_ALIASES.get(split, [split]):
        p = root / c
        if p.is_dir():
            return p
    return None


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
        a_candidates: Optional[List[str]] = None,
        b_candidates: Optional[List[str]] = None,
        label_candidates: Optional[List[str]] = None,
    ) -> None:
        self.root       = Path(root)
        self.split      = split
        self.size       = image_size
        self.do_augment = augment and split == "train"

        a_cands = a_candidates or _A_CANDS
        b_cands = b_candidates or _B_CANDS
        l_cands = label_candidates or _LABEL_CANDS

        # --- Layout B: flat + split text files ---
        split_file = self.root / f"{split}.txt"
        if split_file.exists():
            self.a_dir   = _detect_dir(self.root, a_cands)
            self.b_dir   = _detect_dir(self.root, b_cands)
            self.lbl_dir = _detect_dir(self.root, l_cands)
            self._check_dirs(self.root)
            self.names = _read_split_file(split_file)
        else:
            # --- Layout A: split sub-folders ---
            split_dir = _find_split_dir(self.root, split)
            if split_dir is not None:
                self.a_dir   = _detect_dir(split_dir, a_cands)
                self.b_dir   = _detect_dir(split_dir, b_cands)
                self.lbl_dir = _detect_dir(split_dir, l_cands)
                self._check_dirs(split_dir)
                self.names = _list_images(self.a_dir)  # type: ignore
            else:
                # --- Layout C: flat, manual split ---
                self.a_dir   = _detect_dir(self.root, a_cands)
                self.b_dir   = _detect_dir(self.root, b_cands)
                self.lbl_dir = _detect_dir(self.root, l_cands)
                self._check_dirs(self.root)
                all_names = _list_images(self.a_dir)  # type: ignore
                rng = random.Random(seed)
                shuffled = list(all_names)
                rng.shuffle(shuffled)
                n_val = max(1, int(len(shuffled) * val_ratio))
                if split == "train":
                    self.names = sorted(shuffled[n_val:])
                elif split == "val":
                    self.names = sorted(shuffled[:n_val])
                else:
                    self.names = all_names

        self.tiles     = None if split == "train" else self._build_tiles()
        self.transform = build_train_transforms(image_size) if self.do_augment else None

    def _check_dirs(self, parent: Path) -> None:
        for attr, label in [("a_dir", "A/t1"), ("b_dir", "B/t2"), ("lbl_dir", "GT/label")]:
            if getattr(self, attr) is None:
                raise FileNotFoundError(
                    f"DSIFNCDDataset [{self.split}]: could not find {label} dir under {parent}."
                )

    def _build_tiles(self) -> List[Tuple[str, int, int]]:
        s, tiles, seen = self.size, [], set()
        for name in self.names:
            try:
                W, H = Image.open(self.a_dir / name).size  # type: ignore
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

    def __getitem__(self, idx: int) -> dict:
        s = self.size
        if self.split == "train":
            name   = self.names[idx]
            a_full = np.array(Image.open(self.a_dir / name).convert("RGB"))  # type: ignore
            b_full = np.array(Image.open(self.b_dir / name).convert("RGB"))  # type: ignore
            l_full = np.array(Image.open(self.lbl_dir / name).convert("L"))  # type: ignore
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
            name, r, c = self.tiles[idx]  # type: ignore
            img_a = np.array(Image.open(self.a_dir / name).convert("RGB"))[r:r+s, c:c+s]  # type: ignore
            img_b = np.array(Image.open(self.b_dir / name).convert("RGB"))[r:r+s, c:c+s]  # type: ignore
            lbl   = np.array(Image.open(self.lbl_dir / name).convert("L"))[r:r+s, c:c+s]  # type: ignore

        # DSIFN GT may be {0, 255} or {0, 1} — handle both
        lbl_bin = (lbl > 127).astype(np.uint8)

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
            "name":    name,
        }
