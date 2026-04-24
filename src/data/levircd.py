"""LEVIR-CD dataset for iteration-based training.

Directory layout expected:
    root/
      train/  A/  B/  label/
      test/   A/  B/  label/

split='train' → random 256×256 crops from 80% of train/.
split='val'   → sliding-window 256×256 tiles from 20% of train/.
split='test'  → sliding-window 256×256 tiles from test/.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALB = True
except ImportError:
    _ALB = False

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _discover(split_dir: Path) -> List[str]:
    exts = {".png", ".jpg", ".tif", ".tiff"}
    return sorted(p.name for p in (split_dir / "A").iterdir() if p.suffix.lower() in exts)


def _split(names: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


class LEVIRCDDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        val_ratio: float = 0.2,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.root       = Path(root)
        self.split      = split
        self.size       = image_size
        self.do_augment = augment and split == "train"

        src = self.root / ("train" if split in ("train", "val") else "test")
        self.a_dir   = src / "A"
        self.b_dir   = src / "B"
        self.lbl_dir = src / "label"

        all_names = _discover(src)
        if split in ("train", "val"):
            train_names, val_names = _split(all_names, val_ratio, seed)
            self.names = train_names if split == "train" else val_names
        else:
            self.names = all_names

        self.tiles: Optional[List[Tuple[str, int, int]]] = (
            None if split == "train" else self._build_tiles()
        )
        self.transform = self._augmentation() if self.do_augment else None

    # ── Tile building ─────────────────────────────────────────────────────────
    def _build_tiles(self) -> List[Tuple[str, int, int]]:
        s, tiles, seen = self.size, [], set()
        for name in self.names:
            W, H = Image.open(self.a_dir / name).size
            rows = list(range(0, H - s + 1, s)) + ([H - s] if H % s else [])
            cols = list(range(0, W - s + 1, s)) + ([W - s] if W % s else [])
            for r in rows:
                for c in cols:
                    if (name, r, c) not in seen:
                        seen.add((name, r, c))
                        tiles.append((name, r, c))
        return tiles

    # ── Augmentation ──────────────────────────────────────────────────────────
    def _augmentation(self):
        if not _ALB:
            return None
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
                A.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
                ToTensorV2(),
            ],
            additional_targets={"image_b": "image"},
        )

    # ── Normalise ─────────────────────────────────────────────────────────────
    def _norm(self, arr: np.ndarray) -> torch.Tensor:
        x = arr.astype(np.float32) / 255.0
        return torch.from_numpy(((x - _MEAN) / _STD).transpose(2, 0, 1))

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.names) if self.split == "train" else len(self.tiles)  # type: ignore

    def __getitem__(self, idx: int) -> dict:
        s = self.size
        if self.split == "train":
            name = self.names[idx]
            a_full = np.array(Image.open(self.a_dir / name).convert("RGB"))
            b_full = np.array(Image.open(self.b_dir / name).convert("RGB"))
            l_full = np.array(Image.open(self.lbl_dir / name).convert("L"))
            H, W   = a_full.shape[:2]
            # Pad if needed
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
            img_a = np.array(Image.open(self.a_dir / name).convert("RGB"))[r:r+s, c:c+s]
            img_b = np.array(Image.open(self.b_dir / name).convert("RGB"))[r:r+s, c:c+s]
            lbl   = np.array(Image.open(self.lbl_dir / name).convert("L"))[r:r+s, c:c+s]

        lbl_bin = (lbl > 127).astype(np.uint8)

        if self.do_augment and self.transform is not None:
            aug  = self.transform(image=img_a, image_b=img_b, mask=lbl_bin)
            ta   = aug["image"].float()
            tb   = aug["image_b"].float()
            tm   = aug["mask"].unsqueeze(0).float()
        else:
            ta = self._norm(img_a)
            tb = self._norm(img_b)
            tm = torch.from_numpy(lbl_bin).unsqueeze(0).float()

        return {
            "image_a": ta,
            "image_b": tb,
            "label":   tm,    # trainer expects "label"
            "mask":    tm,    # evaluator uses "mask"
            "id":      name,
            "name":    name,
        }
