"""Clean paired-image change detection dataset.

Expected layout:
    root/split/A/
    root/split/B/
    root/split/Mask/
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils.torchvision_compat import patch_register_fake

patch_register_fake()
import torchvision.transforms.functional as TF

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _files_by_stem(path: Path) -> dict[str, Path]:
    return {p.stem: p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS}


class ChangeDetectionDataset(Dataset):
    def __init__(self, root, split, cfg, transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.threshold = cfg.data.binary_threshold
        self.image_size = cfg.data.image_size

        a_dir = self.root / cfg.data.a_folder
        b_dir = self.root / cfg.data.b_folder
        mask_dir = self.root / cfg.data.mask_folder

        for d in [a_dir, b_dir, mask_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Missing directory: {d}")

        a_stems = _files_by_stem(a_dir)
        b_stems = _files_by_stem(b_dir)
        mask_stems = _files_by_stem(mask_dir)

        common = sorted(set(a_stems) & set(b_stems) & set(mask_stems))
        if len(common) == 0:
            raise ValueError(f"No matching A/B/Mask pairs found in {self.root}")

        missing_b = set(a_stems) - set(b_stems)
        missing_m = set(a_stems) - set(mask_stems)
        if missing_b:
            raise ValueError(f"Missing B images for: {sorted(missing_b)[:5]}")
        if missing_m:
            raise ValueError(f"Missing Mask images for: {sorted(missing_m)[:5]}")

        self.samples = [(a_stems[s], b_stems[s], mask_stems[s], s) for s in common]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a_path, b_path, mask_path, name = self.samples[idx]

        image_a = Image.open(a_path).convert("RGB")
        image_b = Image.open(b_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image_a, image_b, mask = self.transform(image_a, image_b, mask)
        else:
            image_a = TF.to_tensor(TF.resize(image_a, [self.image_size, self.image_size]))
            image_b = TF.to_tensor(TF.resize(image_b, [self.image_size, self.image_size]))
            mask = TF.to_tensor(TF.resize(mask, [self.image_size, self.image_size],
                                          interpolation=TF.InterpolationMode.NEAREST))

        mask = (mask > (self.threshold / 255.0)).float()

        return {"image_a": image_a, "image_b": image_b, "mask": mask, "name": name}
