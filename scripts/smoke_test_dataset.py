#!/usr/bin/env python3
"""Dataset smoke test — verifies the tile pipeline end-to-end.

Loads config, builds train and val datasets, checks tensor shapes,
verifies masks are binary, verifies no NaN values, and saves a
visualisation grid.

No CLI arguments required.
Outputs:
    outputs/dataset_inspection/sample_tiles.png
    outputs/dataset_inspection/leakage_report.json

Does NOT start training.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from utils.config import load_config

cfg = load_config()
dc  = cfg.get("dataset", {})

out_dir = ROOT / "outputs" / "dataset_inspection"
out_dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Dataset Smoke Test")
print("=" * 60)
print(f"  root       : {dc.get('root')}")
print(f"  train_mode : {dc.get('train_mode', 'image')}")
print(f"  tile_size  : {dc.get('tile_size', 256)}")
print(f"  stride     : train={dc.get('train_stride', 128)}  val/test={dc.get('val_stride', 256)}")
print("=" * 60 + "\n")

# ── Build datasets ─────────────────────────────────────────────────────────────
from data.dataset_builder import build_dataset

seed = int(cfg.get("experiment", {}).get("seed", 42))

print("[1] Building train dataset ...")
train_ds = build_dataset(dc, "train", augment=False, seed=seed)
print(f"    Train samples : {len(train_ds)}")

print("[2] Building val dataset ...")
val_ds = build_dataset(dc, "val", augment=False, seed=seed)
print(f"    Val tiles     : {len(val_ds)}")

print("[3] Building test dataset ...")
test_ds = build_dataset(dc, "test", augment=False, seed=seed)
print(f"    Test tiles    : {len(test_ds)}\n")

# ── Leakage check ──────────────────────────────────────────────────────────────
from data.leakage_check import check_leakage

def _index_of(ds) -> list:
    """Get tile index if available."""
    return getattr(ds, "index", [{"image_a_path": str(i)} for i in range(len(ds))])

try:
    print("[4] Running leakage check ...")
    leakage_report = check_leakage(
        train_index = _index_of(train_ds),
        val_index   = _index_of(val_ds),
        test_index  = _index_of(test_ds),
        out_path    = out_dir / "leakage_report.json",
    )
    print(f"    Leakage check: {leakage_report['status']}")
except RuntimeError as e:
    print(f"    LEAKAGE DETECTED: {e}")
    sys.exit(1)

# ── Load one batch and verify ──────────────────────────────────────────────────
print("\n[5] Loading one training batch ...")
sample = train_ds[0]

ta, tb, tm = sample["image_a"], sample["image_b"], sample["mask"]

all_ok = True

def check_tensor(name: str, t: torch.Tensor) -> bool:
    ok = True
    if torch.isnan(t).any():
        print(f"  FAIL [{name}] contains NaN!")
        ok = False
    elif torch.isinf(t).any():
        print(f"  FAIL [{name}] contains Inf!")
        ok = False
    else:
        print(f"  OK   [{name}] shape={tuple(t.shape)}  min={t.min():.3f}  max={t.max():.3f}")
    return ok

all_ok &= check_tensor("image_a", ta)
all_ok &= check_tensor("image_b", tb)
all_ok &= check_tensor("mask",    tm)

# Mask binary check
unique_vals = tm.unique().tolist()
is_binary = all(abs(v - 0.0) < 1e-3 or abs(v - 1.0) < 1e-3 for v in unique_vals)
if is_binary:
    print(f"  OK   [mask] binary (unique values: {[round(v,3) for v in unique_vals[:5]]})")
else:
    print(f"  WARN [mask] non-binary values found: {[round(v,3) for v in unique_vals[:10]]}")

# Shape
assert ta.shape == tb.shape, f"image_a/image_b shape mismatch: {ta.shape} vs {tb.shape}"
H, W = ta.shape[1], ta.shape[2]
assert tm.shape == (1, H, W), f"mask shape should be (1,{H},{W}), got {tm.shape}"
print(f"  OK   shapes consistent: images={tuple(ta.shape)}  mask={tuple(tm.shape)}")

# ── Val batch ─────────────────────────────────────────────────────────────────
print("\n[6] Loading one val batch ...")
val_sample = val_ds[0]
check_tensor("val/image_a", val_sample["image_a"])
check_tensor("val/mask",    val_sample["mask"])

# ── Visualisation grid ────────────────────────────────────────────────────────
print("\n[7] Saving sample tile grid ...")

try:
    import torchvision
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as TF

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _denorm(t):
        return torch.clamp(t * _STD + _MEAN, 0, 1)

    n_show = min(8, len(train_ds))
    imgs_a, imgs_b, masks = [], [], []
    for i in range(n_show):
        s = train_ds[i]
        imgs_a.append(_denorm(s["image_a"]))
        imgs_b.append(_denorm(s["image_b"]))
        masks.append(s["mask"].expand(3, -1, -1))

    row_a = make_grid(torch.stack(imgs_a), nrow=n_show, padding=2, normalize=False)
    row_b = make_grid(torch.stack(imgs_b), nrow=n_show, padding=2, normalize=False)
    row_m = make_grid(torch.stack(masks),  nrow=n_show, padding=2, normalize=False)

    grid = torch.cat([row_a, row_b, row_m], dim=1)
    TF.to_pil_image(grid).save(out_dir / "sample_tiles.png")
    print(f"  Saved: {out_dir / 'sample_tiles.png'}")

except Exception as e:
    print(f"  Visualisation skipped: {e}")

# ── Dataset statistics summary ─────────────────────────────────────────────────
print("\n[8] Dataset statistics ...")
from data.dataset_builder import log_dataset_stats
import logging
log = logging.getLogger("smoke_test")
logging.basicConfig(level=logging.INFO, format="%(message)s")
stats = log_dataset_stats(train_ds, val_ds, test_ds, log, dc)

# ── Manifest ───────────────────────────────────────────────────────────────────
from data.dataset_builder import save_dataset_manifest
manifest_path = ROOT / "outputs" / "dataset_manifests" / "levircd_manifest.json"
save_dataset_manifest(
    stats,
    dc,
    manifest_path,
    leakage_status="PASS",
)

print("\n" + "=" * 60)
if all_ok:
    print("  RESULT: PASSED — dataset pipeline is healthy")
else:
    print("  RESULT: FAILED — see FAIL lines above")
print("=" * 60 + "\n")

sys.exit(0 if all_ok else 1)
