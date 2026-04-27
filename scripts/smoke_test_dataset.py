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
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from utils.config import load_config
from data.second import inspect_second_dataset

cfg = load_config()
dc  = cfg.get("dataset", {})
dataset_name = str(dc.get("name", "unknown"))
dataset_mode = str(dc.get("mode", "binary")).lower()

out_dir = ROOT / "outputs" / "dataset_inspection"
out_dir.mkdir(parents=True, exist_ok=True)
manifest_dir = ROOT / "outputs" / "dataset_manifests"
manifest_dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Dataset Smoke Test")
print("=" * 60)
print(f"  dataset    : {dataset_name}")
print(f"  root       : {dc.get('root')}")
print(f"  mode       : {dataset_mode}")
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
    leakage_supported = dataset_name.upper() in {"LEVIR-CD", "WHU-CD"}
    if not leakage_supported:
        print(f"[4] Leakage check skipped for {dataset_name} (split semantics make filename overlap non-diagnostic).")
        leakage_report = {"status": "SKIPPED"}
    else:
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

ta = sample["image_a"]
tb = sample["image_b"]
tm = sample.get("mask", sample.get("change_mask", sample.get("label")))

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

if dataset_name.upper() == "SECOND" and dataset_mode == "semantic":
    label_a = sample.get("label_a")
    label_b = sample.get("label_b")
    if label_a is None or label_b is None:
        print("  FAIL [semantic] label_a/label_b missing in semantic mode")
        all_ok = False
    else:
        if label_a.dtype != torch.long or label_b.dtype != torch.long:
            print(f"  FAIL [semantic] label tensors must be long, got {label_a.dtype} / {label_b.dtype}")
            all_ok = False
        else:
            print(f"  OK   [semantic] label_a/label_b dtype={label_a.dtype}")

# ── Val batch ─────────────────────────────────────────────────────────────────
print("\n[6] Loading one val batch ...")
val_sample = val_ds[0]
check_tensor("val/image_a", val_sample["image_a"])
check_tensor("val/mask",    val_sample.get("mask", val_sample.get("change_mask", val_sample.get("label"))))

# ── Visualisation grid ────────────────────────────────────────────────────────
print("\n[7] Saving sample tile grid ...")

try:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _denorm(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().cpu().float().permute(1, 2, 0).numpy()
        arr = np.clip(arr * std + mean, 0, 1)
        return (arr * 255).astype(np.uint8)

    def _binary_rgb(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().cpu().float()
        if arr.dim() == 3:
            arr = arr.squeeze(0)
        arr = (arr > 0.5).numpy().astype(np.uint8) * 255
        return np.stack([arr, arr, arr], axis=-1)

    def _label_rgb(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().cpu().numpy().astype(np.int64)
        palette = np.array([
            [0, 0, 0], [220, 20, 60], [65, 105, 225], [34, 139, 34],
            [255, 140, 0], [148, 0, 211], [255, 215, 0], [255, 255, 255],
        ], dtype=np.uint8)
        safe = arr.copy()
        safe[safe < 0] = 0
        safe[safe >= len(palette)] = len(palette) - 1
        return palette[safe]

    n_show = min(4, len(train_ds))
    rows = []
    for i in range(n_show):
        s = train_ds[i]
        panel1 = _denorm(s["image_a"])
        panel2 = _denorm(s["image_b"])
        if dataset_name.upper() == "SECOND" and dataset_mode == "semantic":
            panel3 = _label_rgb(s["label_a"])
            panel4 = _label_rgb(s["label_b"])
            panel5 = _binary_rgb(s["change_mask"])
        else:
            mask = s.get("mask", s.get("change_mask", s.get("label")))
            panel3 = _binary_rgb(mask)
            panel4 = _binary_rgb(mask)
            panel5 = _binary_rgb(mask)
        rows.append(np.concatenate([panel1, panel2, panel3, panel4, panel5], axis=1))

    grid = np.concatenate(rows, axis=0)
    grid_name = "SECOND_sample_grid.png" if dataset_name.upper() == "SECOND" else "sample_tiles.png"
    Image.fromarray(grid).save(out_dir / grid_name)
    print(f"  Saved: {out_dir / grid_name}")

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
if dataset_name.upper() == "SECOND":
    manifest_path = manifest_dir / "SECOND_manifest.json"
    second_manifest = inspect_second_dataset(dc)
    with open(manifest_path, "w") as f:
        json.dump(second_manifest, f, indent=2)
else:
    manifest_path = manifest_dir / f"{dataset_name.lower().replace('/', '_').replace(' ', '_')}_manifest.json"
    save_dataset_manifest(
        stats,
        dc,
        manifest_path,
        leakage_status=leakage_report.get("status", "PASS"),
    )
print(f"  Manifest    : {manifest_path}")

print("\n" + "=" * 60)
if all_ok:
    print("  RESULT: PASSED — dataset pipeline is healthy")
else:
    print("  RESULT: FAILED — see FAIL lines above")
print("=" * 60 + "\n")

sys.exit(0 if all_ok else 1)
