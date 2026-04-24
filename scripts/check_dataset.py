"""Dataset integrity checker and manifest generator.

Checks all four dataset configs and saves manifests to:
    outputs/dataset_manifests/<dataset_name>_manifest.json

No CLI arguments needed. Edit DATASET_CONFIG_PATHS below to change which
configs to check.

Run:
    conda activate mamba_new
    cd MambaRefine-CD
    python scripts/check_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config_loader import load_config

# ── Configure which dataset configs to check ─────────────────────────────────
DATASET_CONFIG_PATHS = [
    "configs/datasets/levircd.yaml",
    "configs/datasets/whucd.yaml",
    "configs/datasets/sysucd.yaml",
    "configs/datasets/dsifncd.yaml",
]
OUTPUT_DIR = ROOT / "outputs" / "dataset_manifests"
# ─────────────────────────────────────────────────────────────────────────────

_EXTS = {".png", ".jpg", ".tif", ".tiff", ".jpeg"}
_SPLIT_ALIASES = {
    "train": ["train", "training", "Train", "trainset"],
    "val":   ["val", "valid", "validation", "Val", "valset"],
    "test":  ["test", "testing", "Test", "testset"],
}
_A_CANDS     = ["A", "t1", "time1", "imageA", "T1", "A_256", "img1"]
_B_CANDS     = ["B", "t2", "time2", "imageB", "T2", "B_256", "img2"]
_LABEL_CANDS = ["label", "labels", "mask", "OUT", "GT", "change_map", "cm"]


def _find_dir(parent: Path, candidates: list[str]) -> Path | None:
    for c in candidates:
        p = parent / c
        if p.is_dir():
            return p
    return None


def _count_images(d: Path | None) -> int:
    if d is None or not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in _EXTS)


def _first_size(d: Path | None) -> str:
    if d is None or not d.exists():
        return "N/A"
    from PIL import Image
    for p in d.iterdir():
        if p.suffix.lower() in _EXTS:
            try:
                w, h = Image.open(p).size
                return f"{w}x{h}"
            except Exception:
                pass
    return "N/A"


def _compute_change_ratio(label_dir: Path | None, sample_n: int = 50) -> float | None:
    """Estimate change ratio from up to sample_n masks."""
    if label_dir is None or not label_dir.exists():
        return None
    try:
        import numpy as np
        from PIL import Image
        files = sorted(p for p in label_dir.iterdir() if p.suffix.lower() in _EXTS)[:sample_n]
        if not files:
            return None
        total_px = total_ch = 0
        for f in files:
            arr = np.array(Image.open(f).convert("L"))
            total_px += arr.size
            total_ch += int((arr > 0).sum())
        return round(total_ch / max(total_px, 1), 6)
    except Exception:
        return None


def check_dataset(ds_cfg: dict) -> dict:
    name = ds_cfg.get("name", "unknown")
    root = Path(ds_cfg.get("root", ""))

    a_cands = ds_cfg.get("image_a_dir_candidates", _A_CANDS)
    b_cands = ds_cfg.get("image_b_dir_candidates", _B_CANDS)
    l_cands = ds_cfg.get("label_dir_candidates",   _LABEL_CANDS)

    manifest: dict = {
        "dataset_name":       name,
        "root":               str(root),
        "root_exists":        root.is_dir(),
        "splits":             {},
        "detected_dirs":      {},
        "first_image_sizes":  {},
        "changed_pixel_ratio": None,
    }

    if not root.is_dir():
        manifest["warning"] = f"Root directory does not exist: {root}"
        return manifest

    # Check each split
    for split in ("train", "val", "test"):
        split_dir = None
        for alias in _SPLIT_ALIASES.get(split, [split]):
            p = root / alias
            if p.is_dir():
                split_dir = p
                break

        if split_dir is not None:
            a_dir = _find_dir(split_dir, a_cands)
            b_dir = _find_dir(split_dir, b_cands)
            l_dir = _find_dir(split_dir, l_cands)
            base  = split_dir
        else:
            a_dir = _find_dir(root, a_cands)
            b_dir = _find_dir(root, b_cands)
            l_dir = _find_dir(root, l_cands)
            base  = root

        count_a = _count_images(a_dir)
        count_b = _count_images(b_dir)
        count_l = _count_images(l_dir)
        size_a  = _first_size(a_dir)

        manifest["splits"][split] = {
            "split_dir":     str(split_dir) if split_dir else None,
            "a_dir":         str(a_dir)    if a_dir    else None,
            "b_dir":         str(b_dir)    if b_dir    else None,
            "label_dir":     str(l_dir)    if l_dir    else None,
            "count_a":       count_a,
            "count_b":       count_b,
            "count_label":   count_l,
            "count_ok":      count_a == count_b == count_l and count_a > 0,
        }
        manifest["first_image_sizes"][split] = size_a

        if split == "train":
            manifest["changed_pixel_ratio"] = _compute_change_ratio(l_dir)

    return manifest


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for cfg_path in DATASET_CONFIG_PATHS:
        full_path = ROOT / cfg_path
        if not full_path.exists():
            print(f"  [SKIP] Config not found: {full_path}")
            continue

        try:
            cfg    = load_config(full_path)
            ds_cfg = cfg.get("dataset", cfg)   # allow flat or nested
        except Exception as e:
            print(f"  [ERR ] Failed to load {full_path}: {e}")
            continue

        manifest = check_dataset(ds_cfg)
        name     = manifest["dataset_name"]

        # Save manifest
        out_file = OUTPUT_DIR / f"{name.replace('/', '-').replace(' ', '_')}_manifest.json"
        with open(out_file, "w") as f:
            json.dump(manifest, f, indent=2)

        # Print summary
        ok = manifest.get("root_exists", False)
        status = "OK  " if ok else "WARN"
        print(f"[{status}] {name}")
        print(f"       Root    : {manifest['root']}")
        for split, info in manifest.get("splits", {}).items():
            cnt = info.get("count_a", 0)
            c_ok = "✓" if info.get("count_ok") else "✗"
            print(f"       {split:5s}  {c_ok}  A/B/mask={info['count_a']}/{info['count_b']}/{info['count_label']}"
                  f"  size={manifest['first_image_sizes'].get(split,'?')}")
        ratio = manifest.get("changed_pixel_ratio")
        if ratio is not None:
            print(f"       Change ratio (sample): {ratio:.4f} ({ratio*100:.2f}%)")
        print(f"       Manifest : {out_file}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All datasets found.")
    else:
        print("Some dataset roots are missing. Update 'root' in the dataset configs.")
    print(f"Manifests saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
