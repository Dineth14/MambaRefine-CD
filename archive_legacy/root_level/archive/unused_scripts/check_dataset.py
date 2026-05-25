"""Dataset integrity checker and manifest generator.

Checks all datasets from configs/global_config.yaml and saves manifests to:
    outputs/dataset_manifests/<dataset_name>_manifest.json

No CLI arguments needed. Edit datasets_catalog in global_config.yaml to change
which datasets are checked.

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

from utils.config import load_config
from data.second import inspect_second_dataset

OUTPUT_DIR = ROOT / "outputs" / "dataset_manifests"

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
    if str(name).upper() == "SECOND":
        return inspect_second_dataset(ds_cfg)
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
    cfg = load_config()
    datasets_catalog = cfg.get("datasets_catalog", {})

    for _, ds_cfg in datasets_catalog.items():

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
            if str(name).upper() == "SECOND":
                print(
                    f"       {split:5s}  A/B/sem/binary="
                    f"{info.get('image_a_count', 0)}/{info.get('image_b_count', 0)}/"
                    f"{info.get('label_a_count', 0)}:{info.get('label_b_count', 0)}/"
                    f"{info.get('binary_label_count', 0)}"
                )
                if info.get("error"):
                    print(f"       {split:5s}  ERROR: {info['error']}")
                if info.get("change_pixel_ratio") is not None:
                    print(f"       {split:5s}  Change ratio : {info['change_pixel_ratio']:.4f}")
                if info.get("ignore_pixel_ratio") is not None:
                    print(f"       {split:5s}  Ignore ratio : {info['ignore_pixel_ratio']:.4f}")
            else:
                c_ok = "✓" if info.get("count_ok") else "✗"
                print(f"       {split:5s}  {c_ok}  A/B/mask={info['count_a']}/{info['count_b']}/{info['count_label']}"
                      f"  size={manifest['first_image_sizes'].get(split,'?')}")
        ratio = manifest.get("changed_pixel_ratio")
        if ratio is not None:
            print(f"       Change ratio (sample): {ratio:.4f} ({ratio*100:.2f}%)")
        for warning in manifest.get("warnings", []):
            print(f"       Warning : {warning}")
        print(f"       Manifest : {out_file}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All datasets found.")
    else:
        print("Some dataset roots are missing. Update 'datasets_catalog' in global_config.yaml.")
    print(f"Manifests saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
