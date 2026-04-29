"""Check train/val/test split overlap for a configured dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from utils.config import load_config
from data.dataset_builder import build_dataset


def _sample_ids(ds) -> set[str]:
    if hasattr(ds, "index"):
        return {Path(entry["image_a_path"]).name for entry in ds.index}
    if hasattr(ds, "names"):
        return {str(name) for name in ds.names}
    ids = set()
    for i in range(len(ds)):
        item = ds[i]
        ids.add(str(item.get("name", item.get("id", i))).split("_")[0])
    return ids


def _print_overlap(label: str, a: set[str], b: set[str]) -> None:
    overlap = sorted(a & b)
    print(f"{label}: {len(overlap)}")
    if overlap:
        print("  first overlaps:", ", ".join(overlap[:20]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check split overlap for a config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg.get("dataset", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    name = str(dc.get("name", "unknown"))
    root = Path(dc.get("root", ""))

    print(f"Dataset: {name}")
    print(f"Root: {root}")

    train_ds = build_dataset(dc, "train", augment=False, seed=seed)
    val_ds = build_dataset(dc, "val", augment=False, seed=seed)
    test_ds = build_dataset(dc, "test", augment=False, seed=seed)

    train_ids = _sample_ids(train_ds)
    val_ids = _sample_ids(val_ds)
    test_ids = _sample_ids(test_ds)

    print(f"Train samples/tiles: {len(train_ds)} | image IDs: {len(train_ids)}")
    print(f"Val samples/tiles:   {len(val_ds)} | image IDs: {len(val_ids)}")
    print(f"Test samples/tiles:  {len(test_ds)} | image IDs: {len(test_ids)}")
    _print_overlap("train-val image ID overlap", train_ids, val_ids)
    _print_overlap("train-test image ID overlap", train_ids, test_ids)
    _print_overlap("val-test image ID overlap", val_ids, test_ids)

    if name.upper().replace("-", "") == "LEVIRCD":
        print("LEVIR note: train and val are split from root/train; test is from root/test.")
        if not (train_ids & val_ids):
            print("No train-val image-level leakage detected for the configured random split.")
        print("Validation split file: none; generated deterministically from train/ with val_ratio and seed.")
        print("Test split file: none; uses all files under test/A, test/B, test/label.")

    for split_name, ds in [("val", val_ds), ("test", test_ds)]:
        if hasattr(ds, "raw_mask_stats"):
            print(f"{split_name} first 5 mask stats:")
            for i in range(min(5, len(ds))):
                print(" ", ds.raw_mask_stats(i))


if __name__ == "__main__":
    main()
