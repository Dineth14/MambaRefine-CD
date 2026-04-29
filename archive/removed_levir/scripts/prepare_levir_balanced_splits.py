"""Scan the LEVIR-CD dataset and compute per-sample change pixel ratios.

Saves a CSV/JSON with split metadata for constructing a balanced sampler.

Usage:
    python scripts/prepare_levir_balanced_splits.py \\
        --data_root /path/to/LEVIR-CD \\
        --out data/LEVIRCD/split_stats.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _compute_change_ratio(mask_path: Path) -> float:
    from PIL import Image as PILImage
    mask = np.array(PILImage.open(mask_path).convert("L"))
    return float((mask > 0).sum()) / mask.size


def _scan_split(root: Path, split: str) -> list[dict]:
    label_dir = root / split / "label"
    if not label_dir.is_dir():
        # Alternate layout: root/label/split/
        label_dir = root / "label" / split
    if not label_dir.is_dir():
        logger.warning(f"label dir not found for split={split}: checked {label_dir}")
        return []

    records = []
    for mask_path in sorted(label_dir.glob("*.png")):
        ratio = _compute_change_ratio(mask_path)
        records.append({
            "split":   split,
            "name":    mask_path.stem,
            "mask_path": str(mask_path),
            "change_ratio": ratio,
        })
    return records


def _print_stats(records: list[dict], split: str) -> None:
    ratios = [r["change_ratio"] for r in records]
    if not ratios:
        return
    arr = np.array(ratios)
    has_change = (arr > 0).sum()
    no_change  = (arr == 0).sum()
    logger.info(f"  {split:5s}: {len(records):5d} samples | "
                f"changed={has_change} ({100*has_change/len(records):.1f}%) | "
                f"unchanged={no_change} | "
                f"mean_ratio={arr.mean():.4f} | "
                f"max_ratio={arr.max():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LEVIR-CD split change ratios.")
    parser.add_argument("--data_root", required=True,
                        help="Path to LEVIR-CD root (has train/ val/ test/ subdirs).")
    parser.add_argument("--out", default="data/LEVIRCD/split_stats.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    root   = Path(args.data_root)
    splits = ["train", "val", "test"]

    all_records: list[dict] = []
    for split in splits:
        recs = _scan_split(root, split)
        _print_stats(recs, split)
        all_records.extend(recs)

    logger.info(f"Total: {len(all_records)} samples")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    with open(out_path, "w", newline="") as f:
        if all_records:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
    logger.info(f"Saved CSV to {out_path}")

    # JSON (split-wise for easy loading)
    json_path = out_path.with_suffix(".json")
    split_data: dict[str, list] = {}
    for r in all_records:
        split_data.setdefault(r["split"], []).append(r)
    with open(json_path, "w") as f:
        json.dump(split_data, f, indent=2)
    logger.info(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
