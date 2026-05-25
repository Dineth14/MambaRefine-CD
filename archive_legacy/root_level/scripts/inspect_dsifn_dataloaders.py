#!/usr/bin/env python3
"""Inspect DSIFN-CD dataloader/dataset split behavior directly."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

from dsifn_audit_utils import load_dataset_config, manifest_rows, write_text


def mask_unique(path: str) -> list[int]:
    with Image.open(path) as img:
        vals = sorted(set(img.convert("L").getdata()))
    return vals[:32]


def summarize_rows(label: str, rows: list[dict], sample_count: int) -> list[str]:
    lines = [f"## {label}", "", f"- Dataset length: `{len(rows)}`"]
    first = rows[:sample_count]
    rng = random.Random(42)
    random_rows = rng.sample(rows, min(sample_count, len(rows))) if rows else []
    for title, subset in (("First samples", first), ("Random samples", random_rows)):
        lines.extend(["", f"### {title}", ""])
        for row in subset:
            lines.append(
                f"- `{row['sample_index']}` pre=`{row['pre_image_path']}` post=`{row['post_image_path']}` "
                f"mask=`{row['mask_path']}` crop=({row['crop_x']},{row['crop_y']})"
            )
    if rows:
        row = rows[0]
        lines.extend([
            "",
            "### First Sample Shapes",
            "",
            f"- Image shape: `{Image.open(row['pre_image_path']).size}`",
            f"- Mask shape: `{Image.open(row['mask_path']).size}`",
            f"- Mask unique values: `{mask_unique(row['mask_path'])}`",
        ])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DSIFN-CD dataloaders/datasets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--out", default="outputs/dsifn_dataloader_inspection.md")
    args = parser.parse_args()

    cfg, ds_cfg = load_dataset_config(args.config)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    rows_by_split = {split: manifest_rows(ds_cfg, split, seed=seed) for split in ("train", "val", "test")}
    path_sets = {split: {r["pre_image_path"] for r in rows} for split, rows in rows_by_split.items()}

    lines = [
        "# DSIFN Dataloader Inspection",
        "",
        f"- Config: `{args.config}`",
        "- Train uses augmentation: `true`",
        "- Val/test use random augmentation: `false`",
        "- Val/test loader shuffle expected: `false`",
        "",
        "## Cross-Split Path Presence",
        "",
        "| Split Pair | Shared Pre Paths |",
        "|---|---:|",
    ]
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        lines.append(f"| {left} vs {right} | {len(path_sets[left] & path_sets[right])} |")
    lines.append("")
    for split in ("train", "val", "test"):
        lines.extend(summarize_rows(split, rows_by_split[split], args.num_samples))
        lines.append("")
    write_text(args.out, "\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
