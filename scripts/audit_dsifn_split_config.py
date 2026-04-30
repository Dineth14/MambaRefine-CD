#!/usr/bin/env python3
"""Audit DSIFN-CD split configuration and resolved sample sources."""
from __future__ import annotations

import argparse
from pathlib import Path

from dsifn_audit_utils import (
    REPO_ROOT,
    load_dataset_config,
    markdown_table,
    resolve_split,
    write_json,
    write_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DSIFN-CD split configuration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--json_out", default="outputs/dsifn_split_config_audit.json")
    parser.add_argument("--md_out", default="outputs/dsifn_split_config_audit.md")
    args = parser.parse_args()

    cfg, ds_cfg = load_dataset_config(args.config)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    root = Path(ds_cfg["root"])
    image_size = int(ds_cfg.get("image_size", 256))
    crop_size = int(cfg.get("eval", cfg.get("evaluation", {})).get("crop_size", image_size))
    overlap = cfg.get("eval", cfg.get("evaluation", {})).get("overlap", None)
    stride = image_size
    if overlap is not None:
        stride = int(crop_size * (1.0 - float(overlap)))

    split_infos = {}
    for split in ("train", "val", "test"):
        res = resolve_split(ds_cfg, split, seed=seed)
        split_infos[split] = {
            "layout": res.layout,
            "source": res.source,
            "base_dir": str(res.base_dir),
            "pre_image_dir": str(res.a_dir),
            "post_image_dir": str(res.b_dir),
            "mask_dir": str(res.mask_dir),
            "num_image_names": len(res.names),
            "first_10_names": res.names[:10],
            "samples_are_tiles": res.samples_are_tiles,
        }

    audit = {
        "config": str((REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)),
        "dataset_name": ds_cfg.get("name"),
        "dataset_root": str(root),
        "split_dir": ds_cfg.get("split_dir"),
        "require_explicit_splits": bool(ds_cfg.get("require_explicit_splits", True)),
        "seed": seed,
        "image_size": image_size,
        "crop_size": crop_size,
        "eval_stride_from_overlap": stride,
        "val_ratio": ds_cfg.get("val_ratio", 0.2),
        "pre_image_path_pattern": "resolved by split source + image_a_dir_candidates + sample filename/stem",
        "post_image_path_pattern": "resolved by split source + image_b_dir_candidates + sample filename/stem",
        "mask_path_pattern": "resolved by split source + label_dir_candidates + sample filename/stem",
        "image_a_dir_candidates": ds_cfg.get("image_a_dir_candidates"),
        "image_b_dir_candidates": ds_cfg.get("image_b_dir_candidates"),
        "label_dir_candidates": ds_cfg.get("label_dir_candidates"),
        "augmentation": {
            "dataset_augmentation": ds_cfg.get("augmentation", ds_cfg.get("augment", True)),
            "augmentation_ops": ds_cfg.get("augmentation_ops", []),
            "train_random_crop": True,
            "val_test_random_augmentation": False,
        },
        "patch_generation": {
            "train": "random crop at __getitem__ when source image is larger than crop",
            "val": "deterministic non-overlapping tiles from resolved val names",
            "test": "deterministic non-overlapping tiles from resolved test names",
        },
        "splits": split_infos,
    }
    write_json(args.json_out, audit)

    rows = []
    for split, info in split_infos.items():
        rows.append([
            split,
            info["layout"],
            info["num_image_names"],
            info["source"],
            info["pre_image_dir"],
            info["post_image_dir"],
            info["mask_dir"],
        ])
    md = [
        "# DSIFN Split Config Audit",
        "",
        f"- Config: `{args.config}`",
        f"- Dataset root: `{root}`",
        f"- Split dir: `{ds_cfg.get('split_dir', root / 'splits')}`",
        f"- Require explicit splits: `{bool(ds_cfg.get('require_explicit_splits', True))}`",
        f"- Seed: `{seed}`",
        f"- Image/crop size: `{image_size}`",
        f"- Validation ratio fallback: `{ds_cfg.get('val_ratio', 0.2)}`",
        f"- Train random crop: `true`",
        f"- Val/test random augmentation: `false`",
        "",
        "## Resolved Split Sources",
        "",
        markdown_table(
            ["Split", "Layout", "Image Names", "Source", "Pre Dir", "Post Dir", "Mask Dir"],
            rows,
        ),
        "",
        "## Notes",
        "",
        "- Flat DSIFN roots must use explicit non-overlapping split files.",
        "- The unsafe fallback where `test` uses all flat images is disabled.",
    ]
    write_text(args.md_out, "\n".join(md) + "\n")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
