#!/usr/bin/env python3
"""Build train/val/test DSIFN-CD sample manifests."""
from __future__ import annotations

import argparse
from pathlib import Path

from dsifn_audit_utils import (
    MANIFEST_FIELDS,
    load_dataset_config,
    manifest_rows,
    resolve_split,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DSIFN-CD split manifests.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="outputs/dsifn_manifests")
    args = parser.parse_args()

    cfg, ds_cfg = load_dataset_config(args.config)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    out_dir = Path(args.out_dir)
    summary = {"config": args.config, "seed": seed, "splits": {}}
    for split in ("train", "val", "test"):
        res = resolve_split(ds_cfg, split, seed=seed)
        rows = manifest_rows(ds_cfg, split, seed=seed)
        out_path = out_dir / f"{split}_manifest.csv"
        write_csv(out_path, rows, fields=MANIFEST_FIELDS)
        summary["splits"][split] = {
            "layout": res.layout,
            "source": res.source,
            "image_names": len(res.names),
            "manifest_rows": len(rows),
            "manifest_path": str(out_path),
        }
        print(f"{split}: names={len(res.names)} rows={len(rows)} -> {out_path}")
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
