#!/usr/bin/env python3
"""Precompute SECOND binary masks from configs/global_config.yaml."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.second import precompute_second_binary_masks
from utils.config import load_config

SUMMARY_PATH = ROOT / "outputs" / "second_binary_masks" / "precompute_summary.json"


def _resolve_second_cfg(cfg: dict) -> dict:
    active = dict(cfg.get("dataset", {}))
    if str(active.get("name", "")).strip().upper() == "SECOND":
        return active
    catalog = cfg.get("datasets_catalog", {})
    second_cfg = catalog.get("SECOND") or catalog.get("second")
    if second_cfg is None:
        raise KeyError("SECOND dataset config was not found in configs/global_config.yaml.")
    return dict(second_cfg)


def main() -> None:
    cfg = load_config()
    second_cfg = _resolve_second_cfg(cfg)
    summary = precompute_second_binary_masks(second_cfg, force=False)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("SECOND binary mask precompute complete")
    print(f"Cache root: {summary['cache_root']}")
    for split, split_info in summary["splits"].items():
        print(
            f"{split:>5s}  created={split_info['created_masks']:4d}  "
            f"total={split_info['total_entries']:4d}  "
            f"changed_pixel_ratio={split_info['changed_pixel_ratio']:.4f}  "
            f"ignored_pixel_ratio={split_info['ignored_pixel_ratio']:.4f}"
        )
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
