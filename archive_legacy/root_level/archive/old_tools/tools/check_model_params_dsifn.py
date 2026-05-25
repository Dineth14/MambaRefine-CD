#!/usr/bin/env python3
"""Check DSIFN-CD ablation parameter counts are strictly increasing."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from utils.ablation import assert_model_matches_config, module_flags, parameter_breakdown
from utils.config import load_config
from models.mambarefinecd import build_model


CONFIGS = [
    "a0_fpn_baseline.yaml",
    "a1_mambavision_fpn.yaml",
    "a2_mambavision_drbi.yaml",
    "a3_mambavision_drbi_signed.yaml",
    "a4_mambavision_drbi_arf.yaml",
    "a5_mambavision_drbi_arf_boundary.yaml",
    "a6_full.yaml",
]


def main() -> None:
    config_dir = REPO / "configs" / "ablations" / "dsifn"
    rows = []
    for name in CONFIGS:
        path = config_dir / name
        cfg = load_config(path)
        build_cfg = cfg.to_dict()
        build_cfg.setdefault("model", {})["pretrained"] = False
        model = build_model(build_cfg)
        assert_model_matches_config(model, build_cfg)
        params = parameter_breakdown(model)
        flags = module_flags(cfg)
        rows.append((name, params, flags))
        print(
            f"{name}: total={params['total_params']} trainable={params['trainable_params']} "
            f"backbone={flags['encoder_name']} drbi={flags['drbi_enabled']} "
            f"signed={flags['signed_diff_enabled']} cram={flags['cram_lite_enabled']} "
            f"arf={flags['arf_fpn_enabled']} boundary_refine={flags['boundary_refine_enabled']} "
            f"boundary_loss={flags['boundary_loss_enabled']}"
        )

    totals = [params["total_params"] for _, params, _ in rows]
    duplicates = sorted({value for value in totals if totals.count(value) > 1})
    if duplicates:
        raise AssertionError(f"Duplicate DSIFN ablation parameter counts found: {duplicates}")

    if any(left >= right for left, right in zip(totals, totals[1:])):
        formatted = ", ".join(f"{name}={params['total_params']}" for name, params, _ in rows)
        raise AssertionError(f"Expected strictly increasing counts a0 < a1 < ... < a6, got: {formatted}")

    print("DSIFN parameter check passed: a0 < a1 < a2 < a3 < a4 < a5 < a6")


if __name__ == "__main__":
    main()
