#!/usr/bin/env python3
"""Verify LEVIR ablation configs instantiate different models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from utils.config import load_config
from utils.ablation import (
    assert_model_matches_config,
    config_fingerprint,
    module_flags,
    parameter_breakdown,
)
from models.mambarefinecd import build_model


EXPECTED = {
    "a0_fpn_baseline.yaml": {
        "mambavision_enabled": False,
        "drbi_enabled": False,
        "signed_diff_enabled": False,
        "cram_lite_enabled": False,
        "arf_fpn_enabled": False,
        "boundary_refine_enabled": False,
        "boundary_loss_enabled": False,
    },
    "a1_mambavision_fpn.yaml": {
        "mambavision_enabled": True,
        "drbi_enabled": False,
        "signed_diff_enabled": False,
        "cram_lite_enabled": False,
        "arf_fpn_enabled": False,
        "boundary_refine_enabled": False,
        "boundary_loss_enabled": False,
    },
    "a2_mambavision_drbi.yaml": {
        "mambavision_enabled": True,
        "drbi_enabled": True,
        "signed_diff_enabled": False,
        "cram_lite_enabled": False,
        "arf_fpn_enabled": False,
        "boundary_refine_enabled": False,
        "boundary_loss_enabled": False,
    },
    "a3_mambavision_drbi_signed.yaml": {
        "mambavision_enabled": True,
        "drbi_enabled": True,
        "signed_diff_enabled": True,
        "cram_lite_enabled": False,
        "arf_fpn_enabled": False,
        "boundary_refine_enabled": False,
        "boundary_loss_enabled": False,
    },
    "a4_mambavision_drbi_arf.yaml": {
        "mambavision_enabled": True,
        "drbi_enabled": True,
        "signed_diff_enabled": True,
        "cram_lite_enabled": False,
        "arf_fpn_enabled": True,
        "boundary_refine_enabled": False,
        "boundary_loss_enabled": False,
    },
    "a5_mambavision_drbi_arf_boundary.yaml": {
        "mambavision_enabled": True,
        "drbi_enabled": True,
        "signed_diff_enabled": True,
        "cram_lite_enabled": False,
        "arf_fpn_enabled": True,
        "boundary_refine_enabled": True,
        "boundary_loss_enabled": False,
    },
    "a6_full.yaml": {
        "mambavision_enabled": True,
        "drbi_enabled": True,
        "signed_diff_enabled": True,
        "cram_lite_enabled": True,
        "arf_fpn_enabled": True,
        "boundary_refine_enabled": True,
        "boundary_loss_enabled": True,
    },
}


def _shape_of_output(output) -> tuple[int, ...]:
    if isinstance(output, dict):
        output = output.get("change_logits")
    if isinstance(output, (tuple, list)):
        output = output[0]
    return tuple(output.shape)


def verify_one(path: Path, device: torch.device, image_size: int) -> dict:
    cfg = load_config(path)
    # Verification should not depend on external pretrained downloads.
    cfg.setdefault("model", {})["pretrained"] = False
    flags = module_flags(cfg)
    expected = EXPECTED.get(path.name)
    if expected is None:
        raise AssertionError(f"Unexpected LEVIR ablation config: {path.name}")
    mismatched = {k: (flags.get(k), v) for k, v in expected.items() if flags.get(k) != v}
    if mismatched:
        raise AssertionError(f"{path.name} flag mismatch: {mismatched}")

    model = build_model(cfg).to(device).eval()
    assert_model_matches_config(model, cfg)
    params = parameter_breakdown(model)
    if flags["drbi_enabled"] and params["drbi_params"] <= 0:
        raise AssertionError(f"{path.name}: D-RBI enabled but drbi_params=0")
    if not flags["drbi_enabled"] and params["drbi_params"] != 0:
        raise AssertionError(f"{path.name}: D-RBI disabled but drbi_params={params['drbi_params']}")
    if flags["arf_fpn_enabled"] and params["arf_params"] <= 0:
        raise AssertionError(f"{path.name}: ARF-FPN enabled but arf_params=0")
    if not flags["arf_fpn_enabled"] and params["arf_params"] != 0:
        raise AssertionError(f"{path.name}: ARF-FPN disabled but arf_params={params['arf_params']}")
    if flags["cram_lite_enabled"] and params["cram_lite_params"] <= 0:
        raise AssertionError(f"{path.name}: CRAM-lite enabled but cram_lite_params=0")
    if not flags["cram_lite_enabled"] and params["cram_lite_params"] != 0:
        raise AssertionError(f"{path.name}: CRAM-lite disabled but cram_lite_params={params['cram_lite_params']}")
    if flags["boundary_refine_enabled"] and params["boundary_refinement_params"] <= 0:
        raise AssertionError(f"{path.name}: boundary refinement enabled but boundary_refinement_params=0")
    if not flags["boundary_refine_enabled"] and params["boundary_refinement_params"] != 0:
        raise AssertionError(f"{path.name}: boundary refinement disabled but boundary_refinement_params={params['boundary_refinement_params']}")
    x1 = torch.zeros(1, 3, image_size, image_size, device=device)
    x2 = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        out = model(x1, x2)
    shape = _shape_of_output(out)
    if shape != (1, 1, image_size, image_size):
        raise AssertionError(f"{path.name} output shape mismatch: {shape}")
    return {
        "config": str(path),
        "name": cfg.get("experiment", {}).get("name", path.stem),
        "fingerprint": config_fingerprint(cfg),
        "flags": flags,
        "params": params,
        "output_shape": shape,
    }


def write_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ablation Config Verification",
        "",
        "Generated by `python tools/verify_ablation_configs.py`.",
        "",
        "## Summary",
        "",
        "| Config | Fingerprint | Encoder | Decoder | D-RBI | Signed | CRAM-lite | ARF-FPN | Boundary Refine | Boundary Loss | Total Params |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        f = row["flags"]
        p = row["params"]
        lines.append(
            f"| {Path(row['config']).name} | {row['fingerprint']} | {f['encoder_name']} | {f['decoder_name']} | "
            f"{f['drbi_enabled']} | {f['signed_diff_enabled']} | {f['cram_lite_enabled']} | "
            f"{f['arf_fpn_enabled']} | {f['boundary_refine_enabled']} | {f['boundary_loss_enabled']} | {p['total_params']} |"
        )
    lines.extend(["", "## Parameter Breakdown", ""])
    for row in rows:
        lines.append(f"### {Path(row['config']).name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({"flags": row["flags"], "params": row["params"], "output_shape": row["output_shape"]}, indent=2))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LEVIR ablation model/config switches.")
    parser.add_argument("--config_dir", default="configs/ablations/levir")
    parser.add_argument("--report", default="docs/ABLATION_CONFIG_VERIFICATION.md")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    paths = sorted((REPO / args.config_dir).glob("*.yaml"))
    if sorted(p.name for p in paths) != sorted(EXPECTED):
        raise AssertionError(
            f"Expected exactly {sorted(EXPECTED)}, found {sorted(p.name for p in paths)}"
        )
    rows = []
    for path in paths:
        row = verify_one(path, device, args.image_size)
        rows.append(row)
        p = row["params"]
        f = row["flags"]
        print(
            f"{path.name}: total={p['total_params']} encoder={p['encoder_params']} "
            f"decoder={p['decoder_params']} drbi={p['drbi_params']} arf={p['arf_params']} "
            f"cram={p['cram_lite_params']} boundary={p['boundary_refinement_params']} flags={f}"
        )
    counts = {row["name"]: row["params"]["total_params"] for row in rows}
    if counts["a0_fpn_baseline"] == counts["a6_full"]:
        raise AssertionError("a0_fpn_baseline and a6_full have identical parameter counts.")
    write_report(rows, REPO / args.report)
    print(f"Saved report: {REPO / args.report}")


if __name__ == "__main__":
    main()
