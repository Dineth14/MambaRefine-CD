#!/usr/bin/env python3
"""Verify DSIFN-CD ablation configs before running publication experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from utils.ablation import (
    assert_model_matches_config,
    config_fingerprint,
    module_flags,
    parameter_breakdown,
)
from utils.config import load_config
from models.mambarefinecd import build_model


EXPECTED_FILES = [
    "a0_fpn_baseline.yaml",
    "a1_mambavision_fpn.yaml",
    "a2_mambavision_drbi.yaml",
    "a3_mambavision_drbi_signed.yaml",
    "a4_mambavision_drbi_arf.yaml",
    "a5_mambavision_drbi_arf_boundary.yaml",
    "a6_full.yaml",
]

EXPECTED_FLAGS = {
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

FAIRNESS_FIELDS = {
    "dataset.name": "DSIFN-CD",
    "dataset.image_size": 256,
    "dataset.batch_size": 8,
    "dataset.augmentation_ops": ["horizontal_flip", "vertical_flip"],
    "dataset.val_ratio": 0.2,
    "training.batch_size": 8,
    "training.optimizer": "Adam",
    "training.lr": 1.0e-4,
    "training.scheduler": "cosine",
    "training.max_iterations": 50000,
    "ema.enabled": True,
    "experiment.seed": 42,
}


def _get(cfg: dict, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _explicit_model_flags(cfg: dict) -> dict[str, bool]:
    model = cfg.get("model", {})
    return {
        "drbi_enabled": bool((model.get("drbi") or {}).get("enabled", False)),
        "signed_diff_enabled": bool((model.get("signed_diff") or {}).get("enabled", False)),
        "cram_lite_enabled": bool((model.get("cram_lite") or {}).get("enabled", False)),
        "arf_fpn_enabled": bool((model.get("arf_fpn") or {}).get("enabled", False)),
        "boundary_refine_enabled": bool((model.get("boundary_refine") or {}).get("enabled", False)),
    }


def _shape_of_output(output) -> tuple[int, ...]:
    if isinstance(output, dict):
        output = output.get("change_logits")
    if isinstance(output, (tuple, list)):
        output = output[0]
    return tuple(output.shape)


def _check_fairness(cfg: dict, path: Path) -> None:
    for dotted, expected in FAIRNESS_FIELDS.items():
        actual = _get(cfg, dotted)
        if actual != expected:
            raise AssertionError(f"{path.name}: {dotted}={actual!r}, expected {expected!r}")
    allowed = cfg.get("metrics", {}).get("allowed")
    if list(allowed) != ["Pre", "Rec", "F1", "IoU", "OA"]:
        raise AssertionError(f"{path.name}: metrics.allowed must be [Pre, Rec, F1, IoU, OA], got {allowed}")
    if bool(cfg.get("resume", {}).get("enabled", False)):
        raise AssertionError(f"{path.name}: resume.enabled must stay false for independent ablations")


def verify_one(path: Path, device: torch.device, image_size: int) -> dict:
    cfg = load_config(path)
    cfg.setdefault("model", {})["pretrained"] = False
    _check_fairness(cfg, path)

    flags = module_flags(cfg)
    expected = EXPECTED_FLAGS[path.name]
    mismatched = {k: (flags.get(k), v) for k, v in expected.items() if flags.get(k) != v}
    if mismatched:
        raise AssertionError(f"{path.name}: runtime flag mismatch {mismatched}")

    explicit = _explicit_model_flags(cfg)
    explicit_expected = {k: v for k, v in expected.items() if k in explicit}
    explicit_mismatched = {
        k: (explicit.get(k), v) for k, v in explicit_expected.items() if explicit.get(k) != v
    }
    if explicit_mismatched:
        raise AssertionError(f"{path.name}: explicit model flag mismatch {explicit_mismatched}")

    model = build_model(cfg).to(device).eval()
    assert_model_matches_config(model, cfg)
    params = parameter_breakdown(model)
    shape: tuple[int, ...] | str
    if device.type == "cpu" and flags["mambavision_enabled"]:
        # MambaVision selective_scan is CUDA-only in the installed dependency.
        # CPU verification still builds the model and counts parameters.
        shape = "skipped_cpu_mambavision_cuda_kernel_required"
    else:
        x1 = torch.zeros(1, 3, image_size, image_size, device=device)
        x2 = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            shape = _shape_of_output(model(x1, x2))
        if shape != (1, 1, image_size, image_size):
            raise AssertionError(f"{path.name}: output shape {shape}, expected {(1, 1, image_size, image_size)}")

    return {
        "config": str(path),
        "name": cfg.get("experiment", {}).get("name", path.stem),
        "fingerprint": config_fingerprint(cfg),
        "flags": flags,
        "explicit_model_flags": explicit,
        "params": params,
        "output_shape": shape,
    }


def _duplicate_signature(row: dict) -> tuple[int, str]:
    relevant_flags = {
        k: row["flags"][k]
        for k in (
            "mambavision_enabled",
            "drbi_enabled",
            "signed_diff_enabled",
            "cram_lite_enabled",
            "arf_fpn_enabled",
            "boundary_refine_enabled",
            "boundary_loss_enabled",
        )
    }
    return row["params"]["total_params"], json.dumps(relevant_flags, sort_keys=True)


def check_duplicates(rows: list[dict]) -> None:
    seen: dict[tuple[int, str], str] = {}
    for row in rows:
        sig = _duplicate_signature(row)
        prior = seen.get(sig)
        if prior is not None:
            raise AssertionError(
                f"{Path(prior).name} and {Path(row['config']).name} have identical "
                "parameter count and identical module flags."
            )
        seen[sig] = row["config"]


def write_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DSIFN-CD Ablation Config Verification",
        "",
        "Generated by `python tools/verify_ablation_configs_dsifn.py --cpu`.",
        "",
        "## Summary",
        "",
        "| Config | Fingerprint | Backbone | Decoder | D-RBI | Signed | CRAM-lite | ARF-FPN | Boundary Refine | Boundary Loss | Total Params | Trainable Params |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        f = row["flags"]
        p = row["params"]
        lines.append(
            f"| {Path(row['config']).name} | {row['fingerprint']} | {f['encoder_name']} | {f['decoder_name']} | "
            f"{f['drbi_enabled']} | {f['signed_diff_enabled']} | {f['cram_lite_enabled']} | "
            f"{f['arf_fpn_enabled']} | {f['boundary_refine_enabled']} | {f['boundary_loss_enabled']} | "
            f"{p['total_params']} | {p['trainable_params']} |"
        )
    lines.extend(["", "## Detail", ""])
    for row in rows:
        lines.append(f"### {Path(row['config']).name}")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "flags": row["flags"],
                    "explicit_model_flags": row["explicit_model_flags"],
                    "params": row["params"],
                    "output_shape": row["output_shape"],
                },
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DSIFN-CD ablation configs.")
    parser.add_argument("--config_dir", default="configs/ablations/dsifn")
    parser.add_argument("--report", default="results/dsifn_config_verification.md")
    parser.add_argument("--json", default="results/dsifn_config_verification.json")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config_dir = REPO / args.config_dir
    paths = [config_dir / name for name in EXPECTED_FILES]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required DSIFN ablation configs: {missing}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    rows = [verify_one(path, device, args.image_size) for path in paths]
    check_duplicates(rows)

    for row in rows:
        p = row["params"]
        f = row["flags"]
        print(
            f"{Path(row['config']).name}: modules="
            f"backbone={f['encoder_name']}, decoder={f['decoder_name']}, "
            f"drbi={f['drbi_enabled']}, signed={f['signed_diff_enabled']}, "
            f"cram={f['cram_lite_enabled']}, arf={f['arf_fpn_enabled']}, "
            f"boundary_refine={f['boundary_refine_enabled']}, boundary_loss={f['boundary_loss_enabled']} | "
            f"params={p['total_params']} trainable={p['trainable_params']}"
        )

    write_report(rows, REPO / args.report)
    json_path = REPO / args.json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved report: {REPO / args.report}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
