#!/usr/bin/env python3
"""Verify DSIFN-CD ablation configs before running publication experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

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
    "training.optimizer": "AdamW",
    "training.lr": 5.0e-5,
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


def _plain(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _explicit_model_flags(cfg: dict) -> dict[str, bool]:
    model = cfg.get("model", {})
    return {
        "drbi_enabled": bool((model.get("drbi") or {}).get("enabled", False)),
        "signed_diff_enabled": bool((model.get("signed_diff") or {}).get("enabled", False)),
        "cram_lite_enabled": bool((model.get("cram_lite") or {}).get("enabled", False)),
        "arf_fpn_enabled": bool((model.get("arf_fpn") or {}).get("enabled", False)),
        "boundary_refine_enabled": bool((model.get("boundary_refine") or {}).get("enabled", False)),
    }


def _raw_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _critical_runtime_flags(cfg: dict) -> dict[str, Any]:
    model = cfg.get("model", {})
    diff = cfg.get("difference", {})
    decoder = cfg.get("decoder", {})
    loss = cfg.get("loss", {})
    loss_boundary = loss.get("boundary", {}) if isinstance(loss.get("boundary", {}), dict) else {}
    return {
        "backbone": model.get("backbone"),
        "model.decoder": model.get("decoder"),
        "difference.enabled": diff.get("enabled"),
        "difference.use_signed_diff": diff.get("use_signed_diff"),
        "model.cram_lite.enabled": (model.get("cram_lite") or {}).get("enabled"),
        "decoder.type": decoder.get("type"),
        "decoder.use_boundary_residual": decoder.get("use_boundary_residual"),
        "loss.boundary.enabled": loss_boundary.get("enabled"),
        "loss.boundary_weight": loss.get("boundary_weight"),
    }


def _structure_for_duplicate_check(cfg: dict) -> dict[str, Any]:
    data = _plain(cfg)
    data.pop("_meta", None)
    data.pop("experiment", None)
    return data


def _critical_signature(cfg: dict) -> str:
    return json.dumps(_structure_for_duplicate_check(cfg), sort_keys=True, default=str)


def _detect_config_issues(path: Path, cfg: dict, expected: dict[str, bool]) -> list[str]:
    issues: list[str] = []
    raw = _raw_yaml(path)
    if "_base_" in raw:
        issues.append("ERROR: _base_ inheritance is present, but this loader does not implement _base_ resolution.")

    required_model_flags = ["drbi", "signed_diff", "cram_lite", "arf_fpn", "boundary_refine"]
    model = raw.get("model", {}) if isinstance(raw.get("model", {}), dict) else {}
    for key in required_model_flags:
        section = model.get(key)
        if not isinstance(section, dict) or "enabled" not in section:
            issues.append(f"ERROR: missing explicit model.{key}.enabled")

    loss = raw.get("loss", {}) if isinstance(raw.get("loss", {}), dict) else {}
    boundary = loss.get("boundary") if isinstance(loss, dict) else None
    if not isinstance(boundary, dict) or "enabled" not in boundary:
        issues.append("ERROR: missing explicit loss.boundary.enabled")

    flags = module_flags(cfg)
    for key, expected_value in expected.items():
        if flags.get(key) != expected_value:
            issues.append(f"ERROR: resolved {key}={flags.get(key)!r}, expected {expected_value!r}")

    explicit = _explicit_model_flags(cfg)
    for key in ("drbi_enabled", "signed_diff_enabled", "cram_lite_enabled", "arf_fpn_enabled", "boundary_refine_enabled"):
        if key in expected and explicit.get(key) != expected[key]:
            issues.append(f"ERROR: explicit {key}={explicit.get(key)!r}, expected {expected[key]!r}")

    runtime = _critical_runtime_flags(cfg)
    if expected["arf_fpn_enabled"] and runtime["model.decoder"] != "adaptive_rf":
        issues.append("ERROR: ARF-FPN enabled but resolved model.decoder is not adaptive_rf")
    if not expected["arf_fpn_enabled"] and runtime["model.decoder"] == "adaptive_rf":
        issues.append("ERROR: ARF-FPN disabled but resolved model.decoder is adaptive_rf")
    if runtime["difference.enabled"] != expected["drbi_enabled"]:
        issues.append("ERROR: model.drbi.enabled did not resolve to difference.enabled")
    if runtime["difference.use_signed_diff"] != expected["signed_diff_enabled"]:
        issues.append("ERROR: model.signed_diff.enabled did not resolve to difference.use_signed_diff")
    if bool(runtime["decoder.use_boundary_residual"]) != expected["boundary_refine_enabled"]:
        issues.append("ERROR: model.boundary_refine.enabled did not resolve to decoder.use_boundary_residual")

    known_model_keys = {
        "mode",
        "backbone",
        "baseline_channels",
        "variant",
        "decoder",
        "pretrained",
        "output_mode",
        "enable_semantic_heads",
        "freeze_backbone",
        "drbi",
        "signed_diff",
        "cram_lite",
        "arf_fpn",
        "boundary_refine",
    }
    unknown_model_keys = sorted(set(model) - known_model_keys)
    for key in unknown_model_keys:
        issues.append(f"WARNING: unknown model key may be unused: model.{key}")

    if not issues:
        issues.append("OK: no missing keys, wrong key names, duplicate structure, or base override conflicts detected.")
    return issues


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

    build_cfg = cfg.to_dict()
    build_cfg.setdefault("model", {})["pretrained"] = False
    model = build_model(build_cfg).to(device).eval()
    assert_model_matches_config(model, build_cfg)
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
        "runtime_flags": _critical_runtime_flags(cfg),
        "resolved_config": _plain(cfg),
        "issues": _detect_config_issues(path, cfg, expected),
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
    resolved_seen: dict[str, str] = {}
    output_roots: dict[str, str] = {}
    for row in rows:
        sig = _duplicate_signature(row)
        prior = seen.get(sig)
        if prior is not None:
            raise AssertionError(
                f"{Path(prior).name} and {Path(row['config']).name} have identical "
                "parameter count and identical module flags."
            )
        seen[sig] = row["config"]
        resolved_sig = _critical_signature(row["resolved_config"])
        resolved_prior = resolved_seen.get(resolved_sig)
        if resolved_prior is not None:
            raise AssertionError(
                f"{Path(resolved_prior).name} and {Path(row['config']).name} resolve to the same config structure."
            )
        resolved_seen[resolved_sig] = row["config"]
        output_root = str(row["resolved_config"].get("experiment", {}).get("output_root", ""))
        output_prior = output_roots.get(output_root)
        if output_prior is not None:
            raise AssertionError(
                f"{Path(output_prior).name} and {Path(row['config']).name} use the same output_root={output_root!r}."
            )
        output_roots[output_root] = row["config"]


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


def write_debug_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DSIFN Ablation Debug Report",
        "",
        "Generated by `python tools/verify_ablation_configs_dsifn.py`.",
        "",
        "## Inheritance",
        "",
        "The seven publication DSIFN configs do not use `_base_` inheritance. They are loaded by deep-merging `configs/global_config.yaml` first, then the ablation config. Explicit `model.*.enabled` flags are now mapped to the runtime keys during normalization, so child flags remain authoritative after the global base merge.",
        "",
        "## Summary",
        "",
        "| Config | Backbone | D-RBI | Signed | CRAM-lite | ARF-FPN | Boundary Refine | Boundary Loss | Params | Issues |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        f = row["flags"]
        issue_text = "OK" if all(issue.startswith("OK:") for issue in row["issues"]) else "ERROR/WARNING"
        lines.append(
            f"| {Path(row['config']).name} | {f['encoder_name']} | {f['drbi_enabled']} | "
            f"{f['signed_diff_enabled']} | {f['cram_lite_enabled']} | {f['arf_fpn_enabled']} | "
            f"{f['boundary_refine_enabled']} | {f['boundary_loss_enabled']} | "
            f"{row['params']['total_params']} | {issue_text} |"
        )

    lines.extend(["", "## Per-Config Diagnostics", ""])
    for row in rows:
        lines.append(f"### {Path(row['config']).name}")
        lines.append("")
        lines.append("Flags:")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "model_flags": row["flags"],
                    "explicit_model_flags": row["explicit_model_flags"],
                    "runtime_flags": row["runtime_flags"],
                    "params": row["params"],
                    "output_shape": row["output_shape"],
                },
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")
        lines.append("Issues:")
        lines.append("")
        for issue in row["issues"]:
            lines.append(f"- {issue}")
        lines.append("")
        lines.append("Full resolved config:")
        lines.append("")
        lines.append("```yaml")
        lines.append(yaml.safe_dump(row["resolved_config"], sort_keys=False))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DSIFN-CD ablation configs.")
    parser.add_argument("--config_dir", default="configs/ablations/dsifn")
    parser.add_argument("--report", default="results/dsifn_config_verification.md")
    parser.add_argument("--debug_report", default="docs/DSIFN_ABLATION_DEBUG_REPORT.md")
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
    write_debug_report(rows, REPO / args.debug_report)
    json_path = REPO / args.json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved report: {REPO / args.report}")
    print(f"Saved debug report: {REPO / args.debug_report}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
