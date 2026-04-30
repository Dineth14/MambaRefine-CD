#!/usr/bin/env python3
"""Compare actual constructed model modules across ablation configs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from models.mambarefinecd import build_model
from utils.config import load_config


FIELDS = [
    "config_path",
    "variant_name",
    "params_M",
    "trainable_params_M",
    "encoder_type",
    "backbone_name",
    "decoder_type",
    "fusion_input_channels",
    "has_drbi",
    "has_adaptive_rf",
    "has_boundary_refiner",
    "has_cram_lite",
    "fusion_terms",
    "dilation_rates",
    "loss_terms",
    "diff_vs_a6_full",
    "status",
    "notes",
]


def _yes(value: Any) -> str:
    return "true" if bool(value) else "false"


def _trace_signature(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "encoder_type": trace.get("encoder_type"),
        "backbone_name": trace.get("backbone_name"),
        "fusion_terms_used": trace.get("fusion_terms_used"),
        "fusion_input_channels": trace.get("fusion_input_channels"),
        "drbi_enabled": trace.get("drbi_enabled"),
        "region_gate_enabled": trace.get("region_gate_enabled"),
        "boundary_gate_enabled": trace.get("boundary_gate_enabled"),
        "decoder_type": trace.get("decoder_type"),
        "adaptive_rf_enabled": trace.get("adaptive_rf_enabled"),
        "dilation_rates": trace.get("dilation_rates"),
        "boundary_residual_enabled": trace.get("boundary_residual_enabled"),
        "cram_lite_enabled": trace.get("cram_lite_enabled"),
        "loss_terms": trace.get("loss_terms"),
    }


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, list[Any]]:
    keys = sorted(set(a) | set(b))
    return {key: [b.get(key), a.get(key)] for key in keys if a.get(key) != b.get(key)}


def build_row(path: Path) -> tuple[dict[str, str], dict[str, Any] | None]:
    row = {key: "" for key in FIELDS}
    row["config_path"] = str(path)
    row["variant_name"] = path.stem
    row["status"] = "FAIL"
    try:
        cfg = load_config(path)
        cfg.setdefault("model", {})["pretrained"] = False
        model = build_model(cfg)
        trace = model.get_ablation_trace() if hasattr(model, "get_ablation_trace") else {}
        row.update({
            "variant_name": str(cfg.get("experiment", {}).get("name", path.stem)),
            "params_M": f"{float(trace.get('params_M', 0.0)):.4f}",
            "trainable_params_M": f"{float(trace.get('trainable_params_M', 0.0)):.4f}",
            "encoder_type": str(trace.get("encoder_type", "")),
            "backbone_name": str(trace.get("backbone_name", "")),
            "decoder_type": str(trace.get("decoder_type", "")),
            "fusion_input_channels": str(trace.get("fusion_input_channels", "")),
            "has_drbi": _yes(trace.get("drbi_enabled", False)),
            "has_adaptive_rf": _yes(trace.get("adaptive_rf_enabled", False)),
            "has_boundary_refiner": _yes(trace.get("boundary_residual_enabled", False)),
            "has_cram_lite": _yes(trace.get("cram_lite_enabled", False)),
            "fusion_terms": json.dumps(trace.get("fusion_terms_used", {}), sort_keys=True),
            "dilation_rates": json.dumps(trace.get("dilation_rates", [])),
            "loss_terms": json.dumps(trace.get("loss_terms", {}), sort_keys=True),
            "status": "PASS",
        })
        return row, _trace_signature(trace)
    except Exception as exc:
        row["notes"] = f"{type(exc).__name__}: {exc}"
        return row, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ablation model construction.")
    parser.add_argument("--config_dir", default="configs/ablations/dsifn")
    parser.add_argument("--out", default="outputs/ablation_model_comparison.csv")
    parser.add_argument("--full_variant", default="a6_full")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = (REPO / config_dir).resolve()
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No YAML configs found in {config_dir}")

    rows: list[dict[str, str]] = []
    traces: dict[str, dict[str, Any]] = {}
    for path in paths:
        row, signature = build_row(path)
        rows.append(row)
        if signature is not None:
            traces[row["variant_name"]] = signature

    full_name = args.full_variant
    full_trace = traces.get(full_name)
    if full_trace is None:
        for name in traces:
            if name.endswith(args.full_variant):
                full_name = name
                full_trace = traces[name]
                break

    if full_trace is not None:
        for row in rows:
            trace = traces.get(row["variant_name"])
            if trace is None:
                continue
            diff = _diff(trace, full_trace)
            row["diff_vs_a6_full"] = json.dumps(diff, sort_keys=True)
            if row["variant_name"] != full_name and not diff:
                row["status"] = "FAIL"
                row["notes"] = (row["notes"] + " " if row["notes"] else "") + "Trace is identical to a6_full."

    signatures = [json.dumps(t, sort_keys=True) for t in traces.values()]
    if signatures and len(set(signatures)) == 1:
        for row in rows:
            row["status"] = "FAIL"
            row["notes"] = (row["notes"] + " " if row["notes"] else "") + (
                "FAIL: ablation configs are not changing model construction."
            )
        print("FAIL: ablation configs are not changing model construction.")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant_name']}: {row['status']} params={row['params_M']} "
            f"encoder={row['encoder_type']} decoder={row['decoder_type']} "
            f"drbi={row['has_drbi']} arf={row['has_adaptive_rf']} "
            f"boundary={row['has_boundary_refiner']} cram={row['has_cram_lite']} "
            f"notes={row['notes']}"
        )
    print(f"Saved model comparison CSV: {out_path}")


if __name__ == "__main__":
    main()
