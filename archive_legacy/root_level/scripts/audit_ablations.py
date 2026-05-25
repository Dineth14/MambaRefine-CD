#!/usr/bin/env python3
"""Audit active binary-CD ablation configs without training.

The script builds each config, runs a dummy forward/loss pass, records compact
model trace metadata, and saves a CSV report under outputs/.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch

from models.mambarefinecd import build_model
from training.losses import build_loss
from training.model_outputs import normalize_model_output
from utils.ablation import module_flags
from utils.config import load_config
from utils.memory import params_m


FIELDS = [
    "config_path",
    "variant_name",
    "dataset",
    "use_raw_pair",
    "use_abs_diff",
    "use_signed_diff",
    "use_feature_product",
    "drbi_enabled",
    "region_gate_enabled",
    "boundary_gate_enabled",
    "decoder_type",
    "boundary_residual_enabled",
    "use_bce",
    "use_dice",
    "use_coarse_loss",
    "use_boundary_loss",
    "params_M",
    "dummy_forward_ok",
    "output_shape",
    "loss_ok",
    "metrics_ok",
    "status",
    "notes",
]


def _bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loss_terms(cfg: dict) -> tuple[bool, bool, bool, bool]:
    loss = cfg.get("loss", {})
    coarse = loss.get("coarse", {}) if isinstance(loss.get("coarse", {}), dict) else {}
    boundary = loss.get("boundary", {}) if isinstance(loss.get("boundary", {}), dict) else {}
    return (
        float(loss.get("bce_weight", 1.0)) > 0.0,
        float(loss.get("dice_weight", 1.0)) > 0.0,
        bool(coarse.get("enabled", False)) and float(coarse.get("weight", 0.0)) > 0.0,
        bool(boundary.get("enabled", False)) and float(boundary.get("weight", 0.0)) > 0.0,
    )


def audit_one(path: Path, device: torch.device, batch_size: int, image_size: int) -> dict:
    cfg = load_config(path)
    cfg.setdefault("model", {})["pretrained"] = False
    cfg.setdefault("debug", {})["ablation_trace"] = True
    flags = module_flags(cfg)
    use_bce, use_dice, use_coarse, use_boundary = _loss_terms(cfg)
    row = {
        "config_path": str(path),
        "variant_name": cfg.get("experiment", {}).get("name", path.stem),
        "dataset": cfg.get("dataset", {}).get("name", "unknown"),
        "use_raw_pair": "unknown",
        "use_abs_diff": "unknown",
        "use_signed_diff": _bool(cfg.get("difference", {}).get("use_signed_diff", False)),
        "use_feature_product": _bool(cfg.get("difference", {}).get("use_product", False)),
        "drbi_enabled": _bool(flags["drbi_enabled"]),
        "region_gate_enabled": _bool(cfg.get("difference", {}).get("use_region_gate", False)),
        "boundary_gate_enabled": _bool(cfg.get("difference", {}).get("use_boundary_gate", False)),
        "decoder_type": flags["decoder_name"],
        "boundary_residual_enabled": _bool(flags["boundary_refine_enabled"]),
        "use_bce": _bool(use_bce),
        "use_dice": _bool(use_dice),
        "use_coarse_loss": _bool(use_coarse),
        "use_boundary_loss": _bool(use_boundary),
        "params_M": "",
        "dummy_forward_ok": "false",
        "output_shape": "",
        "loss_ok": "false",
        "metrics_ok": "false",
        "status": "FAIL",
        "notes": "",
    }
    notes: list[str] = []
    try:
        model = build_model(cfg).to(device)
        model.eval()
        row["params_M"] = f"{params_m(model):.4f}"
        x1 = torch.zeros(batch_size, 3, image_size, image_size, device=device)
        x2 = torch.zeros(batch_size, 3, image_size, image_size, device=device)
        with torch.inference_mode():
            out = model(x1, x2)
        normalized = normalize_model_output(out)
        logits = normalized["change_logits"]
        trace = model.get_ablation_trace() if hasattr(model, "get_ablation_trace") else {}
        terms_raw = trace.get("fusion_terms_used", {})
        if isinstance(terms_raw, dict):
            terms = {key for key, enabled in terms_raw.items() if enabled}
        else:
            terms = set(terms_raw)
        row["use_raw_pair"] = _bool("raw_pair" in terms)
        row["use_abs_diff"] = _bool("abs_diff" in terms)
        row["use_signed_diff"] = _bool("signed_diff" in terms)
        row["use_feature_product"] = _bool("feature_product" in terms or "product" in terms)
        row["region_gate_enabled"] = _bool(trace.get("region_gate_enabled", flags["drbi_enabled"]))
        row["boundary_gate_enabled"] = _bool(trace.get("boundary_gate_enabled", flags["drbi_enabled"]))
        row["boundary_residual_enabled"] = _bool(trace.get("boundary_residual_enabled", flags["boundary_refine_enabled"]))
        row["output_shape"] = "x".join(str(v) for v in logits.shape)
        row["dummy_forward_ok"] = _bool(tuple(logits.shape) == (batch_size, 1, image_size, image_size))

        model.train()
        loss_fn = build_loss(cfg)
        target = torch.zeros(batch_size, 1, image_size, image_size, device=device)
        total, _, _ = loss_fn(logits, target)
        row["loss_ok"] = _bool(torch.isfinite(total).item())
        allowed = set(cfg.get("metrics", {}).get("allowed", ["Pre", "Rec", "F1", "IoU", "OA"]))
        row["metrics_ok"] = _bool(allowed == {"Pre", "Rec", "F1", "IoU", "OA"})
        if row["dummy_forward_ok"] == "true" and row["loss_ok"] == "true" and row["metrics_ok"] == "true":
            row["status"] = "PASS"
        if flags["drbi_enabled"] is False and row["use_abs_diff"] == "true":
            notes.append("D-RBI disabled; decoder legacy path still uses abs-diff+sum fusion.")
        if flags["boundary_refine_enabled"] is False and row["boundary_residual_enabled"] == "true":
            notes.append("Boundary residual unexpectedly active.")
    except Exception as exc:
        notes.append(f"{type(exc).__name__}: {exc}")
    row["notes"] = " ".join(notes)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ablation configs for active binary CD.")
    parser.add_argument("--config_dir", default="configs/ablations/dsifn")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    config_dir = (REPO / args.config_dir).resolve() if not Path(args.config_dir).is_absolute() else Path(args.config_dir)
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No YAML configs found in {config_dir}")
    device = _device(args.device)
    dataset_slug = config_dir.name
    out_path = Path(args.out) if args.out else REPO / "outputs" / f"ablation_audit_{dataset_slug}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [audit_one(path, device, args.batch_size, args.image_size) for path in paths]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{Path(row['config_path']).name}: {row['status']} params_M={row['params_M']} output={row['output_shape']} notes={row['notes']}")
    print(f"Saved audit CSV: {out_path}")


if __name__ == "__main__":
    main()
