#!/usr/bin/env python3
"""Short synthetic training check to catch identical ablation behavior/checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch

from models.mambarefinecd import build_model
from training.losses import build_loss
from training.model_outputs import normalize_model_output
from utils.checkpoint_identity import sha256_file
from utils.config import load_config


FIELDS = [
    "variant_name",
    "params_M",
    "initial_loss",
    "final_loss",
    "output_mean_initial",
    "output_std_initial",
    "output_mean_final",
    "output_std_final",
    "grad_norm",
    "checkpoint_sha256",
    "status",
    "notes",
]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        value = float(p.grad.detach().float().norm(2).item())
        total += value * value
    return math.sqrt(total)


def _params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def run_one(path: Path, args: argparse.Namespace, device: torch.device, ckpt_dir: Path) -> dict:
    row = {key: "" for key in FIELDS}
    row["variant_name"] = path.stem
    row["status"] = "FAIL"
    notes: list[str] = []
    try:
        torch.manual_seed(int(args.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed))
        cfg = load_config(path)
        cfg.setdefault("model", {})["pretrained"] = False
        cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
        cfg.setdefault("dataset", {})["image_size"] = int(args.image_size)
        variant = str(cfg.get("experiment", {}).get("name", path.stem))
        row["variant_name"] = variant

        model = build_model(cfg).to(device)
        model.train()
        loss_fn = build_loss(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)

        ia = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        ib = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        target = (torch.rand(args.batch_size, 1, args.image_size, args.image_size, device=device) > 0.5).float()

        row["params_M"] = f"{_params_m(model):.4f}"
        initial_loss = None
        final_loss = None
        initial_mean = initial_std = final_mean = final_std = None
        grad_norm = 0.0
        for iteration in range(int(args.iters)):
            optimizer.zero_grad(set_to_none=True)
            outputs = normalize_model_output(model(ia, ib))
            logits = outputs["change_logits"]
            total, _, _ = loss_fn(logits, target)
            if not torch.isfinite(total):
                raise RuntimeError(f"NaN/Inf loss at iter {iteration}: {float(total.detach().item())}")
            if iteration == 0:
                initial_loss = float(total.detach().item())
                initial_mean = float(logits.detach().float().mean().item())
                initial_std = float(logits.detach().float().std().item())
            total.backward()
            grad_norm = _grad_norm(model)
            optimizer.step()
            final_loss = float(total.detach().item())
            final_mean = float(logits.detach().float().mean().item())
            final_std = float(logits.detach().float().std().item())

        ckpt_path = ckpt_dir / f"{variant}_quick_iter{args.iters}.pth"
        torch.save({"model": model.state_dict(), "variant_name": variant, "iters": int(args.iters)}, ckpt_path)
        row.update({
            "initial_loss": f"{initial_loss:.6f}",
            "final_loss": f"{final_loss:.6f}",
            "output_mean_initial": f"{initial_mean:.6f}",
            "output_std_initial": f"{initial_std:.6f}",
            "output_mean_final": f"{final_mean:.6f}",
            "output_std_final": f"{final_std:.6f}",
            "grad_norm": f"{grad_norm:.6f}",
            "checkpoint_sha256": sha256_file(ckpt_path),
            "status": "PASS",
        })
    except Exception as exc:
        notes.append(f"{type(exc).__name__}: {exc}")
    row["notes"] = " ".join(notes)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short ablation divergence sanity check.")
    parser.add_argument("--config_dir", default="configs/ablations/dsifn")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="outputs/quick_ablation_divergence_test.csv")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = (REPO / config_dir).resolve()
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No YAML configs found in {config_dir}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / "quick_ablation_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)

    rows = [run_one(path, args, device, ckpt_dir) for path in paths]
    hashes: dict[str, list[str]] = {}
    signatures: dict[str, list[str]] = {}
    for row in rows:
        if row["checkpoint_sha256"]:
            hashes.setdefault(row["checkpoint_sha256"], []).append(row["variant_name"])
        sig = json.dumps({
            "output_mean_initial": row["output_mean_initial"],
            "output_std_initial": row["output_std_initial"],
            "output_mean_final": row["output_mean_final"],
            "output_std_final": row["output_std_final"],
        }, sort_keys=True)
        signatures.setdefault(sig, []).append(row["variant_name"])

    for row in rows:
        duplicate_hashes = [names for names in hashes.values() if row["variant_name"] in names and len(names) > 1]
        duplicate_outputs = [names for names in signatures.values() if row["variant_name"] in names and len(names) > 1]
        if duplicate_hashes or duplicate_outputs:
            row["status"] = "FAIL"
            extra = []
            if duplicate_hashes:
                extra.append(f"Identical checkpoint hash shared by {duplicate_hashes[0]}.")
            if duplicate_outputs:
                extra.append(f"Identical output stats shared by {duplicate_outputs[0]}.")
            row["notes"] = (row["notes"] + " " if row["notes"] else "") + " ".join(extra)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant_name']}: {row['status']} "
            f"loss {row['initial_loss']} -> {row['final_loss']} "
            f"hash={row['checkpoint_sha256'][:12]} notes={row['notes']}"
        )
    print(f"Saved quick divergence CSV: {out_path}")


if __name__ == "__main__":
    main()
