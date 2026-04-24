"""Standalone model evaluation script.

Edit CONFIG_PATH to switch which evaluation config to use.
No CLI arguments needed.

Outputs:
    outputs/eval_runs/<name>/eval_metrics.json
    outputs/eval_runs/<name>/eval_metrics.csv
    (+ predictions if evaluation.save_predictions: true)

Run:
    conda activate mamba_new
    cd MambaRefine-CD
    python scripts/evaluate.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from utils.config_loader      import load_config
from utils.seed               import set_seed
from data.dataset_builder     import build_test_loader
from models.cd_model          import build_model
from training.evaluator       import Evaluator
from training.checkpoint      import peek as peek_ckpt, load as load_ckpt
from training.logger          import get_logger

# ── Change this to switch evaluation target ───────────────────────────────────
CONFIG_PATH = "configs/experiments/eval_levir.yaml"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg  = load_config(ROOT / CONFIG_PATH)
    exp  = cfg.get("experiment", {})
    hw   = cfg.get("hardware", {})

    set_seed(int(exp.get("seed", 42)))

    # ── Output directory ──────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / exp.get("output_root", "outputs/eval_runs") / f"eval_{ts}_{exp.get('name', 'eval')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(exp.get("name", "eval"), out_dir / "logs")
    logger.info(f"Evaluation  : {CONFIG_PATH}")
    logger.info(f"Output dir  : {out_dir}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(hw.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Device      : {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = cfg.get("dataset", {})
    split  = cfg.get("evaluation", {}).get("split", "test")
    logger.info(f"Dataset     : {ds_cfg.get('name', 'unknown')}  split={split}")

    loader = build_test_loader(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    ckpt_path = cfg.get("checkpoint", {}).get("path")
    if not ckpt_path:
        raise ValueError(
            "checkpoint.path is null in eval config. "
            "Set it to a trained checkpoint, e.g.:\n"
            "  checkpoint:\n"
            "    path: outputs/benchmark_runs/run_XXX/checkpoints/best.pth"
        )
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_info = peek_ckpt(ckpt_path)
    logger.info(f"Checkpoint  : {ckpt_path}")
    logger.info(f"  Iteration : {ckpt_info.get('iteration', '?')}")
    logger.info(f"  Best F1   : {ckpt_info.get('best_metric', '?')}")

    model.load_state_dict(ckpt_info["model"], strict=True)
    model.eval()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator = Evaluator(cfg, device, logger=logger)
    amp       = bool(hw.get("mixed_precision", True))
    dataset_name = ds_cfg.get("name", "unknown")

    logger.info(f"\nRunning evaluation on {dataset_name} [{split}] ...")
    results = evaluator.evaluate(model, loader, dataset_name=dataset_name, amp=amp)

    # ── Print table ───────────────────────────────────────────────────────────
    evaluator.print_table(results, title=f"\n── Evaluation Results [{dataset_name}] ──")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results_meta = {
        "config":      CONFIG_PATH,
        "checkpoint":  str(ckpt_path),
        "dataset":     dataset_name,
        "split":       split,
        "timestamp":   ts,
        "metrics":     {k: v for k, v in results.items() if isinstance(v, (int, float, str))},
    }
    json_path = out_dir / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_meta, f, indent=2)
    logger.info(f"\nSaved JSON  → {json_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "eval_metrics.csv"
    metric_keys = ["f1", "iou", "miou", "precision", "recall", "oa",
                   "boundary_f1", "edge_iou", "pred_positive_ratio", "gt_positive_ratio"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split", "checkpoint", "timestamp"] + metric_keys)
        w.writerow(
            [dataset_name, split, str(ckpt_path), ts]
            + [results.get(k, "") for k in metric_keys]
        )
    logger.info(f"Saved CSV   → {csv_path}")


if __name__ == "__main__":
    main()
