#!/usr/bin/env python3
"""Sequential ablation runner for MambaRefine-CD."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.pipeline import run_training_pipeline
from utils.config import TRAIN_CONFIG_DIR, apply_ablation, load_config

ABLATION_CONFIG_PATH = ROOT / "configs" / "ablation_config.yaml"
OUTPUT_ROOT = ROOT / "outputs" / "ablation"
DATASET_CONFIG_MAP = {
    "LEVIR-CD": TRAIN_CONFIG_DIR / "levir_cd.yaml",
    "WHU-CD": TRAIN_CONFIG_DIR / "whu_cd.yaml",
    "DSIFN-CD": TRAIN_CONFIG_DIR / "dsifn_cd.yaml",
    "SECOND": TRAIN_CONFIG_DIR / "second_semantic.yaml",
}


def _slug(value: str) -> str:
    return value.lower().replace("/", "-").replace(" ", "_")


def _safe_copy(src: Path, dst: Path) -> Path:
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    stamped = dst.with_name(f"{dst.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{dst.suffix}")
    shutil.copy2(src, stamped)
    return stamped


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _prepare_cfg(dataset_name: str, common_cfg: dict[str, Any], experiment_name: str, override: dict[str, Any]):
    cfg = load_config(DATASET_CONFIG_MAP[dataset_name])
    cfg = apply_ablation(cfg, common_cfg)
    cfg = apply_ablation(cfg, override)
    cfg = apply_ablation(
        cfg,
        {
            "experiment": {
                "name": f"ablation_{_slug(dataset_name)}_{experiment_name}",
            },
            "dataset": {
                "name": dataset_name,
            },
        },
    )
    return cfg


def main() -> None:
    spec = yaml.safe_load(ABLATION_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    ablation_cfg = spec.get("ablation", {})
    if not bool(ablation_cfg.get("enable", False)):
        raise SystemExit("ablation.enable=false in configs/ablation_config.yaml")

    dataset_names = list(spec.get("datasets", {}).get("active", []))
    experiments = dict(ablation_cfg.get("experiments", {}))
    common_cfg = {
        "experiment": spec.get("experiment", {}),
        "training": spec.get("training", {}),
    }

    for dataset_name in dataset_names:
        if dataset_name not in DATASET_CONFIG_MAP:
            raise KeyError(f"Unsupported ablation dataset: {dataset_name}")
        for experiment_name, override in experiments.items():
            print(f"Running ablation: {experiment_name}")
            experiment_root = OUTPUT_ROOT / _slug(dataset_name) / experiment_name
            run_dir = experiment_root / "runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir.parent.mkdir(parents=True, exist_ok=True)

            cfg = _prepare_cfg(dataset_name, common_cfg, experiment_name, override)
            config_path = experiment_root / "config_used.yaml"
            if config_path.exists():
                config_path = config_path.with_name(f"config_used_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            _write_yaml(config_path, cfg.to_dict())

            result = run_training_pipeline(cfg, output_dir=run_dir, config_source_path=ABLATION_CONFIG_PATH)
            metrics_source = run_dir / "test_results" / "test_metrics.json"
            checkpoint_source = run_dir / "checkpoints" / "best.pth"
            final_metrics = result.get("final_metrics") or {}

            (experiment_root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "experiment": experiment_name,
                        "description": override.get("description"),
                        "run_dir": str(run_dir.relative_to(ROOT)),
                        "config_used": str(config_path.relative_to(ROOT)),
                        "final_metrics_available": bool(final_metrics),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if metrics_source.exists():
                metrics_copy = _safe_copy(metrics_source, experiment_root / "metrics.json")
                eval_copy = _safe_copy(metrics_source, experiment_root / "eval_metrics.json")
                print(f"  metrics -> {metrics_copy.relative_to(ROOT)}")
                print(f"  eval    -> {eval_copy.relative_to(ROOT)}")

            if checkpoint_source.exists():
                checkpoint_copy = _safe_copy(checkpoint_source, experiment_root / "best_checkpoint.pth")
                print(f"  ckpt    -> {checkpoint_copy.relative_to(ROOT)}")


if __name__ == "__main__":
    main()