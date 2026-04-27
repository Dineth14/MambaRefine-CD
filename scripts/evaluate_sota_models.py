#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.cd_model import build_model
from sota_eval.base_adapter import AdapterUnavailableError, BaseAdapter
from sota_eval.evaluator import evaluate_with_adapter
from sota_eval.registry import build_adapter
from training.checkpoint import peek as peek_ckpt
from utils.config import load_config

CONFIG_PATH = ROOT / "configs" / "sota_reproduce_config.resolved.yaml"
BASE_CONFIG_PATH = ROOT / "configs" / "sota_reproduce_config.yaml"
OUT_ROOT = ROOT / "outputs" / "sota_reproduced_eval"


def _load_cfg() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.exists() else BASE_CONFIG_PATH
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _device(cfg: dict) -> torch.device:
    requested = str(cfg.get("hardware", {}).get("device", "cuda"))
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested if ":" in requested else "cuda:0")
    return torch.device("cpu")


class OurModelAdapter(BaseAdapter):
    model_name = "MambaRefine-CD"
    expected_normalization = "imagenet"
    expected_output_type = "binary_logits"

    def is_available(self):
        if not self.checkpoint_path:
            return False, "checkpoint path missing in config"
        return super().is_available()

    def build_model(self):
        checkpoint = peek_ckpt(self.checkpoint_path)
        eval_cfg = checkpoint.get("config")
        if not isinstance(eval_cfg, dict):
            eval_cfg = load_config().to_dict()
        eval_cfg.setdefault("dataset", {})
        eval_cfg.setdefault("model", {})
        eval_cfg.setdefault("hardware", {})
        eval_cfg["dataset"]["name"] = self.dataset_name
        eval_cfg["dataset"]["mode"] = "binary"
        eval_cfg["model"]["output_mode"] = "binary"
        eval_cfg["model"]["pretrained"] = False
        eval_cfg["hardware"]["device"] = str(self.device)
        model = build_model(eval_cfg).to(self.device)
        return model

    def load_checkpoint(self, model):
        checkpoint = peek_ckpt(self.checkpoint_path)
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}

    def preprocess_batch(self, batch):
        batch = dict(batch)
        batch["image_a"] = batch["image_a"].to(self.device)
        batch["image_b"] = batch["image_b"].to(self.device)
        return batch

    def forward(self, model, batch):
        return model(batch["image_a"], batch["image_b"])[0]


def _write_failed_status(path: Path, stage: str, reason: str, checkpoint_path: str | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "status.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "reason": reason,
                "stage": stage,
                "checkpoint_path": checkpoint_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    cfg = _load_cfg()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = _device(cfg)
    datasets = cfg.get("datasets", {}).get("active", [])

    if bool(cfg.get("our_model", {}).get("enabled", True)):
        for dataset_name in datasets:
            checkpoint = cfg.get("our_model", {}).get("checkpoint_paths", {}).get(dataset_name)
            out_dir = OUT_ROOT / "MambaRefine-CD" / dataset_name
            adapter = OurModelAdapter(cfg, {"repo_dir": None, "weights_dir": None}, dataset_name, checkpoint, device)
            try:
                if not checkpoint:
                    raise AdapterUnavailableError("checkpoint not configured")
                evaluate_with_adapter(adapter, cfg, cfg["datasets"][dataset_name], out_dir)
            except Exception as exc:
                _write_failed_status(out_dir, "eval", str(exc), checkpoint)

    for model_name, model_cfg in cfg.get("external_models", {}).items():
        if not bool(model_cfg.get("enabled", True)):
            continue
        for dataset_name in datasets:
            checkpoint = model_cfg.get("resolved_checkpoints", {}).get(dataset_name)
            out_dir = OUT_ROOT / model_name / dataset_name
            if not checkpoint:
                status = "MANUAL_REQUIRED"
                weight_spec = model_cfg.get("official_weights", {}).get(dataset_name, {})
                weight_type = str(weight_spec.get("type", "missing"))
                if weight_type == "missing":
                    status = "MISSING_CHECKPOINT"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "status.json").write_text(
                    json.dumps(
                        {
                            "status": status,
                            "reason": weight_spec.get("note") or "checkpoint not discovered",
                            "stage": "load",
                            "checkpoint_path": None,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                continue
            checkpoint_abs = ROOT / checkpoint if not str(checkpoint).startswith("/") else Path(checkpoint)
            adapter = build_adapter(model_cfg.get("adapter"), cfg, model_cfg, dataset_name, str(checkpoint_abs), device)
            try:
                evaluate_with_adapter(adapter, cfg, cfg["datasets"][dataset_name], out_dir)
            except Exception as exc:
                _write_failed_status(out_dir, "eval", str(exc), str(checkpoint_abs))


if __name__ == "__main__":
    main()
