"""Single global configuration loader.

All entry scripts load a single config file:
    configs/global_config.yaml

The returned object supports both dict-style and dot-style access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
GLOBAL_CONFIG_PATH = ROOT / "configs" / "global_config.yaml"


class Config(dict):
    """Recursive mapping with dot access.

    Keeps compatibility with existing dict-based internals while allowing:
        cfg.training.lr
        cfg["training"]["lr"]
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        super().__init__()
        data = data or {}
        for key, value in data.items():
            super().__setitem__(key, self._wrap(value))

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def to_dict(self) -> dict[str, Any]:
        def unwrap(obj: Any) -> Any:
            if isinstance(obj, Config):
                return {k: unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [unwrap(v) for v in obj]
            return obj

        return unwrap(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _match_dataset_entry(catalog: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    target = dataset_name.lower().strip()
    for key, value in catalog.items():
        if key.lower() == target:
            return value
        if isinstance(value, dict) and str(value.get("name", "")).lower().strip() == target:
            return value
    return None


def _normalize_config(data: dict[str, Any]) -> dict[str, Any]:
    data.setdefault("experiment", {})
    data.setdefault("hardware", {})
    data.setdefault("model", {})
    data.setdefault("training", {})
    data.setdefault("dataset", {})
    data.setdefault("evaluation", {})
    data.setdefault("resume", {})
    data.setdefault("loss", {})
    data.setdefault("ema", {})
    data.setdefault("debug", {})
    data.setdefault("validation", {})
    data.setdefault("checkpoint", {})
    data.setdefault("decoder", {})
    data.setdefault("boundary_metrics", {})
    data.setdefault("post_training", {})

    # Resolve active dataset from one-file catalog.
    catalog = data.get("datasets_catalog", {})
    dataset = dict(data.get("dataset", {}))
    ds_name = str(dataset.get("name", "")).strip()
    if catalog and ds_name:
        base_ds = _match_dataset_entry(catalog, ds_name)
        if base_ds is not None:
            data["dataset"] = _deep_merge(base_ds, dataset)

    # EMA lives in one top-level section, but training internals still expect keys under training.
    ema = data.get("ema", {})
    training = data.get("training", {})
    training["use_ema"] = bool(ema.get("enabled", training.get("use_ema", False)))
    training["ema_decay"] = float(ema.get("decay", training.get("ema_decay", 0.999)))
    training.setdefault("non_blocking_transfer", True)
    data["training"] = training

    # Normalize optimizer names for torch.optim getattr usage.
    opt_name = str(training.get("optimizer", "AdamW"))
    opt_map = {
        "adamw": "AdamW",
        "adam": "Adam",
        "sgd": "SGD",
        "rmsprop": "RMSprop",
    }
    training["optimizer"] = opt_map.get(opt_name.lower(), opt_name)

    # Keep device and gpu_ids consistent. If device is bare cuda, use first gpu_ids entry.
    hardware = data.get("hardware", {})
    gpu_ids = hardware.get("gpu_ids", [0])
    device = str(hardware.get("device", "cuda"))
    if device == "cuda" and gpu_ids:
        hardware["device"] = f"cuda:{int(gpu_ids[0])}"
    data["hardware"] = hardware

    model = data.get("model", {})
    model.setdefault("output_mode", "binary")
    model.setdefault("num_classes", 1)
    model.setdefault("semantic_num_classes", int(data.get("dataset", {}).get("num_classes", 7)))
    data["model"] = model

    # Preserve checkpoint path for eval/validate in one place.
    checkpoint = data.get("checkpoint", {})
    checkpoint.setdefault("path", None)
    checkpoint.setdefault("monitor", "f1")
    checkpoint.setdefault("mode", "max")
    checkpoint.setdefault("save_best_only", True)
    data["checkpoint"] = checkpoint

    validation = data.get("validation", {})
    validation.setdefault("batch_size", training.get("batch_size", 8))
    validation.setdefault("save_samples", True)
    validation.setdefault("sample_count", 16)
    validation.setdefault("split", "val")
    data["validation"] = validation

    dataset_cfg = data.get("dataset", {})
    dataset_cfg.setdefault("num_workers", 8)
    dataset_cfg.setdefault("pin_memory", True)
    dataset_cfg.setdefault("persistent_workers", True)
    dataset_cfg.setdefault("prefetch_factor", 4)
    dataset_cfg.setdefault("precompute_second_binary_masks", False)
    dataset_cfg.setdefault("second_binary_cache_dir", "outputs/second_binary_masks")
    dataset_cfg.setdefault("cache_images_in_ram", False)
    dataset_cfg.setdefault("cache_masks_in_ram", False)
    data["dataset"] = dataset_cfg

    evaluation = data.get("evaluation", {})
    evaluation.setdefault("split", "test")
    evaluation.setdefault("threshold_list", [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    evaluation.setdefault("tta_augmentations", ["original", "hflip", "vflip", "rot90"])
    evaluation.setdefault("threshold_select_metric", "mF1")
    evaluation.setdefault("second_metrics", False)
    evaluation.setdefault("compute_SeK", True)
    evaluation.setdefault("sek_binary_fallback", False)
    data["evaluation"] = evaluation

    loss_cfg = data.get("loss", {})
    loss_cfg.setdefault("type", "bce_dice")
    loss_cfg.setdefault("dice_weight", 1.0)
    loss_cfg.setdefault("focal_weight", 0.3)
    loss_cfg.setdefault("focal_gamma", 1.5)
    loss_cfg.setdefault("bce_weight", 1.0)
    loss_cfg.setdefault("sek_weight", 0.05)
    loss_cfg.setdefault("sek_mode", str(dataset_cfg.get("mode", "binary")))
    loss_cfg.setdefault("sek_eps", 1e-6)
    loss_cfg.setdefault("sek_separate_nochange", False)
    data["loss"] = loss_cfg

    debug = data.get("debug", {})
    debug.setdefault("name", "memory_debug")
    debug.setdefault("output_root", "outputs/memory_debug")
    debug.setdefault("steps", 3)
    debug.setdefault("batch_size", training.get("batch_size", 8))
    debug.setdefault("image_size", data["dataset"].get("image_size", 256))
    debug.setdefault("use_amp", hardware.get("mixed_precision", True))
    debug.setdefault("profile_torch_ops", True)
    debug.setdefault("profile_one_step_only", True)
    debug.setdefault("compare_modes", [
        "baseline_forward_only",
        "train_forward_loss_backward",
        "train_with_return_features_false",
        "train_with_return_features_true",
        "train_with_ema_disabled",
        "train_with_ema_enabled",
        "train_with_tta_disabled",
        "eval_no_grad",
        "eval_with_tta",
        "decoder_baseline",
        "decoder_refinement",
        "decoder_adaptive_rf",
    ])
    debug.setdefault("save_memory_summary", True)
    debug.setdefault("seed", data["experiment"].get("seed", 42))
    data["debug"] = debug

    benchmark = data.get("benchmark", {})
    benchmark.setdefault("model_name", data["experiment"].get("name", "model"))
    benchmark.setdefault("output_dir", "outputs/benchmark_runs/summary")
    benchmark.setdefault("datasets", ["LEVIR-CD", "WHU-CD", "SYSU-CD", "DSIFN-CD", "SECOND"])
    benchmark.setdefault("eval_split", "test")
    benchmark.setdefault("main_dataset", data["dataset"].get("name", "LEVIR-CD"))
    benchmark.setdefault("checkpoints", {})
    data["benchmark"] = benchmark

    return data


def load_config() -> Config:
    with GLOBAL_CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(_normalize_config(raw))
