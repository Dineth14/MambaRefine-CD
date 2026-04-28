"""Single global configuration loader.

All entry scripts load a single config file:
    configs/global_config.yaml

Training can optionally layer a dataset-specific override config on top of the
global base config.

The returned object supports both dict-style and dot-style access.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
GLOBAL_CONFIG_PATH = ROOT / "configs" / "global_config.yaml"
TRAIN_CONFIG_DIR = ROOT / "configs" / "train"


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


def _plain_dict(data: dict[str, Any] | Config) -> dict[str, Any]:
    if isinstance(data, Config):
        return data.to_dict()
    return deepcopy(dict(data))


def _apply_ablation_aliases(merged: dict[str, Any], override: dict[str, Any]) -> None:
    """Translate paper-facing ablation switches to runtime config keys."""
    training_override = override.get("training", {}) if isinstance(override.get("training", {}), dict) else {}
    model_override = override.get("model", {}) if isinstance(override.get("model", {}), dict) else {}
    decoder_override = override.get("decoder", {}) if isinstance(override.get("decoder", {}), dict) else {}
    loss_override = override.get("loss", {}) if isinstance(override.get("loss", {}), dict) else {}

    merged.setdefault("training", {})
    merged.setdefault("hardware", {})
    merged.setdefault("ema", {})
    merged.setdefault("model", {})
    merged.setdefault("difference", {})
    merged.setdefault("decoder", {})
    merged.setdefault("loss", {})
    merged.setdefault("boundary_metrics", {})

    if "max_iter" in training_override:
        merged["training"]["max_iterations"] = training_override["max_iter"]
    if "val_every" in training_override:
        merged["training"]["validate_every"] = training_override["val_every"]
    if "amp" in training_override:
        merged["hardware"]["mixed_precision"] = bool(training_override["amp"])
    if "ema" in training_override:
        merged["ema"]["enabled"] = bool(training_override["ema"])

    if bool(model_override.get("disable_drbi", False)):
        merged["difference"]["enabled"] = False

    if bool(model_override.get("disable_boundary", False)):
        merged["difference"]["use_boundary_gate"] = False

    if bool(model_override.get("disable_boundary_refinement", False)):
        merged["decoder"]["use_boundary_residual"] = False

    if bool(model_override.get("disable_diff_features", False)):
        merged["difference"]["use_absdiff"] = False
        merged["difference"]["use_product"] = False

    decoder_type = model_override.get("decoder_type")
    if decoder_type is not None:
        decoder_name = "baseline" if str(decoder_type).lower() == "simple" else str(decoder_type)
        merged["model"]["decoder"] = decoder_name
        merged["decoder"]["type"] = decoder_name

    if "alpha" in decoder_override:
        merged["decoder"]["residual_scale"] = decoder_override["alpha"]

    if loss_override:
        use_dice = loss_override.get("use_dice")
        use_boundary = loss_override.get("use_boundary")
        if use_dice is False and use_boundary is False:
            merged["loss"]["type"] = "bce_dice"
            merged["loss"]["bce_weight"] = 1.0
            merged["loss"]["dice_weight"] = 0.0
            merged["loss"]["boundary_weight"] = 0.0
            merged["loss"]["focal_weight"] = 0.0
            merged["loss"]["sek_weight"] = 0.0
        elif use_dice is True and use_boundary is False:
            merged["loss"]["type"] = "bce_dice"
            merged["loss"]["bce_weight"] = 1.0
            merged["loss"]["dice_weight"] = 1.0
            merged["loss"]["boundary_weight"] = 0.0
            merged["loss"]["focal_weight"] = 0.0
            merged["loss"]["sek_weight"] = 0.0
        elif use_dice is True and use_boundary is True:
            merged["loss"]["type"] = merged["loss"].get("full_type", "dice_focal_sek")
            merged["loss"]["dice_weight"] = 1.0
            merged["loss"].setdefault("focal_weight", 0.2)
            merged["loss"].setdefault("sek_weight", 0.05)


def apply_ablation(cfg: dict[str, Any] | Config, ablation_cfg: dict[str, Any] | Config) -> Config:
    """Return a config with ablation overrides recursively applied.

    The merge is non-destructive for the input objects and preserves Config
    dot-access semantics on the returned object.
    """
    base = _plain_dict(cfg)
    override = _plain_dict(ablation_cfg)
    merged = _deep_merge(base, override)
    _apply_ablation_aliases(merged, override)
    return Config(_normalize_config(merged))


def _resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    return config_path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    train_alias = data.get("train", {})
    train_ema = train_alias.get("ema", {}) if isinstance(train_alias, dict) else {}
    if isinstance(train_ema, dict) and train_ema:
        data.setdefault("ema", {})
        data["ema"]["enabled"] = bool(train_ema.get("enabled", data["ema"].get("enabled", False)))
        data["ema"]["decay"] = float(train_ema.get("decay", data["ema"].get("decay", 0.999)))
        if "save_best_by" in train_alias:
            data.setdefault("checkpoint", {})["selection_metric"] = train_alias["save_best_by"]
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
    if "gpu_ids" not in hardware and "gpu_id" in hardware:
        raw_gpu_id = hardware.get("gpu_id")
        hardware["gpu_ids"] = raw_gpu_id if isinstance(raw_gpu_id, list) else [raw_gpu_id]
    gpu_ids = hardware.get("gpu_ids", [0])
    device = str(hardware.get("device", "cuda"))
    if device == "cuda" and gpu_ids:
        hardware["device"] = f"cuda:{int(gpu_ids[0])}"
    data["hardware"] = hardware

    model = data.get("model", {})
    model.setdefault("output_mode", "binary")
    model.setdefault("num_classes", 1)
    model.setdefault("semantic_num_classes", int(data.get("dataset", {}).get("num_classes", 7)))
    model.setdefault("enable_semantic_heads", False)
    model.setdefault("semantic_head_type", "lightweight")
    data["model"] = model

    # Preserve checkpoint path for eval/validate in one place.
    checkpoint = data.get("checkpoint", {})
    checkpoint.setdefault("path", None)
    if "selection_metric" in checkpoint and "monitor" not in checkpoint:
        checkpoint["monitor"] = checkpoint["selection_metric"]
    checkpoint.setdefault(
        "monitor",
        "Fscd" if str(data.get("task", "")).lower() == "semantic_cd" or str(data.get("dataset", {}).get("name", "")).upper() == "SECOND" else "f1",
    )
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
    dataset_cfg.setdefault("task_type", "semantic_change" if str(dataset_cfg.get("mode", "binary")).lower() == "semantic" else "binary_change")
    dataset_cfg.setdefault("precompute_second_binary_masks", False)
    dataset_cfg.setdefault("second_binary_cache_dir", "outputs/second_binary_masks")
    dataset_cfg.setdefault("cache_images_in_ram", False)
    dataset_cfg.setdefault("cache_masks_in_ram", False)
    dataset_cfg.setdefault("second_label_palette", None)
    data["dataset"] = dataset_cfg

    evaluation = data.get("evaluation", {})
    eval_alias = data.get("eval", {})
    if eval_alias:
        evaluation.update(eval_alias)
    evaluation.setdefault("split", "test")
    evaluation.setdefault("threshold", 0.5)
    evaluation.setdefault("use_ema", training.get("use_ema", False))
    evaluation.setdefault("threshold_list", [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    threshold_sweep = evaluation.get("threshold_sweep", False)
    if isinstance(threshold_sweep, dict):
        evaluation["threshold_sweep"] = {
            "enabled": bool(threshold_sweep.get("enabled", False)),
            "values": threshold_sweep.get("values", evaluation.get("threshold_list")),
        }
    else:
        evaluation.setdefault("threshold_sweep", False)
    evaluation.setdefault("tta_augmentations", ["original", "hflip", "vflip", "rot90"])
    evaluation.setdefault("threshold_select_metric", "mF1")
    evaluation.setdefault("inference_mode", "patch")
    evaluation.setdefault("crop_size", dataset_cfg.get("image_size", 256))
    evaluation.setdefault("overlap", 0.25)
    evaluation.setdefault("second_metrics", False)
    evaluation.setdefault("compute_SeK", True)
    evaluation.setdefault("sek_binary_fallback", False)
    data["evaluation"] = evaluation
    data["eval"] = evaluation

    metrics = data.get("metrics", {})
    metrics.setdefault("average_mode", "global")
    data["metrics"] = metrics

    loss_cfg = data.get("loss", {})
    final_cfg = loss_cfg.get("final", {}) if isinstance(loss_cfg.get("final", {}), dict) else {}
    if final_cfg:
        loss_cfg["type"] = final_cfg.get("type", loss_cfg.get("type", "bce_dice"))
        loss_cfg["bce_weight"] = float(final_cfg.get("bce_weight", loss_cfg.get("bce_weight", 1.0)))
        loss_cfg["dice_weight"] = float(final_cfg.get("dice_weight", loss_cfg.get("dice_weight", 1.0)))
        # A nested final loss means the ablation intended the simple binary loss
        # family, so clear inherited focal/SeK weights unless explicitly set at
        # the same nested level.
        loss_cfg["focal_weight"] = float(final_cfg.get("focal_weight", 0.0))
        loss_cfg["sek_weight"] = float(final_cfg.get("sek_weight", 0.0))
    boundary_cfg = loss_cfg.get("boundary", {}) if isinstance(loss_cfg.get("boundary", {}), dict) else {}
    if boundary_cfg:
        loss_cfg["boundary_weight"] = float(boundary_cfg.get("weight", 0.0)) if bool(boundary_cfg.get("enabled", False)) else 0.0
    coarse_cfg = loss_cfg.get("coarse", {}) if isinstance(loss_cfg.get("coarse", {}), dict) else {}
    if coarse_cfg:
        data.setdefault("decoder", {})["aux_weight"] = float(coarse_cfg.get("weight", 0.0)) if bool(coarse_cfg.get("enabled", False)) else 0.0
    loss_cfg.setdefault("type", "bce_dice")
    loss_cfg.setdefault("dice_weight", 1.0)
    loss_cfg.setdefault("boundary_weight", 0.0)
    loss_cfg.setdefault("focal_weight", 0.3)
    loss_cfg.setdefault("focal_gamma", 1.5)
    loss_cfg.setdefault("bce_weight", 1.0)
    loss_cfg.setdefault("sek_weight", 0.05)
    loss_cfg.setdefault("change_loss_weight", 1.0)
    loss_cfg.setdefault("semantic_loss_weight", 0.5)
    loss_cfg.setdefault("consistency_loss_weight", 0.2)
    loss_cfg.setdefault("sek_loss_weight", 0.3)
    loss_cfg.setdefault("ce_weight", 1.0)
    loss_cfg.setdefault("sek_mode", str(dataset_cfg.get("mode", "binary")))
    loss_cfg.setdefault("sek_eps", 1e-6)
    loss_cfg.setdefault("sek_separate_nochange", False)
    loss_cfg.setdefault("consistency_detach_semantic", True)
    loss_cfg.setdefault("consistency_loss_type", "bce")
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


def load_config(path: str | Path | None = None) -> Config:
    config_path = GLOBAL_CONFIG_PATH if path is None else _resolve_config_path(path)
    if config_path == GLOBAL_CONFIG_PATH:
        raw = _read_yaml(config_path)
    else:
        raw = _deep_merge(_read_yaml(GLOBAL_CONFIG_PATH), _read_yaml(config_path))
    normalized = _normalize_config(raw)
    from utils.ablation import config_fingerprint
    normalized["_meta"] = {
        "config_path": str(config_path),
        "config_fingerprint": config_fingerprint(normalized),
    }
    return Config(normalized)
