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


def _has_enabled_flag(section: Any) -> bool:
    return isinstance(section, dict) and "enabled" in section


def _enabled(section: dict[str, Any]) -> bool:
    return bool(section.get("enabled", False))


def _apply_explicit_model_module_flags(data: dict[str, Any]) -> None:
    """Map explicit ablation flags under model.* to runtime keys.

    Runtime modules are instantiated from historical keys:
      - D-RBI: difference.enabled
      - signed diff: difference.use_signed_diff
      - ARF-FPN: model.decoder / decoder.type
      - boundary refinement: decoder.use_boundary_residual
      - boundary loss: loss.boundary.enabled

    Publication configs additionally define model.drbi.enabled,
    model.signed_diff.enabled, model.arf_fpn.enabled, and
    model.boundary_refine.enabled. When those explicit keys are present, they
    are authoritative so a base config cannot silently re-enable modules.
    """
    model = data.setdefault("model", {})
    difference = data.setdefault("difference", {})
    decoder = data.setdefault("decoder", {})

    drbi = model.get("drbi")
    if _has_enabled_flag(drbi):
        difference["enabled"] = _enabled(drbi)

    signed = model.get("signed_diff")
    if _has_enabled_flag(signed):
        difference["use_signed_diff"] = _enabled(signed)

    arf = model.get("arf_fpn")
    if _has_enabled_flag(arf):
        decoder_name = "adaptive_rf" if _enabled(arf) else "baseline"
        model["decoder"] = decoder_name
        decoder["type"] = decoder_name
        if not _enabled(arf):
            decoder["dilation_rates"] = []

    boundary_refine = model.get("boundary_refine")
    if _has_enabled_flag(boundary_refine):
        decoder["use_boundary_residual"] = _enabled(boundary_refine)
        if not _enabled(boundary_refine):
            decoder["residual_scale"] = 0.0


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
    data.setdefault("efficiency", {})
    data.setdefault("profiling", {})
    data.setdefault("dataloader", {})
    data.setdefault("logging", {})
    data.setdefault("optimizer", {})

    # Resolve active dataset from one-file catalog.
    catalog = data.get("datasets_catalog", {})
    dataset = dict(data.get("dataset", {}))
    ds_name = str(dataset.get("name", "")).strip()
    if catalog and ds_name:
        base_ds = _match_dataset_entry(catalog, ds_name)
        if base_ds is not None:
            data["dataset"] = _deep_merge(base_ds, dataset)

    efficiency = data.get("efficiency", {})
    hardware = data.get("hardware", {})
    training = data.get("training", {})
    if "amp" in efficiency:
        hardware["mixed_precision"] = bool(efficiency.get("amp"))
    if "gradient_checkpointing" in efficiency:
        training["gradient_checkpointing"] = bool(efficiency.get("gradient_checkpointing"))
    efficiency.setdefault("amp", bool(hardware.get("mixed_precision", True)))
    efficiency.setdefault("amp_dtype", "fp16")
    efficiency.setdefault("gradient_checkpointing", bool(training.get("gradient_checkpointing", False)))
    efficiency.setdefault("channels_last", False)
    efficiency.setdefault("compile", False)
    efficiency.setdefault("compile_mode", "reduce-overhead")
    efficiency.setdefault("fast_mode", False)
    efficiency.setdefault("channels_multiplier", 1.0)
    data["efficiency"] = efficiency
    data["hardware"] = hardware
    data["training"] = training

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
    training["use_ema"] = bool(ema.get("enabled", training.get("use_ema", False)))
    training["ema_decay"] = float(ema.get("decay", training.get("ema_decay", 0.999)))
    training.setdefault("non_blocking_transfer", True)
    training.setdefault("gradient_checkpointing", bool(efficiency.get("gradient_checkpointing", False)))
    training.setdefault("overwrite_output_dir", False)
    training.setdefault("allow_resume_for_ablation", False)
    if "val_interval" in training and "validate_every" not in training:
        training["validate_every"] = training["val_interval"]
    training.setdefault("validate_every", training.get("val_interval", 5000))
    data["training"] = training

    resume = data.get("resume", {})
    resume.setdefault("enabled", False)
    resume.setdefault("checkpoint_path", None)
    resume.setdefault("strict", True)
    data["resume"] = resume

    optimizer_cfg = data.get("optimizer", {})
    optimizer_cfg.setdefault("grad_clip_norm", training.get("gradient_clip", None))
    if optimizer_cfg.get("grad_clip_norm", None) is None:
        training["gradient_clip"] = 0.0
    else:
        training["gradient_clip"] = float(optimizer_cfg["grad_clip_norm"])
    data["optimizer"] = optimizer_cfg

    logging_cfg = data.get("logging", {})
    logging_cfg.setdefault("train_metrics_every_iter", False)
    logging_cfg.setdefault("save_visualizations", False)
    logging_cfg.setdefault("log_interval", training.get("log_every", 20))
    training["log_every"] = int(logging_cfg.get("log_interval", training.get("log_every", 20)))
    data["logging"] = logging_cfg

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
    data["model"] = model
    _apply_explicit_model_module_flags(data)

    decoder = data.get("decoder", {})
    if "decoder_channels" in decoder:
        decoder["channels"] = int(decoder["decoder_channels"])
    else:
        channel_multiplier = float(efficiency.get("channels_multiplier", 1.0))
        if channel_multiplier != 1.0:
            decoder["channels"] = max(1, int(round(int(decoder.get("channels", 256)) * channel_multiplier)))
            difference = data.setdefault("difference", {})
            difference["out_channels"] = max(1, int(round(int(difference.get("out_channels", 256)) * channel_multiplier)))
    data["decoder"] = decoder

    # Preserve checkpoint path for eval/validate in one place.
    checkpoint = data.get("checkpoint", {})
    checkpoint.setdefault("path", None)
    if "selection_metric" in checkpoint and "monitor" not in checkpoint:
        checkpoint["monitor"] = checkpoint["selection_metric"]
    checkpoint.setdefault("monitor", "f1")
    checkpoint.setdefault("mode", "max")
    checkpoint.setdefault("save_best_only", True)
    checkpoint.setdefault("save_best", True)
    checkpoint.setdefault("save_last", True)
    checkpoint.setdefault("save_every", None)
    checkpoint.setdefault("save_latest", bool(checkpoint.get("save_last", True)))
    checkpoint.setdefault("latest_every", training.get("validate_every", 5000))
    data["checkpoint"] = checkpoint

    validation = data.get("validation", {})
    validation.setdefault("batch_size", training.get("batch_size", 8))
    validation.setdefault("save_samples", True)
    validation.setdefault("sample_count", 16)
    validation.setdefault("split", "val")
    data["validation"] = validation

    dataset_cfg = data.get("dataset", {})
    dataloader_cfg = data.get("dataloader", {})
    dataloader_cfg.setdefault("num_workers", dataset_cfg.get("num_workers", 8))
    dataloader_cfg.setdefault("pin_memory", dataset_cfg.get("pin_memory", True))
    dataloader_cfg.setdefault("persistent_workers", dataset_cfg.get("persistent_workers", True))
    dataloader_cfg.setdefault("prefetch_factor", dataset_cfg.get("prefetch_factor", 4))
    dataloader_cfg.setdefault("drop_last", True)
    dataset_cfg["num_workers"] = int(dataloader_cfg["num_workers"])
    dataset_cfg["pin_memory"] = bool(dataloader_cfg["pin_memory"])
    dataset_cfg["persistent_workers"] = bool(dataloader_cfg["persistent_workers"])
    dataset_cfg["prefetch_factor"] = int(dataloader_cfg["prefetch_factor"])
    dataset_cfg.setdefault("task_type", "binary_change")
    dataset_cfg.setdefault("cache_images_in_ram", False)
    dataset_cfg.setdefault("cache_masks_in_ram", False)
    data["dataset"] = dataset_cfg
    data["dataloader"] = dataloader_cfg

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
    evaluation.setdefault("threshold_select_metric", "F1_1")
    evaluation.setdefault("inference_mode", "patch")
    evaluation.setdefault("crop_size", dataset_cfg.get("image_size", 256))
    evaluation.setdefault("overlap", 0.25)
    evaluation.setdefault("save_predictions", False)
    evaluation.setdefault("save_visualizations", False)
    evaluation.setdefault("memory_efficient", True)
    evaluation.setdefault("use_cached_predictions", False)
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
        # family, so clear inherited focal weight unless explicitly set there.
        loss_cfg["focal_weight"] = float(final_cfg.get("focal_weight", 0.0))
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
    data["loss"] = loss_cfg

    debug = data.get("debug", {})
    debug.setdefault("ablation_trace", False)
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

    profiling = data.get("profiling", {})
    profiling.setdefault("enabled", False)
    profiling.setdefault("warmup_iters", 20)
    profiling.setdefault("profile_iters", 100)
    profiling.setdefault("log_interval", 10)
    data["profiling"] = profiling

    benchmark = data.get("benchmark", {})
    benchmark.setdefault("model_name", data["experiment"].get("name", "model"))
    benchmark.setdefault("output_dir", "outputs/benchmark_runs/summary")
    benchmark.setdefault("datasets", ["DSIFN-CD", "WHU-CD"])
    benchmark.setdefault("eval_split", "test")
    benchmark.setdefault("main_dataset", data["dataset"].get("name", "DSIFN-CD"))
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
