"""GPU memory profiler for MERCon change-detection pipeline.

No CLI arguments.
Config-driven via configs/global_config.yaml.
"""
from __future__ import annotations

import copy
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_builder import build_dataset
from models.cd_model import build_model
from models.decoders import DECODER_REGISTRY
from training.ema import EMA
from training.evaluator import Evaluator
from training.losses import build_loss
from training.model_outputs import normalize_model_output
from training.tta import apply_tta, build_tta_augmentations
from utils.config import GLOBAL_CONFIG_PATH, load_config
from utils.memory_debug import (
    BugCheck,
    LayerMetadataRecorder,
    append_records,
    bool_status,
    cuda_snapshot,
    dataloader_info,
    ensure_dir,
    find_biggest_jump,
    format_markdown_table,
    has_preexisting_hooks,
    memory_summary_text,
    model_output_to_logits,
    peak_from_rows,
    pick_label_tensor,
    reset_cuda_peak,
    summarize_ema_devices,
    top_layer_rows,
    write_csv,
    write_json,
)

def _build_loader(cfg: dict, split: str) -> DataLoader:
    dc = cfg.get("dataset", {})
    dataset_cfg = {
        "name": dc.get("name", "LEVIR-CD"),
        "root": dc["root"],
        "image_size": int(dc.get("image_size", 256)),
        "val_ratio": float(dc.get("val_ratio", 0.2)),
        "num_workers": int(dc.get("num_workers", 2)),
        "image_a_dir_candidates": dc.get("image_a_dir_candidates"),
        "image_b_dir_candidates": dc.get("image_b_dir_candidates"),
        "label_dir_candidates": dc.get("label_dir_candidates"),
    }
    dataset_cfg = {k: v for k, v in dataset_cfg.items() if v is not None}

    seed = int(cfg.get("debug", {}).get("seed", cfg.get("experiment", {}).get("seed", 42)))
    ds = build_dataset(dataset_cfg, split=split, augment=(split == "train"), seed=seed)

    hw = cfg.get("hardware", {})
    pin = str(hw.get("device", "cuda")).startswith("cuda")
    loader = DataLoader(
        ds,
        batch_size=int(dc.get("batch_size", cfg.get("debug", {}).get("batch_size", 8))),
        shuffle=(split == "train"),
        num_workers=int(dc.get("num_workers", 2)),
        pin_memory=pin,
        drop_last=True,
        persistent_workers=int(dc.get("num_workers", 2)) > 0,
    )
    return loader


def _to_device_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
    return out


def _autocast_enabled(cfg: dict, device: torch.device) -> bool:
    dbg = cfg.get("debug", {})
    hw = cfg.get("hardware", {})
    use_amp = bool(dbg.get("use_amp", hw.get("mixed_precision", True)))
    return use_amp and device.type == "cuda"


def _build_model_loss_opt(cfg: dict, device: torch.device, decoder: str | None = None):
    cfg_local = copy.deepcopy(cfg)
    if decoder is not None:
        cfg_local.setdefault("model", {})["decoder"] = decoder
    model = build_model(cfg_local).to(device)

    if bool(cfg_local.get("model", {}).get("freeze_backbone", False)):
        for p in model.encoder.parameters():
            p.requires_grad_(False)

    loss_fn = build_loss(cfg_local).to(device)

    lr = float(cfg_local.get("training", {}).get("lr", 1e-4))
    wd = float(cfg_local.get("training", {}).get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd,
    )
    return cfg_local, model, loss_fn, optimizer


def _loss_from_outputs(
    cfg: dict,
    loss_fn: torch.nn.Module,
    outputs: Any,
    batch_gpu: Dict[str, torch.Tensor],
) -> torch.Tensor:
    normalized = normalize_model_output(outputs)
    if str(cfg.get("loss", {}).get("type", "")).lower().replace("-", "_") == "second_semantic_cd":
        semantic_batch = {
            "change_mask": batch_gpu["change_mask"],
            "label_a": batch_gpu["label_a"],
            "label_b": batch_gpu["label_b"],
            "valid_mask": batch_gpu.get("valid_mask"),
        }
        return loss_fn(normalized, semantic_batch)
    logits = normalized["change_logits"]
    total, _, _ = loss_fn(logits, pick_label_tensor(batch_gpu))
    return total


def _single_train_step_snapshots(
    cfg: dict,
    batch_gpu: Dict[str, torch.Tensor],
    test_name: str,
    decoder: str | None = None,
    return_features_flag: bool | None = None,
    ema_enabled: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device(cfg.get("hardware", {}).get("device", "cuda"))
    amp = _autocast_enabled(cfg, device)
    cfg_local, model, loss_fn, optimizer = _build_model_loss_opt(cfg, device, decoder=decoder)
    model.train()

    ia = batch_gpu["image_a"]
    ib = batch_gpu["image_b"]

    snaps: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    ema = EMA(model, decay=float(cfg.get("training", {}).get("ema_decay", 0.999))) if ema_enabled else None

    reset_cuda_peak()
    snaps.append(cuda_snapshot("start", notes="fresh model + batch ready"))

    with torch.amp.autocast("cuda", enabled=amp):
        if return_features_flag is None:
            output = model(ia, ib)
        else:
            output = model(ia, ib, return_features=return_features_flag)
        logits, out_type, has_features = model_output_to_logits(output)
    meta["output_type"] = out_type
    meta["output_has_features"] = bool(has_features)
    snaps.append(cuda_snapshot("after_forward_grad", notes=f"output_type={out_type}"))

    with torch.amp.autocast("cuda", enabled=amp):
        total = _loss_from_outputs(cfg_local, loss_fn, output, batch_gpu)
    snaps.append(cuda_snapshot("after_loss", notes=f"loss={float(total.detach().item()):.5f}"))

    total.backward()
    snaps.append(cuda_snapshot("after_backward"))

    optimizer.step()
    snaps.append(cuda_snapshot("after_optimizer_step"))

    if ema is not None:
        ema.update(model)
        snaps.append(cuda_snapshot("after_ema_update", notes="ema_enabled=True"))

    optimizer.zero_grad(set_to_none=True)
    snaps.append(cuda_snapshot("after_zero_grad", notes="set_to_none=True"))

    del total, logits, output
    torch.cuda.empty_cache()
    snaps.append(cuda_snapshot("after_cleanup"))

    return snaps, meta


def _forward_only_no_grad_snapshots(cfg: dict, batch_gpu: Dict[str, torch.Tensor], test_name: str) -> List[Dict[str, Any]]:
    device = torch.device(cfg.get("hardware", {}).get("device", "cuda"))
    amp = _autocast_enabled(cfg, device)
    _, model, _, _ = _build_model_loss_opt(cfg, device)
    model.eval()
    ia = batch_gpu["image_a"]
    ib = batch_gpu["image_b"]

    snaps: List[Dict[str, Any]] = []
    reset_cuda_peak()
    snaps.append(cuda_snapshot("start"))
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=amp):
            output = model(ia, ib)
            logits, _, _ = model_output_to_logits(output)
    snaps.append(cuda_snapshot("after_forward_no_grad", notes=f"shape={list(logits.shape)}"))
    del output, logits
    torch.cuda.empty_cache()
    snaps.append(cuda_snapshot("after_cleanup"))
    return snaps


def _eval_tta_snapshots(cfg: dict, batch_gpu: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    device = torch.device(cfg.get("hardware", {}).get("device", "cuda"))
    amp = _autocast_enabled(cfg, device)
    _, model, _, _ = _build_model_loss_opt(cfg, device)
    model.eval()
    ia = batch_gpu["image_a"]
    ib = batch_gpu["image_b"]

    snaps: List[Dict[str, Any]] = []
    reset_cuda_peak()
    snaps.append(cuda_snapshot("start"))
    with torch.no_grad():
        outputs = normalize_model_output(apply_tta(
            model,
            ia,
            ib,
            amp=amp,
            augmentations=build_tta_augmentations(cfg),
        ))
        logits = outputs["change_logits"]
    snaps.append(cuda_snapshot("after_tta_no_grad", notes=f"shape={list(logits.shape)}"))
    del logits
    torch.cuda.empty_cache()
    snaps.append(cuda_snapshot("after_cleanup"))
    return snaps


def _ema_snapshots(cfg: dict, batch_gpu: Dict[str, torch.Tensor]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device(cfg.get("hardware", {}).get("device", "cuda"))
    _, model, _, _ = _build_model_loss_opt(cfg, device)

    snaps: List[Dict[str, Any]] = []
    reset_cuda_peak()
    snaps.append(cuda_snapshot("start"))

    ema = EMA(model, decay=float(cfg.get("training", {}).get("ema_decay", 0.999)))
    snaps.append(cuda_snapshot("after_ema_init"))
    ema_info = summarize_ema_devices(ema._shadow)

    store_on_cpu = bool(cfg.get("ema", {}).get("store_on_cpu", True))
    if store_on_cpu:
        for k, v in list(ema._shadow.items()):
            ema._shadow[k] = v.detach().cpu()
        snaps.append(cuda_snapshot("after_ema_moved_to_cpu", notes="debug-only copy"))
        ema_info["after_cpu_move"] = summarize_ema_devices(ema._shadow)

    return snaps, ema_info


def _profiler_one_step(cfg: dict, batch_gpu: Dict[str, torch.Tensor], out_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"enabled": False}
    if not bool(cfg.get("debug", {}).get("profile_torch_ops", False)):
        return result

    device = torch.device(cfg.get("hardware", {}).get("device", "cuda"))
    amp = _autocast_enabled(cfg, device)
    _, model, loss_fn, optimizer = _build_model_loss_opt(cfg, device)
    model.train()

    ia = batch_gpu["image_a"]
    ib = batch_gpu["image_b"]

    trace_dir = ensure_dir(out_dir / "profiler_trace")
    table_path = trace_dir / "profiler_table.txt"
    chrome_path = trace_dir / "trace.json"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.amp.autocast("cuda", enabled=amp):
            output = model(ia, ib)
            logits, _, _ = model_output_to_logits(output)
            total = _loss_from_outputs(cfg, loss_fn, output, batch_gpu)
        total.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=120)
    table_path.write_text(table, encoding="utf-8")
    try:
        prof.export_chrome_trace(str(chrome_path))
    except Exception as exc:
        (trace_dir / "trace_export_error.txt").write_text(str(exc), encoding="utf-8")

    result.update({
        "enabled": True,
        "table_path": str(table_path),
        "trace_path": str(chrome_path),
    })
    return result


def _bug_checks(
    cfg: dict,
    model_output_meta: Dict[str, Any],
    ema_info: Dict[str, Any],
    loader: DataLoader,
    preexisting_hook_names: List[str],
) -> List[BugCheck]:
    checks: List[BugCheck] = []

    return_features = bool(cfg.get("model", {}).get("return_features", False))
    checks.append(BugCheck(
        "return_features enabled during training",
        bool_status(not return_features, warn=return_features),
        f"model.return_features={return_features}",
    ))

    has_features = bool(model_output_meta.get("output_has_features", False))
    out_type = model_output_meta.get("output_type", "unknown")
    checks.append(BugCheck(
        "model output includes feature payload",
        bool_status(not has_features, warn=has_features),
        f"output_type={out_type}, has_features={has_features}",
    ))

    tta_enabled = bool(cfg.get("evaluation", {}).get("use_tta", False))
    checks.append(BugCheck(
        "TTA enabled in config",
        bool_status(not tta_enabled, warn=tta_enabled),
        f"evaluation.use_tta={tta_enabled} (should stay false during train)",
    ))

    src = inspect.getsource(Evaluator.evaluate)
    has_no_grad = "@torch.no_grad" in src
    checks.append(BugCheck(
        "validation uses torch.no_grad",
        bool_status(has_no_grad),
        "Evaluator.evaluate has @torch.no_grad decorator" if has_no_grad else "No @torch.no_grad decorator detected",
    ))

    amp_enabled = bool(cfg.get("debug", {}).get("use_amp", cfg.get("hardware", {}).get("mixed_precision", True)))
    checks.append(BugCheck(
        "AMP autocast enabled",
        bool_status(amp_enabled, warn=not amp_enabled),
        f"use_amp={amp_enabled}",
    ))

    gpu_ema = int(ema_info.get("gpu_tensor_count", 0))
    checks.append(BugCheck(
        "EMA shadow tensors on GPU",
        bool_status(gpu_ema == 0, warn=gpu_ema > 0),
        f"gpu_tensor_count={gpu_ema}, cpu_tensor_count={ema_info.get('cpu_tensor_count', 0)}",
    ))

    checks.append(BugCheck(
        "pre-existing hooks may store tensors",
        bool_status(len(preexisting_hook_names) == 0, warn=len(preexisting_hook_names) > 0),
        "none" if not preexisting_hook_names else f"existing hooks on modules: {preexisting_hook_names[:10]}",
    ))

    trainer_text = (ROOT / "src/training/trainer.py").read_text(encoding="utf-8")
    has_append_logits = ("append(logits" in trainer_text) or ("append(features" in trainer_text)
    checks.append(BugCheck(
        "training loop appends logits/features lists",
        bool_status(not has_append_logits, warn=has_append_logits),
        "No append(logits/features) found in trainer.py" if not has_append_logits else "Potential list retention found in trainer.py",
    ))

    has_zero_none = "zero_grad(set_to_none=True)" in trainer_text
    checks.append(BugCheck(
        "optimizer.zero_grad uses set_to_none=True",
        bool_status(has_zero_none),
        "set_to_none=True found" if has_zero_none else "set_to_none=True not found",
    ))

    li = dataloader_info(loader)
    loader_warn = (li["num_workers"] > 8) or (li["prefetch_factor"] not in (None, 2))
    checks.append(BugCheck(
        "DataLoader worker/pin/prefetch settings",
        bool_status(not loader_warn, warn=loader_warn),
        json.dumps(li),
    ))

    return checks


def _detect_suspected_cause(records: List[Dict[str, Any]], checks: List[BugCheck]) -> str:
    by_test: Dict[str, float] = {}
    for r in records:
        name = str(r["test_name"])
        by_test[name] = max(by_test.get(name, 0.0), float(r["max_allocated_gb"]))

    if not by_test:
        return "No records generated."

    top_test = max(by_test.items(), key=lambda x: x[1])
    warnings = [c.details for c in checks if c.status == "WARN"]

    if "ema" in top_test[0].lower():
        return f"Highest peak is EMA-related ({top_test[0]}={top_test[1]:.3f} GiB). EMA shadow placement is likely contributing."
    if "tta" in top_test[0].lower():
        return f"Highest peak is TTA-related ({top_test[0]}={top_test[1]:.3f} GiB). TTA multiplies inference passes and memory."
    if "backward" in top_test[0].lower() or "train" in top_test[0].lower():
        return f"Peak occurs in training backward path ({top_test[0]}={top_test[1]:.3f} GiB), indicating activation + optimizer state overhead."
    if warnings:
        return f"Main suspicion from checks: {warnings[0]}"
    return f"Peak memory is observed in '{top_test[0]}' at {top_test[1]:.3f} GiB."


def _write_markdown_report(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
    layer_rows: List[Dict[str, Any]],
    bug_checks: List[BugCheck],
    return_features_cmp: Dict[str, Any],
    ema_info: Dict[str, Any],
    tta_info: Dict[str, Any],
    amp_enabled: bool,
    biggest_jump_text: str,
    suspected_cause: str,
) -> None:
    peak_by_test: Dict[str, float] = {}
    for r in summary_rows:
        name = str(r["test_name"])
        peak_by_test[name] = max(peak_by_test.get(name, 0.0), float(r["max_allocated_gb"]))

    peak_rows = [{"test_name": k, "peak_max_allocated_gb": round(v, 4)} for k, v in sorted(peak_by_test.items(), key=lambda x: x[1], reverse=True)]

    check_rows = [
        {"check_name": c.check_name, "status": c.status, "details": c.details}
        for c in bug_checks
    ]

    md = []
    md.append("# GPU Memory Debug Report")
    md.append("")
    md.append("## 1. Summary")
    md.append(f"- Total memory snapshots: {len(summary_rows)}")
    md.append(f"- AMP enabled: {amp_enabled}")
    md.append(f"- Biggest memory jump: {biggest_jump_text}")
    md.append("")
    md.append("## 2. Peak memory by test mode")
    md.append(format_markdown_table(peak_rows, ["test_name", "peak_max_allocated_gb"]))
    md.append("")
    md.append("## 3. Biggest memory jump")
    md.append(f"- {biggest_jump_text}")
    md.append("")
    md.append("## 4. Suspected cause")
    md.append(f"- {suspected_cause}")
    md.append("")
    md.append("## 5. EMA device check")
    md.append(f"- GPU EMA tensors: {ema_info.get('gpu_tensor_count', 0)}")
    md.append(f"- CPU EMA tensors: {ema_info.get('cpu_tensor_count', 0)}")
    md.append(f"- Total EMA params size: {round(float(ema_info.get('total_param_gb', 0.0)), 4)} GiB")
    if "after_cpu_move" in ema_info:
        ac = ema_info["after_cpu_move"]
        md.append(f"- After debug CPU move: gpu={ac.get('gpu_tensor_count', 0)}, cpu={ac.get('cpu_tensor_count', 0)}")
    md.append("")
    md.append("## 6. return_features comparison")
    md.append(format_markdown_table([
        {"mode": "return_features=False", "peak_gb": return_features_cmp.get("false_peak_gb", "n/a")},
        {"mode": "return_features=True", "peak_gb": return_features_cmp.get("true_peak_gb", "n/a")},
        {"mode": "delta", "peak_gb": return_features_cmp.get("delta_gb", "n/a")},
        {"mode": "notes", "peak_gb": return_features_cmp.get("notes", "")},
    ], ["mode", "peak_gb"]))
    md.append("")
    md.append("## 7. TTA check")
    md.append(format_markdown_table([
        {"item": "evaluation.use_tta", "value": tta_info.get("config_use_tta", False)},
        {"item": "tta_peak_gb", "value": tta_info.get("tta_peak_gb", "n/a")},
        {"item": "notes", "value": tta_info.get("notes", "")},
    ], ["item", "value"]))
    md.append("")
    md.append("## 8. AMP check")
    md.append(f"- AMP/autocast active: {amp_enabled}")
    md.append("")
    md.append("## 9. Layer output metadata table")
    md.append(format_markdown_table(top_layer_rows(layer_rows, top_k=30), [
        "module_name", "output_shape", "dtype", "device", "size_mb", "requires_grad", "output_type"
    ]))
    md.append("")
    md.append("## 10. Recommendations")
    md.append("- Freeze backbone for warmup (`model.freeze_backbone: true`) to reduce activation+optimizer footprint.")
    md.append("- Reduce `dataset.batch_size`/`training.batch_size` if peak memory exceeds GPU budget.")
    md.append("- Keep TTA disabled during training; use only in eval.")
    md.append("- Keep `optimizer.zero_grad(set_to_none=True)` and AMP enabled.")
    md.append("- Move EMA shadow tensors to CPU if memory is constrained.")
    md.append("")
    md.append("## Bug Checks")
    md.append(format_markdown_table(check_rows, ["check_name", "status", "details"]))

    (out_dir / "memory_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    cfg = load_config()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU memory debug.")

    dbg = cfg.get("debug", {})
    debug_name = str(dbg.get("name", "tiny_memory_debug"))
    output_root = ROOT / str(dbg.get("output_root", "outputs/memory_debug"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(output_root / f"run_{ts}_{debug_name}")

    # Save resolved config for reproducibility.
    (out_dir / "config.yaml").write_text(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    device = torch.device(cfg.get("hardware", {}).get("device", "cuda"))
    if device.type != "cuda":
        raise RuntimeError(f"This script expects CUDA device, got {device}")

    torch.cuda.set_device(int(cfg.get("hardware", {}).get("gpu_ids", [0])[0]))
    torch.cuda.empty_cache()
    reset_cuda_peak()

    # Memory summary before any heavy allocations.
    (out_dir / "memory_summary_before.txt").write_text(memory_summary_text(), encoding="utf-8")

    split = str(cfg.get("dataset", {}).get("split", "train"))
    loader = _build_loader(cfg, split=split)
    batch_cpu = next(iter(loader))
    batch_gpu = _to_device_batch(batch_cpu, device)

    # Hook metadata capture from one forward.
    cfg_tmp, model_tmp, _, _ = _build_model_loss_opt(cfg, device)
    preexisting = has_preexisting_hooks(model_tmp)
    recorder = LayerMetadataRecorder()
    recorder.register(model_tmp)
    with torch.no_grad():
        _ = model_tmp(batch_gpu["image_a"], batch_gpu["image_b"])
    recorder.clear()
    layer_rows = recorder.rows

    records: List[Dict[str, Any]] = []
    mode_list = [str(x) for x in dbg.get("compare_modes", [])]
    mode_set = set(mode_list)

    def mode_on(name: str) -> bool:
        return (not mode_set) or (name in mode_set)

    # A. Model-only memory
    reset_cuda_peak()
    snaps_a = [cuda_snapshot("start")]
    _, model_a, _, _ = _build_model_loss_opt(cfg, device)
    snaps_a.append(cuda_snapshot("after_model_to_gpu"))
    append_records(records, "model_only_memory", snaps_a)

    # B. Batch-only memory
    reset_cuda_peak()
    snaps_b = [cuda_snapshot("start")]
    _batch_only = _to_device_batch(batch_cpu, device)
    snaps_b.append(cuda_snapshot("after_batch_to_gpu"))
    append_records(records, "batch_only_memory", snaps_b)

    # C. Forward-only no-grad
    if mode_on("baseline_forward_only"):
        append_records(records, "baseline_forward_only", _forward_only_no_grad_snapshots(cfg, batch_gpu, "baseline_forward_only"))

    # D/E/F/G/H. Grad path with loss/backward/step/zero_grad
    out_meta: Dict[str, Any] = {}
    if mode_on("train_forward_loss_backward"):
        snaps_train, out_meta = _single_train_step_snapshots(cfg, batch_gpu, "train_forward_loss_backward")
        append_records(records, "train_forward_loss_backward", snaps_train)
    else:
        # Keep metadata checks available even when this mode is disabled.
        _, out_meta = _single_train_step_snapshots(cfg, batch_gpu, "metadata_probe")

    # I. Eval validation-style no-grad
    if mode_on("eval_no_grad"):
        append_records(records, "eval_no_grad", _forward_only_no_grad_snapshots(cfg, batch_gpu, "eval_no_grad"))

    # J. TTA memory
    if mode_on("eval_with_tta"):
        try:
            tta_snaps = _eval_tta_snapshots(cfg, batch_gpu)
            append_records(records, "eval_with_tta", tta_snaps)
            tta_peak = peak_from_rows([
                {"max_allocated_gb": r["max_allocated_gb"]}
                for r in records if r["test_name"] == "eval_with_tta"
            ])
            tta_info = {
                "config_use_tta": bool(cfg.get("evaluation", {}).get("use_tta", False)),
                "tta_peak_gb": round(tta_peak, 4),
                "notes": "TTA measured in eval/no_grad mode",
            }
        except Exception as exc:
            tta_info = {
                "config_use_tta": bool(cfg.get("evaluation", {}).get("use_tta", False)),
                "tta_peak_gb": "n/a",
                "notes": f"TTA measurement skipped: {exc}",
            }
    else:
        tta_info = {
            "config_use_tta": bool(cfg.get("evaluation", {}).get("use_tta", False)),
            "tta_peak_gb": "disabled",
            "notes": "eval_with_tta mode disabled in debug.compare_modes",
        }

    # K. EMA memory
    ema_snaps, ema_info = _ema_snapshots(cfg, batch_gpu)
    append_records(records, "ema_memory", ema_snaps)

    if mode_on("train_with_ema_disabled"):
        snaps_ema_off, _ = _single_train_step_snapshots(
            cfg,
            batch_gpu,
            "train_with_ema_disabled",
            ema_enabled=False,
        )
        append_records(records, "train_with_ema_disabled", snaps_ema_off)

    if mode_on("train_with_ema_enabled"):
        snaps_ema_on, _ = _single_train_step_snapshots(
            cfg,
            batch_gpu,
            "train_with_ema_enabled",
            ema_enabled=True,
        )
        append_records(records, "train_with_ema_enabled", snaps_ema_on)

    if mode_on("train_with_tta_disabled"):
        snaps_tta_off, _ = _single_train_step_snapshots(
            cfg,
            batch_gpu,
            "train_with_tta_disabled",
            ema_enabled=False,
        )
        append_records(records, "train_with_tta_disabled", snaps_tta_off)

    # L. return_features comparison
    rf_false_peak = 0.0
    rf_true_peak = 0.0
    if mode_on("train_with_return_features_false"):
        rf_false_snaps, _rf_false_meta = _single_train_step_snapshots(
            cfg,
            batch_gpu,
            "train_with_return_features_false",
            return_features_flag=False,
        )
        append_records(records, "train_with_return_features_false", rf_false_snaps)
        rf_false_peak = peak_from_rows([
            {"max_allocated_gb": r["max_allocated_gb"]}
            for r in records if r["test_name"] == "train_with_return_features_false"
        ])

    if mode_on("train_with_return_features_true"):
        rf_true_snaps, _rf_true_meta = _single_train_step_snapshots(
            cfg,
            batch_gpu,
            "train_with_return_features_true",
            return_features_flag=True,
        )
        append_records(records, "train_with_return_features_true", rf_true_snaps)
        rf_true_peak = peak_from_rows([
            {"max_allocated_gb": r["max_allocated_gb"]}
            for r in records if r["test_name"] == "train_with_return_features_true"
        ])

    signature = inspect.signature(type(model_tmp).forward)
    supports_flag = ("return_features" in signature.parameters)
    return_features_cmp = {
        "false_peak_gb": round(rf_false_peak, 4),
        "true_peak_gb": round(rf_true_peak, 4),
        "delta_gb": round(rf_true_peak - rf_false_peak, 4),
        "notes": "forward supports return_features" if supports_flag else "forward does not expose return_features; flag likely ignored",
    }

    # M. Decoder comparison
    for dec in ["baseline", "refinement", "adaptive_rf"]:
        if dec not in DECODER_REGISTRY:
            continue
        mode_name = f"decoder_{dec}"
        if mode_on(mode_name):
            dec_snaps, _ = _single_train_step_snapshots(cfg, batch_gpu, mode_name, decoder=dec)
            append_records(records, mode_name, dec_snaps)

    # Profile one step if requested.
    profiler_info = _profiler_one_step(cfg, batch_gpu, out_dir)

    # Bug checks.
    bug_checks = _bug_checks(cfg, out_meta, ema_info, loader, preexisting)

    # Biggest jump.
    jump_delta, jump_pair = find_biggest_jump(records, key="allocated_gb")
    biggest_jump_text = "n/a"
    if jump_pair is not None:
        biggest_jump_text = f"{jump_pair[0]} -> {jump_pair[1]} : +{jump_delta:.4f} GiB"

    suspected = _detect_suspected_cause(records, bug_checks)

    # Save structured outputs.
    write_csv(
        out_dir / "memory_report.csv",
        records,
        [
            "test_name",
            "stage",
            "allocated_gb",
            "reserved_gb",
            "max_allocated_gb",
            "max_reserved_gb",
            "notes",
        ],
    )

    write_csv(
        out_dir / "layer_memory.csv",
        layer_rows,
        [
            "module_name",
            "output_shape",
            "dtype",
            "device",
            "size_mb",
            "requires_grad",
            "output_type",
        ],
    )

    bug_rows = [{"check_name": c.check_name, "status": c.status, "details": c.details} for c in bug_checks]
    write_csv(out_dir / "bug_checks.csv", bug_rows, ["check_name", "status", "details"])

    payload = {
        "output_dir": str(out_dir),
        "config_path": str(GLOBAL_CONFIG_PATH.relative_to(ROOT)),
        "records": records,
        "ema_info": ema_info,
        "return_features_comparison": return_features_cmp,
        "tta_info": tta_info,
        "bug_checks": bug_rows,
        "profiler": profiler_info,
        "biggest_jump": biggest_jump_text,
        "suspected_cause": suspected,
        "dataloader": dataloader_info(loader),
    }
    write_json(out_dir / "memory_report.json", payload)

    _write_markdown_report(
        out_dir=out_dir,
        summary_rows=records,
        layer_rows=layer_rows,
        bug_checks=bug_checks,
        return_features_cmp=return_features_cmp,
        ema_info=ema_info,
        tta_info=tta_info,
        amp_enabled=_autocast_enabled(cfg, device),
        biggest_jump_text=biggest_jump_text,
        suspected_cause=suspected,
    )

    (out_dir / "memory_summary_after.txt").write_text(memory_summary_text(), encoding="utf-8")

    print(f"Memory debug complete. Reports saved to: {out_dir}")


if __name__ == "__main__":
    main()
