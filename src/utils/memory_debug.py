"""GPU memory debugging helpers.

This module intentionally stores only lightweight metadata (never full tensors)
so debugging itself does not inflate GPU memory usage.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

GiB = 1024 ** 3


@dataclass
class BugCheck:
    check_name: str
    status: str
    details: str


class LayerMetadataRecorder:
    """Forward-hook recorder that keeps only output metadata."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self._handles: List[Any] = []

    @staticmethod
    def _tensor_meta(module_name: str, tensor: torch.Tensor, output_type: str) -> Dict[str, Any]:
        size_mb = tensor.numel() * tensor.element_size() / (1024 ** 2)
        return {
            "module_name": module_name,
            "output_shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "size_mb": float(size_mb),
            "requires_grad": bool(tensor.requires_grad),
            "output_type": output_type,
        }

    def _record_any(self, module_name: str, out: Any, output_type: str) -> None:
        if torch.is_tensor(out):
            self.rows.append(self._tensor_meta(module_name, out, output_type))
            return
        if isinstance(out, (list, tuple)):
            for item in out:
                self._record_any(module_name, item, type(out).__name__)
            return
        if isinstance(out, dict):
            for item in out.values():
                self._record_any(module_name, item, "dict")

    def _hook(self, module_name: str):
        def fn(_: nn.Module, __: Tuple[Any, ...], output: Any) -> None:
            self._record_any(module_name, output, type(output).__name__)
        return fn

    def register(self, model: nn.Module, patterns: Optional[List[str]] = None) -> int:
        pats = patterns or [
            "encoder",
            "patch_embed",
            "levels",
            "decoder",
            "refinement",
            "loss",
        ]
        count = 0
        for name, module in model.named_modules():
            if not name:
                continue
            lname = name.lower()
            if any(p in lname for p in pats):
                self._handles.append(module.register_forward_hook(self._hook(name)))
                count += 1
        return count

    def clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cuda_snapshot(tag: str, notes: str = "") -> Dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "tag": tag,
        "allocated_gb": torch.cuda.memory_allocated() / GiB,
        "reserved_gb": torch.cuda.memory_reserved() / GiB,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / GiB,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / GiB,
        "notes": notes,
    }


def reset_cuda_peak() -> None:
    torch.cuda.reset_peak_memory_stats()


def memory_summary_text() -> str:
    return torch.cuda.memory_summary()


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def peak_from_rows(rows: List[Dict[str, Any]], key: str = "max_allocated_gb") -> float:
    if not rows:
        return 0.0
    return float(max(r.get(key, 0.0) for r in rows))


def find_biggest_jump(rows: List[Dict[str, Any]], key: str = "allocated_gb") -> Tuple[float, Optional[Tuple[str, str]]]:
    if len(rows) < 2:
        return 0.0, None
    best_delta = -1e9
    best_pair: Optional[Tuple[str, str]] = None
    def _label(row: Dict[str, Any]) -> str:
        if "tag" in row and row["tag"]:
            return str(row["tag"])
        if "stage" in row and row["stage"]:
            return str(row["stage"])
        return "?"

    prev = rows[0]
    for cur in rows[1:]:
        delta = float(cur.get(key, 0.0)) - float(prev.get(key, 0.0))
        if delta > best_delta:
            best_delta = delta
            best_pair = (_label(prev), _label(cur))
        prev = cur
    return best_delta, best_pair


def summarize_ema_devices(shadow: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    gpu_count = 0
    cpu_count = 0
    gpu_names: List[str] = []
    total_bytes = 0
    for name, tensor in shadow.items():
        total_bytes += tensor.numel() * tensor.element_size()
        if tensor.is_cuda:
            gpu_count += 1
            gpu_names.append(name)
        else:
            cpu_count += 1
    return {
        "gpu_tensor_count": gpu_count,
        "cpu_tensor_count": cpu_count,
        "total_param_gb": total_bytes / GiB,
        "gpu_tensor_examples": gpu_names[:10],
    }


def format_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "(no rows)"
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---" for _ in columns]) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def has_preexisting_hooks(model: nn.Module) -> List[str]:
    names: List[str] = []
    for name, module in model.named_modules():
        if getattr(module, "_forward_hooks", None):
            if len(module._forward_hooks) > 0:
                names.append(name or "<root>")
    return names


def pick_label_tensor(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "label" in batch:
        return batch["label"]
    if "mask" in batch:
        return batch["mask"]
    raise KeyError("Batch has neither 'label' nor 'mask'.")


def model_output_to_logits(output: Any) -> Tuple[torch.Tensor, str, bool]:
    """Return logits tensor + output type + whether features are present."""
    contains_features = False
    if torch.is_tensor(output):
        return output, "tensor", False
    if isinstance(output, tuple):
        logits = output[0]
        # In this repo, tuple output is usually (logits, aux_logits).
        # Treat aux logits as expected output, not feature payload retention.
        contains_features = False
        return logits, "tuple", contains_features
    if isinstance(output, list):
        logits = output[0]
        contains_features = False
        return logits, "list", contains_features
    if isinstance(output, dict):
        if "logits" in output:
            contains_features = any(k for k in output.keys() if k != "logits")
            return output["logits"], "dict", contains_features
        first = next(iter(output.values()))
        return first, "dict", len(output) > 1
    raise TypeError(f"Unsupported model output type: {type(output)}")


def bool_status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "PASS"
    if warn:
        return "WARN"
    return "FAIL"


def top_layer_rows(rows: List[Dict[str, Any]], top_k: int = 25) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: float(r.get("size_mb", 0.0)), reverse=True)
    return ordered[:top_k]


def to_float4(x: Any) -> float:
    return round(float(x), 4)


def dataloader_info(loader: Any) -> Dict[str, Any]:
    return {
        "num_workers": int(getattr(loader, "num_workers", 0)),
        "pin_memory": bool(getattr(loader, "pin_memory", False)),
        "prefetch_factor": getattr(loader, "prefetch_factor", None),
        "persistent_workers": bool(getattr(loader, "persistent_workers", False)),
        "batch_size": int(getattr(loader, "batch_size", 0) or 0),
    }


def append_records(records: List[Dict[str, Any]], test_name: str, snaps: Iterable[Dict[str, Any]]) -> None:
    for snap in snaps:
        records.append({
            "test_name": test_name,
            "stage": snap["tag"],
            "allocated_gb": to_float4(snap["allocated_gb"]),
            "reserved_gb": to_float4(snap["reserved_gb"]),
            "max_allocated_gb": to_float4(snap["max_allocated_gb"]),
            "max_reserved_gb": to_float4(snap["max_reserved_gb"]),
            "notes": snap.get("notes", ""),
        })
