"""Helpers for normalizing model outputs across binary and semantic modes."""
from __future__ import annotations

from typing import Any

import torch


def normalize_model_output(output: Any) -> dict[str, torch.Tensor | None]:
    """Normalize model outputs to a common dictionary shape.

    Supported inputs:
    * ``Tensor`` -> binary change logits only
    * ``(change_logits, aux_logits)`` -> legacy decoder path
    * ``dict`` containing ``change_logits`` and optional semantic logits
    """
    if isinstance(output, dict):
        change_logits = output.get("change_logits", output.get("binary_change_logits"))
        if change_logits is None:
            raise KeyError("Model output dict is missing 'change_logits' or 'binary_change_logits'.")
        return {
            "change_logits": change_logits,
            "aux_logits": output.get("aux_logits"),
            "sem_logits_t1": output.get("sem_logits_t1", output.get("semantic_t1_logits")),
            "sem_logits_t2": output.get("sem_logits_t2", output.get("semantic_t2_logits")),
        }

    if isinstance(output, tuple):
        if len(output) == 2:
            change_logits, aux_logits = output
        elif len(output) == 1:
            change_logits, aux_logits = output[0], None
        else:
            raise ValueError(f"Unsupported model output tuple length: {len(output)}")
        return {
            "change_logits": change_logits,
            "aux_logits": aux_logits,
            "sem_logits_t1": None,
            "sem_logits_t2": None,
        }

    if torch.is_tensor(output):
        return {
            "change_logits": output,
            "aux_logits": None,
            "sem_logits_t1": None,
            "sem_logits_t2": None,
        }

    raise TypeError(f"Unsupported model output type: {type(output)!r}")
