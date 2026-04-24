"""Exponential Moving Average (EMA) of model parameters.

Usage in training loop
----------------------
    from training.ema import EMA

    ema = EMA(model, decay=0.999)

    # After each optimizer step:
    ema.update(model)

    # During validation:
    ema.apply_shadow(model)
    run_validation(model)
    ema.restore(model)

Config keys
-----------
    training:
      use_ema: true
      ema_decay: 0.999
"""
from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """Maintains an EMA copy of model parameters.

    The shadow parameters are updated with:
        shadow = decay * shadow + (1 - decay) * param

    During validation ``apply_shadow`` swaps live parameters with EMA
    parameters.  ``restore`` brings back the originals.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay   = decay
        self._shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}

        # Initialise shadow from current model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone()

    # ------------------------------------------------------------------
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters after an optimiser step."""
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._shadow:
                # New parameter added after construction (rare)
                self._shadow[name] = param.data.clone()
                continue
            self._shadow[name] = d * self._shadow[name] + (1.0 - d) * param.data

    # ------------------------------------------------------------------
    def apply_shadow(self, model: nn.Module) -> None:
        """Swap live params with EMA shadow.  Call ``restore`` afterward."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._shadow:
                continue
            self._backup[name] = param.data.clone()
            param.data.copy_(self._shadow[name])

    # ------------------------------------------------------------------
    def restore(self, model: nn.Module) -> None:
        """Restore the live parameters saved by ``apply_shadow``."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self._shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay   = state["decay"]
        self._shadow = state["shadow"]
