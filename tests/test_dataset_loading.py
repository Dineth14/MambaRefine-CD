"""Dataset loading tests for active DSIFN-CD and WHU-CD datasets."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))


def _root(env_var: str) -> str | None:
    value = os.environ.get(env_var)
    return value if value and Path(value).exists() else None


def _check_binary_sample(sample: dict) -> None:
    for key in ("image_a", "image_b"):
        tensor = sample[key]
        assert isinstance(tensor, torch.Tensor), f"{key} not a tensor"
        assert tensor.ndim == 3, f"{key} should be [C,H,W], got {tensor.shape}"
        assert tensor.shape[0] == 3, f"{key} should have 3 channels"
        assert tensor.dtype == torch.float32
    mask = sample.get("mask", sample.get("label"))
    assert isinstance(mask, torch.Tensor), "mask not a tensor"
    assert mask.ndim in (2, 3), f"mask should be 2D or 3D, got {mask.shape}"


@pytest.mark.skipif(_root("WHU_ROOT") is None, reason="WHU_ROOT not set")
class TestWHUDataset:
    def _ds(self):
        from datasets.whu import build_whu_dataset

        cfg = {"dataset": {"name": "WHU-CD", "root": _root("WHU_ROOT"), "image_size": 256}}
        return build_whu_dataset(cfg, split="val")

    def test_loads(self) -> None:
        assert len(self._ds()) > 0

    def test_sample_keys_and_shapes(self) -> None:
        _check_binary_sample(self._ds()[0])


@pytest.mark.skipif(_root("DSIFN_ROOT") is None, reason="DSIFN_ROOT not set")
class TestDSIFNDataset:
    def _ds(self):
        from datasets.dsifn import build_dsifn_dataset

        cfg = {"dataset": {"name": "DSIFN-CD", "root": _root("DSIFN_ROOT"), "image_size": 256}}
        return build_dsifn_dataset(cfg, split="val")

    def test_loads(self) -> None:
        assert len(self._ds()) > 0

    def test_sample_keys_and_shapes(self) -> None:
        _check_binary_sample(self._ds()[0])
