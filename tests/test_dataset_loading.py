"""Unit tests for dataset loading.

Tests that each dataset (LEVIR, WHU, DSIFN, SECOND) returns
correctly typed and shaped tensors.

These tests are SKIPPED automatically if the dataset root does not exist.
Set the environment variables LEVIR_ROOT, WHU_ROOT, DSIFN_ROOT, SECOND_ROOT
to the actual dataset paths to enable the tests.

Usage:
    python -m pytest tests/test_dataset_loading.py -v
    # or with explicit roots:
    LEVIR_ROOT=/path/to/LEVIR python -m pytest tests/test_dataset_loading.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import pytest
import torch


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _root(env_var: str) -> str | None:
    v = os.environ.get(env_var)
    if v and Path(v).exists():
        return v
    return None


def _check_binary_sample(sample: dict) -> None:
    """Validate a single binary-CD sample dict."""
    for key in ("image_a", "image_b"):
        t = sample[key]
        assert isinstance(t, torch.Tensor), f"{key} not a tensor"
        assert t.ndim == 3,                 f"{key} should be 3D [C,H,W], got {t.shape}"
        assert t.shape[0] == 3,             f"{key} should have 3 channels, got {t.shape}"
        assert t.dtype == torch.float32,    f"{key} dtype should be float32"

    mask = sample.get("mask", sample.get("label"))
    assert mask is not None,                "no 'mask' or 'label' key found"
    assert isinstance(mask, torch.Tensor),  "mask not a tensor"
    assert mask.ndim in (2, 3),             f"mask should be 2D or 3D, got {mask.shape}"


def _check_second_sample(sample: dict) -> None:
    """Validate a single SECOND semantic-CD sample dict."""
    for key in ("image_a", "image_b"):
        t = sample[key]
        assert isinstance(t, torch.Tensor), f"{key} not a tensor"
        assert t.ndim == 3 and t.shape[0] == 3

    for key in ("label_t1", "label_t2", "sem_label_t1", "sem_label_t2"):
        if key in sample:
            t = sample[key]
            assert isinstance(t, torch.Tensor), f"{key} not a tensor"
            assert t.ndim == 2,                 f"{key} should be 2D, got {t.shape}"
            break
    else:
        pytest.fail("No semantic label key found (expected label_t1/label_t2 or sem_label_t1/sem_label_t2)")


# --------------------------------------------------------------------------
# LEVIR-CD
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    _root("LEVIR_ROOT") is None,
    reason="LEVIR_ROOT not set or path does not exist",
)
class TestLEVIRDataset:
    def _ds(self) -> object:
        from datasets.levir import build_levir_dataset
        cfg = {
            "dataset": {
                "name": "LEVIR-CD",
                "root": _root("LEVIR_ROOT"),
                "image_size": 256,
            }
        }
        return build_levir_dataset(cfg, split="val")

    def test_loads(self) -> None:
        ds = self._ds()
        assert len(ds) > 0

    def test_sample_keys_and_shapes(self) -> None:
        ds = self._ds()
        _check_binary_sample(ds[0])

    def test_image_size(self) -> None:
        ds = self._ds()
        s  = ds[0]
        assert s["image_a"].shape[-1] == 256


# --------------------------------------------------------------------------
# WHU-CD
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    _root("WHU_ROOT") is None,
    reason="WHU_ROOT not set or path does not exist",
)
class TestWHUDataset:
    def _ds(self) -> object:
        from datasets.whu import build_whu_dataset
        cfg = {
            "dataset": {
                "name": "WHU-CD",
                "root": _root("WHU_ROOT"),
                "image_size": 256,
            }
        }
        return build_whu_dataset(cfg, split="val")

    def test_loads(self) -> None:
        assert len(self._ds()) > 0

    def test_sample_keys_and_shapes(self) -> None:
        _check_binary_sample(self._ds()[0])


# --------------------------------------------------------------------------
# DSIFN-CD
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    _root("DSIFN_ROOT") is None,
    reason="DSIFN_ROOT not set or path does not exist",
)
class TestDSIFNDataset:
    def _ds(self) -> object:
        from datasets.dsifn import build_dsifn_dataset
        cfg = {
            "dataset": {
                "name": "DSIFN-CD",
                "root": _root("DSIFN_ROOT"),
                "image_size": 256,
            }
        }
        return build_dsifn_dataset(cfg, split="val")

    def test_loads(self) -> None:
        assert len(self._ds()) > 0

    def test_sample_keys_and_shapes(self) -> None:
        _check_binary_sample(self._ds()[0])


# --------------------------------------------------------------------------
# SECOND
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    _root("SECOND_ROOT") is None,
    reason="SECOND_ROOT not set or path does not exist",
)
class TestSECONDDataset:
    def _ds(self) -> object:
        from datasets.second import build_second_dataset
        cfg = {
            "dataset": {
                "name": "SECOND",
                "root": _root("SECOND_ROOT"),
                "image_size": 256,
                "num_classes": 7,
                "ignore_index": 255,
            }
        }
        return build_second_dataset(cfg, split="val")

    def test_loads(self) -> None:
        assert len(self._ds()) > 0

    def test_sample_keys_and_shapes(self) -> None:
        _check_second_sample(self._ds()[0])
