"""Unit tests for SECONDSCDMetrics.

Tests:
  - Metric restriction: only OA, mIoU, SeK, Fscd returned
  - All values in [0, 100]
  - Perfect prediction
  - OA value correctness
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import torch
import pytest
from metrics.second_scd_metrics import SECONDSCDMetrics

EPS      = 0.1   # 0.1 percentage point tolerance
N_CLS    = 7     # SECOND default
IGN_IDX  = 255
H, W     = 4, 4


def _perfect_sample(n_cls: int = 3, h: int = H, w: int = W) -> dict:
    """Return perfect prediction where pred == gt at both times."""
    gt_s1 = torch.randint(0, n_cls, (1, h, w))
    gt_s2 = torch.randint(0, n_cls, (1, h, w))
    return dict(pred_s1=gt_s1.clone(), pred_s2=gt_s2.clone(),
                gt_s1=gt_s1, gt_s2=gt_s2)


class TestSECONDMetricsKeys:
    """Only OA, mIoU, SeK, Fscd allowed."""

    def test_only_allowed_keys(self) -> None:
        m  = SECONDSCDMetrics(num_classes=N_CLS, ignore_index=IGN_IDX)
        s  = _perfect_sample(n_cls=N_CLS)
        m.update(s["pred_s1"], s["pred_s2"], s["gt_s1"], s["gt_s2"])
        res = m.compute()
        assert set(res.keys()) == {"OA", "mIoU", "SeK", "Fscd"}, \
            f"Unexpected keys: {set(res.keys())}"


class TestSECONDMetricsRange:
    """All values in [0, 100]."""

    def test_values_in_range(self) -> None:
        m  = SECONDSCDMetrics(num_classes=N_CLS, ignore_index=IGN_IDX)
        gt_s1 = torch.zeros(1, H, W, dtype=torch.long)
        gt_s2 = torch.zeros(1, H, W, dtype=torch.long)
        pred_s1 = torch.ones(1, H, W, dtype=torch.long)
        pred_s2 = torch.zeros(1, H, W, dtype=torch.long)
        m.update(pred_s1, pred_s2, gt_s1, gt_s2)
        res = m.compute()
        for k, v in res.items():
            assert 0.0 <= v <= 100.0 + 1e-5, f"{k}={v} out of [0, 100]"


class TestSECONDMetricsPerfect:
    """Perfect prediction should yield high OA."""

    def test_perfect_oa(self) -> None:
        m = SECONDSCDMetrics(num_classes=N_CLS, ignore_index=IGN_IDX)
        s = _perfect_sample(n_cls=N_CLS)
        m.update(s["pred_s1"], s["pred_s2"], s["gt_s1"], s["gt_s2"])
        res = m.compute()
        assert res["OA"] > 99.0, f"Expected OA~100, got {res['OA']}"

    def test_tiny_second_example_perfect(self) -> None:
        m = SECONDSCDMetrics(num_classes=4, ignore_index=255)
        gt_t1 = torch.tensor([[[1, 1], [2, 2]]], dtype=torch.long)
        gt_t2 = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.long)
        m.update(gt_t1, gt_t2, gt_t1, gt_t2)
        res = m.compute()
        assert abs(res["OA"] - 100.0) < EPS
        assert abs(res["mIoU"] - 100.0) < EPS
        assert abs(res["Fscd"] - 100.0) < EPS
        assert res["SeK"] > 99.0

    def test_tiny_second_example_wrong_prediction_decreases(self) -> None:
        perfect = SECONDSCDMetrics(num_classes=4, ignore_index=255)
        wrong = SECONDSCDMetrics(num_classes=4, ignore_index=255)
        gt_t1 = torch.tensor([[[1, 1], [2, 2]]], dtype=torch.long)
        gt_t2 = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.long)
        pred_t1 = gt_t1.clone()
        pred_t2 = gt_t2.clone()
        pred_t2[0, 0, 1] = 1
        perfect.update(gt_t1, gt_t2, gt_t1, gt_t2)
        wrong.update(pred_t1, pred_t2, gt_t1, gt_t2)
        perfect_res = perfect.compute()
        wrong_res = wrong.compute()
        assert wrong_res["OA"] < perfect_res["OA"]
        assert wrong_res["mIoU"] < perfect_res["mIoU"]
        assert wrong_res["Fscd"] < perfect_res["Fscd"]


class TestSECONDMetricsOAFormula:
    """Manually verify OA with known confusion matrix."""

    def test_known_oa(self) -> None:
        # With no GT changes, a single predicted semantic change makes one
        # timestamp wrong in the SCD label stream: 7/8 correct.
        m = SECONDSCDMetrics(num_classes=3, ignore_index=255)
        gt_s1   = torch.zeros(1, 2, 2, dtype=torch.long)
        gt_s2   = torch.zeros(1, 2, 2, dtype=torch.long)
        pred_s1 = gt_s1.clone()
        pred_s2 = gt_s2.clone()
        pred_s2[0, 0, 0] = 1  # one wrong pixel
        m.update(pred_s1, pred_s2, gt_s1, gt_s2)
        res = m.compute()
        assert abs(res["OA"] - 87.5) < EPS, f"Expected OA=87.5, got {res['OA']}"


class TestSECONDMetricsReset:
    """Reset clears accumulator."""

    def test_reset(self) -> None:
        m = SECONDSCDMetrics(num_classes=N_CLS, ignore_index=IGN_IDX)
        # First round: random
        m.update(torch.zeros(1, H, W, dtype=torch.long),
                 torch.ones(1,  H, W, dtype=torch.long),
                 torch.zeros(1, H, W, dtype=torch.long),
                 torch.zeros(1, H, W, dtype=torch.long))
        m.reset()
        # Second round: perfect
        s = _perfect_sample(n_cls=N_CLS)
        m.update(s["pred_s1"], s["pred_s2"], s["gt_s1"], s["gt_s2"])
        res = m.compute()
        assert res["OA"] > 99.0


class TestSECONDMetricsWithChangeMask:
    """Verify that passing an explicit change mask does not raise."""

    def test_with_change_mask(self) -> None:
        m = SECONDSCDMetrics(num_classes=N_CLS, ignore_index=IGN_IDX)
        gt_s1  = torch.zeros(1, H, W, dtype=torch.long)
        gt_s2  = torch.zeros(1, H, W, dtype=torch.long)
        ch     = torch.zeros(1, H, W, dtype=torch.bool)
        ch[0, :2, :] = True
        m.update(gt_s1, gt_s2, gt_s1, gt_s2, change_mask=ch)
        res = m.compute()
        assert isinstance(res, dict)
        assert set(res.keys()) == {"OA", "mIoU", "SeK", "Fscd"}


class TestSECONDMetricsIgnoreIndex:
    """Ignore pixels must not affect global accumulators."""

    def test_ignore_pixels_do_not_hurt_perfect_valid_pixels(self) -> None:
        m = SECONDSCDMetrics(num_classes=4, ignore_index=255)
        gt_t1 = torch.tensor([[[1, 255], [2, 2]]], dtype=torch.long)
        gt_t2 = torch.tensor([[[1, 255], [2, 3]]], dtype=torch.long)
        pred_t1 = gt_t1.clone()
        pred_t2 = gt_t2.clone()
        pred_t1[0, 0, 1] = 0
        pred_t2[0, 0, 1] = 1
        m.update(pred_t1, pred_t2, gt_t1, gt_t2)
        res = m.compute()
        assert abs(res["OA"] - 100.0) < EPS
        assert abs(res["Fscd"] - 100.0) < EPS


class TestSECONDMetricsSemanticSpecificity:
    """Perfect binary change localization is not enough for perfect Fscd."""

    def test_binary_perfect_but_wrong_semantic_classes_not_perfect(self) -> None:
        m = SECONDSCDMetrics(num_classes=4, ignore_index=255)
        gt_t1 = torch.tensor([[[1, 1], [2, 2]]], dtype=torch.long)
        gt_t2 = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.long)
        pred_t1 = gt_t1.clone()
        pred_t2 = gt_t2.clone()
        # The changed pixels remain changed, but their semantic destination
        # classes are wrong, so SCD F-score must drop.
        pred_t2[0, 0, 1] = 3
        pred_t2[0, 1, 1] = 1
        m.update(pred_t1, pred_t2, gt_t1, gt_t2)
        res = m.compute()
        assert res["Fscd"] < 100.0
