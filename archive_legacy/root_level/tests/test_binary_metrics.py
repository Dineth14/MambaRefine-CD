"""Unit tests for BinaryMetrics.

Tests:
  - Correct Pre/Rec/F1/IoU/OA for known TP/FP/TN/FN values
  - Metric restriction: no extra keys returned
  - All-positive prediction edge case
  - All-negative prediction edge case
  - Perfect prediction
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import torch
import pytest
from metrics.binary_cd_metrics import BinaryMetrics

EPS = 1e-2  # half a percentage point tolerance


def _make_logits(pred_binary: list[int]) -> torch.Tensor:
    """Convert binary list to logits [1,1,N] that threshold at 0.5 correctly."""
    # sigmoid(logit): > 0 → prob > 0.5 → pred = 1
    vals = [1.0 if p else -1.0 for p in pred_binary]
    n    = len(vals)
    return torch.tensor(vals, dtype=torch.float32).view(1, 1, 1, n)


def _make_gt(gt_binary: list[int]) -> torch.Tensor:
    n = len(gt_binary)
    return torch.tensor(gt_binary, dtype=torch.long).view(1, 1, n)


class TestBinaryMetricsKeys:
    """Verify only the 5 allowed keys are returned."""

    def test_only_allowed_keys(self) -> None:
        m = BinaryMetrics(threshold=0.5)
        m.update(_make_logits([0, 0, 1, 1]), _make_gt([0, 1, 0, 1]))
        result = m.compute()
        assert set(result.keys()) == {"Pre", "Rec", "F1", "IoU", "OA"}, \
            f"Unexpected keys: {set(result.keys())}"


class TestBinaryMetricsValues:
    """Numerical correctness with TP=TN=FP=FN=1."""

    def setup_method(self) -> None:
        # gt:   [0, 0, 1, 1]
        # pred: [0, 1, 0, 1]  → TN, FP, FN, TP  (each=1)
        self.m = BinaryMetrics(threshold=0.5)
        self.m.update(_make_logits([0, 1, 0, 1]), _make_gt([0, 0, 1, 1]))
        self.res = self.m.compute()

    def test_precision(self) -> None:
        assert abs(self.res["Pre"] - 50.0) < EPS

    def test_recall(self) -> None:
        assert abs(self.res["Rec"] - 50.0) < EPS

    def test_f1(self) -> None:
        assert abs(self.res["F1"]  - 50.0) < EPS

    def test_iou(self) -> None:
        expected = 100.0 / 3.0  # 33.33...
        assert abs(self.res["IoU"] - expected) < EPS

    def test_oa(self) -> None:
        assert abs(self.res["OA"]  - 50.0) < EPS


class TestBinaryMetricsPerfect:
    """Perfect prediction: all metrics = 100."""

    def test_perfect_prediction(self) -> None:
        m = BinaryMetrics(threshold=0.5)
        m.update(_make_logits([0, 0, 1, 1]), _make_gt([0, 0, 1, 1]))
        res = m.compute()
        for key in ("Pre", "Rec", "F1", "IoU", "OA"):
            assert abs(res[key] - 100.0) < EPS, f"{key}={res[key]}"


class TestBinaryMetricsAllNegative:
    """All-negative prediction: TP=FP=0; Recall=0, Pre=0."""

    def test_all_negative_pred(self) -> None:
        m = BinaryMetrics(threshold=0.5)
        # gt has change, pred all 0
        m.update(_make_logits([0, 0, 0, 0]), _make_gt([0, 0, 1, 1]))
        res = m.compute()
        assert res["Pre"] < EPS or abs(res["Pre"] - 0.0) < EPS
        assert res["Rec"] < EPS or abs(res["Rec"] - 0.0) < EPS

    def test_no_change_in_gt(self) -> None:
        """If gt is all-zero, OA should be 100."""
        m = BinaryMetrics(threshold=0.5)
        m.update(_make_logits([0, 0, 0, 0]), _make_gt([0, 0, 0, 0]))
        res = m.compute()
        assert abs(res["OA"] - 100.0) < EPS


class TestBinaryMetricsRange:
    """All values should be in [0, 100]."""

    def test_values_in_range(self) -> None:
        m = BinaryMetrics(threshold=0.5)
        m.update(_make_logits([0, 1, 0, 1, 1, 0]), _make_gt([1, 1, 0, 0, 1, 0]))
        res = m.compute()
        for k, v in res.items():
            assert 0.0 <= v <= 100.0 + 1e-5, f"{k}={v} out of [0,100]"


class TestBinaryMetricsReset:
    """Reset clears accumulator."""

    def test_reset(self) -> None:
        m = BinaryMetrics(threshold=0.5)
        m.update(_make_logits([1, 1, 1, 1]), _make_gt([0, 0, 0, 0]))
        m.reset()
        m.update(_make_logits([0, 0, 1, 1]), _make_gt([0, 0, 1, 1]))
        res = m.compute()
        assert abs(res["OA"] - 100.0) < EPS
