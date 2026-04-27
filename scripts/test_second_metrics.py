"""Synthetic checks for SECOND evaluation metrics.

Run:
    PYTHONPATH=src conda run -n mamba_new python scripts/test_second_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.second_metrics import SECONDMetrics


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-6) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def test_perfect_binary_metrics() -> None:
    metric = SECONDMetrics(num_classes=3, ignore_index=255, compute_sek=True, sek_binary_fallback=False)
    change = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)
    metric.update(change_pred=change, change_gt=change)
    result = metric.compute()
    assert_close("binary OA", result["OA"], 1.0)
    assert_close("binary Fscd", result["Fscd"], 1.0)
    assert_close("binary mIoU", result["mIoU"], 1.0)
    if result["SeK"] is not None:
        raise AssertionError(f"Expected SeK to be unavailable in binary mode, got {result['SeK']}")


def test_perfect_semantic_metrics() -> None:
    metric = SECONDMetrics(num_classes=3, ignore_index=255, compute_sek=True)
    label_a = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    label_b = torch.tensor([[0, 2], [1, 2]], dtype=torch.long)
    change = label_a != label_b
    metric.update(
        change_pred=change,
        change_gt=change,
        pred_label_t1=label_a,
        pred_label_t2=label_b,
        label_t1=label_a,
        label_t2=label_b,
    )
    result = metric.compute()
    assert_close("semantic OA", result["OA"], 1.0)
    assert_close("semantic Fscd", result["Fscd"], 1.0)
    assert_close("semantic mIoU", result["mIoU"], 1.0)
    assert_close("semantic SeK", float(result["SeK"]), 1.0)


def test_all_background_prediction_penalizes_fscd() -> None:
    metric = SECONDMetrics(num_classes=3, ignore_index=255, compute_sek=False)
    gt_change = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)
    pred_change = torch.zeros_like(gt_change)
    metric.update(change_pred=pred_change, change_gt=gt_change)
    result = metric.compute()
    if result["Fscd"] >= 0.5:
        raise AssertionError(f"Expected low Fscd for all-background prediction, got {result['Fscd']}")


def test_ignore_index_is_excluded() -> None:
    metric = SECONDMetrics(num_classes=3, ignore_index=255, compute_sek=False)
    gt_change = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
    pred_change = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
    ignore_mask = torch.tensor([[0, 0], [1, 0]], dtype=torch.bool)
    metric.update(change_pred=pred_change, change_gt=gt_change, ignore_mask=ignore_mask)
    result = metric.compute()
    assert_close("ignore OA", result["OA"], 1.0)
    assert_close("ignore Fscd", result["Fscd"], 1.0)


def test_binary_sek_fallback_behavior() -> None:
    gt_change = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)
    pred_change = gt_change.clone()

    metric = SECONDMetrics(num_classes=3, ignore_index=255, compute_sek=True, sek_binary_fallback=False)
    metric.update(change_pred=pred_change, change_gt=gt_change)
    result = metric.compute()
    if result["SeK"] is not None or result["binary_kappa"] is not None:
        raise AssertionError("Expected no SeK or binary_kappa when fallback is disabled.")

    fallback_metric = SECONDMetrics(num_classes=3, ignore_index=255, compute_sek=True, sek_binary_fallback=True)
    fallback_metric.update(change_pred=pred_change, change_gt=gt_change)
    fallback_result = fallback_metric.compute()
    if fallback_result["SeK"] is not None:
        raise AssertionError("Expected SeK to remain unavailable in binary fallback mode.")
    assert_close("binary_kappa fallback", float(fallback_result["binary_kappa"]), 1.0)


def main() -> None:
    test_perfect_binary_metrics()
    test_perfect_semantic_metrics()
    test_all_background_prediction_penalizes_fscd()
    test_ignore_index_is_excluded()
    test_binary_sek_fallback_behavior()
    print("SECOND metric synthetic checks passed.")


if __name__ == "__main__":
    main()