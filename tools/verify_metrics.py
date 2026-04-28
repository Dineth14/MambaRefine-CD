"""Metric sanity checker.

Constructs synthetic predictions with known TP/FP/TN/FN counts,
then verifies that BinaryMetrics and SECONDSCDMetrics produce
numerically correct outputs.

Usage:
    python tools/verify_metrics.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

EPS = 1e-6


def _check_close(name: str, got: float, expected: float, tol: float = 0.5) -> bool:
    ok = abs(got - expected) < tol
    status = "OK" if ok else "FAIL"
    logger.info(f"  [{status}] {name}: expected={expected:.4f}  got={got:.4f}")
    return ok


def _test_binary() -> bool:
    """
    2×2 pixel, batch=1.
    Pixel layout (gt | pred_class):
        0 | 0   -> TN
        0 | 1   -> FP
        1 | 0   -> FN
        1 | 1   -> TP
    So TP=1, TN=1, FP=1, FN=1.
    Expected:
        Pre  = 1/(1+1) = 50.0
        Rec  = 1/(1+1) = 50.0
        F1   = 2*0.5*0.5/(0.5+0.5) = 50.0
        IoU  = 1/(1+1+1) = 33.333...
        OA   = (1+1)/(1+1+1+1) = 50.0
    """
    from metrics.binary_cd_metrics import BinaryMetrics

    # logits: > 0 → pred=1
    logits = torch.tensor([[[[  -1.0,  1.0],
                               [ -1.0,  1.0]]]])   # shape [1,1,2,2]
    gt     = torch.tensor([[[0, 0],
                              [1, 1]]],             # shape [1,2,2]
                           dtype=torch.long)

    m = BinaryMetrics(threshold=0.5)
    m.update(logits, gt)
    res = m.compute()

    logger.info("=== BinaryMetrics sanity check ===")
    all_ok = True
    all_ok &= _check_close("Pre",  res["Pre"],  50.0)
    all_ok &= _check_close("Rec",  res["Rec"],  50.0)
    all_ok &= _check_close("F1",   res["F1"],   50.0)
    all_ok &= _check_close("IoU",  res["IoU"],  100.0 / 3.0, tol=0.1)
    all_ok &= _check_close("OA",   res["OA"],   50.0)

    # Verify ONLY these 5 keys are present
    allowed = {"Pre", "Rec", "F1", "IoU", "OA"}
    extra   = set(res.keys()) - allowed
    if extra:
        logger.error(f"  [FAIL] BinaryMetrics returned extra keys: {extra}")
        all_ok = False
    else:
        logger.info(f"  [OK]   No extra metric keys returned.")

    return all_ok


def _test_second() -> bool:
    from metrics.second_scd_metrics import SECONDSCDMetrics

    gt_s1 = torch.tensor([[[1, 1], [2, 2]]], dtype=torch.long)
    gt_s2 = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.long)
    m = SECONDSCDMetrics(num_classes=4, ignore_index=255, threshold=0.5)
    m.update(gt_s1, gt_s2, gt_s1, gt_s2)
    res = m.compute()

    logger.info("=== SECONDSCDMetrics sanity check ===")
    all_ok  = True
    allowed = {"OA", "mIoU", "SeK", "Fscd"}

    all_ok &= _check_close("OA",  res["OA"],   100.0)
    all_ok &= _check_close("mIoU",  res["mIoU"],   100.0)
    all_ok &= _check_close("Fscd",  res["Fscd"],   100.0)
    if res["SeK"] <= 0.0:
        logger.error(f"  [FAIL] SeK should be positive/perfect, got {res['SeK']:.4f}")
        all_ok = False

    wrong = SECONDSCDMetrics(num_classes=4, ignore_index=255, threshold=0.5)
    pred_s1 = gt_s1.clone()
    pred_s2 = gt_s2.clone()
    pred_s2[0, 0, 1] = 1
    wrong.update(pred_s1, pred_s2, gt_s1, gt_s2)
    wrong_res = wrong.compute()
    if not (wrong_res["OA"] < res["OA"] and wrong_res["mIoU"] < res["mIoU"] and wrong_res["Fscd"] < res["Fscd"]):
        logger.error(f"  [FAIL] Wrong prediction did not decrease metrics: {wrong_res}")
        all_ok = False

    extra = set(res.keys()) - allowed
    if extra:
        logger.error(f"  [FAIL] SECONDSCDMetrics returned extra keys: {extra}")
        all_ok = False
    else:
        logger.info(f"  [OK]   No extra metric keys returned.")

    # All values should be in [0, 100]
    for k, v in res.items():
        if not (0.0 <= v <= 100.0):
            logger.error(f"  [FAIL] {k}={v:.4f} out of [0, 100]")
            all_ok = False
        else:
            logger.info(f"  [OK]   {k}={v:.4f} in [0, 100]")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["all", "binary", "second"], default="all")
    args = parser.parse_args()
    ok1 = True if args.task == "second" else _test_binary()
    ok2 = True if args.task == "binary" else _test_second()
    logger.info("=" * 50)
    if ok1 and ok2:
        logger.info("All metric sanity checks PASSED.")
    else:
        logger.error("Some metric sanity checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
