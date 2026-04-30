#!/usr/bin/env python3
"""Self-check active binary change-detection metrics."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch

from metrics.binary_cd_metrics import BinaryMetrics


def main() -> None:
    # Predictions at threshold 0.5:
    # pred = [[1, 1, 0],
    #         [0, 1, 0]]
    # gt   = [[1, 0, 0],
    #         [1, 1, 0]]
    # TP=2, FP=1, TN=2, FN=1
    probs = torch.tensor([[[[0.9, 0.8, 0.2], [0.1, 0.7, 0.3]]]], dtype=torch.float32)
    labels = torch.tensor([[[[1, 0, 0], [1, 1, 0]]]], dtype=torch.float32)
    metric = BinaryMetrics(threshold=0.5)
    metric.update(probs, labels)
    got = metric.compute()

    tp, fp, tn, fn = 2.0, 1.0, 2.0, 1.0
    eps = 1e-6
    pre = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2.0 * pre * rec / (pre + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    expected = {
        "Pre": round(pre * 100.0, 4),
        "Rec": round(rec * 100.0, 4),
        "F1": round(f1 * 100.0, 4),
        "IoU": round(iou * 100.0, 4),
        "OA": round(oa * 100.0, 4),
    }
    print(f"TP={int(tp)}")
    print(f"FP={int(fp)}")
    print(f"TN={int(tn)}")
    print(f"FN={int(fn)}")
    for key in ("Pre", "Rec", "F1", "IoU", "OA"):
        print(f"{key}={got[key]:.4f} expected={expected[key]:.4f}")
    ok = all(math.isclose(got[k], expected[k], rel_tol=0.0, abs_tol=1e-4) for k in expected)
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
