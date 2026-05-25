# Metrics Reference

This document defines the metrics used for all datasets.

---

## Binary Change Detection (LEVIR-CD, WHU-CD, DSIFN-CD)

Only the following **5 metrics** are logged, saved, and reported.
No other metrics (Kappa, mIoU, FWIoU, SeK) are computed or shown.

| Metric | Formula | Notes |
|--------|---------|-------|
| **Pre** (Precision) | $\frac{TP}{TP + FP + \varepsilon} \times 100$ | Change class only |
| **Rec** (Recall) | $\frac{TP}{TP + FN + \varepsilon} \times 100$ | Change class only |
| **F1** | $\frac{2 \cdot Pre \cdot Rec}{Pre + Rec + \varepsilon}$ | Harmonic mean of Pre and Rec |
| **IoU** | $\frac{TP}{TP + FP + FN + \varepsilon} \times 100$ | Intersection over Union (change class) |
| **OA** (Overall Accuracy) | $\frac{TP + TN}{TP + TN + FP + FN + \varepsilon} \times 100$ | All pixels |

- **Threshold**: `0.5` for sigmoid probability map → binary mask.
- All values are in the range **[0, 100]** (percentages).
- $\varepsilon = 10^{-6}$ for numerical stability.
- Best checkpoint is selected by **F1**.

### Class: `BinaryMetrics`

```python
from metrics.binary_cd_metrics import BinaryMetrics

m = BinaryMetrics(threshold=0.5)
m.update(logits, gt_mask)   # logits: [B,1,H,W], gt_mask: [B,H,W] long
results = m.compute()       # {"Pre": float, "Rec": float, "F1": float, "IoU": float, "OA": float}
m.reset()
```

---

## Semantic Change Detection (SECOND)

Only the following **4 metrics** are logged, saved, and reported.
No binary-only metrics (Pre/Rec/F1/IoU) are used.

| Metric | Formula | Notes |
|--------|---------|-------|
| **OA** (Overall Accuracy) | $\frac{\sum_i H_{ii}}{\sum_{ij} H_{ij}} \times 100$ | Combined t1+t2 confusion matrix |
| **mIoU** | $\frac{1}{C} \sum_i \frac{H_{ii}}{H_{i\cdot} + H_{\cdot i} - H_{ii}} \times 100$ | Mean over all $C$ semantic classes |
| **SeK** (Semantic Change Kappa) | $\frac{\kappa_{n0} \cdot e^{IoU_{\text{change}}}}{e}$ | See protocol below |
| **Fscd** | $\frac{1}{C-1} \sum_{i>0} \frac{2 H_{ii}}{H_{i\cdot} + H_{\cdot i}} \times 100$ | Diagonal F1 for semantic-change classes only |

- **SeK protocol** follows MambaFCS / SECOND paper:
  1. Build combined 2T confusion matrix $H$ from $(pred_{t1}, gt_{t1})$ and $(pred_{t2}, gt_{t2})$
  2. $\kappa_{n0}$: set $H[0,0] = 0$ (ignore no-change pixels), then compute Cohen's kappa
  3. $IoU_{\text{change}} = \frac{TP_{\text{change}}}{TP + FP + FN}$ from a separate binary change head or from semantic change pixels
  4. $SeK = \kappa_{n0} \cdot e^{IoU_{\text{change}} - 1}$
- All values in **[0, 100]** (percentages).
- Best checkpoint is selected by **Fscd** (or **SeK** if configured).

### Class: `SECONDSCDMetrics`

```python
from metrics.second_scd_metrics import SECONDSCDMetrics

m = SECONDSCDMetrics(num_classes=7, ignore_index=255, threshold=0.5)
m.update(pred_sem1, pred_sem2, gt_sem1, gt_sem2, change_mask=None)
results = m.compute()   # {"OA": float, "mIoU": float, "SeK": float, "Fscd": float}
m.reset()
```

- `pred_sem1`, `pred_sem2`: `[B, H, W]` long tensors (argmax of semantic logits)
- `gt_sem1`, `gt_sem2`: `[B, H, W]` long tensors
- `change_mask`: optional `[B, H, W]` bool tensor from the binary change head

---

## Summary

| Dataset | Allowed Metrics | Selection Criterion |
|---------|----------------|---------------------|
| LEVIR-CD | Pre, Rec, F1, IoU, OA | F1 |
| WHU-CD   | Pre, Rec, F1, IoU, OA | F1 |
| DSIFN-CD | Pre, Rec, F1, IoU, OA | F1 |
| SECOND   | OA, mIoU, SeK, Fscd  | Fscd |
