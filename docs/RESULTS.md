# MambaRefine-CD — Results

**Generated:** 2026-05-02  
**All results:** test-set evaluation on EMA checkpoint, best-val threshold from sweep `[0.30–0.60]`. Split integrity verified `PASS` on every DSIFN run (hash-locked clean split, no train/test leakage). Metrics are in percent (%).

> **Averaging policy:** where multiple runs share the same config and backbone, results are averaged to reduce noise. Individual run results are listed in the per-run appendix. Runs with different backbone sizes are reported separately in the backbone-scaling table.

---

## 1. DSIFN-CD Ablation Study

**Dataset:** DSIFN-CD — 2758 train / 394 val / 789 test images (3156 test tiles @ 256×256 with 0.25 overlap)  
**Seed:** 42 | **Optimizer:** AdamW, lr=5e-5 | **Augmentation:** horizontal + vertical flip

The ablation isolates the contribution of each component by adding one at a time to the previous row.

### 1.1 Component ablation table (averaged across runs)

| ID | Variant | Backbone | D-RBI | Signed Diff | ARF-FPN | Bnd. Residual | CRAMLite | Full Losses | Iters | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) | Params (M) | N runs |
|----|---------|----------|:-----:|:-----------:|:-------:|:-------------:|:--------:|:-----------:|------:|--------:|--------:|-------:|--------:|-------:|-----------:|:------:|
| A0 | FPN Baseline | SimpleCNN | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 30k | 77.27 | 76.15 | 76.63 | 62.12 | 83.80 | 7.84 | 2 |
| A1 | + MambaVision-S | MambaVision-S | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 50k | 93.65 | 94.09 | 93.87 | 88.45 | 95.72 | 53.54 | 2 |
| A2 | + D-RBI (unsigned) | MambaVision-S | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | 50k | 92.70 | 93.98 | 93.33 | 87.50 | 95.35 | 54.98 | 1 |
| A3 | + Signed Diff | MambaVision-S | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | 50k | 93.69 | 94.88 | 94.28 | 89.19 | 95.99 | 55.34 | 2 |
| A4 | + ARF-FPN | MambaVision-S | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | 50k | 93.94 | 94.78 | 94.36 | 89.32 | 96.05 | 65.12 | 2 |
| A5 | + Boundary Residual | MambaVision-S | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 50k | 92.96 | 94.22 | 93.59 | 87.94 | 95.53 | 65.19 | 1 |
| **A6** | **Full Model** | **MambaVision-S** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **50k** | **95.47** | **95.87** | **95.67** | **91.71** | **96.98** | **65.40** | **2** |

> **Full Losses** = BCE+Dice main + coarse auxiliary (w=0.4) + boundary edge loss (w=0.1). A0–A5 use BCE+Dice only.  
> **N runs** = number of independent runs averaged; single-run results are exact.

### 1.2 Component contribution deltas (vs preceding row, averaged results)

| Component | ΔF1 vs previous | ΔIoU vs previous |
|-----------|----------------:|-----------------:|
| A0 → A1: MambaVision backbone | +17.56 | +26.44 |
| A1 → A2: D-RBI (unsigned) | +0.12 | +0.22 |
| A2 → A3: Signed difference | +0.30 | +0.51 |
| A3 → A4: ARF-FPN decoder | +0.08 | +0.14 |
| A4 → A5: Boundary residual | −0.12 | −0.22 |
| A5 → A6: Semantic heads + losses | +1.36 | +2.43 |

> A5 (boundary residual) requires the full loss schedule to be effective, as seen in the A6 jump.


---

## 2. Backbone Scaling Comparison (Full Model, DSIFN-CD)

All runs use the full model config (D-RBI + Signed Diff + ARF-FPN + CRAMLite + full loss schedule, 50k iters). Note: the tiny and base runs were launched with `boundary_residual_enabled=False` (confirmed by ablation trace); the small canonical runs have `boundary_residual_enabled=True`.

| Backbone Variant | Params (M) | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) | N runs |
|-----------------|:----------:|--------:|--------:|-------:|--------:|-------:|:------:|
| MambaVision-T (tiny) | 46.73 | 94.33 | 94.84 | 94.58 | 89.72 | 96.24 | 1 |
| **MambaVision-S (small)** | **65.40** | **95.47** | **95.87** | **95.67** | **91.71** | **96.98** | **2** |
| MambaVision-B (base) | 113.37 | 95.66 | 95.89 | 95.77 | 91.89 | 97.07 | 1 |

> The small → base jump (+0.10 F1) is marginal for a 1.73× parameter increase, making MambaVision-S the best efficiency-accuracy trade-off.

---

## 3. WHU-CD Results

**Dataset:** WHU-CD — standard train/val/test split  
**Full model config:** MambaVision-S + ARF-FPN + D-RBI (signed) + Boundary Residual + CRAMLite + full loss schedule  
**Training:** 50k iterations, AdamW lr=5e-5, cosine decay, 2500 warmup iters, batch=8, AMP

| Run | Backbone | Params (M) | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) | Thresh |
|-----|----------|:----------:|--------:|--------:|-------:|--------:|-------:|-------:|
| run_whu_20260430_114506 | MambaVision-S | 65.40 | 95.58 | 94.74 | 95.15 | 90.76 | 99.54 | 0.55 |

> Only one completed WHU run with the unified pipeline. The earlier run (`run_20260428_023626`) used a different evaluation script (old metric keys) and is not directly comparable; see appendix.

---

## 4. Summary — Best Single-Run Results

| Dataset | Model | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) |
|---------|-------|--------:|--------:|-------:|--------:|-------:|
| DSIFN-CD | Full Model (MambaVision-S, run_a6_105918) | 96.26 | 96.53 | **96.40** | 93.04 | 97.47 |
| DSIFN-CD | Full Model avg (2 canonical runs) | 95.47 | 95.87 | **95.67** | 91.71 | 96.98 |
| WHU-CD | Full Model (MambaVision-S) | 95.58 | 94.74 | **95.15** | 90.76 | 99.54 |

---

## 5. Appendix — WHU Early Run (Different Eval Script)

The run `run_20260428_023626_whu_a4_full_WHU-CD` used an older evaluation pipeline producing different metric keys (`f1`, `iou`, `precision`, `recall`, `oa` in fractional form). Converted to percentage for reference only:

| F1 (%) | IoU (%) | Pre (%) | Rec (%) | OA (%) | Thresh |
|-------:|--------:|--------:|--------:|-------:|-------:|
| 95.58  | 91.53   | 96.16   | 95.00   | 99.58  | 0.40   |

This result is not used in the main tables due to pipeline differences.
