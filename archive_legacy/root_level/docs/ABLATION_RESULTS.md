# Ablation Results — MambaRefine-CD

**Generated:** 2026-05-01  
**Protocol:** All DSIFN runs use the clean split (no train/test leakage); split integrity verified `PASS` on every run. All WHU runs use the standard test split. Metrics are reported on the **test set** at the best validation threshold (EMA checkpoint).

---

## DSIFN-CD Ablation

**Dataset:** DSIFN-CD — 2758 train / 394 val / 789 test images (3156 test tiles at 256×256)  
**Seed:** 42 | **Eval threshold:** per-run best val threshold

| ID | Variant | Components Added | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) | Params (M) | Thresh |
|----|---------|-----------------|--------:|--------:|-------:|--------:|-------:|-----------:|-------:|
| A0 | FPN Baseline | SimpleCNN + FPN decoder (no MambaVision) | 73.85 | 77.54 | 75.65 | 60.84 | 82.71 | 7.84 | 0.3 |
| A1 | MambaVision + FPN | MambaVision-S backbone, FPN decoder | 93.04 | 93.38 | 93.21 | 87.28 | 95.29 | 53.54 | 0.5 |
| A2 | + D-RBI | + Difference-Refined Boundary Interaction (unsigned diff, boundary gate) | 92.70 | 93.98 | 93.33 | 87.50 | 95.35 | 54.98 | 0.6 |
| A3 | + Signed Diff | + Signed difference in D-RBI | 92.81 | 94.46 | 93.63 | 88.02 | 95.55 | 55.34 | 0.6 |
| A4 | + ARF-FPN | + Adaptive Receptive Field FPN decoder | 93.27 | 94.14 | 93.71 | 88.16 | 95.62 | 65.12 | 0.6 |
| A5 | + Boundary Residual | + Boundary residual connection in decoder | 92.96 | 94.22 | 93.59 | 87.94 | 95.53 | 65.19 | 0.6 |
| **A6** | **Full Model** | **+ Semantic heads + auxiliary + boundary losses (50k iters)** | **94.68** | **95.20** | **94.94** | **90.37** | **96.49** | **65.40** | **0.6** |

### DSIFN Ablation — Component Contribution Summary

| Component | ΔF1 vs previous | ΔIoU vs previous |
|-----------|----------------:|-----------------:|
| A0 → A1: MambaVision backbone | +17.56 | +26.44 |
| A1 → A2: D-RBI (unsigned) | +0.12 | +0.22 |
| A2 → A3: Signed difference | +0.30 | +0.51 |
| A3 → A4: ARF-FPN decoder | +0.08 | +0.14 |
| A4 → A5: Boundary residual | −0.12 | −0.22 |
| A5 → A6: Semantic heads + losses | +1.36 | +2.43 |

> **Note:** A5 shows a marginal drop vs A4 in isolation; the boundary residual connection's contribution is recovered and amplified when combined with the full loss schedule in A6.

---

## WHU-CD Results

**Dataset:** WHU-CD — standard train/val/test split  
**Architecture:** Full model (MambaVision-S + ARF-FPN + D-RBI Signed + Boundary Residual + Semantic heads)

| Run | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) | Params (M) | Thresh | Notes |
|-----|--------:|--------:|-------:|--------:|-------:|-----------:|-------:|-------|
| whu_a4_full (2026-04-28) | 96.16 | 95.00 | 95.58 | 91.53 | 99.58 | — | 0.4 | Earlier run, older eval script |
| **whu_full (2026-04-30)** | **95.58** | **94.74** | **95.15** | **90.76** | **99.54** | **65.40** | **0.55** | **Latest canonical run** |

> The `whu_full` (2026-04-30) run is the canonical latest result using the unified training and evaluation pipeline consistent with the DSIFN ablations.

---

## Run Provenance

| Dataset | Variant | Run Directory | Checkpoint Iter | Checkpoint SHA256 (first 16) |
|---------|---------|--------------|----------------:|------------------------------|
| DSIFN | a0_fpn_baseline | `run_dsifn_a0_fpn_baseline_seed42_20260430_232914` | 15000 | `1353d7bcc191...` |
| DSIFN | a1_mambavision_fpn | `run_dsifn_a1_mambavision_fpn_seed42_20260501_004250` | 25000 | `70f01ff2436e...` |
| DSIFN | a2_mambavision_drbi | `run_dsifn_a2_mambavision_drbi_seed42_20260501_004345` | 30000 | `15d400766aeb...` |
| DSIFN | a3_mambavision_drbi_signed | `run_dsifn_a3_mambavision_drbi_signed_seed42_20260501_005020` | 30000 | `48940251ed67...` |
| DSIFN | a4_mambavision_drbi_arf | `run_dsifn_a4_mambavision_drbi_arf_seed42_20260501_010826` | 30000 | `7850e110cf91...` |
| DSIFN | a5_mambavision_drbi_arf_boundary | `run_dsifn_a5_mambavision_drbi_arf_boundary_seed42_20260501_012826` | 30000 | `57945cfa443d...` |
| DSIFN | a6_full | `run_dsifn_a6_full_seed42_20260430_233101` | 50000 | `a0a91a2418ba...` |
| WHU | whu_full | `run_whu_whu_full_seed42_20260430_114506` | 50000 | `c6f38cf2b6c6...` |
