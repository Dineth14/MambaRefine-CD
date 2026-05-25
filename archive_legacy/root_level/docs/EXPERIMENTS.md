# Experiments

**Updated:** 2026-05-02 — Rebuilt from actual run outputs, configs, and ablation traces.

---

## 1. Active Datasets

### DSIFN-CD
- **Root:** `/storage2/ChangeDetection/MV/Datasets/DSIFN-CD/DSIFN`
- **Split directory:** `splits/` (hash-locked; integrity verified on every run)
- **Split sizes:** 2758 train / 394 val / 789 test images
- **Test tiles:** 3156 (256×256 patches, 0.25 overlap)
- **Split hashes:**
  - train: `984a678c19b035bc...`
  - val: `3fdbaf77e6fa86f2...`
  - test: `ec436aa88e9571f0...`
- **Integrity:** `PASS` verified on all runs (`old_leakage_protocol_used: false`)

### WHU-CD
- **Root:** `/storage2/ChangeDetection/MV/Datasets/WHU-CD`
- **Split:** standard train/val/test

---

## 2. Training Protocol

All runs share this protocol unless noted in the per-run table:

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| LR schedule | Cosine decay |
| Warmup iterations | 2500 |
| Weight decay | 0.01 |
| Gradient clip (norm) | 0.5 |
| Batch size | 8 |
| Image size | 256×256 |
| Mixed precision | AMP (torch.amp, BF16/FP16) |
| EMA decay | 0.999 |
| Augmentation | Horizontal flip + vertical flip |
| Checkpoint metric | F1 (max, EMA weights) |
| Val frequency | every 5000 iters |
| Threshold sweep | [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60] |
| Inference | Patch-based, crop=256, overlap=0.25 |

Ablation runs A1–A5 use 30 000 iterations. Full model (A6) and baseline (A0) use 50 000 iterations.

---

## 3. Model Configuration per Ablation

All ablation runs use `model.mode: dual` → `DRBISiameseMambaCD`.  
Config files are in `configs/ablations/dsifn/`.

| ID | Config file | Backbone | D-RBI enabled | Signed diff | ARF-FPN decoder | Boundary residual | CRAMLite | Coarse loss (w) | Boundary loss (w) |
|----|-------------|----------|:-------------:|:-----------:|:---------------:|:-----------------:|:--------:|:---------------:|:-----------------:|
| A0 | `a0_fpn_baseline.yaml` | simple_cnn [64,128,256,512] | ✗ | ✗ | ✗ (baseline) | ✗ | ✗ | 0 | 0 |
| A1 | `a1_mambavision_fpn.yaml` | MambaVision-S | ✗ | ✗ | ✗ (baseline) | ✗ | ✗ | 0 | 0 |
| A2 | `a2_mambavision_drbi.yaml` | MambaVision-S | ✓ (absdiff only) | ✗ | ✗ (baseline) | ✗ | ✗ | 0 | 0 |
| A3 | `a3_mambavision_drbi_signed.yaml` | MambaVision-S | ✓ | ✓ | ✗ (baseline) | ✗ | ✗ | 0 | 0 |
| A4 | `a4_mambavision_drbi_arf.yaml` | MambaVision-S | ✓ | ✓ | ✓ (no bnd_res) | ✗ | ✗ | 0 | 0 |
| A5 | `a5_mambavision_drbi_arf_boundary.yaml` | MambaVision-S | ✓ | ✓ | ✓ | ✓ | ✗ | 0 | 0 |
| A6 | `a6_full.yaml` | MambaVision-T¹ | ✓ | ✓ | ✓ | ✓ | ✓ (stages 0,1,2) | 0.4 | 0.1 |

> ¹ The `a6_full.yaml` config specifies `variant: tiny` (MambaVision-T). The two canonical A6 runs (20260430_105918 and 20260430_233101) confirmed `backbone_name: small` in their ablation traces, indicating the variant was overridden at launch via a CLI argument or environment. The experiment-level backbone variant comparison runs (tiny/small/base) explicitly set the variant in their launch commands.
>
> **D-RBI parameters (all D-RBI runs):** `out_channels=256`, `gate_hidden_ratio=0.25`, `region_gate=[0.2, 0.8]`, `boundary_gate=[0.0, 0.4]`, `pre_norm=True`, `use_depthwise=True`, `use_product=False`.  
> **ARF-FPN dilation rates (all ARF runs):** `[1, 2, 4, 8]`, `residual_scale=0.1`, `aux_weight=0.4` (A6 only).  
> **CRAMLite (A6):** `alpha_init=0.5`, `apply_stages=[0, 1, 2]`, `attention_type=spatial`.  
> **Boundary loss (A6):** `type=bce_dice`, `target_type=sobel`, `edge_width=3`.

---

## 4. WHU-CD Full Model Configuration

Config: `configs/experiments/whu_full.yaml`

| Setting | Value |
|---------|-------|
| Backbone | MambaVision-S (`variant: small`) |
| D-RBI | ✓ (absdiff + signed diff) |
| ARF-FPN | ✓ (dilation [1,2,4,8]) |
| Boundary residual | ✓ |
| CRAMLite | ✓ (stages [0,1,2], α=0.5) |
| Coarse loss weight | 0.4 |
| Boundary loss weight | 0.1 |
| Max iterations | 50 000 |
| LR | 5e-5 |
| Warmup | 2500 iters |
| Val every | 5000 iters |
| skip_nan_steps | true |

Identical to A6 full model except backbone is explicitly `small` and `boundary_refine.enabled: true` in the model stanza.

---

## 5. All Completed Runs — DSIFN-CD

### A0: FPN Baseline (SimpleCNN + Baseline Decoder)

| Run | Date | Iters at ckpt | Pre | Rec | F1 | IoU | OA | Thresh |
|-----|------|:-------------:|----:|----:|---:|----:|---:|-------:|
| run_dsifn_a0_fpn_baseline_seed42_20260430_105337 | 2026-04-30 | 35 000 | 80.70 | 74.75 | 77.61 | 63.41 | 84.89 | 0.30 |
| run_dsifn_a0_fpn_baseline_seed42_20260430_232914 | 2026-04-30 | 15 000 | 73.85 | 77.54 | 75.65 | 60.84 | 82.71 | 0.30 |
| **Average** | | | **77.27** | **76.15** | **76.63** | **62.12** | **83.80** | |

Note: A0 run 1 stopped early at iter 35k (higher F1 than the 15k run, likely due to different warm start or checkpoint selection). Run 2 stopped at iter 15k. Both ran for up to 50k configured iterations.

### A1: MambaVision-S + Baseline FPN Decoder

| Run | Date | Iters at ckpt | Pre | Rec | F1 | IoU | OA | Thresh |
|-----|------|:-------------:|----:|----:|---:|----:|---:|-------:|
| run_dsifn_a1_mambavision_fpn_seed42_20260430_160816 | 2026-04-30 | 30 000 | 94.25 | 94.80 | 94.53 | 89.62 | 96.15 | 0.50 |
| run_dsifn_a1_mambavision_fpn_seed42_20260501_004250 | 2026-05-01 | 25 000 | 93.04 | 93.38 | 93.21 | 87.28 | 95.29 | 0.50 |
| **Average** | | | **93.65** | **94.09** | **93.87** | **88.45** | **95.72** | |

### A2: + D-RBI (unsigned diff + gates)

| Run | Date | Iters at ckpt | Pre | Rec | F1 | IoU | OA | Thresh |
|-----|------|:-------------:|----:|----:|---:|----:|---:|-------:|
| run_dsifn_a2_mambavision_drbi_seed42_20260430_172810 | 2026-04-30 | 30 000 | — | — | — | — | — | — |
| run_dsifn_a2_mambavision_drbi_seed42_20260501_004345 | 2026-05-01 | 30 000 | 92.70 | 93.98 | 93.33 | 87.50 | 95.35 | 0.60 |

> Run 20260430_172810 used `backbone=base` (102.89M) — wrong config at launch. Excluded from A2 average. Only the small-backbone run is reported.

### A3: + Signed Temporal Difference in D-RBI

| Run | Date | Iters at ckpt | Pre | Rec | F1 | IoU | OA | Thresh |
|-----|------|:-------------:|----:|----:|---:|----:|---:|-------:|
| run_dsifn_a3_mambavision_drbi_signed_seed42_20260430_194044 | 2026-04-30 | 30 000 | 94.58 | 95.30 | 94.94 | 90.37 | 96.44 | 0.60 |
| run_dsifn_a3_mambavision_drbi_signed_seed42_20260501_005020 | 2026-05-01 | 30 000 | 92.81 | 94.46 | 93.63 | 88.02 | 95.55 | 0.60 |
| **Average** | | | **93.69** | **94.88** | **94.28** | **89.19** | **95.99** | |

### A4: + ARF-FPN Decoder (no boundary residual)

| Run | Date | Iters at ckpt | Pre | Rec | F1 | IoU | OA | Thresh |
|-----|------|:-------------:|----:|----:|---:|----:|---:|-------:|
| run_dsifn_a4_mambavision_drbi_arf_seed42_20260430_194139 | 2026-04-30 | 30 000 | 94.61 | 95.41 | 95.01 | 90.49 | 96.49 | 0.60 |
| run_dsifn_a4_mambavision_drbi_arf_seed42_20260501_010826 | 2026-05-01 | 30 000 | 93.27 | 94.14 | 93.71 | 88.16 | 95.62 | 0.60 |
| **Average** | | | **93.94** | **94.78** | **94.36** | **89.32** | **96.05** | |

### A5: + Boundary Residual Head (no CRAMLite, no full losses)

| Run | Date | Iters at ckpt | Pre | Rec | F1 | IoU | OA | Thresh |
|-----|------|:-------------:|----:|----:|---:|----:|---:|-------:|
| run_dsifn_a5_mambavision_drbi_arf_boundary_seed42_20260501_012826 | 2026-05-01 | 30 000 | 92.96 | 94.22 | 93.59 | 87.94 | 95.53 | 0.60 |

### A6: Full Model (CRAMLite + coarse/boundary losses, 50k iters)

| Run | Date | Backbone | bnd_res | Iters | Pre | Rec | F1 | IoU | OA | Params (M) | Thresh |
|-----|------|----------|:-------:|------:|----:|----:|---:|----:|---:|-----------:|-------:|
| run_dsifn_a6_full_seed42_20260430_105918 | 2026-04-30 | small | ✓ | 50 000 | 96.26 | 96.53 | **96.40** | 93.04 | 97.47 | 65.40 | 0.60 |
| run_dsifn_a6_full_seed42_20260430_233101 | 2026-04-30 | small | ✓ | 50 000 | 94.68 | 95.20 | 94.94 | 90.37 | 96.49 | 65.40 | 0.60 |
| run_dsifn_a6_full_seed42_20260501_095055 | 2026-05-01 | small | ✗¹ | 50 000 | 94.66 | 95.30 | 94.98 | 90.44 | 96.51 | 65.33 | 0.60 |
| run_dsifn_a6_full_seed42_20260501_095733 | 2026-05-01 | base | ✗¹ | 50 000 | 95.66 | 95.89 | 95.77 | 91.89 | 97.07 | 113.37 | 0.60 |
| run_dsifn_a6_full_seed42_20260501_095853 | 2026-05-01 | tiny | ✗¹ | 50 000 | 94.33 | 94.84 | 94.58 | 89.72 | 96.24 | 46.73 | 0.60 |
| **Canonical avg (small, bnd_res=✓)** | | | | | **95.47** | **95.87** | **95.67** | **91.71** | **96.98** | **65.40** | |

> ¹ Runs 095055, 095733, 095853 have `boundary_residual_enabled=False` in their ablation traces (config/code mismatch at launch time). These are used only for the backbone scaling comparison.

---

## 6. All Completed Runs — WHU-CD

| Run | Date | Backbone | Iters at ckpt | Pre | Rec | F1 | IoU | OA | Params (M) | Thresh | Notes |
|-----|------|----------|:-------------:|----:|----:|---:|----:|---:|-----------:|-------:|-------|
| run_20260428_023626_whu_a4_full_WHU-CD | 2026-04-28 | — | — | 96.16 | 95.00 | 95.58 | 91.53 | 99.58 | — | 0.40 | Old eval script |
| run_whu_whu_full_seed42_20260430_114409 | 2026-04-30 | base | — | — | — | — | — | — | 113.44 | — | No test results (incomplete run) |
| **run_whu_whu_full_seed42_20260430_114506** | **2026-04-30** | **small** | **50 000** | **95.58** | **94.74** | **95.15** | **90.76** | **99.54** | **65.40** | **0.55** | **Canonical** |

---

## 7. Run Commands

```bash
# DSIFN ablation runs
python scripts/train.py --config configs/ablations/dsifn/a0_fpn_baseline.yaml
python scripts/train.py --config configs/ablations/dsifn/a1_mambavision_fpn.yaml
python scripts/train.py --config configs/ablations/dsifn/a2_mambavision_drbi.yaml
python scripts/train.py --config configs/ablations/dsifn/a3_mambavision_drbi_signed.yaml
python scripts/train.py --config configs/ablations/dsifn/a4_mambavision_drbi_arf.yaml
python scripts/train.py --config configs/ablations/dsifn/a5_mambavision_drbi_arf_boundary.yaml
python scripts/train.py --config configs/ablations/dsifn/a6_full.yaml

# Full model experiments
python scripts/train.py --config configs/experiments/dsifn_full.yaml
python scripts/train.py --config configs/experiments/whu_full.yaml

# Test / eval
python scripts/test.py --config configs/ablations/dsifn/a6_full.yaml --ckpt outputs/dsifn/a6_full/<run>/checkpoints/best.pth
python scripts/test.py --config configs/experiments/whu_full.yaml --ckpt outputs/whu/full/<run>/checkpoints/best.pth
```

---

## 8. Checkpoint Provenance

| Dataset | Variant | Run (short) | SHA256 (first 16 chars) | Iter | Params (M) |
|---------|---------|-------------|------------------------|-----:|-----------:|
| DSIFN | a0_fpn_baseline | 20260430_232914 | `1353d7bcc191...` | 15k | 7.84 |
| DSIFN | a1_mambavision_fpn | 20260501_004250 | `70f01ff2436e...` | 25k | 53.54 |
| DSIFN | a2_mambavision_drbi | 20260501_004345 | `15d400766aeb...` | 30k | 54.98 |
| DSIFN | a3_mambavision_drbi_signed | 20260501_005020 | `48940251ed67...` | 30k | 55.34 |
| DSIFN | a4_mambavision_drbi_arf | 20260501_010826 | `7850e110cf91...` | 30k | 65.12 |
| DSIFN | a5_mambavision_drbi_arf_boundary | 20260501_012826 | `57945cfa443d...` | 30k | 65.19 |
| DSIFN | a6_full (best run) | 20260430_105918 | `a0a91a2418ba...`¹ | 50k | 65.40 |
| DSIFN | a6_full (run 2) | 20260430_233101 | `a0a91a2418ba...`¹ | 50k | 65.40 |
| WHU | whu_full | 20260430_114506 | `c6f38cf2b6c6...` | 50k | 65.40 |

> ¹ Both canonical A6 DSIFN runs share the same checkpoint SHA prefix — the run producing F1=96.40 (105918) is the best single checkpoint.

---

## 9. Notes on Anomalies

1. **A0 iteration counts:** Run 1 best checkpoint at iter 35k, run 2 at iter 15k (both configured for 50k). The lower early checkpoint in run 2 suggests an early plateau or validation-based early selection.

2. **A2 wrong backbone run:** `run_20260430_172810` has `backbone=base` (102.89M) in its ablation trace despite being in the `a2_mambavision_drbi` directory — the variant flag was not overridden back to small. This run is excluded from A2 analysis.

3. **A6 boundary_residual mismatch (095xxx runs):** Runs launched on 2026-05-01 for backbone scaling tests (`095055`, `095733`, `095853`) have `boundary_residual_enabled=False` in their runtime ablation traces, despite the config specifying `decoder.use_boundary_residual: true`. The models were likely loaded from a checkpoint or code state where the flag was not propagated. These runs are valid for backbone comparison but not as full-model canonical results.

4. **WHU base run incomplete:** `run_20260430_114409` (MambaVision-B, 113.4M) has no test results — training did not complete or the test step was not triggered.

5. **Threshold selection:** Test threshold is inherited from the best-F1 threshold found during validation. DSIFN ablation A0 consistently selects threshold=0.30 (model under-confident); A1+ select 0.50–0.60 (well-calibrated predictions).

