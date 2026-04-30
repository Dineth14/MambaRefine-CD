# DSIFN-CD Small-Variant Ablation Report

Generated: 2026-04-30

This report summarizes the DSIFN-CD ablation runs found under `outputs/dsifn/`.
Metrics are final test metrics unless explicitly marked as validation-only.
All reported test metrics use the binary change-detection paper metrics:
Precision, Recall, F1, IoU, and OA.

## Scope

The requested scope is DSIFN ablations trained with the MambaVision `small`
variant. The completed final-test runs show that most relevant MambaVision
ablations are `small`, but two rows are not directly small-variant comparable:

- `a2_mambavision_drbi` was run with MambaVision `base`, not `small`.
- `a5_mambavision_drbi_arf_boundary` was run with MambaVision `base` and does
  not have final test metrics in the current output folder.

The table keeps those rows visible and labels them clearly so the ablation
sequence is not misread.

## Final Test Results

| Ablation | Backbone Variant | Key Modules | Params (M) | Threshold | Pre | Rec | F1 | IoU | OA | EMA |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `a0_fpn_baseline` | baseline CNN | baseline FPN | 7.8375 | 0.30 | 80.6968 | 74.7528 | 77.6111 | 63.4136 | 84.8945 | yes |
| `a1_mambavision_fpn` | small | MambaVision + baseline FPN | 53.5366 | 0.50 | 94.2543 | 94.7987 | 94.5257 | 89.6197 | 96.1542 | yes |
| `a2_mambavision_drbi` | base | MambaVision + D-RBI | 102.8907 | 0.60 | 95.7163 | 95.9525 | 95.8342 | 92.0017 | 97.0783 | yes |
| `a3_mambavision_drbi_signed` | small | MambaVision + D-RBI + signed diff | 55.3440 | 0.60 | 94.5795 | 95.3048 | 94.9407 | 90.3688 | 96.4425 | yes |
| `a4_mambavision_drbi_arf` | small | MambaVision + D-RBI + signed diff + ARF | 65.1204 | 0.60 | 94.6117 | 95.4065 | 95.0074 | 90.4896 | 96.4881 | yes |
| `a6_full` | small | MambaVision + D-RBI + signed diff + ARF + CRAM-lite + boundary refinement/loss | 65.3969 | 0.60 | 96.2591 | 96.5340 | 96.3963 | 93.0434 | 97.4721 | yes |

## Validation-Only / Incomplete Run

| Ablation | Backbone Variant | Status | Val Pre | Val Rec | Val F1 | Val IoU | Val OA | Best Threshold |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `a5_mambavision_drbi_arf_boundary` | base | no `test_results/` found | 94.3023 | 95.3664 | 94.8313 | 90.1708 | 96.3216 | 0.60 |

`a5` should be tested before using it in the final ablation table. It should
also be rerun with `model.variant: small` if the goal is a strictly
small-variant DSIFN ablation study.

## Small-Variant Comparable Subset

For strict small-variant comparison, use `a1`, `a3`, `a4`, and `a6`.
The CNN baseline `a0` is useful as a lower-capacity reference, but it is not a
MambaVision-small model. The current `a2` result is informative, but it is
not directly comparable because it uses the larger `base` backbone.

| Ablation | F1 | IoU | Delta F1 vs `a1` | Delta IoU vs `a1` |
|---|---:|---:|---:|---:|
| `a1_mambavision_fpn` | 94.5257 | 89.6197 | 0.0000 | 0.0000 |
| `a3_mambavision_drbi_signed` | 94.9407 | 90.3688 | +0.4150 | +0.7491 |
| `a4_mambavision_drbi_arf` | 95.0074 | 90.4896 | +0.4817 | +0.8699 |
| `a6_full` | 96.3963 | 93.0434 | +1.8706 | +3.4237 |

## Run Paths

| Ablation | Run Directory |
|---|---|
| `a0_fpn_baseline` | `outputs/dsifn/a0_fpn_baseline/run_dsifn_a0_fpn_baseline_seed42_20260430_105337` |
| `a1_mambavision_fpn` | `outputs/dsifn/a1_mambavision_fpn/run_dsifn_a1_mambavision_fpn_seed42_20260430_160816` |
| `a2_mambavision_drbi` | `outputs/dsifn/a2_mambavision_drbi/run_dsifn_a2_mambavision_drbi_seed42_20260430_172810` |
| `a3_mambavision_drbi_signed` | `outputs/dsifn/a3_mambavision_drbi_signed/run_dsifn_a3_mambavision_drbi_signed_seed42_20260430_194044` |
| `a4_mambavision_drbi_arf` | `outputs/dsifn/a4_mambavision_drbi_arf/run_dsifn_a4_mambavision_drbi_arf_seed42_20260430_194139` |
| `a5_mambavision_drbi_arf_boundary` | `outputs/dsifn/a5_mambavision_drbi_arf_boundary/run_20260429_183105_a5_mambavision_drbi_arf_boundary_DSIFN-CD` |
| `a6_full` | `outputs/dsifn/a6_full/run_dsifn_a6_full_seed42_20260430_105918` |

## Observations

1. Replacing the CNN baseline with MambaVision-small plus baseline FPN gives a
   large improvement: F1 rises from 77.6111 to 94.5257.
2. In the strict small-variant subset, adding signed D-RBI improves F1 from
   94.5257 to 94.9407.
3. Adding ARF gives a small additional gain over signed D-RBI: F1 increases
   from 94.9407 to 95.0074.
4. The full small model gives the best completed small-variant result:
   F1 96.3963 and IoU 93.0434.
5. The current `a2` and `a5` rows use the larger `base` variant and should not
   be mixed into a strictly small-variant ablation claim without rerunning them
   as `small`.

## Recommended Next Steps

1. Rerun `a2_mambavision_drbi` with `model.variant: small`.
2. Rerun or test `a5_mambavision_drbi_arf_boundary` with `model.variant: small`.
3. Use the small-variant comparable subset table for the paper until all rows
   have matching backbone variants and final test metrics.
