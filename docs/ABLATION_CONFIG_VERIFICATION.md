# Ablation Config Verification

Run:

```bash
python tools/verify_ablation_configs.py
```

The script loads every `configs/ablations/levir/*.yaml`, builds the model, checks module presence/absence, runs a dummy forward pass, verifies output shape, prints parameter breakdowns, and rewrites this report with measured parameter counts.

## Expected Switch Matrix

| Config | Encoder | Decoder | D-RBI | Signed Diff | CRAM-lite | ARF-FPN | Boundary Refine | Boundary Loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `a0_fpn_baseline.yaml` | simple_cnn | baseline | false | false | false | false | false | false |
| `a1_mambavision_fpn.yaml` | mambavision | baseline | false | false | false | false | false | false |
| `a2_mambavision_drbi.yaml` | mambavision | baseline | true | false | false | false | false | false |
| `a3_mambavision_drbi_signed.yaml` | mambavision | baseline | true | true | false | false | false | false |
| `a4_mambavision_drbi_arf.yaml` | mambavision | adaptive_rf | true | true | false | true | false | false |
| `a5_mambavision_drbi_arf_boundary.yaml` | mambavision | adaptive_rf | true | true | false | true | true | false |
| `a6_full.yaml` | mambavision | adaptive_rf | true | true | true | true | true | true |

## Current Known Issue Fixed

The previous LEVIR ablation configs did not isolate modules cleanly. The old baseline still enabled MambaVision, D-RBI, adaptive RF decoding, and boundary residual refinement. The global loss type also remained active unless nested ablation loss settings were normalized. The new configs and runtime checks prevent those silent overlaps.

## Expected Parameter Counts

The counts below are calculated from the configured module definitions. The verification script recomputes them from instantiated PyTorch modules and should be treated as the authoritative runtime check.

| Config | Total | Encoder | Decoder | D-RBI | ARF | CRAM-lite | Boundary Refine |
|---|---:|---:|---:|---:|---:|---:|---:|
| `a0_fpn_baseline.yaml` | 7,837,505 | 4,687,296 | 3,150,209 | 0 | 0 | 0 | 0 |
| `a1_mambavision_fpn.yaml` | 101,327,017 | 97,685,288 | 3,641,729 | 0 | 0 | 0 | 0 |
| `a2_mambavision_drbi.yaml` | 102,890,665 | 97,685,288 | 3,182,977 | 2,022,400 | 0 | 0 | 0 |
| `a3_mambavision_drbi_signed.yaml` | 103,382,185 | 97,685,288 | 3,182,977 | 2,513,920 | 0 | 0 | 0 |
| `a4_mambavision_drbi_arf.yaml` | 113,158,585 | 97,685,288 | 12,959,377 | 2,513,920 | 9,512,208 | 0 | 0 |
| `a5_mambavision_drbi_arf_boundary.yaml` | 113,227,724 | 97,685,288 | 13,028,516 | 2,513,920 | 9,512,208 | 0 | 69,139 |
| `a6_full.yaml` | 113,435,090 | 97,685,288 | 13,028,516 | 2,513,920 | 9,512,208 | 207,366 | 69,139 |
