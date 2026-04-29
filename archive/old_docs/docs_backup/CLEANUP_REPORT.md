# MambaRefineCD Repository Cleanup Report

Generated: 2026-04-28

---

## Overview

This report documents the state of the MambaRefineCD repository after inspection,
organized by action category. Nothing is permanently deleted; unused scripts are
moved to `archive/unused_scripts/` for reference.

---

## Files to KEEP (core, actively used)

| File/Directory | Reason |
|---|---|
| `src/` | Core backend: models, training, data, utils |
| `src/models/cd_model.py` | Main model: SiameseMambaCD, DRBISiameseMambaCD, build_model |
| `src/models/modules/differential_region_boundary.py` | D-RBI module (updated: signed diff added) |
| `src/models/modules/temporal_mamba.py` | Retained but disabled; kept for checkpoint compat |
| `src/models/backbone/mambavision_builder.py` | MambaVision backbone builder |
| `src/models/decoders/adaptive_rf_decoder.py` | Primary decoder with boundary residual |
| `src/models/decoders/refinement_decoder.py` | Refinement decoder (boundary-guided) |
| `src/models/decoders/semantic_heads.py` | LightweightSemanticHead for SECOND |
| `src/training/trainer.py` | Training loop (iteration-based) |
| `src/training/evaluator.py` | Evaluation with threshold sweep + TTA |
| `src/training/losses.py` | Binary CD losses (BCE, Dice, Focal, SEK) |
| `src/training/second_loss.py` | SECOND semantic change detection loss |
| `src/training/sek_loss.py` | SeK surrogate loss |
| `src/training/metrics.py` | StreamingMetrics (binary CD) |
| `src/training/second_metrics.py` | MambaFCS-aligned SECOND metrics |
| `src/training/checkpoint.py` | Checkpoint save/load |
| `src/training/ema.py` | Exponential Moving Average |
| `src/training/logger.py` | Table logging utilities |
| `src/training/model_outputs.py` | Output normalization |
| `src/training/tta.py` | Test-time augmentation |
| `src/training/boundary_metrics.py` | Boundary / edge IoU metrics |
| `src/training/pipeline.py` | Full training pipeline entry |
| `src/data/levircd.py` | LEVIR-CD dataset |
| `src/data/whucd.py` | WHU-CD dataset |
| `src/data/dsifncd.py` | DSIFN-CD dataset |
| `src/data/second.py` | SECOND SCD dataset |
| `src/data/transforms.py` | Data augmentation transforms |
| `src/data/sampler.py` | Balanced change sampler |
| `src/data/dataset_builder.py` | Dataset factory |
| `src/data/factory.py` | DataLoader factory |
| `src/utils/` | Visualization, misc utilities |
| `configs/train/levir_cd.yaml` | LEVIR-CD training config (kept, not replaced) |
| `configs/train/whu_cd.yaml` | WHU-CD training config |
| `configs/train/dsifn_cd.yaml` | DSIFN-CD training config |
| `configs/train/second_semantic.yaml` | SECOND SCD training config |
| `configs/ablation_config.yaml` | Ablation config (kept as reference) |
| `configs/global_config.yaml` | Global experiment config |
| `requirements.txt` | Python dependencies |
| `infer.py` | Root-level inference script |
| `sek_before_fix/` | Historical SeK implementation before fix |

---

## Files to MODIFY (updated in this pass)

| File | Change |
|---|---|
| `src/models/modules/differential_region_boundary.py` | Added `use_signed_diff` parameter |
| `src/models/cd_model.py` | Wire `use_signed_diff` from config into D-RBI |

---

## New Files ADDED (organized interface layer)

| File | Purpose |
|---|---|
| `metrics/binary_cd_metrics.py` | Clean standalone binary CD metrics (Pre/Rec/F1/IoU/OA) |
| `metrics/second_scd_metrics.py` | Clean standalone SECOND SCD metrics (OA/mIoU/SeK/Fscd) |
| `datasets/levir.py` | Thin wrapper over src/data/levircd.py |
| `datasets/whu.py` | Thin wrapper over src/data/whucd.py |
| `datasets/dsifn.py` | Thin wrapper over src/data/dsifncd.py |
| `datasets/second.py` | Thin wrapper over src/data/second.py |
| `datasets/samplers.py` | Change-ratio balanced sampler |
| `datasets/transforms.py` | Re-exports from src/data/transforms.py |
| `losses/binary_losses.py` | BCE+Dice+Focal for binary CD |
| `losses/boundary_loss.py` | Sobel/morph boundary supervised loss |
| `losses/semantic_losses.py` | CE+Dice for SECOND |
| `losses/sek_loss.py` | SeK differentiable loss |
| `models/mambarefinecd.py` | Clean model entry point |
| `models/modules/drbi.py` | Re-exports DifferentialRegionBoundaryInteraction |
| `models/modules/arf_fpn.py` | Re-exports AdaptiveRFDecoder |
| `models/modules/cram_lite.py` | NEW: CRAMLite attention module |
| `models/modules/boundary_refine.py` | Re-exports RefinementDecoder |
| `models/modules/semantic_head.py` | Re-exports LightweightSemanticHead |
| `configs/datasets/levir.yaml` | Organized dataset config |
| `configs/datasets/whu.yaml` | Organized dataset config |
| `configs/datasets/dsifn.yaml` | Organized dataset config |
| `configs/datasets/second.yaml` | Organized dataset config |
| `configs/models/mambarefinecd_base.yaml` | Base model config |
| `configs/models/mambarefinecd_full.yaml` | Full model config (all features on) |
| `configs/models/mambarefinecd_light.yaml` | Lightweight model config |
| `configs/ablations/levir/a{0..4}_*.yaml` | LEVIR ablation configs |
| `configs/ablations/whu/a{0..4}_*.yaml` | WHU ablation configs |
| `configs/ablations/dsifn/a{0..4}_*.yaml` | DSIFN ablation configs |
| `configs/ablations/second/a{0..4}_*.yaml` | SECOND ablation configs |
| `scripts/train.py` | Clean training script (replaces old) |
| `scripts/evaluate.py` | Clean evaluation script |
| `scripts/test.py` | Test on held-out test set |
| `scripts/infer.py` | Inference on image pairs |
| `scripts/count_params_flops.py` | Model FLOPs/Params counter |
| `scripts/prepare_levir_balanced_splits.py` | LEVIR balanced split analysis |
| `tools/validate_dataset.py` | Dataset integrity check |
| `tools/verify_metrics.py` | Metric sanity checker |
| `tools/export_results_table.py` | Export results as CSV table |
| `tests/test_binary_metrics.py` | Unit tests for binary metrics |
| `tests/test_second_metrics.py` | Unit tests for SECOND metrics |
| `tests/test_dataset_loading.py` | Dataset loading smoke tests |
| `docs/METRICS.md` | Metrics documentation with formulas |
| `docs/RUNNING_EXPERIMENTS.md` | How to run all experiments |

---

## Files ARCHIVED (moved to archive/unused_scripts/)

These scripts are not part of the core train/evaluate/test workflow.
They belong to SOTA comparison, website generation, one-time diagnostics, or debug sessions.

| Script | Reason |
|---|---|
| `scripts/benchmark_all.py` | SOTA benchmark runner, not part of main workflow |
| `scripts/check_dataset.py` | One-time check, replaced by tools/validate_dataset.py |
| `scripts/check_training_validation.py` | Debug utility, not needed in clean repo |
| `scripts/clone_sota_repos.py` | SOTA reproduction, archived |
| `scripts/collect_website_qualitative.py` | Website content generation |
| `scripts/compare_runs.py` | Ad-hoc run comparison |
| `scripts/debug_gpu_memory.py` | Debug utility |
| `scripts/discover_sota_checkpoints.py` | SOTA reproduction |
| `scripts/download_sota_weights.py` | SOTA reproduction |
| `scripts/evaluate_sota_models.py` | SOTA reproduction |
| `scripts/extract_ablation_results.py` | Post-hoc extraction, replaced by tools/ |
| `scripts/extract_all_results_for_website.py` | Website generation |
| `scripts/extract_external_sota_results.py` | SOTA extraction |
| `scripts/extract_mambacd_protocol_results.py` | SOTA protocol |
| `scripts/inspect_dataset_structure.py` | One-time, replaced by tools/validate_dataset.py |
| `scripts/model_efficiency.py` | Replaced by scripts/count_params_flops.py |
| `scripts/plot_ablation.py` | Post-hoc visualization |
| `scripts/precompute_second_masks.py` | One-time utility |
| `scripts/profile_second_speed.py` | One-time profiling |
| `scripts/run_ablation.py` | Old ablation runner (replaced by configs + scripts/train.py) |
| `scripts/run_sota_reproduction_pipeline.py` | SOTA reproduction |
| `scripts/validate_ablation.py` | Old validation |
| `scripts/validate_mambacd_protocol_tables.py` | SOTA protocol |
| `scripts/validate_website.py` | Website validation |
| `scripts/write_sota_tables.py` | SOTA table generation |

---

## Files NOT TOUCHED

| File | Reason |
|---|---|
| `sek_before_fix/` | Historical reference, do not modify |
| `outputs/` | Generated outputs, not tracked |
| `results/` | Generated results, not tracked |
| `website/` | Static website content, not part of training |
| `.git/` | Version control, never modify |
| `.gitignore` | Version control settings |

---

## Structural Notes

1. The `src/` directory remains the **backend** for all model, training, and data logic.
2. The new top-level directories (`metrics/`, `datasets/`, `losses/`, `models/`, `scripts/`, `tools/`, `tests/`) are the **clean interface layer**.
3. The new `scripts/` scripts import from both the new interface layer and `src/` (via PYTHONPATH).
4. All metric restrictions are enforced:
   - LEVIR-CD / WHU-CD / DSIFN-CD → only `Pre`, `Rec`, `F1`, `IoU`, `OA`
   - SECOND → only `OA`, `mIoU`, `SeK`, `Fscd`
