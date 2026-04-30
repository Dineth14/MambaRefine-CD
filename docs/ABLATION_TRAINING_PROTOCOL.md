# Ablation Training Protocol

This repository is binary change detection only. Report only `Pre`, `Rec`, `F1`, `IoU`, and `OA`; `F1` is the change-class F1.

## Stage 0: Audit Only

No training. Verify that configs resolve correctly, models are actually different, and each ablation forward path matches the intended component change.

```bash
python scripts/debug_config_resolution.py --config configs/ablations/dsifn/a2_mambavision_drbi.yaml
python scripts/debug_config_resolution.py --config configs/ablations/dsifn/a6_full.yaml
python scripts/audit_ablations.py --config_dir configs/ablations/dsifn
python scripts/compare_ablation_models.py --config_dir configs/ablations/dsifn
```

Required artifacts before trusting a run:

- `resolved_config.yaml`
- `ablation_trace.json`
- `forward_trace_first_batch.json` when `debug.ablation_trace: true`
- unique checkpoint path
- checkpoint SHA-256 in evaluation results

## Stage 1: Sanity Training

Train each ablation for 500-1000 iterations. The purpose is not paper results; it is to confirm that losses, output distributions, traces, and checkpoint hashes differ.

```bash
python scripts/quick_ablation_divergence_test.py \
  --config_dir configs/ablations/dsifn \
  --iters 1000 \
  --batch_size 2 \
  --image_size 256
```

## Stage 2: Screening Training

Train DSIFN ablations for 15k-20k iterations to screen which components matter. Use unique run directories and evaluate each ablation only with its own checkpoint.

## Stage 3: Final Paper Ablations

Train important DSIFN ablations for the full budget, usually 40k-50k iterations. Train only key WHU confirmation ablations unless you need a full second-dataset ablation table.

## Checkpoint Safety

Backbone pretrained weights are allowed. Resuming a full model checkpoint for an ablation is blocked by default. Only enable it intentionally:

```yaml
training:
  allow_resume_for_ablation: true
```

Do not use any ablation result in the paper unless:

- audit passes
- checkpoint hash is unique
- resolved config is saved
- ablation trace is saved
- forward trace confirms the intended path
- metrics are computed from the correct checkpoint

## Evaluation Safety

Evaluation writes results to a unique directory and records:

- config path
- variant name
- run directory
- checkpoint path
- checkpoint file size
- checkpoint SHA-256
- checkpoint iteration/best metric when available

If two variants use the same checkpoint hash, treat the table as invalid until the checkpoint paths are corrected.
