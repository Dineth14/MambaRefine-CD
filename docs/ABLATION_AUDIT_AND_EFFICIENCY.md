# Ablation Audit And Efficiency

This repository is active for binary change detection on DSIFN-CD and WHU-CD.
Do not use ablation results in the paper until the audit passes and results are
verified from checkpoints.

## Active Metrics

Only these paper metrics should be reported:

- Pre
- Rec
- F1
- IoU
- OA

`F1` is the change-class F1. It is not macro F1 or mean F1.

## Ablation Intent

- `a0_fpn_baseline`: simple CNN encoder and simple FPN decoder.
- `a1_mambavision_fpn`: shared MambaVision encoder with simple FPN decoder.
- `a2_mambavision_drbi`: adds D-RBI with absolute difference, no signed diff.
- `a3_mambavision_drbi_signed`: adds signed temporal difference.
- `a4_mambavision_drbi_arf`: replaces simple decoder with adaptive RF decoder.
- `a5_mambavision_drbi_arf_boundary`: adds boundary residual refinement.
- `a6_full`: full model with CRAM-lite, coarse loss, and boundary loss.

The current active ablation configs are under:

```bash
configs/ablations/dsifn/
```

There is currently no active `configs/ablations/whu/` directory.

## Audit Before Training

Run:

```bash
python scripts/audit_ablations.py --config_dir configs/ablations/dsifn
```

The script writes:

```bash
outputs/ablation_audit_dsifn.csv
```

It records fusion terms, D-RBI gates, decoder type, boundary residual status,
loss terms, parameter count, dummy forward shape, and pass/fail status.

For a 256x256 dummy forward:

```bash
python scripts/audit_ablations.py --config_dir configs/ablations/dsifn --batch_size 2 --image_size 256
```

## Metric Correctness Check

Run:

```bash
python scripts/check_binary_metrics.py
```

The script uses a tiny hand-computed confusion matrix and prints TP, FP, TN,
FN, Pre, Rec, F1, IoU, OA, and PASS/FAIL.

## Compare Ablation Results

After training/testing exports CSV summaries, run:

```bash
python scripts/compare_ablation_results.py --csv outputs/ablation_dsifn_summary.csv --full_variant A0
```

Outputs:

```bash
outputs/ablation_effectiveness_report.csv
outputs/ablation_effectiveness_report.md
```

Interpretation flags are heuristic. If a removed component improves F1 or has a
very small delta, re-run seeds before drawing conclusions.

## AMP

AMP is controlled by:

```yaml
efficiency:
  amp: true
```

The config loader maps this to `hardware.mixed_precision`. Training uses
`torch.amp.GradScaler`; evaluation uses autocast when AMP is enabled. If AMP
training produces NaN/Inf, training raises an error with config path,
iteration, loss, AMP state, and batch index.

## Gradient Checkpointing

Gradient checkpointing is opt-in:

```yaml
efficiency:
  gradient_checkpointing: false
```

It lowers activation memory at the cost of additional compute. The current
implementation applies checkpointing only to known safe stage-style modules.
It is disabled during inference/test.

## Lightweight Configs

Run:

```bash
python scripts/train.py --config configs/experiments/dsifn_lightweight.yaml
python scripts/test.py --config configs/experiments/dsifn_lightweight.yaml --ckpt <checkpoint>
```

WHU:

```bash
python scripts/train.py --config configs/experiments/whu_lightweight.yaml
python scripts/test.py --config configs/experiments/whu_lightweight.yaml --ckpt <checkpoint>
```

Lightweight configs keep the shared MambaVision encoder, binary output, binary
loss family, dataset split, crop size, training schedule, and metric
computation. They reduce fusion/decoder channels and use fewer adaptive RF
dilation branches.

## Memory Logs

Memory is measured with PyTorch CUDA counters:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

Training writes:

```bash
<run_dir>/memory_summary.json
```

Test and final evaluation results include:

- `peak_test_mem_GB`
- `params_M`

Training summary includes:

- `peak_train_mem_GB`
- `peak_val_mem_GB`
- `batch_size`
- `crop_size`
- `amp_enabled`
- `gradient_checkpointing_enabled`
- `params_M`

These numbers are per process and per current CUDA device. They are not
`nvidia-smi` totals.
