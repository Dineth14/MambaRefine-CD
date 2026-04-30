# Training Speed And GPU Efficiency

This document covers safe profiling and efficiency options for active binary
change detection on DSIFN-CD and WHU-CD.

The verified full configs are still the reference configs. Fast configs are
efficiency variants, not verified full-model results.

## Profiler Mode

Run a short profiler pass and exit:

```bash
python scripts/train.py --config configs/experiments/dsifn_full.yaml --profile
```

Config:

```yaml
profiling:
  enabled: false
  warmup_iters: 20
  profile_iters: 100
  log_interval: 10
```

Profiler output:

```bash
<run_dir>/profiling_summary.json
<run_dir>/profiling_summary.csv
```

It times data loading, host-to-device transfer, forward, loss, backward,
optimizer, validation if triggered, total iteration time, samples/sec, and peak
GPU memory. CUDA synchronization is used around timing regions.

## Interpreting Bottlenecks

- High data-loading percentage: increase workers, use pinned memory, inspect
  image/mask decode cost.
- High H2D percentage: verify pinned memory and `non_blocking=True`.
- High forward percentage: backbone, D-RBI, adaptive RF decoder, and boundary
  refinement are likely contributors.
- High backward percentage: use AMP, test checkpointing, or reduce batch size.
- High validation/logging percentage: validate less often and avoid saving
  predictions/visualizations.

If data loading is bottlenecked and `num_workers=0`, the profiler logs a
warning.

## Recommended DataLoader Settings

```yaml
dataloader:
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
  drop_last: true
```

`persistent_workers` and `prefetch_factor` are used only when
`num_workers > 0`.

## AMP

```yaml
efficiency:
  amp: true
  amp_dtype: fp16
```

`amp_dtype: bf16` is supported when the installed CUDA/PyTorch stack reports
BF16 support. Training uses `GradScaler`; eval also uses autocast when AMP is
enabled. Non-finite AMP losses stop safely with config path, iteration, loss,
AMP status, and loss stats.

Warning: Mamba/selective-scan kernels may need FP32 wrapping depending on the
installed implementation. Test before using AMP results in a paper.

## Channels Last

```yaml
efficiency:
  channels_last: false
```

When enabled, the model and image tensors use channels-last memory format.
Masks remain in normal tensor format. If conversion fails in the training
pipeline, it is disabled with a warning.

Keep this off by default until validated on your GPU and MambaVision build.

## Gradient Checkpointing

```yaml
efficiency:
  gradient_checkpointing: false
```

Checkpointing reduces activation memory but can slow training. It is disabled
during eval/test. The current implementation applies it conservatively only to
known safe stage-style modules.

## Torch Compile

```yaml
efficiency:
  compile: false
  compile_mode: reduce-overhead
```

When enabled and supported by PyTorch, the model is compiled after model
inspection/logging and before optimizer creation. Compile failures fall back to
eager mode. First iterations can be slower due to compilation.

## Fast Configs

Fast configs are practical efficiency variants:

```bash
python scripts/train.py --config configs/experiments/dsifn_fast.yaml
python scripts/train.py --config configs/experiments/whu_fast.yaml
```

They keep:

- shared MambaVision encoder
- binary output
- D-RBI
- boundary residual refinement
- binary BCE + Dice loss family
- dataset splits
- Pre, Rec, F1, IoU, OA metrics

They reduce fusion/decoder channels and adaptive RF dilation branches. They are
not the verified full model.

## Benchmark Full Vs Fast

```bash
python scripts/benchmark_training_speed.py \
  --configs configs/experiments/dsifn_full.yaml configs/experiments/dsifn_fast.yaml \
  --iters 200 \
  --warmup 20
```

Output:

```bash
outputs/training_speed_benchmark.csv
```

Columns include config, batch size, image size, AMP, channels-last,
checkpointing, compile, fast mode, average iteration time, samples/sec, peak
GPU memory, parameter count, status, and notes.

## Find Max Batch Size

```bash
python scripts/find_max_batch_size.py \
  --config configs/experiments/dsifn_full.yaml \
  --start_batch 2 \
  --max_batch 16 \
  --image_size 256
```

Output:

```bash
outputs/max_batch_size_report.csv
```

The script uses synthetic tensors, catches CUDA OOM, clears cache after OOM,
and does not write checkpoints.

## Validation And Checkpoint Efficiency

Defaults are intended to avoid unnecessary overhead:

```yaml
training:
  validate_every: 5000

checkpoint:
  save_last: true
  save_best: true
  save_every: null

eval:
  memory_efficient: true
  save_predictions: false
  save_visualizations: false
```

Validation/test uses streaming metrics when threshold sweep is off. Prediction
maps and visualizations are saved only when explicitly enabled.

## Warnings

- Do not report fast-config results as verified full-model results.
- Do not change dataset splits when comparing speed.
- AMP may require FP32 wrappers for specific Mamba kernels.
- Checkpointing can reduce memory but slow training.
- `torch.compile` first iterations can be slower.
- Keep F1 as change-class F1 only.
