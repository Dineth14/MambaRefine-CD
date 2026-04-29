# SECOND Pipeline Fix Report

Date: 2026-04-28

## Current Paths

- Dataset loader: `src/data/second.py`, via `datasets/second.py` and `src/data/dataset_builder.py`.
- Dataset config: `configs/datasets/second.yaml`.
- Main config: `configs/ablations/second/a4_full.yaml`.
- Model: `src/models/cd_model.py` with `LightweightSemanticHead` in `src/models/decoders/semantic_heads.py`.
- Scripts: `scripts/train.py`, `scripts/evaluate.py`, `scripts/test.py`.
- Shared evaluator: `src/training/evaluator.py`.
- Loss: `src/training/second_loss.py`, re-exported by `losses/second_losses.py`.
- Metrics: `metrics/second_scd_metrics.py`.

## Findings

- SECOND already had semantic heads, but the config and training metadata were not consistently wired.
- `checkpoint.selection_metric: Fscd` was not normalized to `checkpoint.monitor`, so training could monitor missing `f1`.
- SECOND A4 used a much smaller 96-channel D-RBI/decoder path and only `[1, 2, 4]` dilation rates, explaining unexpectedly low parameter count.
- Dataset samples did not expose the requested `image_t1`, `image_t2`, and `sample_id` aliases.
- `loss.second.*` settings were not consumed by the loss builder.
- Semantic Dice loss was missing from the SECOND loss.
- SECOND prediction saving was not split-aware and `test.py` did not default to saving SECOND predictions.

## Fixes Applied

- SECOND dataset now exposes `image_t1`, `image_t2`, `label_t1`, `label_t2`, `change_mask`, `change_mask_long`, and `sample_id` while preserving existing training keys.
- RGB color-coded labels are decoded through the SECOND palette; masks are not normalized or divided by 255.
- Added `dataset.debug_stats` and `tools/validate_second_dataset.py`.
- Model config now supports nested `model.semantic_head.*`.
- SECOND A4 now uses base backbone, 256-channel D-RBI/decoder, `[1, 2, 4, 8]`, and a 128-channel semantic head.
- SECOND loss now supports semantic CE, semantic Dice, auxiliary binary change loss, consistency loss, and existing SeK-style surrogate.
- Metrics remain semantic-output based and return only `OA`, `mIoU`, `SeK`, `Fscd`.
- Checkpoints now save `best_metric_name`, `iter`, `epoch`, AMP `scaler`, `ema_enabled`, validation metrics, and best SECOND metric metadata.
- Config normalization maps `train.ema.*` to the existing training EMA keys and `selection_metric` to `monitor`.
- `evaluate.py` and `test.py` use the shared evaluator and print only SECOND paper metrics in the main table.
- SECOND predictions are saved under `predictions/<split>/` and visualizations under `visualizations/<split>/`.

## Output Contract

SECOND forward returns:

- `sem_logits_t1`: `[B, C, H, W]`
- `sem_logits_t2`: `[B, C, H, W]`
- `change_logits`: optional auxiliary `[B, 1, H, W]`
- `aux_logits`: optional coarse change logits

Official prediction logic:

- `pred_t1 = argmax(sem_logits_t1, dim=1)`
- `pred_t2 = argmax(sem_logits_t2, dim=1)`
- `pred_change = pred_t1 != pred_t2`

The auxiliary binary head is not used for official SECOND metrics.

## Parameter Breakdown

For `configs/ablations/second/a4_full.yaml`:

- total: `114,912,857`
- backbone: `97,685,288`
- decoder: `13,028,516`
- D-RBI: `3,005,440`
- semantic head: `986,247`
- binary head: `295,297`

## Verification

Passed:

```bash
conda run -n mamba_new python -m pytest tests/test_second_metrics.py -q
conda run -n mamba_new python -m pytest tests -q
conda run -n mamba_new python tools/validate_second_dataset.py --config configs/datasets/second.yaml --sample_limit 2
```

Results:

- SECOND metric tests: `10 passed`
- Full tests: `21 passed, 9 skipped`
- Dataset counts: train `2968`, val `2372`, test `6776`

Dry run:

```bash
conda run -n mamba_new python scripts/train.py --config configs/ablations/second/a4_full.yaml --dry_run
```

The dry run built the model and dataset and printed parameter breakdown, then failed on forward because this session has no CUDA device and the Mamba selective-scan kernel requires CUDA tensors:

`RuntimeError: Expected u.is_cuda() to be true`

Run validation/test equivalence on a CUDA-visible session.

## Palette Assumption

The repo uses the existing SECOND palette:

- `0`: white/no-change/background
- `1`: green
- `2`: gray
- `3`: bright green
- `4`: blue
- `5`: dark red
- `6`: red

Class count is configured as `7`, ignore index as `255`.
