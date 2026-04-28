# LEVIR Validation/Test Mismatch Report

Date: 2026-04-28

## Scope

This report tracks the validation/test mismatch investigation for `configs/ablations/levir/a4_full.yaml`.
The model architecture was not changed.

## Dataset And Split

| Item | Validation | Test |
| --- | --- | --- |
| Dataset root | `/storage2/ChangeDetection/MV/Datasets/LEVIRCD` | `/storage2/ChangeDetection/MV/Datasets/LEVIRCD` |
| Image path | `train/A`, `train/B` | `test/A`, `test/B` |
| Mask path | `train/label` | `test/label` |
| Split file | none | none |
| Split construction | deterministic random image-level split from `train/` using `val_ratio=0.2`, seed `42` | all files under `test/` |
| Image/tile size | `256 x 256` | `256 x 256` |
| Samples/tiles checked | 2032 tiles, 127 image IDs | 5568 tiles, 348 image IDs |

`tools/check_splits.py` found:

- train-val image ID overlap: `0`
- train-test image ID overlap: `0`
- val-test image ID overlap: `0`
- no configured train-val image-level leakage
- validation global mask positive ratio: about `4.73%`
- test global mask positive ratio: about `4.07%`

## Preprocessing

| Item | Validation | Test |
| --- | --- | --- |
| Transform pipeline | deterministic crop/tile read, RGB conversion, ImageNet normalization | deterministic crop/tile read, RGB conversion, ImageNet normalization |
| Random augmentation | disabled | disabled |
| Image scaling | `uint8 -> float32 / 255.0` | `uint8 -> float32 / 255.0` |
| Normalization mean | `[0.485, 0.456, 0.406]` | `[0.485, 0.456, 0.406]` |
| Normalization std | `[0.229, 0.224, 0.225]` | `[0.229, 0.224, 0.225]` |
| Mask conversion | raw grayscale mask, `mask > 127 -> 0/1` | raw grayscale mask, `mask > 127 -> 0/1` |
| Mask interpolation | not resized in patch mode; nearest-neighbor rule preserved in dataset/tools | not resized in patch mode; nearest-neighbor rule preserved in dataset/tools |

First 5 validation/test masks were checked. Raw values are `[0, 255]`, converted values are `[0, 1]`, shape is `[256, 256]`.

## Inference And Metrics

| Item | Validation | Test |
| --- | --- | --- |
| Shared evaluator | `src/training/evaluator.py` | `src/training/evaluator.py` |
| Entrypoint | `Trainer._validate`, `scripts/evaluate.py --split val`, `scripts/test.py --split val` | `scripts/evaluate.py --split test`, `scripts/test.py --split test` |
| Inference mode | `patch` | `patch` |
| Crop size | `256` | `256` |
| Overlap | `0.25` configured; not used in patch mode | `0.25` configured; not used in patch mode |
| Sliding-window support | implemented and configurable | implemented and configurable |
| Logits averaged | only in `sliding_window` mode | only in `sliding_window` mode |
| Threshold base | `eval.threshold=0.5` | `eval.threshold=0.5` |
| Threshold sweep | enabled for validation | disabled on test |
| Best validation threshold | `0.30` in verification with the existing checkpoint | not tuned on test |
| Binary prediction | `sigmoid(logits) > threshold` | `sigmoid(logits) > threshold` |
| Metric class | `StreamingMetrics` through shared evaluator | `StreamingMetrics` through shared evaluator |
| Average mode | global accumulated TP/FP/TN/FN | global accumulated TP/FP/TN/FN |
| Final paper metrics | `Pre`, `Rec`, `F1`, `IoU`, `OA` | `Pre`, `Rec`, `F1`, `IoU`, `OA` |

Metric formulas are global:

- `Pre = TP / (TP + FP + eps)`
- `Rec = TP / (TP + FN + eps)`
- `F1 = 2 * Pre * Rec / (Pre + Rec + eps)`
- `IoU = TP / (TP + FP + FN + eps)`
- `OA = (TP + TN) / (TP + TN + FP + FN + eps)`

## EMA And Checkpoint

Training validation used EMA when `training.use_ema` was enabled. The existing LEVIR checkpoint used for verification is an older checkpoint and does not contain EMA weights or validation-threshold metadata.

Checkpoint inspected:

`outputs/levir/a4_full/run_20260428_023133_levir_a4_full_LEVIR-CD/checkpoints/best.pth`

Observed load metadata:

- checkpoint iteration: `40000`
- best metric: `0.9233782398447543`
- EMA weights found: `false`
- EMA used: `false`
- checkpoint best threshold: missing
- missing keys: `[]`
- unexpected keys: `[]`

Fix applied for future checkpoints:

- best checkpoints now save `ema`, `best_threshold`, and `val_metrics` metadata when available.
- evaluation now prints loaded checkpoint path, iteration, best metric, threshold source, EMA requested/found/used, missing keys, and unexpected keys.
- evaluation can load EMA weights when checkpoint contains them and `eval.use_ema=true` or `--use_ema` is passed.

## Detected Mismatches

1. Validation, `evaluate.py`, and `test.py` had separate evaluation loops.
2. Test/evaluate did not default to the best validation threshold from the checkpoint.
3. Existing best checkpoint did not store `best_threshold`.
4. Existing best checkpoint did not store EMA weights, even though training validation used EMA.
5. Test/evaluate checkpoint loading did not clearly report EMA availability/use.
6. Checkpoint key mismatches could be hidden unless strict loading behavior was inspected.
7. Validation/test mask raw-value diagnostics were not printed from the same evaluation path.
8. Sliding-window settings existed conceptually, but the shared evaluator now implements logit averaging if that mode is selected.

No split leakage or raw mask format mismatch was found for the configured LEVIR split.

## Fixes Applied

- Unified validation, `scripts/evaluate.py`, `scripts/test.py`, and final evaluation through `src/training/evaluator.py`.
- Added `eval.threshold`, `eval.threshold_sweep.enabled`, `eval.threshold_sweep.values`, `eval.threshold_select_metric`.
- Added checkpoint threshold resolution: command line > checkpoint > config.
- Disabled threshold sweep on test to avoid tuning on test labels.
- Added `--threshold`, `--use_ema`, `--no_ema`, `--strict`, `--non_strict`, `--save_debug`, and `--num_workers` CLI handling.
- Added `eval.use_ema` and EMA-aware checkpoint loading.
- Added checkpoint metadata saving for EMA state, best threshold, and validation metrics.
- Added strict missing/unexpected key reporting for evaluation loads.
- Added mask debug logs for first 5 samples with raw unique values, converted unique values, shape, and positive ratio.
- Added debug visual output support under `outputs/debug_levir_eval/<split>/`.
- Added shared sliding-window inference with configurable crop size and overlap, accumulating logits and dividing by a count map before thresholding.
- Added `tools/check_splits.py`.
- Added `tools/compare_val_test_pipeline.py`.
- Restricted binary final printed metrics to `Pre`, `Rec`, `F1`, `IoU`, `OA`.

## Verification Commands

The CUDA Mamba op could not run inside the sandbox CPU path, and sandboxed multiprocessing workers were blocked. Verification was run outside the sandbox with `--num_workers 0`.

```bash
/userhomes/keshawa17/anaconda3/envs/mamba_new/bin/python tools/check_splits.py --config configs/ablations/levir/a4_full.yaml

/userhomes/keshawa17/anaconda3/envs/mamba_new/bin/python scripts/test.py \
  --config configs/ablations/levir/a4_full.yaml \
  --ckpt outputs/levir/a4_full/run_20260428_023133_levir_a4_full_LEVIR-CD/checkpoints/best.pth \
  --split val \
  --num_workers 0

/userhomes/keshawa17/anaconda3/envs/mamba_new/bin/python scripts/evaluate.py \
  --config configs/ablations/levir/a4_full.yaml \
  --ckpt outputs/levir/a4_full/run_20260428_023133_levir_a4_full_LEVIR-CD/checkpoints/best.pth \
  --split val \
  --num_workers 0

/userhomes/keshawa17/anaconda3/envs/mamba_new/bin/python scripts/test.py \
  --config configs/ablations/levir/a4_full.yaml \
  --ckpt outputs/levir/a4_full/run_20260428_023133_levir_a4_full_LEVIR-CD/checkpoints/best.pth \
  --split test \
  --num_workers 0
```

Additional verification:

```bash
/userhomes/keshawa17/anaconda3/envs/mamba_new/bin/python -m py_compile \
  src/training/evaluator.py scripts/evaluate.py scripts/test.py \
  tools/check_splits.py tools/compare_val_test_pipeline.py

/userhomes/keshawa17/anaconda3/envs/mamba_new/bin/python -m pytest tests -q
```

Result: `19 passed, 9 skipped`.

## Before And After Metrics

User-provided validation reference:

| Split/path | Pre | Rec | F1 | IoU | OA |
| --- | ---: | ---: | ---: | ---: | ---: |
| training validation reference | 92.73 | 91.86 | 92.29 | 85.69 | 99.27 |

Previous separate test/evaluate path observed before the shared evaluator:

| Split/path | Threshold | Pre | Rec | F1 | IoU | OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LEVIR test, old path | 0.50 | 85.4989 | 85.4483 | 85.4735 | 74.6322 | 98.8166 |

After the shared evaluator:

| Split/path | Threshold source | Threshold | Pre | Rec | F1 | IoU | OA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `scripts/test.py --split val` | validation sweep | 0.30 | 93.0814 | 91.7072 | 92.3892 | 85.8550 | 99.2848 |
| `scripts/evaluate.py --split val` | validation sweep | 0.30 | 93.0814 | 91.7072 | 92.3892 | 85.8550 | 99.2848 |
| `scripts/test.py --split test` | config | 0.50 | 86.5451 | 85.8841 | 86.2133 | 75.7676 | 98.8809 |

The validation-through-test-script and validation-through-evaluate-script results now match exactly.

## Conclusion

The main validation/test mismatch was pipeline inconsistency, not model architecture:

- validation/test code paths were not shared;
- EMA validation could not be reproduced from the existing checkpoint because EMA weights were not saved;
- best validation threshold could not be reused from the existing checkpoint because threshold metadata was not saved;
- test used threshold `0.50` by config because the existing checkpoint has no saved threshold;
- no LEVIR split leakage or mask value-format issue was found.

Future checkpoints produced after this fix will carry EMA and best validation threshold metadata, so test defaults can match the validation-time evaluation settings without tuning on the test set.
