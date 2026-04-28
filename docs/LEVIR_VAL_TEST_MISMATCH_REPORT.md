# LEVIR Validation/Test Mismatch Report

Date: 2026-04-28

## Current Update

This pass re-checked the existing mismatch fix and found one remaining inconsistency:
`src/training/final_eval.py` still allowed final held-out test evaluation to run with the normal evaluator config, which could sweep thresholds on `test` and save internal metric keys. That has now been fixed.

Additional fixes applied in this pass:

- final post-training test evaluation now resolves threshold as `checkpoint best_threshold -> config threshold`, disables threshold sweep on `test`, and logs threshold source.
- final post-training test evaluation now saves and prints binary paper metrics as `Pre`, `Rec`, `F1`, `IoU`, `OA`, with threshold/EMA as metadata.
- `scripts/test.py` and `scripts/evaluate.py` now print the final metric table with only the allowed paper metrics; threshold and EMA are printed separately as metadata.
- checkpoint load logging now includes the stored checkpoint threshold.
- binary metric accumulation now robustly converts masks from either `0/255` or `0/1` to binary before TP/FP/TN/FN accumulation.
- the shared evaluator now writes `metrics.json` for validation/evaluation outputs, not only CSV and threshold JSON.

Current environment verification:

- `python -m py_compile` passed for the changed scripts/modules.
- `pytest tests -q` passed: `19 passed, 9 skipped`.
- `tools/check_splits.py --config configs/ablations/levir/a4_full.yaml` passed with no train/val/test image-ID overlap.
- Full `scripts/test.py --split val` and `scripts/evaluate.py --split val` both reached the same shared evaluator path but could not complete in this session because CUDA is unavailable (`torch.cuda.is_available() == False`, device count `0`) and the Mamba selective-scan kernel requires CUDA tensors.

Current checkpoint inspected:

`outputs/levir/a4_full/run_20260428_023133_levir_a4_full_LEVIR-CD/checkpoints/best.pth`

- checkpoint iteration: `50000`
- best metric: `0.9238958813619399`
- checkpoint best threshold: missing
- EMA weights found: `false`
- variant: `base`

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

The full model evaluation commands require CUDA because the Mamba selective-scan kernel does not run on CPU. In this session, CUDA was not available, so the commands below are the expected verification commands; syntax/unit tests and split checks were run successfully, while full metric reproduction was blocked by the CUDA requirement.

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

Current unit-test result: `19 passed, 9 skipped`.

## Before And After Metrics

User-provided validation reference:

| Split/path | Pre | Rec | F1 | IoU | OA |
| --- | ---: | ---: | ---: | ---: | ---: |
| training validation reference | 92.73 | 91.86 | 92.29 | 85.69 | 99.27 |

Previous separate test/evaluate path observed before the shared evaluator:

| Split/path | Threshold | Pre | Rec | F1 | IoU | OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LEVIR test, old path | 0.50 | 85.4989 | 85.4483 | 85.4735 | 74.6322 | 98.8166 |

Historical after-fix metrics from the existing output files:

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
