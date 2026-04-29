# LEVIR Validation-Test Gap Diagnosis

## Current Metrics

Validation at iter 50000:

| Split | Pre | Rec | F1 | IoU | OA | Threshold | EMA | Samples |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| val | 0.9288 | 0.9192 | 0.9240 | 0.8587 | 0.9928 | 0.3000 | true | 2032 |

Final test:

| Split | Pre | Rec | F1 | IoU | OA | Threshold | EMA | Samples |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| test | 85.5712 | 86.5705 | 86.0679 | 75.5432 | 98.8581 | 0.3000 | true | 5568 |

Threshold and EMA are consistent. The remaining gap is therefore not explained by threshold or EMA mismatch.

## Split Integrity

Checked with:

```bash
python tools/check_levir_splits.py --config configs/ablations/levir/a6_full.yaml --write_metadata
```

Resolved dataset root:

```text
/storage2/ChangeDetection/MV/Datasets/LEVIRCD
```

Detected folders:

```text
train/: present
val/:   absent
test/:  present
```

The repository does not use a physical official `val/` folder for this LEVIR root. Validation is derived from `train/` using `val_ratio=0.2` and `seed=42`.

| Split | Source Folder | Image Count | Tile/Sample Count | Mode |
|---|---|---:|---:|---|
| train | `train/` subset | 510 | 8160 tile metadata rows | full-image/random-crop training by default |
| val | `train/` holdout | 127 | 2032 | tile |
| test | `test/` | 348 | 5568 | tile |

Overlap checks:

| Overlap | Count |
|---|---:|
| train-val filenames | 0 |
| train-test filenames | 0 |
| val-test filenames | 0 |
| train-val original IDs | 0 |
| train-test original IDs | 0 |
| val-test original IDs | 0 |

Conclusion: no direct tile or original-image ID leakage was detected between train, val, and test. However, validation is an internal holdout from the training image distribution, not a separate official validation folder.

## Pairing Check

First test triplets follow the same sample ID across T1, T2, and mask:

```text
test/A/train_638.png
test/B/train_638.png
test/label/train_638.png
...
test/A/train_657.png
test/B/train_657.png
test/label/train_657.png
```

Pairing errors:

| Split | Errors |
|---|---:|
| train | 0 |
| val | 0 |
| test | 0 |

Conclusion: no alphabetical pairing bug, off-by-one pairing, or mixed train/test directory issue was detected.

## Mask Conversion

For val and test, raw masks use values `[0, 255]` or `[0]`. The conversion rule is:

```text
mask = raw > 127
```

After conversion, masks contain `[0, 1]` or `[0]`.

Conclusion: test mask conversion is correct for binary masks and uses the same conversion as validation.

## Val/Test Preprocessing

Validation and test are both evaluated as 256x256 tiles with deterministic normalization.

| Split | Mode | Image Shape Before | Image Shape After | Mask Shape Before | Mask Shape After |
|---|---|---|---|---|---|
| val | tile | full source image, tiled to 256x256 | `[3, 256, 256]` | full source mask, tiled to 256x256 | `[1, 256, 256]` |
| test | tile | full source image, tiled to 256x256 | `[3, 256, 256]` | full source mask, tiled to 256x256 | `[1, 256, 256]` |

ImageNet-normalized tensor statistics from sampled val/test tiles:

| Split | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|
| val | -2.1179 | 2.6400 | -0.4746 | 0.8976 |
| test | -2.1179 | 2.6400 | -0.4919 | 0.9316 |

Conclusion: preprocessing appears consistent. Test images have slightly higher normalized variance, but no scale or normalization mismatch was detected.

## Distribution Comparison

Using tile metadata:

| Split | Samples | Changed Pixels | No-change Pixels | GT Positive Ratio | Mean Change Area / Tile |
|---|---:|---:|---:|---:|---:|
| val | 2032 | 6,303,697 | 126,865,455 | 0.047336 | 3102.21 |
| test | 5568 | 14,867,412 | 350,037,036 | 0.040743 | 2670.15 |

The test split has a lower positive ratio and lower mean changed area per tile than validation. This makes test somewhat more sparse and likely harder, but the difference alone may not explain the full F1 gap.

## Evaluator Consistency

The final training log shows final test evaluation used:

```text
Evaluation inference mode: patch
crop_size=256
stride=192
overlap=0.25
logits averaged=False
threshold=0.3000 from checkpoint
EMA used=true
EMA found=true
```

The current `scripts/test.py` and `scripts/evaluate.py` both route through `training.evaluator.Evaluator`. They now also print config fingerprints and module flags to catch checkpoint/config mismatches.

Commands to run in the project environment with PyTorch installed:

```bash
python scripts/test.py \
  --config configs/ablations/levir/a6_full.yaml \
  --ckpt outputs/levir/a4_full/run_20260428_184455_levir_a4_full_LEVIR-CD/checkpoints/best.pth \
  --split val \
  --use_ema \
  --save_debug

python scripts/evaluate.py \
  --config configs/ablations/levir/a6_full.yaml \
  --ckpt outputs/levir/a4_full/run_20260428_184455_levir_a4_full_LEVIR-CD/checkpoints/best.pth \
  --split test \
  --use_ema \
  --save_debug
```

This shell could not run those commands because the available `python3` has no PyTorch installed:

```text
ModuleNotFoundError: No module named 'torch'
```

## Debug Visualizations

The evaluator debug path has been aligned to:

```text
debug/levir/val/
debug/levir/test/
```

When `--save_debug` is used, it saves up to 50 samples per split:

```text
image_t1/
image_t2/
gt/
pred/
prob/
error_map/
```

Error map colors:

```text
TP: green
FP: orange/red
FN: blue
TN: black
```

Use these visualizations to inspect shifted masks, inverted masks, wrong image order, resizing errors, or systematic missing small objects.

## Final Diagnosis

Current evidence does not indicate:

- threshold mismatch
- EMA mismatch
- test mask conversion bug
- test path pairing bug
- direct train/val/test filename overlap
- direct original-image ID leakage
- val/test normalization mismatch
- val patch vs test full-image mismatch

The main confirmed issue is protocol-related: validation is a random image-level holdout from the `train/` folder, while final test is from the separate `test/` folder. The validation score is therefore an internal training-distribution validation score, not necessarily representative of official test performance.

## Recommended Reporting

- Use validation only for model selection and threshold selection.
- Do not tune threshold on test.
- Report final LEVIR test honestly if it remains around F1 86 after evaluator consistency is confirmed.
- Clearly mark the current validation result as validation-only.
- If official image-level split metadata is available, prefer official train/val/test folders or split original image pairs before tiling.
