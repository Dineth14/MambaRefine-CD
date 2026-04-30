# DSIFN-CD Split Audit

Generated: 2026-04-30

## Verdict

**FAIL: confirmed split leakage found.**

**Invalidation warning:** The previous DSIFN results obtained before explicit
split files were introduced are invalid as held-out test results because the
test set overlapped with train and validation images. All DSIFN models must be
retrained or at least re-evaluated on the corrected held-out test split before
reporting results in a paper.

The active DSIFN-CD root is a flat layout:

```text
/storage2/ChangeDetection/MV/Datasets/DSIFN-CD/DSIFN/
  t1/
  t2/
  mask/
```

No `trainset/`, `valset/`, `testset/`, `train.txt`, `val.txt`, or `test.txt`
was found. In this repository's current `DSIFNCDDataset` fallback path:

- `train` is an 80% image-level split from all flat `t1/` images.
- `val` is a 20% image-level split from all flat `t1/` images.
- `test` is **all flat `t1/` images**.

Therefore the current test split contains both training and validation images.
The high DSIFN-CD scores from this protocol should not be treated as clean
held-out test performance.

## Commands Run

```bash
python3 scripts/audit_dsifn_data_integrity.py --config configs/experiments/dsifn_full.yaml
python3 scripts/check_binary_metrics.py
```

`check_binary_metrics.py` could not run in this shell because PyTorch is not
installed:

```text
ModuleNotFoundError: No module named 'torch'
```

## Generated Audit Files

```text
outputs/dsifn_split_config_audit.json
outputs/dsifn_split_config_audit.md
outputs/dsifn_manifests/train_manifest.csv
outputs/dsifn_manifests/val_manifest.csv
outputs/dsifn_manifests/test_manifest.csv
outputs/dsifn_overlap_report.csv
outputs/dsifn_overlap_report.json
outputs/dsifn_overlap_report.md
outputs/dsifn_near_duplicate_report.csv
outputs/dsifn_near_duplicate_report.md
outputs/dsifn_dataloader_inspection.md
outputs/dsifn_data_integrity_summary.json
outputs/dsifn_data_integrity_summary.md
```

## Split Resolution

| Split | Layout | Image Names | Manifest Rows | Source |
|---|---|---:|---:|---|
| train | flat manual split | 3153 | 3153 | `all_names[788:]` after seed 42, `val_ratio=0.2` |
| val | flat manual split | 788 | 3152 | `all_names[:788]` after seed 42, `val_ratio=0.2` |
| test | flat manual split | 3941 | 15764 | all names from flat `t1/` directory |

Validation and test are tiled into four 256x256 patches per 512x512 image.
Training uses random 256x256 crops from the selected training images.

## Exact Overlap Summary

| Split Pair | Overlap Type | Count |
|---|---|---:|
| train vs val | pre/post/mask path overlap | 0 |
| train vs val | original scene ID overlap | 0 |
| train vs test | pre path overlap | 3153 |
| train vs test | post path overlap | 3153 |
| train vs test | mask path overlap | 3153 |
| train vs test | original scene ID overlap | 3153 |
| val vs test | pre path overlap | 788 |
| val vs test | post path overlap | 788 |
| val vs test | mask path overlap | 788 |
| val vs test | pair-key tile overlap | 3152 |
| val vs test | mask-key tile overlap | 3152 |
| val vs test | original scene ID overlap | 788 |

This is confirmed leakage, not only a near-duplicate warning.

## Near-Duplicate Summary

The near-duplicate audit reported:

| Risk | Count |
|---|---:|
| confirmed leakage | 6304 |
| high-risk leakage | 6990 |
| suspicious | 1940 |

The confirmed leakage is driven by identical sample keys and identical files
across validation/test and train/test due to the flat-layout test fallback.

## Patch Generation Policy

Current behavior:

1. Train/val are split at image-name level from the flat image list.
2. Train samples random 256x256 crops at `__getitem__`.
3. Val/test generate deterministic non-overlapping 256x256 tiles.
4. Test uses all flat images, so test tiles are generated from train and val
   images as well.

The patch extraction order is not the main issue. The main issue is that test
does not have an independent file list or folder in the current flat layout.

## Dataloader Behavior

From `outputs/dsifn_dataloader_inspection.md`:

| Split Pair | Shared Pre Paths |
|---|---:|
| train vs val | 0 |
| train vs test | 3153 |
| val vs test | 788 |

Train uses augmentation. Val/test use deterministic evaluation transforms and
do not use random augmentation.

## Threshold Policy

The repository policy is clean in principle:

- Validation may sweep thresholds.
- Best threshold is saved in the checkpoint.
- Test uses the checkpoint threshold and disables threshold sweep.

However, because the current DSIFN test split overlaps train/val, final DSIFN
test results are not clean even though the threshold policy itself is not the
leakage source.

## Result Metadata

`scripts/test.py` and `src/training/final_eval.py` now include these fields in
new test result JSON:

- `dataset_name`
- `split`
- `num_samples`
- `dataset_root`
- `config_path`
- `checkpoint_path`
- `checkpoint_sha256`
- `threshold`
- `threshold_source`

Existing historical result files were not rewritten.

## Fix Applied

The unsafe DSIFN flat-layout fallback is disabled. A flat DSIFN root now
requires explicit non-overlapping split files:

```text
DSIFN/
  t1/
  t2/
  mask/
  splits/
    train.txt
    val.txt
    test.txt
```

If split files are missing, the loader stops with:

```text
DSIFN flat layout requires explicit non-overlapping split files. Refusing to use all images as test because this causes train/test leakage.
```

If any image ID appears in more than one split, training/evaluation stops with:

```text
DATA LEAKAGE FOUND: refusing to train/evaluate.
```

## Post-Fix Audit

After creating explicit deterministic split files with seed 42:

```bash
python3 scripts/create_dsifn_splits.py \
  --root /storage2/ChangeDetection/MV/Datasets/DSIFN-CD/DSIFN \
  --out_dir /storage2/ChangeDetection/MV/Datasets/DSIFN-CD/DSIFN/splits \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --seed 42
```

the split counts are:

| Split | Image IDs | Manifest Rows |
|---|---:|---:|
| train | 2758 | 2758 |
| val | 394 | 1576 |
| test | 789 | 3156 |

The clean audit command:

```bash
python3 scripts/audit_dsifn_data_integrity.py --config configs/experiments/dsifn_full.yaml
```

returned:

```text
Final verdict: PASS WITH WARNINGS
```

Confirmed identity overlap is now zero:

| Split Pair | Path/Stem/Pair/Scene Overlap |
|---|---:|
| train vs val | 0 |
| train vs test | 0 |
| val vs test | 0 |

The warning is caused by duplicate content hashes across different image IDs,
mainly repeated masks and post-event images. These are reported as warnings,
not confirmed leakage, because paths, stems, pair keys, mask keys, and original
scene IDs are disjoint.

## Recommended Fix

Do not report the current DSIFN final test results as held-out test scores.

Recommended protocol:

1. Obtain or restore the official DSIFN train/test split if available.
2. If only flat data is available, create explicit image-level split files:
   `train.txt`, `val.txt`, and `test.txt`.
3. Ensure each original image ID appears in exactly one split.
4. Generate train crops only from train IDs.
5. Generate validation tiles only from val IDs.
6. Generate test tiles only from test IDs.
7. Rerun:

```bash
python3 scripts/audit_dsifn_data_integrity.py --config configs/experiments/dsifn_full.yaml
```

Only treat the result as clean if the verdict becomes `PASS` or, with
explainable non-leakage warnings only, `PASS WITH WARNINGS`.
