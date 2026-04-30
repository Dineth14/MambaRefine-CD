# DSIFN-CD Clean Split Protocol

This protocol replaces the leaked flat-layout fallback. Do not use old DSIFN
checkpoints or result files as paper evidence unless they are re-evaluated on a
clean held-out test split and clearly documented.

## 1. Create Deterministic Splits

```bash
python scripts/create_dsifn_splits.py \
  --root /storage2/ChangeDetection/MV/Datasets/DSIFN-CD/DSIFN \
  --out_dir /storage2/ChangeDetection/MV/Datasets/DSIFN-CD/DSIFN/splits \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --seed 42
```

The script splits original image IDs first. Patches are not generated before
splitting.

## 2. Audit

```bash
python scripts/audit_dsifn_data_integrity.py \
  --config configs/experiments/dsifn_full.yaml
```

Proceed only if the verdict is `PASS` or `PASS WITH WARNINGS` with no exact
train/val/test overlap.

## 3. Train Full Model From Scratch

```bash
python scripts/train.py \
  --config configs/experiments/dsifn_full.yaml
```

## 4. Test Clean Held-Out Split

```bash
python scripts/test.py \
  --config configs/experiments/dsifn_full.yaml \
  --ckpt <new_checkpoint_from_clean_training>
```

## 5. Run Ablations After Audit Passes

```bash
python scripts/audit_ablations.py \
  --config_dir configs/ablations/dsifn
```

Then train and test each ablation with the same clean split files.

## Reporting Rules

- Report only `Pre`, `Rec`, `F1`, `IoU`, and `OA`.
- `F1` is the change-class F1.
- Do not tune thresholds on the test split.
- Use validation for checkpoint selection and threshold selection.
- Use only the corrected test split for final DSIFN results.
