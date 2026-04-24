# Experiments

## Dataset: LEVIR-CD

| Property | Value |
|---|---|
| Image pairs | 637 (train) + 49 (test) |
| Image size | 1024 × 1024 px |
| Patch size | 256 × 256 (random crop / sliding window) |
| Bands | RGB (3-channel) |
| Change type | Building footprint appearance / disappearance |
| Positive ratio | ~4–6% of pixels |

Directory layout:
```
Datasets/LEVIRCD/
  train/  A/  B/  label/
  test/   A/  B/  label/
```

Train / val split: 80% / 20% of `train/` (deterministic, seed=42).

---

## Metrics

| Metric | Formula | Notes |
|---|---|---|
| F1 | $2\text{TP} / (2\text{TP} + \text{FP} + \text{FN})$ | Primary metric |
| mIoU | $\text{TP} / (\text{TP} + \text{FP} + \text{FN})$ | Change class IoU |
| Precision | $\text{TP} / (\text{TP} + \text{FP})$ | |
| Recall | $\text{TP} / (\text{TP} + \text{FN})$ | |
| OA | $(\text{TP} + \text{TN}) / \text{Total}$ | Overall accuracy |
| Boundary F1 | F1 within boundary band of GT mask | Measures edge quality |

---

## Config Reference

### Model options

```yaml
model:
  backbone: mambavision
  variant: small          # tiny | tiny2 | small | base | large
  decoder: refinement     # baseline | adaptive_rf | refinement | global_local
  pretrained: true
  freeze_backbone: false  # freeze backbone weights
```

### Training options

```yaml
training:
  max_iterations: 50000
  batch_size: 8
  lr: 1e-4
  weight_decay: 0.01
  optimizer: AdamW
  warmup_iterations: 3000
  validate_every: 5000
  log_every: 20
  gradient_clip: 1.0
```

### Resume options

```yaml
resume:
  enabled: true
  checkpoint_path: outputs/run_XXX/checkpoints/best.pth
  strict: true    # false = warn on variant mismatch instead of raising
```

### Hardware options

```yaml
hardware:
  device: cuda       # cuda | cuda:0 | cuda:1 | cpu
  mixed_precision: true
```

### Decoder-specific options

```yaml
decoder:
  channels: 256              # FPN internal width
  dilation_rates: [1,2,4,8]  # adaptive_rf only
  aux_weight: 0.4            # refinement decoder coarse auxiliary loss weight
```

---

## Planned Experiments

| ID | Config | Variant | Purpose |
|---|---|---|---|
| E1 | baseline | tiny | Reference |
| E2 | adaptive_rf | tiny | RF ablation |
| E3 | refinement | small | MERCon paper |
| E4 | refinement | base | Scale ablation |
| E5 | global_local | small | Two-branch comparison |

---

## Expected Outputs

After `python scripts/train.py` with `refinement_decoder.yaml`:

```
outputs/
  run_20260424_XXXXXX_refinement_decoder/
    config.yaml
    model_info.json      ← variant, total_params, encoder_channels
    logs/train.log
    tensorboard/
    checkpoints/best.pth
    validation/metrics.csv
    samples/iter_0005000.png
    ...
```

Training log example:
```
[5000/50000] loss=0.4182 bce=0.2801 dice=0.1381 lr=9.98e-05
── Validation @ iter 5000 ──
+---------------+----------+
| Metric        |    Value |
+---------------+----------+
| F1            |   0.8920 |
| mIoU          |   0.8051 |
| Prec          |   0.9112 |
| Recall        |   0.8740 |
| OA            |   0.9923 |
| Bnd F1        |   0.7841 |
+---------------+----------+
  ✓ New best f1=0.8920 saved.
```
