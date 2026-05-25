# Running SECOND

Validate dataset decoding and palette handling:

```bash
python tools/validate_second_dataset.py --config configs/datasets/second.yaml
```

Dry-run model, dataset, and loss wiring:

```bash
python scripts/train.py --config configs/ablations/second/a4_full.yaml --dry_run
```

Train:

```bash
python scripts/train.py --config configs/ablations/second/a4_full.yaml
```

Evaluate validation split with EMA:

```bash
python scripts/evaluate.py \
  --config configs/ablations/second/a4_full.yaml \
  --ckpt outputs/second/a4_full/best.pth \
  --split val \
  --use_ema \
  --save_predictions \
  --save_visualizations
```

Test with EMA:

```bash
python scripts/test.py \
  --config configs/ablations/second/a4_full.yaml \
  --ckpt outputs/second/a4_full/best.pth \
  --split test \
  --use_ema
```

Check SECOND metrics:

```bash
pytest tests/test_second_metrics.py
```

Notes:

- SECOND metrics are semantic-change metrics from `sem_logits_t1` and `sem_logits_t2`.
- The auxiliary binary `change_logits` is used for loss/optional visualization only.
- Test evaluation must not tune thresholds on the test split.
- Prediction maps are saved under `outputs/second/<experiment>/predictions/<split>/`.
