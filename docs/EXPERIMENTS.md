# Experiments

## 1. Active Datasets

- DSIFN-CD
- WHU-CD

## 2. Active Metrics

- Pre
- Rec
- F1
- IoU
- OA

## 3. Current Verified Results

| Dataset | Pre | Rec | F1 | IoU | OA |
|---|---:|---:|---:|---:|---:|
| DSIFN-CD | 96.86 | 97.20 | 97.03 | 94.23 | 97.93 |
| WHU-CD | 96.16 | 95.00 | 95.58 | 91.53 | 99.58 |

## 4. Training Commands

```bash
python scripts/train.py --config configs/experiments/dsifn_full.yaml
python scripts/train.py --config configs/experiments/whu_full.yaml
```

## 5. Testing Commands

```bash
python scripts/test.py --config configs/experiments/dsifn_full.yaml --ckpt <checkpoint>
python scripts/test.py --config configs/experiments/whu_full.yaml --ckpt <checkpoint>
```

## 6. Ablations

DSIFN ablation configs remain active under:

```text
configs/ablations/dsifn/
```

Status: TODO, keep verification before using these results in a paper table.
