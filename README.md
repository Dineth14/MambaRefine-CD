# MambaRefine-CD

Efficient remote-sensing change detection with MambaVision and region-boundary interaction modeling.

## Active Datasets

- DSIFN-CD
- WHU-CD

Earlier experimental support for other datasets has been archived.

## Method Modules

- MambaVision backbone
- D-RBI
- Signed temporal difference
- ARF decoder
- Boundary residual refinement

## Results

| Dataset | Pre | Rec | F1 | IoU | OA |
|---|---:|---:|---:|---:|---:|
| DSIFN-CD | 96.86 | 97.20 | 97.03 | 94.23 | 97.93 |
| WHU-CD | 96.16 | 95.00 | 95.58 | 91.53 | 99.58 |

## Installation

```bash
git clone TODO_REPOSITORY_URL
cd MambaRefine-CD
pip install -r requirements.txt
```

## Dataset Preparation

Update dataset roots in:

```text
configs/datasets/dsifn.yaml
configs/datasets/whu.yaml
configs/experiments/dsifn_full.yaml
configs/experiments/whu_full.yaml
```

Expected active layout:

```text
data/
  DSIFN-CD/
  WHU-CD/
```

## Training

```bash
python scripts/train.py --config configs/experiments/dsifn_full.yaml
python scripts/train.py --config configs/experiments/whu_full.yaml
```

## Testing

```bash
python scripts/test.py --config configs/experiments/dsifn_full.yaml --ckpt <checkpoint>
python scripts/test.py --config configs/experiments/whu_full.yaml --ckpt <checkpoint>
```

## Ablations

DSIFN ablations are kept under:

```text
configs/ablations/dsifn/
```

Verify configs before running ablations:

```bash
python tools/verify_ablation_configs.py --config_dir configs/ablations/dsifn/
python tools/check_model_params.py
```

## Project Structure

```text
MambaRefine-CD/
  configs/
    datasets/
    models/
    experiments/
    ablations/dsifn/
  docs/
  scripts/
  src/
  tools/
  tests/
  archive/
```

## Citation

```bibtex
@article{mambarefinecd,
  title={MambaRefine-CD: Efficient Remote-Sensing Change Detection with MambaVision and Region-Boundary Interaction Modeling},
  author={Anonymous},
  year={2026}
}
```
