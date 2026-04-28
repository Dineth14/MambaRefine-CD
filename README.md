# MambaRefine-CD

Efficient change detection with MambaVision and region-boundary interaction modeling.

![Paper](https://img.shields.io/badge/Paper-TODO-lightgrey)
![Code](https://img.shields.io/badge/Code-PyTorch-blue)
![License](https://img.shields.io/badge/License-TODO-lightgrey)

## Overview

MambaRefine-CD is a research codebase for remote-sensing change detection from bi-temporal imagery. The task is to identify changed regions while preserving accurate object boundaries, which is important for urban monitoring, land-cover analysis, and disaster assessment. The key idea is to decouple change evidence into region-level cues for structural extent and boundary-level cues for contour refinement. The model uses a MambaVision backbone for efficient visual representation learning and adds region-boundary interaction modules for dense change prediction. The repository reports only verified binary change detection metrics; SECOND semantic change detection results are not included because that pipeline is not currently stable.

## Key Features

- MambaVision backbone for bi-temporal feature extraction.
- Differential Region-Boundary Interaction (D-RBI).
- Signed temporal difference modeling.
- Adaptive receptive-field decoder.
- Boundary refinement for sharper change masks.

## Architecture

The pipeline takes two registered images as input, extracts multi-scale temporal features with a shared MambaVision backbone, and forms temporal difference representations. D-RBI separates the fused evidence into region and boundary streams. The adaptive receptive-field decoder aggregates multi-scale region features, while the boundary refinement branch adjusts predictions near object contours.

![Pipeline](figures/pipeline.png)

## Results

Binary change detection results are reported using only Pre, Rec, F1, IoU, and OA. LEVIR-CD values are validation results and are marked accordingly.

| Dataset | Pre | Rec | F1 | IoU | OA |
|---|---:|---:|---:|---:|---:|
| DSIFN-CD | 96.86 | 97.20 | 97.03 | 94.23 | 97.93 |
| WHU-CD | 96.16 | 95.00 | 95.58 | 91.53 | 99.58 |
| LEVIR-CD (validation) | TODO | TODO | 92.29 | 85.69 | 99.27 |

SECOND results are not reported because the current SECOND pipeline is under verification.

## Installation

```bash
git clone TODO_REPOSITORY_URL
cd MambaRefine-CD
pip install -r requirements.txt
```

## Dataset Preparation

Prepare the datasets under `data/` or update the corresponding config paths.

```text
data/
  LEVIR-CD/
  WHU-CD/
  DSIFN-CD/
```

Expected dataset-specific details such as split files, preprocessing options, and patch settings should be verified against the config used for each experiment.

## Training

Example training command:

```bash
python scripts/train.py --config configs/ablations/levir/a6_full.yaml
```

For other datasets, replace the config path with the corresponding WHU-CD or DSIFN-CD configuration.

## Evaluation

Example evaluation command:

```bash
python scripts/test.py --config configs/ablations/levir/a6_full.yaml --ckpt TODO_CKPT_PATH
```

Reported binary metrics:

- Pre
- Rec
- F1
- IoU
- OA

Use the same preprocessing, checkpoint, threshold, and evaluation protocol when comparing validation and test results.

## Ablation Studies

Ablation studies should compare the baseline model against the full model under the same dataset split and evaluation pipeline. Recommended comparisons include:

- Baseline vs. full model.
- Effect of D-RBI.
- Effect of signed temporal difference.
- Effect of boundary refinement.

All missing ablation values should remain as `TODO` until verified.

## Visualization

Recommended qualitative outputs for inspecting predictions:

- Binary prediction maps.
- Boundary maps.
- Error maps or comparison panels, if enabled by the evaluation script.

Visualization outputs should be used for diagnosis only and should not replace quantitative evaluation.

## Project Structure

```text
MambaRefine-CD/
  configs/
  datasets/
  models/
  scripts/
  losses/
  metrics/
  tools/
  tests/
  outputs/
```

## TODO

- Improve and verify SECOND dataset support.
- Add model compression experiments.
- Add more benchmarks.
- Fill missing LEVIR-CD test metrics after verification.
- Add verified paper and citation links when available.

## License

TODO

## Acknowledgement

This work builds upon prior research in change detection and vision models.

## Citation

```bibtex
@article{mambarefinecd,
  title={MambaRefine-CD: Efficient Change Detection with MambaVision and Region-Boundary Interaction Modeling},
  author={Anonymous},
  year={2026}
}
```
