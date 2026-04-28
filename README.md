# MERCon: Efficient Region-Boundary Interaction for Remote-Sensing Change Detection

## Overview

MERCon is a modular PyTorch codebase for remote-sensing change detection. The repository supports binary change detection on LEVIR-CD, WHU-CD, and DSIFN-CD, and semantic change detection on SECOND. The design is intended to provide a clean, reproducible implementation of efficient region-boundary interaction modules and is designed to compete with recent Mamba-based change detection methods.

The codebase separates binary change detection and semantic change detection protocols. Dataset selection, MambaVision variant selection, ablation selection, metric selection, loss weights, output paths, and training schedules are controlled from `configs/base.yaml`, with thin dataset presets under `configs/datasets/` and thin ablation presets under `configs/ablations/`.

## Key Contributions

- A unified configuration-driven pipeline for binary and semantic remote-sensing change detection.
- A lightweight MERCon architecture with explicit region and boundary streams.
- Dataset-specific metric protocols that avoid mixing binary CD metrics with SECOND semantic-change metrics.
- Iteration-based training with automatic validation, qualitative outputs, best-checkpoint selection, and post-training testing.
- Mamba-CD-style 256 x 256 non-overlapping patch preparation utilities for binary CD datasets.

## Model Architecture

MERCon consists of:

1. Siamese shared-weight original MambaVision encoder
2. Differential Region-Boundary Interaction module
3. Adaptive Receptive Field FPN decoder
4. CRAM-lite change-region attention
5. Boundary-supervised residual refinement
6. SECOND-specific semantic decoder with optional frequency fusion and SeK loss

Binary MERCon predicts a final binary change logit, a coarse binary change logit, and an optional boundary logit. MERCon-SECOND predicts a binary change logit, semantic maps for T1 and T2, a semantic-change map, and an optional boundary logit.

The encoder is loaded from the original MambaVision implementation configured by `model.backbone.repo`. The default path is `/storage2/ChangeDetection/MV/MambaVisionCD`. The repository does not use a CNN substitute or any silent fallback backbone.

## Supported Datasets

| Dataset | Task | Model | Main Metrics |
|---|---|---|---|
| LEVIR-CD | Binary CD | `mercon_binary` | Pre, Rec, F1, IoU, OA |
| WHU-CD | Binary CD | `mercon_binary` | Pre, Rec, F1, IoU, OA |
| DSIFN-CD | Binary CD | `mercon_binary` | Pre, Rec, F1, IoU, OA |
| SECOND | Semantic CD | `mercon_second` | OA, mIoU, F_scd, SeK |

## Dataset Preparation

Binary datasets are expected in this processed layout:

```text
dataset_root/
  train/A/
  train/B/
  train/label/
  val/A/
  val/B/
  val/label/
  test/A/
  test/B/
  test/label/
```

Alternative folder names such as `t1`, `t2`, `mask`, `image1`, and `image2` are supported by the dataset loader.

SECOND uses semantic labels for both timestamps:

```text
dataset_root/
  train/A/
  train/B/
  train/label_a/
  train/label_b/
  val/A/
  val/B/
  val/label_a/
  val/label_b/
  test/A/
  test/B/
  test/label_a/
  test/label_b/
```

For binary datasets, prepare Mamba-CD-style non-overlapping 256 x 256 patches:

```text
python tools/prepare_binary_cd_patches.py --dataset levir --raw-root /data/raw/LEVIR-CD --out-root /data/processed/LEVIR-CD --patch-size 256 --stride 256 --non-overlap true --split mamba_cd
python tools/prepare_binary_cd_patches.py --dataset whu --raw-root /data/raw/WHU-CD --out-root /data/processed/WHU-CD --patch-size 256 --stride 256 --non-overlap true --split mamba_cd
python tools/prepare_binary_cd_patches.py --dataset dsifn --raw-root /data/raw/DSIFN-CD --out-root /data/processed/DSIFN-CD --patch-size 256 --stride 256 --non-overlap true --split mamba_cd
```

Verify a processed dataset:

```text
python tools/check_dataset.py --dataset levir --root /data/processed/LEVIR-CD
python tools/check_dataset.py --dataset whu --root /data/processed/WHU-CD
```

LEVIR-CD follows the Mamba-CD 256 x 256 non-overlapping patch protocol with 7120/1024/2048 train/validation/test pairs. WHU-CD follows Mamba-CD-style 256 x 256 non-overlapping patching with 6096 training pairs.

## Metric Protocol

For LEVIR-CD, WHU-CD, and DSIFN-CD, MERCon reports only:

```text
Pre, Rec, F1, IoU, OA
```

For SECOND, MERCon reports only:

```text
OA, mIoU, F_scd, SeK
```

Binary metrics are implemented in `metrics/binary_cd_metrics.py`. SECOND semantic-change metrics are implemented separately in `metrics/second_scd_metrics.py`.

## Training

Training is iteration-based. The default schedule is `train.max_iters: 50000`. Validation runs every `train.val_interval` iterations, and training loss is logged every `train.log_interval` iterations. Only the best checkpoint is saved.

The active dataset and ablation are selected in `configs/base.yaml`:

```text
config:
  dataset_profile: levir   # levir | whu | dsifn | second
  ablation_profile: null   # null | A0_baseline_fpn | ... | S4_second_full
```

The active MambaVision encoder is also selected in `configs/base.yaml`:

```text
model:
  variant: small           # tiny | tiny2 | small | base | large
```

```text
python train.py
```

Optional overrides use `key=value` syntax:

```text
python train.py config.dataset_profile=whu model.variant=tiny2 output.experiment_name=whu_tiny2
```

MambaVision variant switching is config-driven. Set either `model.variant=<tiny|tiny2|small|base|large>` or `model.backbone.variant=<...>` and the loader will synchronize the matching decoder width, official pretrained URL, and expected feature channels automatically.

```text
python train.py model.variant=tiny2 output.experiment_name=levir_tiny2
python train.py config.dataset_profile=whu model.variant=large output.experiment_name=whu_large
```

To pre-download official checkpoints into `weights/mambavision/`:

```text
python tools/download_mambavision_weights.py --variant all
```

GPU selection is config-driven through `device` and `gpu_id`. For example, `device: cuda` with `gpu_id: 1` uses `cuda:1`. MERCon uses the original MambaVision encoder and therefore requires CUDA for training and inference; no CPU or CNN fallback is used. The same override mechanism can be used at launch:

```text
python train.py gpu_id=1
```

At startup, training writes a hardware and model summary to `run.log` and `startup_summary.json`, including experiment name, dataset, MERCon variant, parameter count, GFLOPs, selected GPU, memory usage, utilization when available, and measured FPS.

Debug run:

```text
python train.py train.max_iters=10 output.experiment_name=debug_levir
```

VS Code launch configurations are provided under `.vscode/launch.json` at the workspace root. You can run train, validate, test, and ablation flows from the Run and Debug panel without typing terminal commands.

## Validation

```text
python validate.py
python validate.py --ckpt outputs/levir_mercon_full/best_checkpoint.pth
```

Validation appends dataset-appropriate metrics to `outputs/{experiment_name}/validation_results.csv`.

## Testing

```text
python test.py
python test.py --ckpt outputs/levir_mercon_full/best_checkpoint.pth
```

After training reaches `train.max_iters`, testing is automatically run with the best checkpoint when `train.auto_test_after_train: true`.

## Ablation Studies

Run one ablation:

```text
python run_ablation.py
python run_ablation.py config.ablation_profile=A7_full_binary
```

`run_ablation.py` accepts the same YAML and `key=value` override interface as `train.py`. When launched from `configs/base.yaml`, set `config.ablation_profile` first.

## Output Structure

Each experiment writes to:

```text
outputs/{experiment_name}/
  config.yaml
  train_log.csv
  train_log.json
  best_checkpoint.pth
  best_metrics.json
  validation_results.csv
  validation_results.json
  test_results.csv
  test_results.json
  dataset_summary.json
  tensorboard/
  qualitative/
  predictions/test/
```

The pipeline saves only the best checkpoint and does not write epoch checkpoints.

## Reproducibility Checklist

- Set `seed` in YAML.
- Set `dataset.root` in YAML or with a command-line override.
- Keep binary CD metrics restricted to `Pre, Rec, F1, IoU, OA`.
- Keep SECOND metrics restricted to `OA, mIoU, F_scd, SeK`.
- Use `train.max_iters` for training length.
- Save all artifacts under `outputs/{experiment_name}/`.
- Use `tools/check_dataset.py` before training to verify split counts.
- Use `tools/count_params_flops.py` to report Params and FLOPs.

## Citation

```bibtex
@misc{mercon2026,
  title  = {MERCon: Efficient Region-Boundary Interaction for Remote-Sensing Change Detection},
  author = {Anonymous},
  year   = {2026},
  note   = {Citation placeholder}
}
```

## Acknowledgements

This repository is designed around established remote-sensing change detection protocols, including binary CD evaluation on LEVIR-CD, WHU-CD, and DSIFN-CD, and semantic change detection evaluation on SECOND. The architecture and experimental interface are intended for reproducible comparison with recent Mamba-based change detection methods.
