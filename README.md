# MERCon-CD: MambaVision Change Detection with D-RBI

**MERCon 2026 — Conference on Machine Intelligence & Remote Sensing**

A clean, research-grade codebase for bi-temporal change detection on LEVIR-CD using the
MambaVision transformer backbone and a novel **Differential Region–Boundary Interaction (D-RBI)**
fusion module with **Adaptive RF Decoder** featuring boundary residual correction.

---

## Problem

Detecting land-cover changes between two satellite images is a fundamental Remote Sensing task.
Key challenges are:

- **Thin boundaries** — narrow changed structures (roads, buildings) require high spatial resolution
- **Scale variation** — changed objects appear at many scales
- **False positives** — illumination/seasonal shifts create spurious responses

---

## Key Contributions

### 1. Differential Region–Boundary Interaction (D-RBI)

Replaces the naive abs-diff fusion with a learned decomposition into two complementary streams:

- **Region stream** — Bounded sigmoid gate $G_r$ emphasises large coherent changed areas
- **Boundary stream** — Sobel-gradient-conditioned gate $G_b$ emphasises thin high-frequency edges

$$D = \phi([F_1 \| F_2 \| |F_2-F_1| \| F_1 \odot F_2])$$
$$R = G_r \odot D, \qquad B = G_b \odot D$$

The Sobel filter is fixed (zero parameters) — only the gate MLPs $\psi_r, \psi_b$ are learned.

### 2. Adaptive RF Decoder with Boundary Residual Correction

Two-stage pipeline:

1. **Coarse prediction** — Adaptive RF-FPN over region features $\{R_i\}$:
   $P_c = \text{ARF-FPN}(\{R_i\})$
2. **Residual correction** — Boundary feature + Sobel edge of $P_c$ drive a small correction:
   $P_f = P_c + \delta \cdot \tanh(\Delta(B_0, P_c, E))$,  $\delta = 0.1$

Zero-initialised final conv ensures $P_f = P_c$ at init — correction is purely additive.

### 3. Stable RF Module

Softmax-gated adaptive dilation avoids collapse to all-background predictions,
giving stable training without GroupNorm workarounds.

> **Note on Temporal Mamba mode**: An experimental Temporal Mamba feature fusion mode
> was attempted but disabled due to NaN instability from the Mamba SSM selective scan
> kernel running in float16 AMP with $T=2$ sequences. The D-RBI module is fully
> convolutional and runs stably under AMP.

---

## Setup

```bash
conda activate mamba_new
cd MambaRefine-CD
pip install -r requirements.txt
```

Pre-trained MambaVision weights are loaded automatically on first use.

---

## Configuration

One runtime config file controls everything:

```text
configs/global_config.yaml
```

Run training with:

```bash
python scripts/train.py
```

Run model efficiency profiling:

```bash
python scripts/model_efficiency.py
```

### Enabling / disabling D-RBI

```yaml
difference:
  enabled: true   # set false to fall back to abs-diff + sum decoder path
```

### Ablation switches

```yaml
difference:
  use_absdiff: true       # include |F2-F1| in input concat
  use_product: true       # include F1⊙F2 in input concat
  use_region_gate: true   # apply learnable region gate G_r
  use_boundary_gate: true # apply Sobel-conditioned gate G_b

decoder:
  use_boundary_residual: true  # apply Δ boundary correction
  residual_scale: 0.1          # tanh clamp scale δ
```

### Change GPU

```yaml
hardware:
  device: cuda
  gpu_ids: [1]
```

### Change model variant

```yaml
model:
  variant: base   # tiny | tiny2 | small | base | large
```

### Change decoder

```yaml
model:
  decoder: adaptive_rf   # baseline | adaptive_rf | refinement | global_local
```

### Change batch size and iterations

```yaml
training:
  batch_size: 8
  max_iterations: 50000
  validate_every: 5000
```

### Change dataset

Edit `dataset.root` and optionally `dataset.name`. Reusable dataset definitions for
LEVIR-CD, WHU-CD, SYSU-CD, and DSIFN-CD are kept inside the same file under
`datasets_catalog:`.

---

## Switching Model Variant

| Variant | Model | Channels | ~Params |
|---|---|---|---|
| `tiny` | mamba_vision_T | [80, 160, 320, 640] | 32M |
| `small` | mamba_vision_S | [96, 192, 384, 768] | 50M |
| `base` | mamba_vision_B | [128, 256, 512, 1024] | 97M |
| `large` | mamba_vision_L | [196, 392, 784, 1568] | 212M |

---

## Enabling Resume

```yaml
resume:
  enabled: true
  checkpoint_path: outputs/run_XXXXXX_refine_mamba_cd/checkpoints/best.pth

```yaml
model:
  decoder: refinement   # baseline | adaptive_rf | refinement | global_local
```

### Change batch size and iterations

```yaml
training:
  batch_size: 8
  max_iterations: 50000
  validate_every: 5000
```

### Change dataset

Edit the active `dataset:` block, or switch `dataset.name` and `dataset.root`.
Reusable dataset definitions for LEVIR-CD, WHU-CD, SYSU-CD, and DSIFN-CD are
kept inside the same file under `datasets_catalog:`.

### Example

```yaml
experiment:
  name: refine_mamba_cd
  output_root: outputs
  seed: 42

hardware:
  device: cuda
  gpu_ids: [0]
  mixed_precision: true

model:
  backbone: mambavision
  variant: tiny
  decoder: refinement
  pretrained: true

training:
  batch_size: 8
  max_iterations: 50000
  validate_every: 5000
```

---

## Switching Model Variant

Edit `model.variant` in `configs/global_config.yaml`:

```yaml
model:
  variant: small   # tiny | tiny2 | small | base | large
```

| Variant | Model | Channels | ~Params |
|---|---|---|---|
| `tiny` | mamba_vision_T | [80, 160, 320, 640] | 32M |
| `small` | mamba_vision_S | [96, 192, 384, 768] | 50M |
| `base` | mamba_vision_B | [128, 256, 512, 1024] | 97M |
| `large` | mamba_vision_L | [196, 392, 784, 1568] | 212M |

To use a specific GPU (e.g. GPU 1):

```yaml
hardware:
  device: cuda
  gpu_ids: [1]
```

---

## Enabling Resume

```yaml
resume:
  enabled: true
  checkpoint_path: outputs/run_XXXXXX_refine_mamba_cd/checkpoints/best.pth
  strict: true
```

Set `checkpoint_path: null` to auto-find the latest checkpoint under `outputs/`.

---

## Validation

```bash
python scripts/validate.py
```

Set the checkpoint and split in `configs/global_config.yaml`:

```yaml
checkpoint:
  path: outputs/run_XXX/checkpoints/best.pth

validation:
  split: val   # val | test
```

---

## Compare Runs

```bash
python scripts/compare_runs.py outputs/run_A outputs/run_B outputs/run_C
```

---

## Multi-Dataset Benchmark Support

MambaRefine-CD supports training and evaluation on **4 benchmark datasets**:

| Dataset | Purpose | Metrics |
|---------|---------|---------|
| LEVIR-CD | Main benchmark | F1, IoU, mIoU, Prec, Recall, OA, BndF1, EdgeIoU |
| WHU-CD | Boundary-sensitive | F1, IoU, mIoU + **boundary emphasis** |
| SYSU-CD | Generalization | F1, IoU, mIoU |
| DSIFN-CD | Generalization | F1, IoU, mIoU |

### Dataset paths (edit to match your storage)
All dataset paths now live in `configs/global_config.yaml` under `dataset:`
and `datasets_catalog:`.

### Train on each dataset
Change the active dataset section in `configs/global_config.yaml`:

```yaml
dataset:
  name: WHU-CD
  root: /storage2/ChangeDetection/MV/Datasets/WHU-CD
```

### Evaluate a single dataset
Set `checkpoint.path` and `evaluation.split` in `configs/global_config.yaml`, then:

```bash
python scripts/evaluate.py
```

### Run full benchmark (all datasets)
After training all four, fill `benchmark.checkpoints` in `configs/global_config.yaml`, then:

```bash
python scripts/benchmark_all.py
```

Outputs saved to `outputs/benchmark_runs/summary/`:
- `benchmark_results.csv` — all metrics for all datasets
- `benchmark_results.md` — Markdown table
- `latex_tables/core_table.tex` — paper-ready LaTeX
- `latex_tables/boundary_table.tex`
- `latex_tables/generalization_table.tex`
- `generalization_summary.json/.md`

### Check dataset integrity
```bash
python scripts/check_dataset.py
```
Saves manifests to `outputs/dataset_manifests/`.

### Boundary Metrics
Tolerance-aware Boundary F1 and Edge IoU are implemented in
`src/training/boundary_metrics.py`. Configure via:
```yaml
boundary_metrics:
  enabled:        true
  boundary_width: 3
  tolerance:      2
```

See [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md) for full metric formulas, dataset layouts, and paper table generation.

---

## Updated Metrics

After this update, validation logs report:

```
+---------------------+----------+
| Metric              |    Value |
+---------------------+----------+
| F1                  |   0.9182 |
| IoU-change          |   0.8487 |
| mIoU                |   0.9210 |
| Precision           |   0.9291 |
| Recall              |   0.9076 |
| OA                  |   0.9923 |
| Boundary F1         |   0.1919 |
| Edge IoU            |   0.1054 |
| Pred Positive Ratio |   0.0481 |
| GT Positive Ratio   |   0.0473 |
+---------------------+----------+
```

Note: `iou` = change-class IoU; `miou` = mean of both class IoUs.

---

## Updated Folder Structure

```
MambaRefine-CD/
├── configs/
│   └── global_config.yaml
├── src/
│   ├── data/
│   │   ├── levircd.py
│   │   ├── whucd.py           ← NEW
│   │   ├── sysucd.py          ← NEW
│   │   ├── dsifncd.py         ← NEW
│   │   ├── dataset_builder.py ← NEW
│   │   ├── transforms.py      ← NEW
│   │   └── factory.py
│   └── training/
│       ├── metrics.py          (updated: iou, miou, gt_positive_ratio)
│       ├── boundary_metrics.py ← NEW
│       ├── evaluator.py        ← NEW
│       ├── generalization_metrics.py ← NEW
│       └── trainer.py          (updated: BoundaryMetrics, extended CSV)
├── scripts/
│   ├── train.py
│   ├── validate.py
│   ├── evaluate.py     ← NEW
│   ├── benchmark_all.py ← NEW
│   ├── check_dataset.py ← NEW
│   └── compare_runs.py
└── docs/
    ├── BENCHMARK_PROTOCOL.md ← NEW
    ├── method.md
    └── experiments.md
```
├── configs/
│   ├── base.yaml                  # shared defaults
│   ├── adaptive_rf.yaml
│   ├── refinement_decoder.yaml    ← MERCon paper
│   └── multiscale_stable.yaml
├── src/
│   ├── models/
│   │   ├── backbone/
│   │   │   └── mambavision_builder.py
│   │   ├── decoders/
│   │   │   ├── baseline_decoder.py
│   │   │   ├── adaptive_rf_decoder.py
│   │   │   ├── refinement_decoder.py  ← NEW
│   │   │   └── global_local_decoder.py
│   │   └── cd_model.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── checkpoint.py
│   │   └── logger.py
│   ├── data/
│   │   ├── levircd.py
│   │   └── factory.py
│   └── utils/
│       ├── config_loader.py
│       ├── seed.py
│       └── visualization.py
├── scripts/
│   ├── train.py
│   ├── validate.py
│   └── compare_runs.py
└── outputs/
    └── run_<date>_<name>/
        ├── config.yaml
        ├── model_info.json
        ├── logs/train.log
        ├── tensorboard/
        ├── checkpoints/best.pth
        ├── validation/metrics.csv
        └── samples/
```

---

## Output Per Run

Every run automatically produces:

| File | Description |
|---|---|
| `config.yaml` | Exact config used |
| `model_info.json` | Variant, param counts, channel sizes |
| `logs/train.log` | Full training log |
| `tensorboard/` | TensorBoard scalars |
| `checkpoints/best.pth` | Best checkpoint |
| `validation/metrics.csv` | Per-validation-step metrics |
| `samples/iter_*.png` | Prediction grids |
