# MERCon-CD: MambaVision Change Detection

**MERCon 2026 — Conference on Machine Intelligence & Remote Sensing**

A clean, research-grade codebase for bi-temporal change detection on LEVIR-CD using the MambaVision transformer backbone and a novel **Localization → Refinement Decoder**.

---

## Problem

Detecting land-cover changes between two satellite images is a fundamental Remote Sensing task. Key challenges are:

- **Thin boundaries** — narrow changed structures (roads, buildings) require high spatial resolution
- **Scale variation** — changed objects appear at many scales
- **False positives** — illumination/seasonal shifts create spurious responses

---

## Key Contributions

### 1. Stable RF Module (from rf_stability_phase)
Softmax-gated adaptive dilation avoids collapse to all-background predictions, giving stable training without GroupNorm workarounds.

### 2. Localization → Refinement Decoder *(new)*
A two-stage decoder that first localises changes coarsely, extracts boundary uncertainty via Sobel gradients, then applies a lightweight residual correction using shallow encoder features:

$$P_f = P_c + \Delta(P_c, E, f_0, f_1)$$

where $E = |\nabla \sigma(P_c)|$ is the boundary edge map and $f_0, f_1$ are the two shallowest encoder scales.

---

## Setup

```bash
conda activate mamba_new
cd mercon_cd_clean
pip install -r requirements.txt
```

Pre-trained MambaVision weights are loaded automatically from `/tmp/mamba_vision_*.pth.tar` (placed there by the existing repo setup).

---

## Training

```bash
# Edit CONFIG_PATH in scripts/train.py, then:
python scripts/train.py
```

Switch the experiment by changing one line in `scripts/train.py`:

```python
CONFIG_PATH = "configs/refinement_decoder.yaml"   # ← change this
```

Available configs:

| Config | Decoder | Variant | Notes |
|---|---|---|---|
| `base.yaml` | baseline | tiny | FPN reference |
| `adaptive_rf.yaml` | adaptive_rf | tiny | Learnable dilation |
| `refinement_decoder.yaml` | refinement | small | **MERCon contribution** |
| `multiscale_stable.yaml` | global_local | small | Two-branch FPN |

---

## Switching Model Variant

In any config file, set `model.variant`:

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
  device: cuda:1
```

---

## Enabling Resume

```yaml
resume:
  enabled: true
  checkpoint_path: outputs/run_XXXXXX_refinement_decoder/checkpoints/best.pth
  strict: true
```

Set `checkpoint_path: null` to auto-find the latest checkpoint under `outputs/`.

---

## Validation

```bash
python scripts/validate.py \
    --config configs/refinement_decoder.yaml \
    --checkpoint outputs/run_XXX/checkpoints/best.pth
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
```yaml
# configs/datasets/levircd.yaml
dataset:
  root: /storage2/ChangeDetection/MV/Datasets/LEVIRCD

# configs/datasets/whucd.yaml
dataset:
  root: /your/path/to/WHU-CD
```

### Train on each dataset
Change one line in `scripts/train.py`:
```python
CONFIG_PATH = "configs/experiments/train_levir_refinement.yaml"   # LEVIR-CD
CONFIG_PATH = "configs/experiments/train_whu_refinement.yaml"     # WHU-CD
CONFIG_PATH = "configs/experiments/train_sysu_refinement.yaml"    # SYSU-CD
CONFIG_PATH = "configs/experiments/train_dsifn_refinement.yaml"   # DSIFN-CD
```

### Evaluate a single dataset
Set `checkpoint.path` in the eval config, then:
```python
# scripts/evaluate.py
CONFIG_PATH = "configs/experiments/eval_levir.yaml"
```
```bash
python scripts/evaluate.py
```

### Run full benchmark (all datasets)
After training all four, fill checkpoint paths in `configs/benchmark_suite.yaml`, then:
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
│   ├── base.yaml
│   ├── datasets/
│   │   ├── levircd.yaml
│   │   ├── whucd.yaml
│   │   ├── sysucd.yaml
│   │   └── dsifncd.yaml
│   ├── experiments/
│   │   ├── train_levir_refinement.yaml
│   │   ├── train_whu_refinement.yaml
│   │   ├── train_sysu_refinement.yaml
│   │   ├── train_dsifn_refinement.yaml
│   │   ├── eval_levir.yaml
│   │   ├── eval_whu.yaml
│   │   ├── eval_sysu.yaml
│   │   └── eval_dsifn.yaml
│   └── benchmark_suite.yaml
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
