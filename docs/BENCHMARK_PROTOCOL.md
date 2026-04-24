# Benchmark Protocol

## Overview

MambaRefine-CD supports multi-dataset benchmarking for change detection.
Three benchmark tiers are used in the MERCon paper:

| Tier | Dataset | Purpose |
|------|---------|---------|
| Main | LEVIR-CD | Primary building change benchmark |
| Boundary | WHU-CD | Boundary-sensitive evaluation |
| Generalization | SYSU-CD / DSIFN-CD | Cross-domain generalization |

---

## Supported Datasets

### 1. LEVIR-CD
- **Type**: Building change detection
- **Images**: 637 image pairs (1024×1024), split into train/test
- **Positive ratio**: ~4–6% of pixels
- **Reference**: Chen et al., 2020

**Expected folder layout:**
```
Datasets/LEVIRCD/
  train/
    A/        ← time-1 RGB images (.png)
    B/        ← time-2 RGB images (.png)
    label/    ← binary change masks (.png)
  test/
    A/  B/  label/
```

### 2. WHU-CD
- **Type**: Building change detection (aerial)
- **Images**: 32507 sub-image pairs (256×256)
- **Positive ratio**: ~3.3%
- **Note**: Boundary metrics are especially important for this dataset

**Expected folder layout:**
```
Datasets/WHU-CD/
  train/   A/  B/  label/
  val/     A/  B/  label/
  test/    A/  B/  label/
```

### 3. SYSU-CD
- **Type**: Multi-class urban change detection
- **Classes**: 6 change types (all merged to binary for CD evaluation)
- **Images**: 20000 pairs (256×256)

**Expected folder layout (multiple variants auto-detected):**
```
Datasets/SYSU-CD/
  train/   A/ (or t1/)  B/ (or t2/)  label/ (or mask/)
  test/    A/  B/  label/
```
OR with split text files:
```
  A/  B/  label/
  train.txt  val.txt  test.txt
```

### 4. DSIFN-CD
- **Type**: Urban change detection (multi-city)
- **Images**: Pairs from 6 cities
- **Reference**: Zhang et al., 2021

**Expected folder layout:**
```
Datasets/DSIFN-CD/
  trainset/ (or train/)
    t1/  t2/  GT/
  testset/ (or test/)
    t1/  t2/  GT/
```

---

## Folder Auto-Detection

All dataset loaders try multiple candidate folder names automatically.
You can override candidates in each dataset config:

```yaml
# configs/datasets/levircd.yaml
dataset:
  image_a_dir_candidates: ["A", "imageA", "t1", "A_256"]
  image_b_dir_candidates: ["B", "imageB", "t2", "B_256"]
  label_dir_candidates:   ["label", "labels", "mask", "OUT"]
```

Run `python scripts/check_dataset.py` to verify detection works for your layout.

---

## Metrics

### Core Metrics

From binary confusion matrix (TP, FP, FN, TN):

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{IoU}_\text{change} = \frac{TP}{TP + FP + FN}$$

$$\text{IoU}_\text{nochange} = \frac{TN}{TN + FP + FN}$$

$$\text{mIoU} = \frac{\text{IoU}_\text{change} + \text{IoU}_\text{nochange}}{2}$$

$$\text{OA} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Ratio Metrics

$$\text{PredRatio} = \frac{\text{predicted change pixels}}{\text{total pixels}}$$

$$\text{GTRatio} = \frac{\text{GT change pixels}}{\text{total pixels}}$$

### Boundary Metrics (src/training/boundary_metrics.py)

**Boundary extraction** via morphological gradient:

$$\text{boundary} = \text{dilate}(M) - \text{erode}(M)$$

where `dilate` and `erode` use a kernel of size $2w+1$ (configurable via `boundary_width`).

**Tolerance-aware Boundary F1:**

A predicted boundary pixel is matched if it falls within $\tau$ pixels of
any GT boundary pixel (and vice-versa):

$$\text{BndPrec} = \frac{|P_b \cap \text{dilate}(G_b)|}{|P_b|}$$

$$\text{BndRecall} = \frac{|G_b \cap \text{dilate}(P_b)|}{|G_b|}$$

$$\text{BndF1} = \frac{2 \cdot \text{BndPrec} \cdot \text{BndRecall}}{\text{BndPrec} + \text{BndRecall}}$$

**Edge IoU** (no tolerance):

$$\text{EdgeIoU} = \frac{|P_b \cap G_b|}{|P_b \cup G_b|}$$

Configure in dataset or experiment config:
```yaml
boundary_metrics:
  enabled:        true
  boundary_width: 3     # morphological kernel half-width
  tolerance:      2     # tolerance in pixels for BndF1
```

---

## Threshold

All metrics are computed at a configurable threshold applied to `sigmoid(logits)`:

```yaml
evaluation:
  threshold: 0.5
```

---

## How to Change Dataset Config

To point a dataset to a different root:

```yaml
# configs/datasets/whucd.yaml
dataset:
  root: /path/to/your/WHU-CD
```

The experiment config references the dataset config via:
```yaml
# configs/experiments/train_whu_refinement.yaml
dataset_config: ../datasets/whucd.yaml
```

---

## Training

### LEVIR-CD
Edit `scripts/train.py`:
```python
CONFIG_PATH = "configs/experiments/train_levir_refinement.yaml"
```
Then:
```bash
conda activate mamba_new
cd MambaRefine-CD
python scripts/train.py
```

### WHU-CD
```python
CONFIG_PATH = "configs/experiments/train_whu_refinement.yaml"
```

### SYSU-CD
```python
CONFIG_PATH = "configs/experiments/train_sysu_refinement.yaml"
```

### DSIFN-CD
```python
CONFIG_PATH = "configs/experiments/train_dsifn_refinement.yaml"
```

To switch the backbone variant in any config:
```yaml
model:
  variant: small   # tiny | tiny2 | small | base | large
```

---

## Evaluation (Single Dataset)

1. Set `checkpoint.path` in the eval config:
```yaml
# configs/experiments/eval_levir.yaml
checkpoint:
  path: outputs/benchmark_runs/run_XXX/checkpoints/best.pth
```

2. Change `CONFIG_PATH` in `scripts/evaluate.py`:
```python
CONFIG_PATH = "configs/experiments/eval_levir.yaml"
```

3. Run:
```bash
python scripts/evaluate.py
```

Outputs are saved to `outputs/eval_runs/<timestamp>/`.

---

## Benchmark All Datasets

1. After training all four datasets, fill in the checkpoint paths in `configs/benchmark_suite.yaml`:
```yaml
checkpoints:
  levir: outputs/benchmark_runs/run_XXX_refinement_levir/checkpoints/best.pth
  whu:   outputs/benchmark_runs/run_XXX_refinement_whu/checkpoints/best.pth
  sysu:  outputs/benchmark_runs/run_XXX_refinement_sysu/checkpoints/best.pth
  dsifn: outputs/benchmark_runs/run_XXX_refinement_dsifn/checkpoints/best.pth
```

2. Run:
```bash
python scripts/benchmark_all.py
```

---

## Paper Table Outputs

After `benchmark_all.py`, tables are saved to:

```
outputs/benchmark_runs/summary/
  benchmark_results.csv           ← all metrics, all datasets
  benchmark_results.md            ← Markdown table
  latex_tables/
    core_table.tex                ← F1, IoU, mIoU, Prec, Recall, OA
    boundary_table.tex            ← BndF1, EdgeIoU, PredRatio, GTRatio
    generalization_table.tex      ← cross-dataset F1, Mean, Std
  generalization_summary.json
  generalization_summary.md
```

The LaTeX tables use `\toprule`, `\midrule`, `\bottomrule` from the
`booktabs` package. Include in your LaTeX preamble:
```latex
\usepackage{booktabs}
```

---

## Checking Dataset Integrity

```bash
python scripts/check_dataset.py
```

This checks all four dataset roots and saves manifests to:
```
outputs/dataset_manifests/
  LEVIR-CD_manifest.json
  WHU-CD_manifest.json
  SYSU-CD_manifest.json
  DSIFN-CD_manifest.json
```

Each manifest includes: root path, A/B/mask counts per split,
image sizes, and estimated change pixel ratio.
