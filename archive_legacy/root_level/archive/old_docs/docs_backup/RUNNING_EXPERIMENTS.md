# Running Experiments

This guide covers training, evaluation, testing, inference, and analysis for all datasets.

---

## Prerequisites

```bash
conda activate mamba_new
cd /storage2/ChangeDetection/MV/MambaRefine-CD
export PYTHONPATH=$PWD:$PWD/src:$PYTHONPATH
```

---

## Training

```bash
# LEVIR-CD (binary CD)
python scripts/train.py --config configs/ablations/levir/a4_full.yaml

# WHU-CD
python scripts/train.py --config configs/ablations/whu/a4_full.yaml

# DSIFN-CD
python scripts/train.py --config configs/ablations/dsifn/a4_full.yaml

# SECOND (semantic CD)
python scripts/train.py --config configs/ablations/second/a4_full.yaml

# Resume from checkpoint
python scripts/train.py --config configs/ablations/levir/a4_full.yaml \
    --resume outputs/levir/a4_full/checkpoints/best.pth

# Dry run (verify model builds and runs one batch)
python scripts/train.py --config configs/ablations/levir/a0_baseline.yaml --dry_run
```

Outputs are saved under the path specified by `experiment.output_root` in the config,
e.g. `outputs/levir/a4_full/`.

### Ablations

| Stage | Config | Notes |
|-------|--------|-------|
| A0 | `configs/ablations/*/a0_baseline.yaml` | No signed diff, no CRAM-lite |
| A1 | `configs/ablations/*/a1_signed_diff.yaml` | + signed difference |
| A2 | `configs/ablations/*/a2_cga_lite.yaml` | + CRAM-lite / CGA-lite |
| A3 | `configs/ablations/*/a3_boundary_loss.yaml` | + boundary loss (binary) |
| A3 | `configs/ablations/second/a3_sek_loss.yaml` | + SeK loss (SECOND) |
| A4 | `configs/ablations/*/a4_full.yaml` | Full model |

---

## Evaluation (val or test split)

```bash
# Val split
python scripts/evaluate.py \
    --config configs/ablations/levir/a4_full.yaml \
    --ckpt   outputs/levir/a4_full/checkpoints/best.pth \
    --split  val

# Test split
python scripts/evaluate.py \
    --config configs/ablations/second/a4_full.yaml \
    --ckpt   outputs/second/a4_full/checkpoints/best.pth \
    --split  test
```

Saves `metrics_val.json` and `metrics_val.csv` (or `_test`) under `experiment.output_root`.

Allowed metrics per dataset:
- LEVIR/WHU/DSIFN: `Pre`, `Rec`, `F1`, `IoU`, `OA`
- SECOND: `OA`, `mIoU`, `SeK`, `Fscd`

---

## Testing (held-out test split + optional predictions)

```bash
python scripts/test.py \
    --config configs/ablations/levir/a4_full.yaml \
    --ckpt   outputs/levir/a4_full/checkpoints/best.pth \
    --save_predictions
```

Saves `metrics.json`, `metrics.csv`, and optionally `predictions/*.png` under `output_root/`.

---

## Inference (no ground truth required)

```bash
# Single image pair
python scripts/infer.py \
    --config configs/ablations/levir/a4_full.yaml \
    --ckpt   outputs/levir/a4_full/checkpoints/best.pth \
    --img_a  /path/to/before.png \
    --img_b  /path/to/after.png \
    --out    /path/to/output/

# Folder with A/ and B/ subfolders
python scripts/infer.py \
    --config configs/ablations/levir/a4_full.yaml \
    --ckpt   outputs/levir/a4_full/checkpoints/best.pth \
    --folder /path/to/pairs/ \
    --out    outputs/infer/levir/
```

Outputs: `prob/*.png` (probability map) and `binary/*.png` (thresholded mask).

---

## Model Parameters and FLOPs

```bash
python scripts/count_params_flops.py \
    --config     configs/ablations/levir/a4_full.yaml \
    --image_size 256
```

Requires `fvcore` (`pip install fvcore`). Falls back to parameter count only if unavailable.

---

## Dataset Preparation

### LEVIR-CD balanced split stats

```bash
python scripts/prepare_levir_balanced_splits.py \
    --data_root /storage2/ChangeDetection/MV/MambaRefine-CD/data/LEVIRCD \
    --out       data/LEVIRCD/split_stats.csv
```

---

## Tools

### Validate dataset loading

```bash
python tools/validate_dataset.py --config configs/datasets/levir.yaml
python tools/validate_dataset.py --all   # check all datasets
```

### Verify metrics are correct

```bash
python tools/verify_metrics.py
```

### Export results table

```bash
# CSV (default)
python tools/export_results_table.py --outputs_root outputs/

# Markdown
python tools/export_results_table.py --outputs_root outputs/ --format markdown
```

---

## Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Metric tests only (no dataset required)
python -m pytest tests/test_binary_metrics.py tests/test_second_metrics.py -v

# Dataset tests (requires dataset paths via environment variables)
LEVIR_ROOT=/path/to/LEVIR   python -m pytest tests/test_dataset_loading.py -v
SECOND_ROOT=/path/to/SECOND python -m pytest tests/test_dataset_loading.py -v
```

---

## Output Directory Structure

```
outputs/
  levir/
    a4_full/
      checkpoints/
        best.pth
        last.pth
      metrics_val.json
      metrics_val.csv
      metrics_test.json
      metrics_test.csv
      params_flops.json
      predictions/          # (if --save_predictions)
  second/
    a4_full/
      ...
```

---

## Config Layout Reference

```
configs/
  datasets/
    levir.yaml       # dataset + metric restriction
    whu.yaml
    dsifn.yaml
    second.yaml
  models/
    mambarefinecd_base.yaml
    mambarefinecd_full.yaml
    mambarefinecd_light.yaml
  ablations/
    levir/a0_baseline.yaml … a4_full.yaml
    whu/  a0_baseline.yaml … a4_full.yaml
    dsifn/a0_baseline.yaml … a4_full.yaml
    second/a0_baseline.yaml … a4_full.yaml
```
