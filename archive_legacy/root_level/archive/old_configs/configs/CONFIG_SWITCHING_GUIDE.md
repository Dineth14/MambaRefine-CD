# Configuration Switching Guide

This guide explains how to switch the active training configuration in `configs/global_config.yaml`, especially when moving between:

- LEVIR-CD
- WHU-CD
- DSIFN-CD
- multiple LEVIR variants such as tile mode, image mode, or different stride settings

It is written for the current codebase, where the training script reads only one file:

`configs/global_config.yaml`

## The most important rule

Keep the top-level `dataset:` block short.

In the cleaned template, the active dataset is resolved by merging the matching `datasets_catalog` entry into the top-level `dataset:` block. That means switching is easy only if the top-level block contains just the active overrides you really want.

This is the safe pattern:

```yaml
dataset:
  name: WHU-CD
  image_size: 256
  num_workers: 4
```

This is the unsafe pattern:

```yaml
dataset:
  name: WHU-CD
  root: /storage2/ChangeDetection/MV/Datasets/LEVIRCD
```

If a stale override like the LEVIR root remains in the active block, it still wins over the catalog.

## How config loading works

The current loader behavior is:

1. Read `configs/global_config.yaml`.
2. Look at the top-level `dataset.name`.
3. Find the matching entry inside `datasets_catalog`.
4. Merge that catalog entry into the top-level `dataset:` block.
5. Let the top-level `dataset:` values win when the same key appears in both places.

So you should think of `datasets_catalog` as the canonical preset library, and the top-level `dataset:` block as a small active override layer.

The trainer uses the resolved top-level `dataset:` block.

## What the training script actually uses

The current training entrypoint:

`scripts/train.py`

loads only:

`configs/global_config.yaml`

It does not take `configs/experiments/*.yaml` from the command line.

So for normal training, the active configuration is always whatever is inside the top-level sections of `global_config.yaml`.

## Safe workflow for switching datasets

When switching to another dataset, update these sections together:

1. `experiment.name`
2. `dataset:` block
3. optionally `hardware.gpu_ids`
4. optionally `training.batch_size`, `num_workers`, or `lr` if the dataset or GPU memory is different

The safest method is:

1. Change `dataset.name` to the dataset you want.
2. Keep only the top-level overrides you actually need for this run.
3. Comment out LEVIR-only routing keys when you switch to WHU, SYSU, or DSIFN.

This avoids leaving stale LEVIR-specific values behind while still keeping switching fast.

## Which parts are dataset-specific

These keys usually change when switching datasets:

- `dataset.name`
- `dataset.root`
- `dataset.num_workers`
- `dataset.train_split`
- `dataset.val_split`
- `dataset.test_split`
- `dataset.image_a_dir_candidates`
- `dataset.image_b_dir_candidates`
- `dataset.label_dir_candidates`
- `experiment.name`

These keys are mostly model/training settings and usually do not need to change just because the dataset changes:

- `model.*`
- `difference.*`
- `decoder.*`
- `loss.*`
- `ema.*`
- `boundary_metrics.*`

These LEVIR-specific tile settings are meaningful only for LEVIR routing and are not the right template for WHU or DSIFN:

- `train_mode`
- `val_mode`
- `test_mode`
- `tile_size`
- `train_stride`
- `val_stride`
- `test_stride`
- `min_change_pixels`
- `include_empty_ratio`
- `use_tile_cache`
- `tile_cache_dir`
- `boundary_aware_sampling`
- `balance_change_tiles`
- `target_change_tile_ratio`

## Recommended pattern

Treat the top-level `dataset:` block as a small active override layer.

Treat `datasets_catalog:` as a library of canonical presets.

When you switch datasets, start by changing `dataset.name`, then keep only the active overrides that still make sense.

## Example: active LEVIR tile training

This is the recommended style for LEVIR tile-based training:

```yaml
experiment:
  name: levir_rf_adaptive_decoder

dataset:
  name: LEVIR-CD
  image_size: 256
  num_workers: 4
  val_ratio: 0.2
  augment: true
  binary: true
  split: train
  # root: /storage2/ChangeDetection/MV/Datasets/LEVIRCD   # uncomment only to override catalog
  train_mode: "tile"
  val_mode: "existing"
  test_mode: "existing"
  tile_size: 256
  train_stride: 128
  val_stride: 256
  test_stride: 256
  min_change_pixels: 1
  include_empty_ratio: 0.25
  use_tile_cache: true
  tile_cache_dir: "outputs/dataset_indices"
  boundary_aware_sampling: true
  balance_change_tiles: true
  target_change_tile_ratio: 0.5
  image_a_dir_candidates: [A, imageA, t1, A_256]
  image_b_dir_candidates: [B, imageB, t2, B_256]
  label_dir_candidates: [label, labels, mask, OUT]
```

## Example: LEVIR image-mode ablation

If you want a second LEVIR configuration without tile training, keep the same dataset but change only the LEVIR-specific routing fields:

```yaml
experiment:
  name: levir_image_mode_ablation

dataset:
  name: LEVIR-CD
  root: /storage2/ChangeDetection/MV/Datasets/LEVIRCD
  image_size: 256
  num_workers: 4
  val_ratio: 0.2
  augment: true
  binary: true
  split: train
  train_split: train
  val_split: null
  test_split: test
  train_mode: "image"
  val_mode: "existing"
  test_mode: "existing"
  tile_size: 256
  train_stride: 128
  val_stride: 256
  test_stride: 256
  min_change_pixels: 1
  include_empty_ratio: 0.25
  use_tile_cache: true
  tile_cache_dir: "outputs/dataset_indices"
  boundary_aware_sampling: true
  balance_change_tiles: true
  target_change_tile_ratio: 0.5
  image_a_dir_candidates: [A, imageA, t1, A_256]
  image_b_dir_candidates: [B, imageB, t2, B_256]
  label_dir_candidates: [label, labels, mask, OUT]
```

Important:

- `train_mode: "tile"` uses the LEVIR tile dataset path
- `train_mode: "image"` uses the original image-level path
- val and test still use the LEVIR tile-aware evaluation path in the current builder

## Example: switch to WHU-CD

In the cleaned template, switching to WHU usually means:

1. set `dataset.name: WHU-CD`
2. change `experiment.name`
3. comment out the LEVIR-only tile routing keys

The active block can be as small as:

```yaml
experiment:
  name: whu_rf_adaptive_decoder

dataset:
  name: WHU-CD
  image_size: 256
  num_workers: 8
  val_ratio: 0.2
  augment: true
  binary: true
  split: train
  # root: /storage2/ChangeDetection/MV/Datasets/WHU-CD   # uncomment only to override catalog
```

Do not keep LEVIR tile-only keys in the active block if you want a clean, readable WHU config.

WHU supported layouts are defined in `src/data/whucd.py`:

```text
root/
  train/A  train/B  train/label
  val/A    val/B    val/label
  test/A   test/B   test/label
```

or:

```text
root/
  A/  B/  label/
```

## Example: switch to DSIFN-CD

In the cleaned template, switching to DSIFN usually means:

1. set `dataset.name: DSIFN-CD`
2. change `experiment.name`
3. comment out the LEVIR-only tile routing keys

The active block can be as small as:

```yaml
experiment:
  name: dsifn_rf_adaptive_decoder

dataset:
  name: DSIFN-CD
  image_size: 256
  num_workers: 8
  val_ratio: 0.2
  augment: true
  binary: true
  split: train
  # root: /storage2/ChangeDetection/MV/Datasets/DSIFN-CD/DSIFN   # uncomment only to override catalog
```

DSIFN supported layouts are defined in `src/data/dsifncd.py` and include:

```text
root/
  trainset/t1  trainset/t2  trainset/GT
  testset/t1   testset/t2   testset/GT
```

or:

```text
root/
  t1/  t2/  GT/
  train.txt  val.txt  test.txt
```

or flat folders with manual split.

## How to manage multiple LEVIR configurations cleanly

If you have several LEVIR setups, do not try to encode all of them inside the current `datasets_catalog.levir` entry.

The clean approach is:

1. Keep one canonical LEVIR reference under `datasets_catalog.levir`.
2. Keep `dataset.name: LEVIR-CD` and edit only the LEVIR-only overrides in the active top-level `dataset:` section.
3. Change `experiment.name` to match that variant.

Examples of LEVIR variants worth naming explicitly:

- `levir_rf_tile_stride128`
- `levir_rf_image_mode`
- `levir_rf_stride256`
- `levir_rf_no_balanced_sampler`

This is safer than leaving one top-level LEVIR block and trying to remember which fields belong to which experiment.

## Minimal field checklist before starting a run

Before training, verify:

1. `experiment.name` matches the run you want.
2. `dataset.name` is the correct dataset.
3. `dataset.root` points to the correct filesystem path.
4. The split fields match the dataset layout.
5. The directory-candidate lists match the dataset folder names.
6. LEVIR-only tile fields are present only when you are intentionally using LEVIR tile mode.
7. `hardware.gpu_ids[0]` is the GPU you want.

## Quick commands

Train:

```bash
cd /storage2/ChangeDetection/MV/MambaRefine-CD
conda run -n mamba_new python scripts/train.py
```

Train on a specific GPU without editing the config:

```bash
cd /storage2/ChangeDetection/MV/MambaRefine-CD
CUDA_VISIBLE_DEVICES=1 conda run -n mamba_new python scripts/train.py
```

## Recommended sanity check before training

If you want to confirm the resolved active dataset after editing `global_config.yaml`, run:

```bash
cd /storage2/ChangeDetection/MV/MambaRefine-CD
conda run -n mamba_new python - <<'EOF'
from pprint import pprint
from src.utils.config import load_config
cfg = load_config()
pprint(cfg["dataset"].to_dict())
print("device:", cfg["hardware"]["device"])
print("experiment:", cfg["experiment"]["name"])
EOF
```

If the printed `dataset.root` still looks like LEVIR after you intended to switch to WHU or DSIFN, a stale override is still present in the active block.

## Short version

If you only remember one thing, remember this:

When switching datasets, change `dataset.name` first, then remove any stale top-level overrides that no longer belong to that dataset.

That is the safest way to keep the cleaned template easy to edit without accidentally carrying LEVIR-specific settings into WHU or DSIFN runs.