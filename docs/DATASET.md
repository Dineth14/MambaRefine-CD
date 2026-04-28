# LEVIR-CD Dataset Pipeline

## Why tile-based training?

Training previously used **510 image-level samples** (one random 256×256 crop per 1024×1024 image per iteration) while validation and test used **sliding-window tiles** (2032 val tiles, 5568 test tiles).  
This created a train/eval distribution mismatch: the model saw far fewer unique spatial patches at training time than at evaluation time.

The tile-based pipeline fixes this by indexing all valid 256×256 crops from every training image at startup, yielding **~25 000 training tiles** from the same 510 source images with `train_stride=128`.

---

## Dataset structure expected

```
root/
  train/
    A/      # pre-change images (1024×1024 PNG)
    B/      # post-change images
    label/  # binary change masks
  test/
    A/
    B/
    label/
```

`val` is derived by an 80/20 random split of the `train/` folder (seeded, reproducible).

---

## Configuration

In `configs/global_config.yaml`:

```yaml
dataset:
  name: LEVIR-CD
  root: /path/to/LEVIRCD

  # Tile pipeline
  train_mode: "tile"        # "tile" | "image" (image = legacy random-crop)
  val_mode:   "existing"    # always non-overlapping tiles from val images
  test_mode:  "existing"    # always non-overlapping tiles from test images

  tile_size:     256
  train_stride:  128    # overlap → more tiles; 256 = non-overlapping
  val_stride:    256    # non-overlapping tiles for val
  test_stride:   256    # non-overlapping tiles for test

  min_change_pixels:   1
  include_empty_ratio: 0.25   # fraction of no-change tiles to retain

  use_tile_cache:  true
  tile_cache_dir:  "outputs/dataset_indices"

  balance_change_tiles:      true
  target_change_tile_ratio:  0.5
```

### Switching back to image-level training

```yaml
dataset:
  train_mode: "image"
```

This reverts to `LEVIRCDDataset` (one random 256×256 crop per image per iteration).

---

## Expected tile counts (LEVIR-CD, stride=128)

| Split | Source images | Tiles per image | Total tiles |
|-------|--------------|-----------------|-------------|
| train (80%) | ~510 | ~49 | ~24 990 |
| val   (20%) | ~127 | 16  | ~2 032  |
| test        | 348  | 16  | ~5 568  |

*49 tiles per 1024×1024 image at stride 128: positions 0,128,…,768 → 7 per axis → 7×7 = 49.*

---

## Balanced sampling

When `balance_change_tiles: true`, a `BalancedChangeSampler` draws training indices so that each epoch contains approximately `target_change_tile_ratio` (default 0.5) change tiles.

LEVIR-CD has roughly 25–30 % changed pixels overall; without balancing, many batches contain only background tiles.

The sampler samples with replacement from the smaller pool when needed.  
**Only the training split** uses balanced sampling; val/test always use sequential ordering.

---

## Leakage prevention

`src/data/leakage_check.py` verifies at smoke-test time:

1. No filename stem appears in more than one split.
2. No train tile's source image path appears in the val or test index.

The check raises `RuntimeError` on any violation.  
Report saved to `outputs/dataset_inspection/leakage_report.json`.

Because val comes from the **same** 20% of `train/` that is held out from train tiles, there is no overlap by construction. Test images come from a completely separate `test/` directory.

---

## Tile caching

The first call to `build_tile_index()` scans every image and its mask, records change-pixel counts, and writes:

```
outputs/dataset_indices/levircd_train_256_128.csv
outputs/dataset_indices/levircd_train_256_128.meta.json
```

Subsequent calls with the same config parameters load the cached CSV.  
The cache key is a SHA-256 hash of `(n_images, first/last filenames, tile_size, stride, min_change_pixels, include_empty_ratio)`.  
If any parameter changes the key mismatches and the index is rebuilt.

---

## Useful commands

```bash
# Inspect dataset structure and estimate tile counts
python scripts/inspect_dataset_structure.py
```

---

## Manifest

After training starts, a manifest is saved to:

```
outputs/dataset_manifests/levircd_manifest.json
```

It records: root, tile_size, stride, tile counts, positive ratios, and leakage-check status.
