# SECOND Speed Optimization

SECOND can train noticeably slower than LEVIR-CD, WHU-CD, or DSIFN-CD even at the same batch size because the bottleneck usually sits on the CPU side. The common causes are split discovery at startup, semantic label decoding, repeated `label_t1 != label_t2` mask generation, and slow host-to-device transfer.

## What Changed

The SECOND path now avoids repeated directory scans by caching split manifests in `outputs/dataset_indices/SECOND_index_train.json`, `SECOND_index_val.json`, and `SECOND_index_test.json`. Each manifest stores the resolved file paths, sample id, image size, and validity flag so training does not need to rebuild the file index every epoch.

For binary-mode runs, cached PNG masks under `outputs/second_binary_masks/<split>/` replace per-sample `label_a != label_b` work inside `__getitem__`. The loader now prefers `cv2.imread` when OpenCV is present, keeps RGB conversion to a single pass, loads masks as grayscale `uint8`, and skips semantic-label decoding entirely on the binary path once a cached binary mask exists.

Optional RAM caches are available through:

```yaml
dataset:
  cache_images_in_ram: false
  cache_masks_in_ram: false
```

When either cache is enabled, the dataset logs the estimated RAM footprint before priming the cache.

## Recommended Config

Start with:

```yaml
dataset:
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
  precompute_second_binary_masks: true
  second_binary_cache_dir: outputs/second_binary_masks
  cache_images_in_ram: false
  cache_masks_in_ram: false

training:
  non_blocking_transfer: true
```

If the profiler still shows the CPU ahead of the GPU, enable mask precompute first. Only enable RAM caches after checking available memory.

## Precompute Masks

Run:

```bash
cd MambaRefine-CD
python3 scripts/precompute_second_masks.py
```

This creates:

- `outputs/second_binary_masks/train/`
- `outputs/second_binary_masks/val/`
- `outputs/second_binary_masks/test/`

and writes `outputs/second_binary_masks/precompute_summary.json` with:

- masks created
- changed pixel ratio
- ignored pixel ratio

## Run The Profiler

Run:

```bash
cd MambaRefine-CD
python3 scripts/profile_second_speed.py
```

The profiler reads `configs/global_config.yaml`, runs a short warmup plus a small number of training iterations, and saves:

- `outputs/profiling/second_speed_profile.json`
- `outputs/profiling/second_speed_profile.csv`

It reports:

- dataloader batch loading time
- CPU-side load, mask, and transform time
- GPU transfer time
- forward time
- backward time
- optimizer step time
- GPU memory
- GPU utilization when `nvidia-smi` is available

## How To Interpret Bottlenecks

`Data loading time / iteration` vs `GPU compute time / iteration` is the main signal.

If `data_loading_time > gpu_compute_time`, the profiler prints:

`GPU is waiting on CPU dataloader. Increase num_workers, enable mask precompute, enable persistent workers, or cache masks.`

That means the GPU is under-fed and the next optimizations should stay on the data path.

If `gpu_compute_time` dominates, the profiler prints:

`Model compute is bottleneck.`

That means the loader is no longer the main limiter and further speedups will need model-side changes rather than dataloader tuning.

## Image Size Notes

The SECOND loader logs sample image sizes during initialization. If all images already match the configured tile size, it avoids resize work on the hot path. If the dataset uses larger uniform images, it crops or tiles them directly rather than resizing every batch. Masks always stay on nearest-neighbor semantics; images stay on bilinear semantics if resize is ever required by a future path.
