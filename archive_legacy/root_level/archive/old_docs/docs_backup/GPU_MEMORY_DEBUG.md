# GPU Memory Debug

This utility profiles GPU memory usage for one/few batches and identifies where memory jumps in the training pipeline.

Run:

```bash
python scripts/debug_gpu_memory.py
```

It reads:

- `configs/global_config.yaml`

No CLI arguments are required.

The memory-debug settings live under:

```yaml
debug:
  memory_debug: true
  output_root: outputs/memory_debug
  compare_modes: [...]
```

## Output location

Each run writes to:

```text
outputs/memory_debug/run_YYYYMMDD_HHMMSS_<debug_name>/
```

Files:

- `memory_report.md`
- `memory_report.csv`
- `memory_report.json`
- `layer_memory.csv`
- `bug_checks.csv`
- `memory_summary_before.txt`
- `memory_summary_after.txt`
- `profiler_trace/` (when profiler is enabled)

## Memory terms

- `allocated`: memory currently held by live tensors.
- `reserved`: memory held by CUDA allocator cache (can be larger than allocated).
- `max_allocated`: peak allocated memory since last `reset_peak_memory_stats`.
- `max_reserved`: peak reserved memory since last `reset_peak_memory_stats`.

## How to interpret key checks

- `return_features` increase:
  If `return_features=true` significantly raises peak memory, intermediate feature tensors may be retained in graph/output payload.

- EMA increase:
  If EMA check reports many shadow tensors on GPU, EMA clone storage is consuming VRAM. Moving shadow weights to CPU reduces GPU usage.

- TTA increase:
  TTA runs multiple augmented forwards. Higher memory in `eval_with_tta` is expected and should be kept out of training mode.

- AMP check:
  If AMP is disabled, activation and gradient memory typically increase.

- Layer metadata table:
  Use top `size_mb` modules to identify heavy outputs (backbone levels, decoder blocks, refinement modules).
