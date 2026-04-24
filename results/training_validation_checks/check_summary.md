# Training Validation Check Report

| Run | Status | Missing Iters | Best F1 | Best Iter | Collapses | Notes |
|-----|--------|---------------|---------|-----------|-----------|-------|
| run_20260101_fail_no_csv | ✗ FAIL | — | — | — | — | validation/val_metrics.csv missing |
| run_20260101_pass | ✓ PASS | — | 0.9190 | 15000 | — | — |
| run_20260101_warn_collapse | ⚠ WARN | — | 0.9100 | 15000 | 1 | COLLAPSE WARNING at iterations: [5000] |
| run_20260101_warn_incomplete | ⚠ WARN | 10000, 15000 | 0.8500 | 5000 | — | Training appears incomplete (last recorded iter=5000). Missing: [10000, 15000] |

## Status Legend

| Status | Meaning |
|--------|---------|
| ✓ PASS | All artefacts present, all expected validation intervals recorded, no collapse |
| ⚠ WARN | Optional files missing, some intervals missing (may be incomplete run), collapse detected |
| ✗ FAIL | config.yaml / val_metrics.csv / best.pth absent, or validation never ran |

## Expected Validation Intervals

Derived from `config.yaml`:
```
training:
  max_iterations: 50000
  validate_every: 5000
# → expects validation at: 5000, 10000, 15000, …, 50000
```