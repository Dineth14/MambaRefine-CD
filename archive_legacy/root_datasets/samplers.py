"""Balanced sampler for binary change detection datasets.

Addresses the common problem of change/no-change imbalance in binary
change detection datasets.

Config keys (dataset.balance section)
---------------------------------------
    enabled:               true
    min_change_ratio:      0.001   # exclude near-empty patches
    max_nochange_fraction: 0.5     # cap fraction of pure no-change samples
    oversample_change:     true    # oversample changed patches
    changed_patch_weight:  2.0     # weight multiplier for changed patches
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch.utils.data import WeightedRandomSampler


def build_balanced_sampler(
    dataset,
    min_change_ratio: float = 0.001,
    changed_patch_weight: float = 2.0,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples changed patches.

    Requires the dataset to expose:
        dataset.change_ratios: List[float]  — per-sample change ratio

    If change_ratios is unavailable, falls back to uniform sampling.

    Args:
        dataset:              Dataset instance.
        min_change_ratio:     Minimum ratio to consider a sample as "changed".
        changed_patch_weight: Weight multiplier for changed samples.
        num_samples:          Number of samples per epoch (default: len(dataset)).
    """
    n = len(dataset)
    if num_samples is None:
        num_samples = n

    change_ratios: Optional[List[float]] = getattr(dataset, "change_ratios", None)
    if change_ratios is None:
        # Fallback: uniform
        weights = [1.0] * n
    else:
        weights = [
            changed_patch_weight if r >= min_change_ratio else 1.0
            for r in change_ratios
        ]

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=num_samples,
        replacement=True,
    )
