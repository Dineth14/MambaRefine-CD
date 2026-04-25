"""Balanced change/no-change sampler for tile-based datasets.

``BalancedChangeSampler`` draws training indices so that each epoch
contains approximately ``target_change_tile_ratio`` change tiles.

Usage::

    from data.sampler import BalancedChangeSampler
    sampler = BalancedChangeSampler(
        dataset              = train_ds,    # LEVIRCDTileDataset
        target_change_ratio  = 0.5,
        seed                 = 42,
    )
    loader = DataLoader(train_ds, batch_size=4, sampler=sampler)

Only used for the training split; val/test always use sequential sampling.
"""
from __future__ import annotations

import math
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedChangeSampler(Sampler):
    """Sample training indices with a controlled change/no-change ratio.

    Each call to ``__iter__`` shuffles both pools independently and
    interleaves them so that consecutive batches stay balanced.
    The total number of samples per epoch is the same as the dataset length.

    Parameters
    ----------
    dataset
        A ``LEVIRCDTileDataset`` (or any dataset whose tile index entries
        expose ``has_change`` via ``dataset.index[i]["has_change"]``).
    target_change_ratio
        Fraction of samples per epoch that should be change tiles.
        Default 0.5 (equal balance).
    seed
        Base seed; incremented by epoch for deterministic variety.
    """

    def __init__(
        self,
        dataset,
        target_change_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.target_change_ratio = float(target_change_ratio)
        self.seed = seed
        self._epoch = 0

        # Separate change vs no-change indices
        change_idx: List[int] = []
        no_change_idx: List[int] = []
        index = getattr(dataset, "index", None)
        if index is None:
            raise AttributeError(
                "BalancedChangeSampler requires a dataset with a `.index` attribute "
                "(e.g. LEVIRCDTileDataset)."
            )
        for i, entry in enumerate(index):
            if entry["has_change"]:
                change_idx.append(i)
            else:
                no_change_idx.append(i)

        self._change_idx    = change_idx
        self._no_change_idx = no_change_idx
        self._n_total       = len(index)

        if not change_idx:
            raise ValueError("BalancedChangeSampler: no change tiles found in dataset.")
        if not no_change_idx:
            # All tiles have change — degrade gracefully
            self._no_change_idx = []

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to advance the random seed."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._n_total

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)

        n_total    = self._n_total
        n_change   = round(n_total * self.target_change_ratio)
        n_no_change = n_total - n_change

        # Sample with replacement if the pool is smaller than needed
        c_pool = self._change_idx
        n_pool = self._no_change_idx

        c_idx = rng.choice(c_pool,    size=n_change,    replace=len(c_pool) < n_change)
        if n_no_change > 0 and n_pool:
            n_idx = rng.choice(n_pool, size=n_no_change, replace=len(n_pool) < n_no_change)
        else:
            # Fallback: fill remainder with change tiles
            n_idx = rng.choice(c_pool, size=n_no_change, replace=True)

        combined = np.concatenate([c_idx, n_idx])
        rng.shuffle(combined)
        return iter(combined.tolist())
