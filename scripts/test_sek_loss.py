#!/usr/bin/env python3
"""Lightweight checks for the SeK-inspired soft-kappa losses."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.sek_loss import binary_soft_kappa_loss


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    torch.manual_seed(0)

    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

    perfect_logits = torch.tensor([[[[-10.0, 10.0], [10.0, -10.0]]]])
    perfect = binary_soft_kappa_loss(perfect_logits, target, eps=1e-6, separate_nochange=True)
    _assert(float(perfect.loss.item()) < 0.02, "Perfect binary prediction should have near-zero loss.")

    random_logits = torch.randn_like(target)
    random_case = binary_soft_kappa_loss(random_logits, target, eps=1e-6, separate_nochange=True)
    _assert(0.05 < float(random_case.loss.item()) < 1.8, "Random prediction should have intermediate loss.")

    changed_target = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    background_logits = torch.full((1, 1, 2, 2), -10.0)
    background_case = binary_soft_kappa_loss(background_logits, changed_target, eps=1e-6, separate_nochange=True)
    _assert(float(background_case.loss.item()) > 0.9, "All-background prediction on changed target should have high loss.")

    ignore_mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    imperfect_logits = torch.tensor([[[[10.0, 10.0], [10.0, -10.0]]]])
    ignore_case = binary_soft_kappa_loss(
        imperfect_logits,
        target,
        valid_mask=(ignore_mask <= 0.5).float(),
        eps=1e-6,
        separate_nochange=True,
    )
    _assert(float(ignore_case.loss.item()) < 0.02, "Ignored pixels should not affect the loss.")

    empty_valid = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    empty_case = binary_soft_kappa_loss(random_logits, target, valid_mask=empty_valid, eps=1e-6)
    _assert(torch.isfinite(empty_case.loss).item(), "Loss should stay finite when no valid pixels remain.")
    _assert(float(empty_case.loss.item()) == 0.0, "Empty valid mask should produce zero loss.")

    print("SeK-inspired loss tests passed.")


if __name__ == "__main__":
    main()
