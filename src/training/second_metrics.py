"""Training-time SECOND semantic change metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from metrics.second_scd_metrics import SECONDSCDMetrics


@dataclass
class SECONDMetrics:
    num_classes: int = 7
    ignore_index: int = 255
    compute_sek: bool = True
    sek_binary_fallback: bool = False

    def __post_init__(self) -> None:
        self.metric = SECONDSCDMetrics(self.num_classes, self.ignore_index)

    def update(
        self,
        *,
        change_pred_binary: Optional[torch.Tensor],
        change_pred_semantic: Optional[torch.Tensor],
        change_gt: torch.Tensor,
        pred_sem1: Optional[torch.Tensor] = None,
        pred_sem2: Optional[torch.Tensor] = None,
        gt_sem1: Optional[torch.Tensor] = None,
        gt_sem2: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        del change_pred_binary, change_pred_semantic, change_gt, valid_mask
        if pred_sem1 is None or pred_sem2 is None or gt_sem1 is None or gt_sem2 is None:
            raise RuntimeError("SECOND metrics require semantic predictions for both timestamps.")
        self.metric.update(pred_sem1, pred_sem2, gt_sem1, gt_sem2)

    def compute(self) -> dict[str, float | str]:
        result = self.metric.compute()
        result["metric_family"] = "second"
        result["second_eval_level"] = "semantic_change"
        return result
