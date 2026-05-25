"""Boundary and edge-IoU metrics for change detection.

Implements tolerance-aware boundary F1 and edge IoU using morphological
operations built from torch.nn.functional max-pool convolutions.

Boundary extraction
-------------------
boundary = dilate(mask) - erode(mask)
  where dilate = max_pool2d(mask, k)
        erode  = -max_pool2d(-mask, k)   (equivalent to min-pool)

Boundary F1 (with tolerance)
-----------------------------
A predicted boundary pixel is matched if it falls within *tolerance* pixels
of any GT boundary pixel (and vice-versa):

  boundary_precision = matched_pred_bnd / (pred_bnd + eps)
  boundary_recall    = matched_gt_bnd   / (gt_bnd   + eps)
  boundary_f1 = 2 * P * R / (P + R + eps)

Edge IoU (without tolerance)
-----------------------------
edge_iou = |pred_bnd ∩ gt_bnd| / |pred_bnd ∪ gt_bnd|

Config example (in dataset or experiment config):
    boundary_metrics:
      enabled:        true
      boundary_width: 3     # morphological kernel half-width
      tolerance:      2     # tolerance for boundary F1 (pixels)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class BoundaryMetrics:
    """Streaming boundary F1 and edge IoU.

    Args:
        boundary_width: half-width of the morphological kernel used to
            extract boundary maps.  Kernel size = 2*w+1.
        tolerance:      tolerance in pixels for boundary F1 matching.
        threshold:      sigmoid threshold to binarise logits.
    """

    def __init__(
        self,
        boundary_width: int = 3,
        tolerance: int = 2,
        threshold: float = 0.5,
    ) -> None:
        self.bw  = boundary_width
        self.tol = tolerance
        self.thr = threshold
        self.reset()

    def reset(self) -> None:
        self.prec_num  = 0.0   # sum of matched predicted boundary pixels
        self.prec_den  = 0.0   # sum of predicted boundary pixels
        self.rec_num   = 0.0   # sum of matched GT boundary pixels
        self.rec_den   = 0.0   # sum of GT boundary pixels
        self.edge_inter = 0.0
        self.edge_union = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary map via morphological gradient."""
        k   = 2 * self.bw + 1
        pad = self.bw
        dilated = F.max_pool2d(mask, k, stride=1, padding=pad)
        eroded  = -F.max_pool2d(-mask, k, stride=1, padding=pad)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _dilate(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Dilate a binary mask by *radius* pixels."""
        if radius == 0:
            return mask
        k   = 2 * radius + 1
        pad = radius
        return (F.max_pool2d(mask, k, stride=1, padding=pad) > 0).float()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate statistics from a batch.

        Args:
            preds:   raw logits OR probabilities, shape (B,1,H,W) or (B,H,W).
            targets: binary ground truth, same spatial shape as preds.
        """
        with torch.no_grad():
            probs  = torch.sigmoid(preds) if not (0 <= preds.min() and preds.max() <= 1) else preds
            pred_b = (probs > self.thr).float()
            tgt    = targets.float()

            # Ensure 4-D (B,1,H,W)
            if pred_b.dim() == 3:
                pred_b = pred_b.unsqueeze(1)
            if tgt.dim() == 3:
                tgt = tgt.unsqueeze(1)

            pred_bnd = self._boundary(pred_b)   # (B,1,H,W)
            gt_bnd   = self._boundary(tgt)

            # Tolerance-dilated versions for matching
            gt_dilated   = self._dilate(gt_bnd,   self.tol)
            pred_dilated = self._dilate(pred_bnd, self.tol)

            # Boundary precision: pred bnd pixels that hit dilated GT bnd
            self.prec_num += (pred_bnd * gt_dilated).sum().item()
            self.prec_den += pred_bnd.sum().item()

            # Boundary recall: GT bnd pixels that hit dilated pred bnd
            self.rec_num += (gt_bnd * pred_dilated).sum().item()
            self.rec_den += gt_bnd.sum().item()

            # Edge IoU (no tolerance)
            inter = (pred_bnd * gt_bnd).sum().item()
            union = ((pred_bnd + gt_bnd).clamp(0, 1)).sum().item()
            self.edge_inter += inter
            self.edge_union  += union

    def compute(self) -> dict:
        """Return a dict with boundary_f1, boundary_precision, boundary_recall, edge_iou."""
        eps = 1e-6
        prec = self.prec_num / (self.prec_den + eps)
        rec  = self.rec_num  / (self.rec_den  + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        iou  = self.edge_inter / (self.edge_union + eps)
        return {
            "boundary_f1":        f1,
            "boundary_precision": prec,
            "boundary_recall":    rec,
            "edge_iou":           iou,
        }


def build_boundary_metrics(cfg: dict) -> BoundaryMetrics:
    """Build a BoundaryMetrics instance from config dict.

    Reads from ``cfg["boundary_metrics"]`` with defaults:
      boundary_width: 3
      tolerance:      2
    and from ``cfg["evaluation"]["threshold"]`` (default 0.5).
    """
    bm_cfg = cfg.get("boundary_metrics", {})
    thr    = cfg.get("evaluation", {}).get("threshold", 0.5)
    return BoundaryMetrics(
        boundary_width = int(bm_cfg.get("boundary_width", 3)),
        tolerance      = int(bm_cfg.get("tolerance", 2)),
        threshold      = float(thr),
    )
