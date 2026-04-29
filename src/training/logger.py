"""Console + file logger with formatted metric tables."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log_table(logger: logging.Logger, metrics: dict, title: str = "") -> None:
    """Print a +---------+--------+ table of metric values."""
    if metrics.get("metric_family") == "second":
        metrics = {key: metrics[key] for key in ("OA", "mIoU", "SeK", "Fscd") if key in metrics}
    elif any(key in metrics for key in ("f1", "iou", "precision", "recall", "oa")):
        metrics = {key: metrics[key] for key in ("precision", "recall", "f1", "iou", "oa") if key in metrics}
    _LABELS = {
        "f1":                  "F1",
        "iou":                 "IoU",
        "precision":           "Precision",
        "recall":              "Recall",
        "oa":                  "OA",
        "boundary_f1":         "Boundary F1",
        "boundary_precision":  "Bnd Precision",
        "boundary_recall":     "Bnd Recall",
        "edge_iou":            "Edge IoU",
        "pred_positive_ratio": "Pred Positive Ratio",
        "gt_positive_ratio":   "GT Positive Ratio",
    }
    rows = [(_LABELS.get(k, k), v) for k, v in metrics.items() if isinstance(v, (int, float))]
    if not rows:
        return
    if title:
        logger.info(title)
    w = max(len(r[0]) for r in rows)
    sep = f"+-{'-' * w}-+-{'-' * 8}-+"
    logger.info(sep)
    logger.info(f"| {'Metric':<{w}} | {'Value':>8} |")
    logger.info(sep)
    for name, val in rows:
        logger.info(f"| {name:<{w}} | {val:>8.4f} |")
    logger.info(sep)
