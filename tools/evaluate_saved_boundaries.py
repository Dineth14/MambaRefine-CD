"""Evaluate ORBIT-compatible BF1 from frozen binary prediction masks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.metrics import boundary_prf_batch


def _load_binary(path: Path, evaluation_size: int | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        if evaluation_size is not None:
            image = image.resize((evaluation_size, evaluation_size), resample=Image.Resampling.NEAREST)
        values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        values = values.reshape(image.height, image.width)
    return (values > 0).float().unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--tolerances", type=int, nargs="+", default=[2])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=None,
        help="Resize predictions and ground truth to this square size with nearest-neighbor interpolation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = {path.stem: path for path in args.pred_dir.iterdir() if path.is_file()}
    ground_truth = {path.stem: path for path in args.gt_dir.iterdir() if path.is_file()}
    if set(predictions) != set(ground_truth):
        missing_pred = sorted(set(ground_truth) - set(predictions))[:5]
        missing_gt = sorted(set(predictions) - set(ground_truth))[:5]
        raise RuntimeError(
            f"Prediction/GT stem mismatch: missing predictions={missing_pred}, missing GT={missing_gt}"
        )
    stems = sorted(predictions)
    if not stems:
        raise RuntimeError("No prediction/ground-truth pairs found")

    sums = {
        tolerance: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0}
        for tolerance in args.tolerances
    }
    for start in range(0, len(stems), args.batch_size):
        batch_stems = stems[start:start + args.batch_size]
        pred = torch.stack([
            _load_binary(predictions[stem], args.evaluation_size) for stem in batch_stems
        ])
        target = torch.stack([
            _load_binary(ground_truth[stem], args.evaluation_size) for stem in batch_stems
        ])
        for tolerance in args.tolerances:
            precision, recall, f1 = boundary_prf_batch(pred, target, tolerance)
            sums[tolerance]["precision"] += float(precision.sum().item())
            sums[tolerance]["recall"] += float(recall.sum().item())
            sums[tolerance]["f1"] += float(f1.sum().item())
            sums[tolerance]["count"] += int(f1.numel())

    results = {}
    for tolerance, values in sums.items():
        count = values["count"]
        results[str(tolerance)] = {
            "boundary_precision": values["precision"] / count,
            "boundary_recall": values["recall"] / count,
            "boundary_f1": values["f1"] / count,
            "sample_count": count,
        }
    payload = {
        "metric": "ORBIT-compatible mean per-image semantic contour BF1",
        "boundary_extraction": "dilation-minus-erosion with radius=tolerance",
        "matching": "symmetric square dilation with radius=tolerance",
        "prediction_dir": str(args.pred_dir.resolve()),
        "ground_truth_dir": str(args.gt_dir.resolve()),
        "evaluation_size": args.evaluation_size,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
