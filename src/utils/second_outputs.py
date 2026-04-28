"""SECOND semantic prediction helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


SECOND_COLOR_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (0, 128, 0),
    2: (128, 128, 128),
    3: (0, 255, 0),
    4: (0, 0, 255),
    5: (128, 0, 0),
    6: (255, 0, 0),
}


def second_semantic_predictions(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pred_t1, pred_t2, semantic-derived change mask."""
    sem1 = outputs.get("sem_logits_t1", outputs.get("semantic_t1_logits"))
    sem2 = outputs.get("sem_logits_t2", outputs.get("semantic_t2_logits"))
    if sem1 is None or sem2 is None:
        raise KeyError("SECOND evaluation requires sem_logits_t1 and sem_logits_t2.")
    pred_t1 = torch.argmax(sem1, dim=1)
    pred_t2 = torch.argmax(sem2, dim=1)
    pred_change = pred_t1 != pred_t2
    return pred_t1, pred_t2, pred_change


def colorize_second(labels: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    arr = arr.astype(np.int64)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for class_id, color in SECOND_COLOR_PALETTE.items():
        out[arr == class_id] = color
    return out


def save_second_prediction_batch(
    *,
    pred_t1: torch.Tensor,
    pred_t2: torch.Tensor,
    pred_change: torch.Tensor,
    sample_ids: Iterable[str],
    output_root: Path,
    binary_head_logits: torch.Tensor | None = None,
    save_visualizations: bool = True,
    save_binary_head_change: bool = False,
    threshold: float = 0.5,
) -> None:
    """Save SECOND predictions in the required directory structure."""
    pred_root = output_root / "predictions"
    vis_root = output_root / "visualizations"
    dirs = {
        "sem_t1": pred_root / "pred_sem_t1",
        "sem_t2": pred_root / "pred_sem_t2",
        "change": pred_root / "pred_change",
        "binary_head": pred_root / "pred_change_binary_head",
        "vis_sem_t1": vis_root / "sem_t1",
        "vis_sem_t2": vis_root / "sem_t2",
        "vis_change": vis_root / "change",
    }
    for key, path in dirs.items():
        if key == "binary_head" and not (binary_head_logits is not None and save_binary_head_change):
            continue
        if key.startswith("vis_") and not save_visualizations:
            continue
        path.mkdir(parents=True, exist_ok=True)

    pred_t1_cpu = pred_t1.detach().cpu()
    pred_t2_cpu = pred_t2.detach().cpu()
    pred_change_cpu = pred_change.detach().cpu().bool()
    binary_head_cpu = binary_head_logits.detach().cpu() if binary_head_logits is not None else None

    for i, raw_id in enumerate(sample_ids):
        sample_id = Path(str(raw_id)).stem
        sem1 = pred_t1_cpu[i].numpy().astype(np.uint8)
        sem2 = pred_t2_cpu[i].numpy().astype(np.uint8)
        change = pred_change_cpu[i].numpy().astype(np.uint8) * 255

        Image.fromarray(sem1).save(dirs["sem_t1"] / f"{sample_id}.png")
        Image.fromarray(sem2).save(dirs["sem_t2"] / f"{sample_id}.png")
        Image.fromarray(change).save(dirs["change"] / f"{sample_id}.png")

        if binary_head_cpu is not None and save_binary_head_change:
            head = (torch.sigmoid(binary_head_cpu[i, 0]) > threshold).numpy().astype(np.uint8) * 255
            Image.fromarray(head).save(dirs["binary_head"] / f"{sample_id}.png")

        if save_visualizations:
            Image.fromarray(colorize_second(sem1)).save(dirs["vis_sem_t1"] / f"{sample_id}.png")
            Image.fromarray(colorize_second(sem2)).save(dirs["vis_sem_t2"] / f"{sample_id}.png")
            Image.fromarray(change, mode="L").convert("RGB").save(dirs["vis_change"] / f"{sample_id}.png")


def assert_second_prediction_dirs(output_root: Path) -> None:
    pred_root = output_root / "predictions"
    required = ["pred_sem_t1", "pred_sem_t2", "pred_change"]
    empty = []
    for name in required:
        folder = pred_root / name
        if not folder.is_dir() or not any(folder.glob("*.png")):
            empty.append(str(folder))
    if empty:
        raise RuntimeError("SECOND prediction folders are empty: " + ", ".join(empty))
