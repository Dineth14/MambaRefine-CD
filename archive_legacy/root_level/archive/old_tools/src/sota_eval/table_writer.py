from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

DATASETS = ["LEVIR-CD", "WHU-CD", "DSIFN-CD"]
COLUMNS = [
    "Model",
    "Dataset",
    "Params_M",
    "FLOPs_G",
    "mF1",
    "F1_1",
    "mIoU",
    "IoU_1",
    "Precision_1",
    "Recall_1",
    "OA",
    "Boundary_F1",
    "Edge_IoU",
    "Threshold",
    "Status",
    "Source",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(COLUMNS) + " |"
    rule = "| " + " | ".join(["---"] * len(COLUMNS)) + " |"
    lines = [header, rule]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = "l" * len(COLUMNS)
    lines = [
        "\\begin{tabular}{" + cols + "}",
        "\\hline",
        " & ".join(COLUMNS) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(str(row.get(column, "")) for column in COLUMNS) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
