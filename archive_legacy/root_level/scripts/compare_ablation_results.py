#!/usr/bin/env python3
"""Compare ablation CSV results against a selected full/baseline variant."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUT_FIELDS = [
    "variant",
    "Pre",
    "Rec",
    "F1",
    "IoU",
    "OA",
    "Delta_F1_vs_Full",
    "Delta_IoU_vs_Full",
    "Delta_OA_vs_Full",
    "params_M",
    "peak_mem_GB",
    "status",
]


def _first(row: dict, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _metric(row: dict, keys: tuple[str, ...]) -> float | None:
    raw = _first(row, keys)
    if raw in ("", "NA", "nan", "None"):
        return None
    value = float(raw)
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def _number(row: dict, keys: tuple[str, ...]) -> float | None:
    raw = _first(row, keys)
    if raw in ("", "NA", "nan", "None"):
        return None
    return float(raw)


def _variant(row: dict) -> str:
    return _first(row, ("variant", "experiment", "variant_name", "method"), "unknown")


def _read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def _find_full(rows: list[dict], full_variant: str) -> dict:
    target = full_variant.lower()
    for row in rows:
        if _variant(row).lower() == target:
            return row
    for row in rows:
        if target in _variant(row).lower():
            return row
    raise ValueError(f"Could not find full_variant={full_variant!r} in input rows.")


def _interpret(delta_f1: float, mem_delta_pct: float | None, variant: str) -> str:
    lower = variant.lower()
    if "light" in lower and mem_delta_pct is not None and mem_delta_pct > -10.0:
        return "Efficiency ablation ineffective."
    if delta_f1 > 0.2:
        return "Component may not be useful or training variance is high. Re-run seed."
    drop = -delta_f1
    if abs(delta_f1) < 0.1:
        return "Weak contribution. Need repeat seeds or better analysis."
    if 0.2 <= drop <= 0.5:
        return "Moderate contribution."
    if drop > 0.5:
        return "Strong contribution."
    return "Small contribution."


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# Ablation Effectiveness Report",
        "",
        "| Variant | Pre | Rec | F1 | IoU | OA | Delta F1 | Delta IoU | Delta OA | Params(M) | Peak Mem(GB) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {Pre} | {Rec} | {F1} | {IoU} | {OA} | {Delta_F1_vs_Full} | "
            "{Delta_IoU_vs_Full} | {Delta_OA_vs_Full} | {params_M} | {peak_mem_GB} | {status} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ablation result CSV files.")
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSV files.")
    parser.add_argument("--full_variant", default="A0", help="Variant name or substring used as the reference.")
    parser.add_argument("--out_csv", default="outputs/ablation_effectiveness_report.csv")
    parser.add_argument("--out_md", default="outputs/ablation_effectiveness_report.md")
    args = parser.parse_args()

    rows = _read_rows([Path(p) for p in args.csv])
    rows = [row for row in rows if _metric(row, ("F1", "F1_1", "f1")) is not None]
    if not rows:
        raise ValueError("No rows with F1/F1_1/f1 metrics found.")
    full = _find_full(rows, args.full_variant)
    full_f1 = _metric(full, ("F1", "F1_1", "f1"))
    full_iou = _metric(full, ("IoU", "IoU_1", "iou"))
    full_oa = _metric(full, ("OA", "oa"))
    full_mem = _number(full, ("peak_mem_GB", "peak_test_mem_GB", "peak_train_mem_GB"))
    assert full_f1 is not None and full_iou is not None and full_oa is not None

    out_rows = []
    for row in rows:
        variant = _variant(row)
        pre = _metric(row, ("Pre", "Precision_1", "precision"))
        rec = _metric(row, ("Rec", "Recall_1", "recall"))
        f1 = _metric(row, ("F1", "F1_1", "f1"))
        iou = _metric(row, ("IoU", "IoU_1", "iou"))
        oa = _metric(row, ("OA", "oa"))
        mem = _number(row, ("peak_mem_GB", "peak_test_mem_GB", "peak_train_mem_GB"))
        params = _first(row, ("params_M", "Params(M)"))
        delta_f1 = (f1 or 0.0) - full_f1
        delta_iou = (iou or 0.0) - full_iou
        delta_oa = (oa or 0.0) - full_oa
        mem_delta_pct = None
        if full_mem and mem is not None:
            mem_delta_pct = (mem - full_mem) / full_mem * 100.0
        out_rows.append({
            "variant": variant,
            "Pre": _fmt(pre),
            "Rec": _fmt(rec),
            "F1": _fmt(f1),
            "IoU": _fmt(iou),
            "OA": _fmt(oa),
            "Delta_F1_vs_Full": _fmt(delta_f1),
            "Delta_IoU_vs_Full": _fmt(delta_iou),
            "Delta_OA_vs_Full": _fmt(delta_oa),
            "params_M": params,
            "peak_mem_GB": _fmt(mem),
            "status": _interpret(delta_f1, mem_delta_pct, variant),
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_markdown(out_rows), encoding="utf-8")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()
