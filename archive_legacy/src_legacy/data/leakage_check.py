"""Data leakage checker for train / val / test splits.

Verifies that:
  1. No filename appears in more than one split.
  2. No train tile is derived from a val/test source image.
  3. No val/test source image path appears anywhere in the train tile index.

Usage::

    from data.leakage_check import check_leakage
    check_leakage(
        train_index = [...],   # list[dict] from build_tile_index
        val_index   = [...],
        test_index  = [...],
        out_path    = Path("outputs/dataset_inspection/leakage_report.json"),
    )

Raises ``RuntimeError`` if any leakage is detected.
Saves a JSON report regardless.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


def check_leakage(
    train_index: List[dict],
    val_index:   List[dict],
    test_index:  List[dict],
    out_path: Optional[Path] = None,
) -> dict:
    """Run all leakage checks and return a report dict.

    Parameters
    ----------
    train_index / val_index / test_index
        Tile index lists as returned by ``build_tile_index``.
        Each entry must have at least the key ``image_a_path``.
    out_path
        If provided, write the report JSON to this path.

    Raises
    ------
    RuntimeError
        If any leakage is found.
    """
    report: dict = {
        "status": "PASS",
        "checks": {},
    }

    # ── Extract stem-level identifiers ────────────────────────────────────────
    def _stems(index: List[dict]) -> set[str]:
        return {Path(e["image_a_path"]).stem for e in index}

    def _abs_paths(index: List[dict]) -> set[str]:
        return {str(Path(e["image_a_path"]).resolve()) for e in index}

    train_stems = _stems(train_index)
    val_stems   = _stems(val_index)
    test_stems  = _stems(test_index)

    train_paths = _abs_paths(train_index)
    val_paths   = _abs_paths(val_index)
    test_paths  = _abs_paths(test_index)

    issues: list[str] = []

    # ── Check 1: filename overlap ─────────────────────────────────────────────
    tv_overlap = train_stems & val_stems
    tt_overlap = train_stems & test_stems
    vt_overlap = val_stems   & test_stems

    report["checks"]["train_val_stem_overlap"]  = sorted(tv_overlap)
    report["checks"]["train_test_stem_overlap"] = sorted(tt_overlap)
    report["checks"]["val_test_stem_overlap"]   = sorted(vt_overlap)

    if tv_overlap:
        issues.append(
            f"Train/val filename overlap ({len(tv_overlap)} files): "
            + ", ".join(sorted(tv_overlap)[:5])
        )
    if tt_overlap:
        issues.append(
            f"Train/test filename overlap ({len(tt_overlap)} files): "
            + ", ".join(sorted(tt_overlap)[:5])
        )

    # ── Check 2: train tile source in val/test source paths ───────────────────
    train_in_val  = train_paths & val_paths
    train_in_test = train_paths & test_paths

    report["checks"]["train_path_in_val"]  = sorted(train_in_val)
    report["checks"]["train_path_in_test"] = sorted(train_in_test)

    if train_in_val:
        issues.append(
            f"Train tile source image(s) also in val index ({len(train_in_val)}): "
            + ", ".join(sorted(train_in_val)[:3])
        )
    if train_in_test:
        issues.append(
            f"Train tile source image(s) also in test index ({len(train_in_test)}): "
            + ", ".join(sorted(train_in_test)[:3])
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    report["train_unique_sources"] = len(train_stems)
    report["val_unique_sources"]   = len(val_stems)
    report["test_unique_sources"]  = len(test_stems)
    report["train_tiles"]  = len(train_index)
    report["val_tiles"]    = len(val_index)
    report["test_tiles"]   = len(test_index)
    report["issues"]       = issues

    if issues:
        report["status"] = "FAIL"

    # ── Save report ───────────────────────────────────────────────────────────
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Leakage report saved to {out_path}")

    if issues:
        raise RuntimeError(
            "DATA LEAKAGE DETECTED:\n" + "\n".join(f"  - {i}" for i in issues)
        )

    print(f"Leakage check: PASS  (train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)} unique sources)")
    return report
