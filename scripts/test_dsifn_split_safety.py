#!/usr/bin/env python3
"""Safety tests for DSIFN-CD explicit split enforcement."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from dsifn_audit_utils import validate_explicit_splits


def make_fake_dsifn(root: Path, names: list[str]) -> None:
    for folder in ("t1", "t2", "mask"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(names):
        Image.new("RGB", (512, 512), (idx * 17 % 255, 20, 40)).save(root / "t1" / name)
        Image.new("RGB", (512, 512), (idx * 17 % 255, 50, 80)).save(root / "t2" / name)
        Image.new("L", (512, 512), 255 if idx % 2 else 0).save(root / "mask" / name)


def write_split(root: Path, train: list[str], val: list[str], test: list[str]) -> Path:
    split_dir = root / "splits"
    split_dir.mkdir(exist_ok=True)
    (split_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    return split_dir


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def try_loader(root: Path, split_dir: Path, split: str):
    try:
        import sys

        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "src"))
        from data.dsifncd import DSIFNCDDataset
    except Exception as exc:
        print(f"SKIP: torch-dependent loader test unavailable ({exc})")
        return None
    return DSIFNCDDataset(root=root, split=split, split_dir=split_dir, require_explicit_splits=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dsifn_split_safety_") as tmp:
        root = Path(tmp) / "DSIFN"
        names = [f"{i:05d}.png" for i in range(10)]
        make_fake_dsifn(root, names)

        missing = validate_explicit_splits({"root": str(root), "split_dir": str(root / "splits")})
        assert_true(missing["verdict"] == "FAIL", "flat layout without split files is rejected")

        split_dir = write_split(root, names[:7], names[7:8], names[8:])
        clean = validate_explicit_splits({"root": str(root), "split_dir": str(split_dir)})
        assert_true(clean["verdict"] == "PASS", "valid explicit split files pass integrity check")

        ds = try_loader(root, split_dir, "test")
        if ds is not None:
            assert_true(len(ds.names) == 2, "test loader image count matches test.txt")
            assert_true(len(ds) == 8, "test loader tile count matches 2 images x 4 tiles")

        write_split(root, names[:7], names[7:8], [names[0], names[8]])
        overlap = validate_explicit_splits({"root": str(root), "split_dir": str(split_dir)})
        assert_true(overlap["verdict"] == "FAIL", "overlapping train/test split files are rejected")
        try:
            ds = try_loader(root, split_dir, "test")
            if ds is not None:
                raise AssertionError("Loader accepted overlapping split files")
        except RuntimeError as exc:
            assert_true("DATA LEAKAGE FOUND" in str(exc), "loader fails when train/test split files overlap")

        write_split(root, names[:7], names[7:8], names)
        all_test = validate_explicit_splits({"root": str(root), "split_dir": str(split_dir)})
        assert_true(all_test["verdict"] == "FAIL", "test split containing all images is rejected through overlap detection")

        write_split(root, names[:7], names[7:8], names[8:])
        final = validate_explicit_splits({"root": str(root), "split_dir": str(split_dir)})
        assert_true(final["verdict"] == "PASS", "clean split audit returns PASS")

    print("DSIFN split safety tests completed.")


if __name__ == "__main__":
    main()
