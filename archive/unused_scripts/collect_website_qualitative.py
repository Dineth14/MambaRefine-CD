#!/usr/bin/env python3
"""Collect qualitative images for the website from local outputs."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
QUAL_DIR = ROOT / "website" / "assets" / "qualitative"
MANIFEST = QUAL_DIR / "manifest.json"

DATASETS = {
    "levir": "LEVIR-CD",
    "whu": "WHU-CD",
    "dsifn": "DSIFN-CD",
    "second": "SECOND",
}
KEYWORDS = ["qualitative", "sample", "samples", "prediction", "pred", "grid", "vis", "result", "eval"]
MODELS = ["mambarefine-cd", "changeformer", "bit", "snunet", "stanet", "mamba-cd"]


def _infer_dataset(path: Path) -> str | None:
    lower = str(path).lower()
    for key, dataset in DATASETS.items():
        if key in lower:
            return dataset
    return None


def _score(path: Path) -> tuple[int, float]:
    lower = str(path).lower()
    score = sum(keyword in lower for keyword in KEYWORDS)
    if "sota_reproduced_eval" in lower:
        score += 5
    if "qualitative_grid" in lower:
        score += 3
    return (score, path.stat().st_mtime)


def _infer_model(path: Path) -> str | None:
    lower = str(path).lower()
    for name in MODELS:
        if name in lower:
            return name
    return None


def main() -> None:
    QUAL_DIR.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Path]] = defaultdict(list)

    for path in OUTPUTS.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        dataset = _infer_dataset(path)
        if not dataset:
            continue
        grouped[dataset].append(path)

    manifest_entries = []
    for dataset in ["LEVIR-CD", "WHU-CD", "DSIFN-CD", "SECOND"]:
        paths = sorted(grouped.get(dataset, []), key=_score, reverse=True)
        copied = 0
        for source in paths:
            if copied >= 8:
                break
            target_name = f"{dataset.lower().replace('-cd', '').replace('-', '_')}_{copied + 1:02d}{source.suffix.lower()}"
            target = QUAL_DIR / target_name
            shutil.copy2(source, target)
            manifest_entries.append(
                {
                    "dataset": dataset,
                    "file": f"assets/qualitative/{target.name}",
                    "source": str(source.resolve()),
                    "caption": f"{dataset} qualitative result" + (f" ({_infer_model(source)})" if _infer_model(source) else ""),
                }
            )
            copied += 1
        if copied == 0:
            manifest_entries.append(
                {
                    "dataset": dataset,
                    "file": None,
                    "source": None,
                    "caption": "Qualitative result pending. Run scripts/collect_website_qualitative.py after evaluation.",
                    "missing": True,
                }
            )

    MANIFEST.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "items": manifest_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
