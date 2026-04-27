#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sota_reproduce_config.yaml"
RESOLVED_PATH = ROOT / "configs" / "sota_reproduce_config.resolved.yaml"
REPORT_DIR = ROOT / "outputs" / "sota_reproduced_eval" / "reports"
SEARCH_ROOTS = [
    ROOT / "external_weights",
    ROOT / "external",
    ROOT / "outputs" / "full_debug_sota_eval",
    ROOT / "outputs" / "sota_reproduced_eval",
]
EXTS = {".pth", ".pt", ".ckpt"}


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _find_ckpts(root: Path, dataset_name: str, model_name: str) -> list[Path]:
    dataset_key = dataset_name.lower().replace("-cd", "").replace("-", "")
    model_aliases = [
        model_name.lower(),
        model_name.lower().replace("-", ""),
        model_name.lower().replace("-", "_"),
    ]
    results = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        lower = str(path).lower().replace("-", "")
        if dataset_key not in lower:
            continue
        if not any(alias.replace("-", "") in lower for alias in model_aliases):
            continue
        results.append(path)
    results.sort(key=lambda item: ("best" not in item.name.lower(), len(str(item)), str(item)))
    return results


def main() -> None:
    cfg = _load_cfg()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    resolved = cfg.copy()
    for model_name, model_cfg in resolved.get("external_models", {}).items():
        discovered = {}
        for dataset_name in cfg.get("datasets", {}).get("active", []):
            candidates = []
            for root in SEARCH_ROOTS:
                if root.exists():
                    candidates.extend(_find_ckpts(root, dataset_name, model_name))
            seen = []
            deduped = []
            for path in candidates:
                key = str(path.resolve())
                if key not in seen:
                    seen.append(key)
                    deduped.append(path)
            discovered[dataset_name] = str(deduped[0].relative_to(ROOT)) if deduped else None
            report.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "resolved_checkpoint": discovered[dataset_name],
                    "candidates": [str(path.relative_to(ROOT)) for path in deduped],
                }
            )
        model_cfg["resolved_checkpoints"] = discovered

    RESOLVED_PATH.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    json_path = REPORT_DIR / "checkpoint_discovery.json"
    md_path = REPORT_DIR / "checkpoint_discovery.md"
    json_path.write_text(json.dumps({"results": report}, indent=2), encoding="utf-8")
    lines = ["# Checkpoint Discovery", "", "| Model | Dataset | Resolved Checkpoint | Candidate Count |", "| --- | --- | --- | --- |"]
    for row in report:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['resolved_checkpoint'] or ''} | {len(row['candidates'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESOLVED_PATH}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
