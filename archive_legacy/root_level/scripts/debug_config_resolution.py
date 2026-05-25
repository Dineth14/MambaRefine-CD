#!/usr/bin/env python3
"""Print and save the fully resolved config used by an ablation run."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from utils.config import GLOBAL_CONFIG_PATH, load_config


class DuplicateKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: DuplicateKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise KeyError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


DuplicateKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _read_raw(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    try:
        yaml.load(text, Loader=DuplicateKeyLoader)
        duplicate_status = "none detected"
    except Exception as exc:
        duplicate_status = f"WARNING: {type(exc).__name__}: {exc}"
    return text, duplicate_status


def _slug(value: str) -> str:
    return str(value).strip().lower().replace("-cd", "").replace("/", "-").replace(" ", "_")


def _resolved_run_dir(cfg: dict) -> Path:
    exp = cfg.get("experiment", {})
    ds = cfg.get("dataset", {})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"debug_{_slug(ds.get('name', 'dataset'))}_{exp.get('name', 'unknown')}_seed{exp.get('seed', 42)}_{ts}"
    return REPO / str(exp.get("output_root", "outputs")) / run_name


def _section(cfg: dict, keys: list[str]) -> dict[str, Any]:
    return {key: cfg.get(key, {}) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug resolved MambaRefine-CD config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", default=None, help="Optional directory for resolved_config.yaml.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    raw_global, global_dupes = _read_raw(GLOBAL_CONFIG_PATH)
    raw_override, override_dupes = _read_raw(config_path)
    cfg = load_config(config_path)
    resolved = cfg.to_dict()

    run_dir = Path(args.run_dir) if args.run_dir else _resolved_run_dir(resolved)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    model_sections = _section(resolved, ["model", "difference", "decoder", "loss", "debug", "efficiency"])
    training_sections = _section(resolved, ["training", "optimizer", "dataloader", "validation", "checkpoint", "evaluation", "eval"])
    resume_sections = _section(resolved, ["resume", "checkpoint"])

    print("=== RAW GLOBAL YAML ===")
    print(raw_global)
    print("=== RAW OVERRIDE YAML ===")
    print(raw_override)
    print("=== DUPLICATE KEY CHECK ===")
    print(f"global_config.yaml: {global_dupes}")
    print(f"{config_path}: {override_dupes}")
    print("=== FULLY RESOLVED CONFIG ===")
    print(yaml.safe_dump(resolved, sort_keys=False))
    print("=== MODEL-RELATED CONFIG ===")
    print(yaml.safe_dump(model_sections, sort_keys=False))
    print("=== TRAINING-RELATED CONFIG ===")
    print(yaml.safe_dump(training_sections, sort_keys=False))
    print("=== RESUME/CHECKPOINT SETTINGS ===")
    print(yaml.safe_dump(resume_sections, sort_keys=False))
    print("=== RUN METADATA ===")
    print(yaml.safe_dump({
        "run_dir": str(run_dir),
        "seed": resolved.get("experiment", {}).get("seed"),
        "variant_name": resolved.get("experiment", {}).get("name"),
        "config_path": str(config_path),
        "resolved_config_path": str(resolved_path),
    }, sort_keys=False))


if __name__ == "__main__":
    main()
