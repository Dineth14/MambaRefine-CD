#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sota_reproduce_config.yaml"
REPORT_DIR = ROOT / "outputs" / "sota_reproduced_eval" / "reports"
CKPT_EXTS = {".pth", ".pt", ".ckpt"}


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _find_ckpts(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in CKPT_EXTS:
            found.append(path)
    return sorted(found)


def _extract_archives(root: Path) -> None:
    for archive in root.glob("*.zip"):
        try:
            with zipfile.ZipFile(archive, "r") as handle:
                handle.extractall(root)
        except Exception:
            continue


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> None:
    cfg = _load_cfg()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name, model_cfg in cfg.get("external_models", {}).items():
        if not bool(model_cfg.get("enabled", True)):
            continue
        base_weights_dir = ROOT / model_cfg["weights_dir"]
        base_weights_dir.mkdir(parents=True, exist_ok=True)
        official = model_cfg.get("official_weights", {})
        for dataset_name, spec in official.items():
            dataset_dir = base_weights_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "model": model_name,
                "dataset": dataset_name,
                "weights_dir": str(dataset_dir.relative_to(ROOT)),
                "status": "",
                "checkpoint": None,
                "message": "",
            }
            existing = _find_ckpts(dataset_dir)
            if existing:
                record["status"] = "FOUND_EXISTING"
                record["checkpoint"] = str(existing[0].relative_to(ROOT))
                results.append(record)
                continue
            _extract_archives(dataset_dir)
            existing = _find_ckpts(dataset_dir)
            if existing:
                record["status"] = "FOUND_EXISTING"
                record["checkpoint"] = str(existing[0].relative_to(ROOT))
                record["message"] = "checkpoint discovered after extracting existing archive"
                results.append(record)
                continue

            download_type = str(spec.get("type", "missing"))
            url = spec.get("url")
            if url and download_type == "github_release_or_manual":
                archive_path = dataset_dir / Path(url).name
                try:
                    if not archive_path.exists():
                        _download(str(url), archive_path)
                    _extract_archives(dataset_dir)
                    checkpoints = _find_ckpts(dataset_dir)
                    if checkpoints:
                        record["status"] = "DOWNLOADED"
                        record["checkpoint"] = str(checkpoints[0].relative_to(ROOT))
                    else:
                        record["status"] = "FAILED"
                        record["message"] = "downloaded archive but no checkpoint file was found"
                except Exception as exc:
                    record["status"] = "FAILED"
                    record["message"] = str(exc)
            elif download_type in {"google_drive_or_manual", "manual"}:
                record["status"] = "MANUAL_REQUIRED"
                note = spec.get("note") or f"Place a checkpoint under {dataset_dir.relative_to(ROOT)}/"
                record["message"] = str(note)
            elif download_type == "missing":
                record["status"] = "MISSING"
                record["message"] = "No official public checkpoint URL is configured."
            else:
                record["status"] = "MANUAL_REQUIRED"
                record["message"] = f"Unsupported automatic download type: {download_type}"
            results.append(record)

    json_path = REPORT_DIR / "weight_download_status.json"
    md_path = REPORT_DIR / "weight_download_status.md"
    json_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    lines = ["# Weight Download Status", "", "| Model | Dataset | Status | Checkpoint | Message |", "| --- | --- | --- | --- | --- |"]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['status']} | {row['checkpoint'] or ''} | {row['message'] or ''} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
