#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sota_reproduce_config.yaml"
REPORT_DIR = ROOT / "outputs" / "sota_reproduced_eval" / "reports"


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=False)


def _safe_fetch(repo_dir: Path) -> tuple[bool, str]:
    if not (repo_dir / ".git").exists():
        return False, "not a git repository"
    status = _run(["git", "status", "--porcelain"], cwd=repo_dir)
    if status.returncode != 0:
        return False, status.stderr.strip() or status.stdout.strip()
    if status.stdout.strip():
        return False, "local modifications present; fetch skipped"
    fetched = _run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
    if fetched.returncode != 0:
        return False, fetched.stderr.strip() or fetched.stdout.strip()
    return True, "git fetch completed"


def main() -> None:
    cfg = _load_cfg()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name, model_cfg in cfg.get("external_models", {}).items():
        if not bool(model_cfg.get("enabled", True)):
            continue
        repo_dir = ROOT / model_cfg["repo_dir"]
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": model_name,
            "repo_url": model_cfg.get("repo_url"),
            "repo_dir": str(repo_dir.relative_to(ROOT)),
            "status": "",
            "error": "",
        }
        if repo_dir.exists():
            ok, message = _safe_fetch(repo_dir)
            record["status"] = "EXISTS"
            record["error"] = "" if ok else message
        else:
            clone = _run(["git", "clone", model_cfg["repo_url"], str(repo_dir)])
            if clone.returncode == 0:
                record["status"] = "CLONED"
            else:
                record["status"] = "FAILED"
                record["error"] = clone.stderr.strip() or clone.stdout.strip()
        results.append(record)

    json_path = REPORT_DIR / "repo_clone_status.json"
    md_path = REPORT_DIR / "repo_clone_status.md"
    json_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    lines = ["# Repo Clone Status", "", "| Model | Status | Repo Dir | Error |", "| --- | --- | --- | --- |"]
    for row in results:
        lines.append(f"| {row['model']} | {row['status']} | {row['repo_dir']} | {row['error'] or ''} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
