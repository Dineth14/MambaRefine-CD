#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "sota_reproduced_eval" / "reports"

STAGES = [
    "scripts/clone_sota_repos.py",
    "scripts/download_sota_weights.py",
    "scripts/discover_sota_checkpoints.py",
    "scripts/evaluate_sota_models.py",
    "scripts/write_sota_tables.py",
    "scripts/collect_website_qualitative.py",
    "scripts/validate_website.py",
]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for stage in STAGES:
        path = ROOT / stage
        if not path.exists():
            results.append({"stage": stage, "status": "SKIPPED", "message": "script missing"})
            continue
        proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), text=True, capture_output=True, check=False)
        results.append(
            {
                "stage": stage,
                "status": "OK" if proc.returncode == 0 else "FAILED",
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
    json_path = REPORT_DIR / "master_status.json"
    md_path = REPORT_DIR / "master_status.md"
    json_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    lines = ["# SOTA Reproduction Pipeline Status", "", "| Stage | Status | Return Code |", "| --- | --- | --- |"]
    for row in results:
        lines.append(f"| {row['stage']} | {row['status']} | {row.get('returncode', '')} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
