#!/usr/bin/env python3
"""Run the complete DSIFN-CD data-integrity audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dsifn_audit_utils import load_dataset_config, repo_path, validate_explicit_splits, write_json, write_text


def run_step(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=repo_path("."), text=True, capture_output=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def load_json(path: str | Path) -> dict:
    p = repo_path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full DSIFN-CD data integrity audit.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    _, ds_cfg = load_dataset_config(args.config)
    split_integrity = validate_explicit_splits(ds_cfg)

    py = sys.executable
    steps = [
        [py, "scripts/audit_dsifn_split_config.py", "--config", args.config],
        [py, "scripts/build_dsifn_manifests.py", "--config", args.config],
        [py, "scripts/check_dsifn_split_overlap.py", "--manifest_dir", "outputs/dsifn_manifests"],
        [py, "scripts/check_dsifn_near_duplicates.py", "--manifest_dir", "outputs/dsifn_manifests", "--max_pairs", "100000"],
        [py, "scripts/inspect_dsifn_dataloaders.py", "--config", args.config, "--num_samples", "20"],
    ]
    results = [run_step(step) for step in steps]
    overlap = load_json("outputs/dsifn_overlap_report.json")
    split_audit = load_json("outputs/dsifn_split_config_audit.json")
    failed_steps = [r for r in results if r["returncode"] != 0]
    overlap_verdict = overlap.get("verdict", "INCONCLUSIVE")
    if split_integrity.get("verdict") == "FAIL":
        verdict = "FAIL"
    elif failed_steps:
        verdict = "INCONCLUSIVE"
    elif overlap_verdict == "FAIL":
        verdict = "FAIL"
    else:
        near_md = repo_path("outputs/dsifn_near_duplicate_report.md").read_text(encoding="utf-8")
        has_near_warnings = any(
            marker not in near_md
            for marker in ("confirmed leakage | 0", "high-risk leakage | 0", "suspicious | 0")
        )
        verdict = "PASS WITH WARNINGS" if has_near_warnings else "PASS"

    summary = {
        "config": args.config,
        "verdict": verdict,
        "steps": results,
        "split_counts": overlap.get("counts", {}),
        "split_integrity": split_integrity,
        "overlap_verdict": overlap_verdict,
        "split_layouts": {
            split: info.get("layout")
            for split, info in split_audit.get("splits", {}).items()
        },
    }
    write_json("outputs/dsifn_data_integrity_summary.json", summary)

    md = [
        "# DSIFN Data Integrity Summary",
        "",
        f"- Config: `{args.config}`",
        f"- Final verdict: `{verdict}`",
        f"- Overlap verdict: `{overlap_verdict}`",
        f"- Explicit split verdict: `{split_integrity.get('verdict', 'INCONCLUSIVE')}`",
        f"- Train samples: `{overlap.get('counts', {}).get('train', 'unknown')}`",
        f"- Val samples: `{overlap.get('counts', {}).get('val', 'unknown')}`",
        f"- Test samples: `{overlap.get('counts', {}).get('test', 'unknown')}`",
        "",
        "## Step Status",
        "",
        "| Command | Return Code |",
        "|---|---:|",
    ]
    for result in results:
        md.append(f"| `{result['cmd']}` | {result['returncode']} |")
    if verdict == "FAIL":
        md.extend(["", "**DATA LEAKAGE RISK FOUND.** See `outputs/dsifn_overlap_report.md`."])
    write_text("outputs/dsifn_data_integrity_summary.md", "\n".join(md) + "\n")
    print(f"Final verdict: {verdict}")
    if failed_steps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
