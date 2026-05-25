"""Compatibility evaluator for active binary CD experiments.

This script forwards to `scripts/test.py` and keeps the older entry point
available for DSIFN-CD and WHU-CD checkpoints.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    test_script = repo / "scripts" / "test.py"
    cmd = [sys.executable, str(test_script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
