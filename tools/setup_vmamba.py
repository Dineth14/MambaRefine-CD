"""Prints instructions for setting up VMamba under third_party/VMamba.

Usage: python tools/setup_vmamba.py
"""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    path = Path("third_party/VMamba")
    if path.exists():
        print(f"VMamba directory found: {path}")
    else:
        print("VMamba directory not found.")
    print("Place the official VMamba repository at third_party/VMamba.")
    print("Install its dependencies following the upstream VMamba instructions.")
    try:
        import vmamba  # noqa: F401
        print("vmamba import: PASS")
    except Exception as exc:
        print(f"vmamba import: not available ({exc})")


if __name__ == "__main__":
    main()
