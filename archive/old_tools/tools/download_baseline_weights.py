#!/usr/bin/env python3
"""Baseline weight helper.

Only downloads URLs that are explicitly configured and verified by the user.
No URL is invented in this script.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


KNOWN_STATUS = {
    "ChangeFormer": "Official repository reports pretrained weights, but no verified direct URL is embedded here. Provide --changeformer-levir-url or --changeformer-dsifn-url after checking the official README.",
    "SNUNet": "No official pretrained weights found; train from scratch.",
    "IFNet": "No reliable official pretrained weights found; train from scratch.",
    "CDMamba": "No verified official LEVIR/WHU/DSIFN weights found; train from scratch unless user provides checkpoint.",
    "M-CD": "No verified pretrained weights found; train from scratch.",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"exists: {dest}")
        return
    print(f"downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    print(f"saved: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download verified external baseline weights.")
    parser.add_argument("--out_dir", default="external_weights")
    parser.add_argument("--changeformer-levir-url", default=None)
    parser.add_argument("--changeformer-dsifn-url", default=None)
    args = parser.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"External weight directory: {out_dir}")

    for model, status in KNOWN_STATUS.items():
        print(f"{model}: {status}")

    if args.changeformer_levir_url:
        download(args.changeformer_levir_url, out_dir / "changeformer" / "levir.pth")
    else:
        print("ChangeFormer LEVIR: no verified URL supplied; skipped.")

    if args.changeformer_dsifn_url:
        download(args.changeformer_dsifn_url, out_dir / "changeformer" / "dsifn.pth")
    else:
        print("ChangeFormer DSIFN: no verified URL supplied; skipped.")

    for path in [
        out_dir / "changeformer" / "levir.pth",
        out_dir / "changeformer" / "dsifn.pth",
    ]:
        print(f"{path}: {'exists' if path.exists() else 'missing'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
