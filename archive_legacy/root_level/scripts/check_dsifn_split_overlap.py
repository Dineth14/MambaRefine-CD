#!/usr/bin/env python3
"""Check exact DSIFN-CD train/val/test overlap from manifests."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from dsifn_audit_utils import read_manifest, write_csv, write_json, write_text


CHECKS = [
    ("pre_path", "pre_image_path"),
    ("post_path", "post_image_path"),
    ("mask_path", "mask_path"),
    ("pre_stem", "pre_stem"),
    ("post_stem", "post_stem"),
    ("mask_stem", "mask_stem"),
    ("pair_key", "pair_key"),
    ("mask_key", "mask_key"),
    ("pre_sha256", "pre_sha256"),
    ("post_sha256", "post_sha256"),
    ("mask_sha256", "mask_sha256"),
    ("original_scene_id", "original_scene_id"),
]
LEAKAGE_CHECKS = {
    "pre_path",
    "post_path",
    "mask_path",
    "pre_stem",
    "post_stem",
    "mask_stem",
    "pair_key",
    "mask_key",
    "original_scene_id",
}
CONTENT_HASH_CHECKS = {"pre_sha256", "post_sha256", "mask_sha256"}


def overlap_values(a_rows: list[dict], b_rows: list[dict], field: str) -> set[str]:
    a = {r[field] for r in a_rows if r.get(field)}
    b = {r[field] for r in b_rows if r.get(field)}
    return a & b


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DSIFN-CD exact split overlap.")
    parser.add_argument("--manifest_dir", default="outputs/dsifn_manifests")
    parser.add_argument("--csv_out", default="outputs/dsifn_overlap_report.csv")
    parser.add_argument("--json_out", default="outputs/dsifn_overlap_report.json")
    parser.add_argument("--md_out", default="outputs/dsifn_overlap_report.md")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    data = {
        split: read_manifest(manifest_dir / f"{split}_manifest.csv")
        for split in ("train", "val", "test")
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    csv_rows = []
    json_report = {
        "manifest_dir": str(manifest_dir),
        "counts": {split: len(rows) for split, rows in data.items()},
        "overlaps": defaultdict(dict),
        "verdict": "PASS",
    }
    leakage = False
    content_hash_warning = False
    for left, right in pairs:
        for label, field in CHECKS:
            vals = sorted(overlap_values(data[left], data[right], field))
            pair_name = f"{left}_vs_{right}"
            json_report["overlaps"][pair_name][label] = {
                "count": len(vals),
                "examples": vals[:50],
            }
            csv_rows.append({
                "split_pair": pair_name,
                "check": label,
                "field": field,
                "overlap_count": len(vals),
                "examples": ";".join(vals[:20]),
            })
            if pair_name in {"train_vs_val", "train_vs_test", "val_vs_test"} and label in LEAKAGE_CHECKS and len(vals) > 0:
                leakage = True
            if label in CONTENT_HASH_CHECKS and len(vals) > 0:
                content_hash_warning = True
    if leakage:
        json_report["verdict"] = "FAIL"
        json_report["warning"] = "DATA LEAKAGE RISK FOUND."
    elif content_hash_warning:
        json_report["warning"] = (
            "Duplicate content hashes found across disjoint image IDs. "
            "Reported as a warning, not confirmed split leakage."
        )

    write_csv(args.csv_out, csv_rows, fields=["split_pair", "check", "field", "overlap_count", "examples"])
    write_json(args.json_out, dict(json_report))

    md = [
        "# DSIFN Exact Split Overlap Report",
        "",
        f"- Manifest dir: `{manifest_dir}`",
        f"- Train samples: `{len(data['train'])}`",
        f"- Val samples: `{len(data['val'])}`",
        f"- Test samples: `{len(data['test'])}`",
        f"- Verdict: `{json_report['verdict']}`",
    ]
    if leakage:
        md.extend(["", "**DATA LEAKAGE RISK FOUND.**"])
    elif content_hash_warning:
        md.extend([
            "",
            "**Warning:** duplicate content hashes were found across disjoint image IDs. "
            "This can occur when masks or post-event images are identical; it is not treated as confirmed split leakage without path/stem/pair/original-scene overlap.",
        ])
    md.extend(["", "## Overlap Counts", ""])
    md.append("| Split Pair | Check | Count | Examples |")
    md.append("|---|---:|---:|---|")
    for row in csv_rows:
        md.append(f"| {row['split_pair']} | {row['check']} | {row['overlap_count']} | {row['examples']} |")
    write_text(args.md_out, "\n".join(md) + "\n")
    if leakage:
        print("DATA LEAKAGE RISK FOUND.")
    elif content_hash_warning:
        print("No identity overlap found; duplicate content hashes reported as warnings.")
    else:
        print("No exact overlap found.")
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
