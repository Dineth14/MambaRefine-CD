#!/usr/bin/env python3
"""Check DSIFN-CD near-duplicate and scene-level leakage risks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from dsifn_audit_utils import avg_hash, hamming_hex, read_manifest, write_csv, write_text


def annotate_hashes(rows: list[dict], max_rows: int | None = None) -> list[dict]:
    out = []
    for idx, row in enumerate(rows):
        if max_rows is not None and idx >= max_rows:
            break
        item = dict(row)
        try:
            item["pre_ahash"] = avg_hash(item["pre_image_path"])
            item["post_ahash"] = avg_hash(item["post_image_path"])
            item["mask_ahash"] = avg_hash(item["mask_path"])
        except Exception as exc:
            item["hash_error"] = str(exc)
        out.append(item)
    return out


def add_group_matches(rows_a: list[dict], rows_b: list[dict], field: str, risk: str, reason: str, results: list[dict], pair: str) -> None:
    map_a = defaultdict(list)
    map_b = defaultdict(list)
    for row in rows_a:
        value = row.get(field)
        if value:
            map_a[value].append(row)
    for row in rows_b:
        value = row.get(field)
        if value:
            map_b[value].append(row)
    for value in sorted(set(map_a) & set(map_b)):
        a = map_a[value][0]
        b = map_b[value][0]
        results.append({
            "split_pair": pair,
            "risk": risk,
            "reason": reason,
            "key": value,
            "left_sample": a.get("pair_key"),
            "right_sample": b.get("pair_key"),
            "left_path": a.get("pre_image_path"),
            "right_path": b.get("pre_image_path"),
            "pre_hamming": "",
            "post_hamming": "",
            "mask_hamming": "",
        })


def add_perceptual_matches(rows_a: list[dict], rows_b: list[dict], max_pairs: int, results: list[dict], pair: str) -> None:
    checked = 0
    for a in rows_a:
        for b in rows_b:
            if checked >= max_pairs:
                return
            checked += 1
            if "pre_ahash" not in a or "pre_ahash" not in b:
                continue
            pre_h = hamming_hex(a["pre_ahash"], b["pre_ahash"])
            post_h = hamming_hex(a["post_ahash"], b["post_ahash"])
            mask_h = hamming_hex(a["mask_ahash"], b["mask_ahash"])
            if pre_h <= 4 and post_h <= 4 and mask_h <= 4:
                results.append({
                    "split_pair": pair,
                    "risk": "suspicious",
                    "reason": "very similar average hashes for pre/post/mask",
                    "key": f"pre_h={pre_h},post_h={post_h},mask_h={mask_h}",
                    "left_sample": a.get("pair_key"),
                    "right_sample": b.get("pair_key"),
                    "left_path": a.get("pre_image_path"),
                    "right_path": b.get("pre_image_path"),
                    "pre_hamming": pre_h,
                    "post_hamming": post_h,
                    "mask_hamming": mask_h,
                })


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DSIFN-CD near duplicates.")
    parser.add_argument("--manifest_dir", default="outputs/dsifn_manifests")
    parser.add_argument("--max_pairs", type=int, default=100000)
    parser.add_argument("--max_hash_rows_per_split", type=int, default=500)
    parser.add_argument("--csv_out", default="outputs/dsifn_near_duplicate_report.csv")
    parser.add_argument("--md_out", default="outputs/dsifn_near_duplicate_report.md")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    data = {
        split: read_manifest(manifest_dir / f"{split}_manifest.csv")
        for split in ("train", "val", "test")
    }
    hashed = {
        split: annotate_hashes(rows, max_rows=args.max_hash_rows_per_split)
        for split, rows in data.items()
    }
    results: list[dict] = []
    for left, right in combinations(("train", "val", "test"), 2):
        pair = f"{left}_vs_{right}"
        add_group_matches(data[left], data[right], "pair_key", "confirmed leakage", "same pair_key across splits", results, pair)
        add_group_matches(data[left], data[right], "mask_key", "confirmed leakage", "same mask_key across splits", results, pair)
        add_group_matches(
            data[left],
            data[right],
            "mask_sha256",
            "suspicious",
            "same mask file content hash across disjoint image IDs",
            results,
            pair,
        )
        add_group_matches(data[left], data[right], "original_scene_id", "high-risk leakage", "same inferred original scene id across splits", results, pair)
        add_perceptual_matches(hashed[left], hashed[right], args.max_pairs, results, pair)

    fields = [
        "split_pair",
        "risk",
        "reason",
        "key",
        "left_sample",
        "right_sample",
        "left_path",
        "right_path",
        "pre_hamming",
        "post_hamming",
        "mask_hamming",
    ]
    write_csv(args.csv_out, results, fields=fields)
    counts = defaultdict(int)
    for row in results:
        counts[row["risk"]] += 1
    md = [
        "# DSIFN Near-Duplicate Report",
        "",
        f"- Manifest dir: `{manifest_dir}`",
        f"- Max perceptual pairs per split pair: `{args.max_pairs}`",
        f"- Perceptual hash rows per split: `{args.max_hash_rows_per_split}`",
        "",
        "## Risk Counts",
        "",
        "| Risk | Count |",
        "|---|---:|",
    ]
    for risk in ("confirmed leakage", "high-risk leakage", "suspicious", "likely safe"):
        md.append(f"| {risk} | {counts.get(risk, 0)} |")
    md.extend(["", "## First Findings", "", "| Split Pair | Risk | Reason | Key |", "|---|---|---|---|"])
    for row in results[:100]:
        md.append(f"| {row['split_pair']} | {row['risk']} | {row['reason']} | {row['key']} |")
    write_text(args.md_out, "\n".join(md) + "\n")
    print(f"Findings: {len(results)}")
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
