"""Compare multiple training runs by reading their validation/metrics.csv files.

Usage:
    python scripts/compare_runs.py outputs/run_A outputs/run_B outputs/run_C

Prints a summary table sorted by best F1.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_best(run_dir: Path) -> dict:
    csv_path = run_dir / "validation" / "metrics.csv"
    if not csv_path.exists():
        return {}
    import csv
    best_f1 = -1.0
    best_row: dict = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                f1 = float(row.get("f1", 0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_row = row
            except ValueError:
                continue
    return best_row


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_runs.py <run_dir> [run_dir ...]")
        sys.exit(1)

    run_dirs = [Path(p) for p in sys.argv[1:]]
    results  = []
    for rd in run_dirs:
        best = _read_best(ROOT / rd if not rd.is_absolute() else rd)
        if best:
            results.append((rd.name, best))

    if not results:
        print("No metrics.csv found in any run directory.")
        return

    results.sort(key=lambda x: float(x[1].get("f1", 0)), reverse=True)
    cols = ["f1", "miou", "precision", "recall", "oa", "boundary_f1", "iteration"]
    col_w = 10

    header = f"{'Run':<50}" + "".join(f"{c:>{col_w}}" for c in cols)
    print()
    print(header)
    print("-" * len(header))
    for name, row in results:
        vals = "".join(f"{float(row.get(c, 0)):>{col_w}.4f}" for c in cols[:-1])
        vals += f"{int(float(row.get('iteration', 0))):>{col_w}}"
        print(f"{name:<50}{vals}")
    print()


if __name__ == "__main__":
    main()
