"""Compare the step-size profile of two reconstructions of the same capture.

Two results in different frames cannot be compared directly, but the shape of
the trajectory can: normalising each by its own median step makes the profiles
comparable, and a region where one has steps ten times the other's is a region
where one of them stretched or compressed the path.

    python diag_step_compare.py <a.csv> <b.csv> [--prefix P] [--bins N]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def profile(p: Path, prefix: str, min_views: int = 8):
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            m = PAT.match(r["#name"])
            if not m or (prefix and m.group(1) != prefix):
                continue
            acc[int(m.group(2))].append(
                np.array([float(r["x"]), float(r["y"]), float(r["alt"])]))
    c = {i: np.mean(v, axis=0) for i, v in acc.items() if len(v) >= min_views}
    keys = sorted(c)
    steps = {b: float(np.linalg.norm(c[b] - c[a]))
             for a, b in zip(keys, keys[1:]) if b - a == 1}
    med = float(np.median(list(steps.values())))
    return {i: d / med for i, d in steps.items()}, med


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    prefix, bins = "", 12
    files = []
    i = 0
    while i < len(argv):
        if argv[i] == "--prefix":
            prefix, i = argv[i + 1], i + 2
        elif argv[i] == "--bins":
            bins, i = int(argv[i + 1]), i + 2
        else:
            files.append(argv[i])
            i += 1

    profs = []
    for f in files:
        s, med = profile(Path(f), prefix)
        v = np.array(list(s.values()))
        print(f"{f}")
        print(f"  {len(s)} steps, median {med:.4f}, "
              f"normalised p50 {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} "
              f"p95 {np.percentile(v, 95):.2f} p99 {np.percentile(v, 99):.2f} "
              f"max {v.max():.2f}")
        print(f"  steps over 3x: {(v > 3).sum()}, over 5x: {(v > 5).sum()}, "
              f"over 10x: {(v > 10).sum()}")
        profs.append(s)

    if len(profs) < 2:
        return 0
    lo = min(min(s) for s in profs)
    hi = max(max(s) for s in profs)
    edges = np.linspace(lo, hi + 1, bins + 1).astype(int)
    print(f"\nmedian normalised step per index range")
    head = "  range".ljust(18) + "".join(f"{Path(f).parent.name:>18}" for f in files)
    print(head)
    for a, b in zip(edges, edges[1:]):
        cells = []
        for s in profs:
            v = [d for i, d in s.items() if a <= i < b]
            cells.append(f"{np.median(v):18.2f}" if v else f"{'-':>18}")
        print(f"  {a:>5}-{b - 1:<11}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
