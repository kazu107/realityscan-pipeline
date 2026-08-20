"""Sanity-check a camera list for collapsed or degenerate regions.

A step profile normalises by the median, so a region whose cameras all sit at
(nearly) the same place shows up as "0.00" and is easy to misread as a rounding
artefact. This prints the raw extent per index range, plus how many cameras land
suspiciously close to the origin, which is what an export writes when it has no
pose to write.

    python diag_extent.py <cameras.csv> [...] [--prefix P] [--bins N]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def load(p: Path, prefix: str):
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            m = PAT.match(r["#name"])
            if not m or (prefix and m.group(1) != prefix):
                continue
            acc[int(m.group(2))].append(
                np.array([float(r["x"]), float(r["y"]), float(r["alt"])]))
    return acc


def main() -> int:
    argv = list(sys.argv[1:])
    prefix, bins, files = "", 14, []
    i = 0
    while i < len(argv):
        if argv[i] == "--prefix":
            prefix, i = argv[i + 1], i + 2
        elif argv[i] == "--bins":
            bins, i = int(argv[i + 1]), i + 2
        else:
            files.append(argv[i])
            i += 1

    for f in files:
        acc = load(Path(f), prefix)
        allp = np.array([p for v in acc.values() for p in v])
        scale = float(np.linalg.norm(allp.max(0) - allp.min(0)))
        near0 = int((np.linalg.norm(allp, axis=1) < 1e-6).sum())
        print(f"{f}")
        print(f"  {len(allp)} cameras, overall extent {scale:.3f}, "
              f"{near0} at the origin")
        keys = sorted(acc)
        edges = np.linspace(keys[0], keys[-1] + 1, bins + 1).astype(int)
        for a, b in zip(edges, edges[1:]):
            pts = np.array([p for i2 in keys if a <= i2 < b for p in acc[i2]])
            if not len(pts):
                print(f"    {a:>5}-{b - 1:<6} no cameras")
                continue
            ext = float(np.linalg.norm(pts.max(0) - pts.min(0)))
            print(f"    {a:>5}-{b - 1:<6} {len(pts):>5} cams  extent "
                  f"{ext:10.4f}  ({ext / scale * 100:5.2f}% of the whole)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
