"""Find cameras a component placed absurdly far from everything else.

-selectMaximalComponent exports the largest component, not a sane one. A patch
over 25 indices came back with two cameras at 27,000 and 88,000 units while the
other 191 sat within 15 - registered, in the component, and completely wrong.
A single such camera wrecks any least-squares fit that uses the component, and
the merge fits every component it imports.

Distance is measured from the component's median position and reported in
multiples of the median camera spacing, so the numbers mean the same thing in
every component's arbitrary scale.

    python diag_outlier_cameras.py <dir with set folders> [...] [--factor 50]
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def load(p: Path):
    names, pos = [], []
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                pos.append([float(r["x"]), float(r["y"]), float(r["alt"])])
            except (KeyError, ValueError):
                continue
            names.append(r["#name"])
    return names, np.array(pos)


def main() -> int:
    argv = list(sys.argv[1:])
    factor = 50.0
    if "--factor" in argv:
        i = argv.index("--factor")
        factor = float(argv[i + 1])
        del argv[i:i + 2]

    grand = 0
    for root in argv:
        rootp = Path(root)
        csvs = sorted(rootp.glob("*/cameras.csv"))
        if not csvs:
            csvs = sorted(rootp.glob("*/*/cameras.csv"))
        print(f"=== {rootp}: {len(csvs)} camera lists")
        hits = []
        for c in csvs:
            names, pos = load(c)
            if len(pos) < 10:
                continue
            d = np.linalg.norm(pos - np.median(pos, axis=0), axis=1)
            med = float(np.median(d)) or 1.0
            bad = np.where(d > factor * med)[0]
            if len(bad):
                worst = bad[np.argsort(d[bad])[::-1]]
                hits.append((c.parent.name, len(bad), len(names),
                             [(names[i], d[i] / med) for i in worst[:3]]))
        grand += sum(h[1] for h in hits)
        if not hits:
            print("   none")
            continue
        for name, n, total, worst in sorted(hits, key=lambda h: -h[3][0][1]):
            print(f"   {name:<28} {n:>3}/{total} camera(s) beyond {factor:g}x")
            for nm, r in worst:
                print(f"        {nm:<26} {r:>12.0f}x the median spread")
    print(f"\n{grand} outlier camera(s) in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
