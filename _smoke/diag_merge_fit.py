"""Which sets did the merge place inconsistently?

For every set component, fit the best similarity from its own frame to the
merged frame using its own cameras. A set the merge placed as a rigid piece
gives a near-zero residual; a set that got bent or misplaced stands out.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from diag_seam import load, umeyama          # noqa: E402

ROOT = Path(r"K:\realityscan\out\1-high")
MERGED = load(ROOT / "_merged" / "cameras.csv")
PREFIX = "1-high"

idx = defaultdict(list)
for k, v in MERGED.items():
    m = re.match(rf"{re.escape(PREFIX)}_(\d+)_", k)
    if m:
        idx[int(m.group(1))].append(v)
ctr = {i: tuple(statistics.fmean(p[k] for p in v) for k in range(3))
       for i, v in idx.items()}
ks = sorted(ctr)
STEP = statistics.median([math.dist(ctr[a], ctr[b]) for a, b in zip(ks, ks[1:])])
print(f"merged: {len(MERGED)} cameras, median index step {STEP:.4f}\n")

print("set                     cams  scale   residual (as % of one index step)")
rows = []
for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")):
    own = load(d / "cameras.csv")
    common = [k for k in own if k in MERGED]
    if len(common) < 10:
        print(f"{d.name:22s}  {len(common):5d}  -- too few shared")
        continue
    s, R, t = umeyama([own[k] for k in common], [MERGED[k] for k in common])
    res = []
    for k in common:
        p = own[k]
        m = [s * sum(R[r][j] * p[j] for j in range(3)) + t[r] for r in range(3)]
        res.append((k, math.dist(m, MERGED[k])))
    vals = [r for _, r in res]
    med, mx = statistics.median(vals), max(vals)
    worst = max(res, key=lambda r: r[1])[0]
    flag = "  <<<" if mx > 0.25 * STEP else ""
    print(f"{d.name:22s}  {len(common):5d}  {s:6.3f}  median {med/STEP*100:6.2f}%  "
          f"max {mx/STEP*100:7.2f}%  worst={worst}{flag}")
    rows.append((d.name, med / STEP, mx / STEP))

print("\n--- per-view positions around the seam in the merged result ---")
for i in (298, 299, 300, 301, 302):
    pts = [(k, v) for k, v in MERGED.items()
           if re.match(rf"{re.escape(PREFIX)}_{i:04d}_", k)]
    c = tuple(statistics.fmean(p[1][j] for p in pts) for j in range(3))
    print(f"  index {i}: {len(pts)} views")
    for k, v in sorted(pts):
        print(f"      {k:24s} offset from centre {math.dist(v, c)/STEP*100:6.2f}% of a step")
