"""Locate discontinuities in a merged camera trajectory.

Reads an Internal/External Camera Parameters CSV, reduces it to one position
per index, and reports where the step between consecutive indices departs from
the local norm. Also measures how much baseline each set hand-off actually had,
since a hand-off over a stretch where the camera barely moved is the usual
reason a chained sequence kinks.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CSV = Path(sys.argv[1] if len(sys.argv) > 1
           else r"K:\realityscan\out\1-high\_merged\cameras.csv")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "1-high"
CHUNK, OVERLAP = 25, 5

pos: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
for line in CSV.open(encoding="utf-8", errors="replace"):
    if line.startswith("#"):
        continue
    c = line.rstrip("\n").split(",")
    m = re.match(rf"{re.escape(PREFIX)}_(\d+)_(\d+)\.", c[0])
    if not m:
        continue
    pos[int(m.group(1))].append((float(c[1]), float(c[2]), float(c[3])))

centre = {i: tuple(statistics.fmean(v[k] for v in p) for k in range(3))
          for i, p in pos.items()}
spread = {i: max(math.dist(v, centre[i]) for v in p) for i, p in pos.items()}
idx = sorted(centre)
print(f"{CSV}\nprefix={PREFIX}  indices {idx[0]}-{idx[-1]} ({len(idx)})  "
      f"cameras {sum(len(v) for v in pos.values())}")

steps = [(a, math.dist(centre[a], centre[b])) for a, b in zip(idx, idx[1:])]
vals = [s for _, s in steps]
med = statistics.median(vals)
print(f"\nstep between consecutive indices: median {med:.3f}, "
      f"mean {statistics.fmean(vals):.3f}, max {max(vals):.3f}")

print("\n--- 20 largest steps (index -> next) ---")
for a, s in sorted(steps, key=lambda t: -t[1])[:20]:
    set_no = a // (CHUNK - OVERLAP)
    boundary = " <- set boundary" if (a + 1) % (CHUNK - OVERLAP) == 0 else ""
    print(f"  {a:4d} -> {a+1:<4d} {s:8.3f}  ({s/med:6.1f}x median){boundary}")

print("\n--- around the reported break ---")
for i in range(290, 312):
    if i not in centre:
        print(f"  {i:4d}  MISSING")
        continue
    step = math.dist(centre[i], centre[i + 1]) if i + 1 in centre else float("nan")
    print(f"  {i:4d}  views={len(pos[i]):3d}  spread={spread[i]:7.3f}  "
          f"step->{i+1}={step:8.3f}  ({step/med:5.1f}x)")

print("\n--- baseline available at each hand-off (the overlap indices) ---")
print("     set  overlap indices     span   step-sum   verdict")
starts = list(range(idx[0], idx[-1] + 1, CHUNK - OVERLAP))
rows = []
for n, start in enumerate(starts[1:], start=1):
    ov = [i for i in range(start, start + OVERLAP) if i in centre]
    if len(ov) < 2:
        continue
    span = max(math.dist(centre[a], centre[b]) for a in ov for b in ov)
    walk = sum(math.dist(centre[a], centre[b]) for a, b in zip(ov, ov[1:]))
    rows.append((n, start, ov[-1], span, walk))
spans = [r[3] for r in rows]
thin = statistics.median(spans) * 0.35
for n, a, b, span, walk in rows:
    flag = "  <-- THIN" if span < thin else ""
    print(f"    {n:3d}  {a:4d}-{b:<4d}  {span:9.3f}  {walk:9.3f}{flag}")
print(f"\n  median hand-off span {statistics.median(spans):.3f}; "
      f"flagged below {thin:.3f}")
