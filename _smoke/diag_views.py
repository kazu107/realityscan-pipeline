"""What angular range does each capture's view set actually cover?

Reads an Internal/External CSV (which carries yaw per camera) and reports, for
every view number, its yaw relative to the heading of its own index. A rig that
covers 360 degrees spreads its views evenly around the circle; a front-only or
rear-only rig bunches them into an arc, which is the classic weak case for
structure from motion along the direction of travel.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CSV = Path(sys.argv[1])
PREFIX = sys.argv[2]

rows = []
for line in CSV.open(encoding="utf-8", errors="replace"):
    if line.startswith("#"):
        continue
    c = line.rstrip("\n").split(",")
    m = re.match(rf"{re.escape(PREFIX)}_(\d+)_(\d+)\.", c[0])
    if m:
        rows.append((int(m.group(1)), int(m.group(2)),
                     float(c[1]), float(c[2]), float(c[3]), float(c[4])))

pos = defaultdict(dict)
yaw = defaultdict(dict)
for i, v, x, y, z, yw in rows:
    pos[i][v] = (x, y, z)
    yaw[i][v] = yw
idx = sorted(pos)
print(f"{CSV.name}  prefix={PREFIX}  {len(rows)} cameras, {len(idx)} indices")

# direction of travel per index, from the index centres
ctr = {i: tuple(statistics.fmean(p[k] for p in pos[i].values()) for k in range(3))
       for i in idx}
heading = {}
for a, b in zip(idx, idx[1:]):
    dx, dy = ctr[b][0] - ctr[a][0], ctr[b][1] - ctr[a][1]
    if math.hypot(dx, dy) > 1e-9:
        heading[a] = math.degrees(math.atan2(dy, dx))

views = sorted({v for i in pos for v in pos[i]})
print("\nview   yaw relative to the direction of travel (median, deg)")
rel_all = {}
for v in views:
    rel = []
    for i in idx:
        if v in yaw[i] and i in heading:
            d = (yaw[i][v] - heading[i] + 180) % 360 - 180
            rel.append(d)
    if rel:
        rel.sort()
        rel_all[v] = rel[len(rel) // 2]
        print(f"  {v:2d}   {rel[len(rel)//2]:8.1f}   (p10 {rel[len(rel)//10]:7.1f}, "
              f"p90 {rel[len(rel)*9//10]:7.1f}, n={len(rel)})")

if rel_all:
    vals = sorted(rel_all.values())
    gaps = [(vals[(k + 1) % len(vals)] - vals[k]) % 360 for k in range(len(vals))]
    print(f"\n  angular span of the views: {max(vals) - min(vals):.0f} deg")
    print(f"  largest uncovered gap between adjacent views: {max(gaps):.0f} deg")

# how even is the walk?
steps = [math.dist(ctr[a], ctr[b]) for a, b in zip(idx, idx[1:])]
med = statistics.median(steps)
cv = statistics.pstdev(steps) / statistics.fmean(steps)
print(f"\nstep between indices: median {med:.4f}, "
      f"coefficient of variation {cv*100:.1f}%")
print(f"  p05 {sorted(steps)[len(steps)//20]/med:.2f}x  "
      f"p95 {sorted(steps)[len(steps)*19//20]/med:.2f}x  "
      f"max {max(steps)/med:.2f}x")
