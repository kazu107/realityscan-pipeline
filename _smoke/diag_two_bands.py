"""1-mid-2-1 and 1-mid-2b are the same walk, so their trajectories must agree.

Both merged components carry the same seeded 1-mid cameras (index 40-64), which
gives an exact similarity transform between the two frames. Applying it lets the
two reconstructions of the very same camera positions be compared directly -
wherever they disagree, at least one of them is wrong.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diag_seam import umeyama                      # noqa: E402

A_CSV = Path(r"K:\realityscan\out\1-mid-2-1\_merged\cameras.csv")
B_CSV = Path(r"K:\realityscan\out\1-mid-2b\_merged\cameras.csv")
A_PREFIX, B_PREFIX = "1-mid-2-repaired", "1-mid-2b"


def load(csv: Path):
    """-> {camera name: xyz}, {index: mean xyz} for the band's own prefix."""
    named, byidx = {}, defaultdict(list)
    for line in csv.open(encoding="utf-8", errors="replace"):
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split(",")
        p = (float(c[1]), float(c[2]), float(c[3]))
        named[c[0]] = p
        m = re.match(r"(.+)_(\d+)_(\d+)\.", c[0])
        if m:
            byidx[(m.group(1), int(m.group(2)))].append(p)
    ctr = {k: tuple(statistics.fmean(q[j] for q in v) for j in range(3))
           for k, v in byidx.items()}
    return named, ctr


A_named, A_ctr = load(A_CSV)
B_named, B_ctr = load(B_CSV)

shared = sorted(set(A_named) & set(B_named))          # the seeded 1-mid cameras
print(f"shared cameras between the two bands: {len(shared)}  "
      f"(e.g. {shared[0] if shared else '-'})")
if len(shared) < 10:
    raise SystemExit("not enough shared cameras to relate the two frames")

s, R, t = umeyama([A_named[k] for k in shared], [B_named[k] for k in shared])
res = []
for k in shared:
    p = A_named[k]
    m = [s * sum(R[r][j] * p[j] for j in range(3)) + t[r] for r in range(3)]
    res.append(math.dist(m, B_named[k]))
print(f"similarity A->B: scale {s:.5f}, residual on the shared cameras "
      f"median {statistics.median(res):.5f} max {max(res):.5f}")


def xform(p):
    return tuple(s * sum(R[r][j] * p[j] for j in range(3)) + t[r] for r in range(3))


idx = sorted(i for (pre, i) in A_ctr if pre == A_PREFIX)
common = [i for i in idx if (B_PREFIX, i) in B_ctr]
print(f"\nindices present in both bands: {len(common)}")

d = {i: math.dist(xform(A_ctr[(A_PREFIX, i)]), B_ctr[(B_PREFIX, i)]) for i in common}
bstep = statistics.median(
    [math.dist(B_ctr[(B_PREFIX, a)], B_ctr[(B_PREFIX, b)])
     for a, b in zip(common, common[1:])])
vals = sorted(d.values())
print(f"disagreement between the two reconstructions of the same position,")
print(f"  in units of one index step ({bstep:.4f}):")
print(f"  median {statistics.median(vals)/bstep*100:6.1f}%   "
      f"p90 {vals[len(vals)*9//10]/bstep*100:6.1f}%   "
      f"p99 {vals[int(len(vals)*0.99)]/bstep*100:6.1f}%   "
      f"max {max(vals)/bstep*100:6.1f}%")

print("\nworst 15 indices")
for i in sorted(common, key=lambda i: -d[i])[:15]:
    print(f"  index {i:5d}   {d[i]/bstep*100:8.1f}% of a step")

print("\nwhere the disagreement lives (mean % of a step per 100 indices)")
for lo in range(0, max(common) + 1, 100):
    seg = [d[i] for i in common if lo <= i < lo + 100]
    if seg:
        bar = "#" * int(statistics.fmean(seg) / bstep * 40)
        print(f"  {lo:5d}-{lo+99:<5d} {statistics.fmean(seg)/bstep*100:7.1f}%  {bar}")
