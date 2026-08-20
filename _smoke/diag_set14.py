"""What shape is the deformation of 1-high_0280-0304 in the merged result?

Fits the set to the merged frame three ways - whole set, first half, second
half - and prints the per-index residual profile. If the merge split the set
into two rigid pieces, each half fits well on its own and the whole-set fit
does not.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diag_seam import load, umeyama          # noqa: E402

ROOT = Path(r"K:\realityscan\out\1-high")
SET = "1-high_0280-0304"
MERGED = load(ROOT / "_merged" / "cameras.csv")
OWN = load(ROOT / SET / "cameras.csv")

idx = defaultdict(list)
for k, v in MERGED.items():
    m = re.match(r"1-high_(\d+)_", k)
    if m:
        idx[int(m.group(1))].append(v)
ctr = {i: tuple(statistics.fmean(p[k] for p in v) for k in range(3))
       for i, v in idx.items()}
ks = sorted(ctr)
STEP = statistics.median([math.dist(ctr[a], ctr[b]) for a, b in zip(ks, ks[1:])])


def index_of(name: str) -> int:
    return int(re.match(r"1-high_(\d+)_", name).group(1))


common = [k for k in OWN if k in MERGED]


def fit(keys, label):
    if len(keys) < 6:
        print(f"{label}: only {len(keys)} cameras")
        return None
    s, R, t = umeyama([OWN[k] for k in keys], [MERGED[k] for k in keys])
    res = {}
    for k in common:                      # evaluate on ALL cameras of the set
        p = OWN[k]
        m = [s * sum(R[r][j] * p[j] for j in range(3)) + t[r] for r in range(3)]
        res[k] = math.dist(m, MERGED[k])
    fitted = [res[k] for k in keys]
    print(f"{label:28s} n={len(keys):3d} scale={s:6.3f}  "
          f"residual on fitted cameras: median {statistics.median(fitted)/STEP*100:6.2f}%  "
          f"max {max(fitted)/STEP*100:7.2f}%")
    return res


print(f"merged median index step {STEP:.4f}; set has {len(common)} cameras "
      f"in the merged component\n")
whole = fit(common, "whole set 280-304")
lo = fit([k for k in common if index_of(k) <= 299], "first part 280-299")
hi = fit([k for k in common if index_of(k) >= 300], "second part 300-304")

print("\nper-index residual with each fit (% of one index step)")
print("  idx  n   whole    fit-on-280-299   fit-on-300-304")
for i in range(280, 305):
    keys = [k for k in common if index_of(k) == i]
    if not keys:
        print(f"  {i}  --")
        continue
    def med(d):
        return statistics.median([d[k] for k in keys]) / STEP * 100 if d else float("nan")
    print(f"  {i:4d} {len(keys):2d}  {med(whole):7.2f}  {med(lo):14.2f}  {med(hi):14.2f}")
