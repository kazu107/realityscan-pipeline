"""Angular coverage per capture, joining the two exports of the same component.

The OpenCV export carries the rotation matrix but renames every image to a
sequential number; the Internal/External export keeps the original names but
only has Euler angles, which are useless here (pitch sits at ~89 deg, so yaw
and roll are degenerate). Both list the component's cameras in the same order,
so joining them row by row recovers name + rotation. The join is verified by
comparing the camera centres the two exports imply.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

PAIRS = {
    "1-mid-2-1 (front)": (
        r"K:\realityscan\out\_fovcheck\1-mid-2-1\opencv.csv",
        r"K:\realityscan\out\1-mid-2-1\1-mid-2-repaired_0000-0024\cameras.csv"),
    "1-mid-2b (rear)": (
        r"K:\realityscan\out\_fovcheck\1-mid-2b\opencv.csv",
        r"K:\realityscan\out\1-mid-2b\1-mid-2b_0000-0024\cameras.csv"),
}


def read_rows(p: Path):
    out = []
    for line in Path(p).open(encoding="utf-8", errors="replace"):
        if line.startswith("#"):
            continue
        out.append(line.rstrip("\n").split(","))
    return out


for label, (ocv_p, ext_p) in PAIRS.items():
    ocv, ext = read_rows(ocv_p), read_rows(ext_p)
    print(f"\n================ {label}")
    if len(ocv) != len(ext):
        print(f"  row counts differ ({len(ocv)} vs {len(ext)}) - cannot join")
        continue

    names, centres, axes = [], [], []
    for a, b in zip(ocv, ext):
        t = [float(a[1]), float(a[2]), float(a[3])]
        R = [[float(a[4 + 3 * r + k]) for k in range(3)] for r in range(3)]
        centres.append([-sum(R[r][k] * t[r] for r in range(3)) for k in range(3)])
        axes.append(R[2])
        names.append(b[0])

    ext_c = [[float(r[1]), float(r[2]), float(r[3])] for r in ext]
    err = [math.dist(c, e) for c, e in zip(centres, ext_c)]
    scale = statistics.median([math.dist(centres[0], c) for c in centres[1:]] or [1])
    print(f"  join check: centre mismatch median {statistics.median(err):.6f}, "
          f"max {max(err):.6f}  (spread of the cloud ~{scale:.3f})")
    if statistics.median(err) > 1e-3 * max(scale, 1):
        print("  centres do not agree - the row order assumption is wrong, stopping")
        continue

    groups = defaultdict(list)
    for name, c, ax in zip(names, centres, axes):
        m = re.match(r"(.+)_(\d+)_(\d+)\.", name)
        if m:
            groups[m.group(1)].append((int(m.group(2)), int(m.group(3)), c, ax))

    for pre, rows in sorted(groups.items()):
        by_idx = defaultdict(list)
        for i, v, c, ax in rows:
            by_idx[i].append(c)
        ctr = {i: [statistics.fmean(p[k] for p in v) for k in range(3)]
               for i, v in by_idx.items()}
        idx = sorted(ctr)
        travel = {}
        for a, b in zip(idx, idx[1:]):
            dx, dy = ctr[b][0] - ctr[a][0], ctr[b][1] - ctr[a][1]
            if math.hypot(dx, dy) > 1e-9:
                travel[a] = math.atan2(dy, dx)

        per_view = defaultdict(list)
        for i, v, c, ax in rows:
            if i in travel:
                az = math.atan2(ax[1], ax[0])
                per_view[v].append(
                    math.degrees((az - travel[i] + math.pi) % (2 * math.pi) - math.pi))
        if not per_view:
            continue
        tag = "reference 360 rig" if pre == "1-mid" else "this capture"
        print(f"\n  {pre}  ({len(rows)} cameras, {len(per_view)} views) - {tag}")
        meds = {}
        for v in sorted(per_view):
            s = sorted(per_view[v])
            meds[v] = s[len(s) // 2]
            print(f"     view {v:2d}: {s[len(s)//2]:7.1f} deg from travel")
        vals = sorted(meds.values())
        gaps = [(vals[(k + 1) % len(vals)] - vals[k]) % 360 for k in range(len(vals))]
        print(f"     span {max(vals)-min(vals):.0f} deg, "
              f"largest gap between neighbouring views {max(gaps):.0f} deg")
