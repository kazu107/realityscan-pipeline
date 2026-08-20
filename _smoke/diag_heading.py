"""Measure how the camera's heading drifts along a reconstruction.

The step profile only sees how far the rig moved between indices, so a block
that is rotated as a whole passes it untouched - which is how a stitched result
with tilted blocks was declared good. Heading catches it: this capture is made
of long straight runs, and runs that are physically parallel must come out
parallel.

Straight runs are found by walking the path and cutting wherever the direction
turns sharply. Each run's heading is then folded into [0, 90) so that runs
along the same line, in either direction, and runs at right angles to it all
land together - what is left is the drift.

    python diag_heading.py <cameras.csv> [...] [--prefix P] [--min-run N]
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def centres(p: Path, prefix: str):
    acc: dict[int, list[np.ndarray]] = {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            m = PAT.match(r["#name"])
            if not m or (prefix and m.group(1) != prefix):
                continue
            acc.setdefault(int(m.group(2)), []).append(
                [float(r["x"]), float(r["y"]), float(r["alt"])])
    idx = sorted(acc)
    return np.array(idx), np.array([np.mean(acc[i], axis=0) for i in idx])


def ground_plane(pts: np.ndarray) -> np.ndarray:
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[:2]


def runs(idx: np.ndarray, xy: np.ndarray, min_run: int, turn_deg: float = 12.0):
    """Split the path into straight runs of at least min_run indices."""
    d = np.diff(xy, axis=0)
    n = np.linalg.norm(d, axis=1)
    ok = n > 0
    ang = np.full(len(d), np.nan)
    ang[ok] = np.degrees(np.arctan2(d[ok, 1], d[ok, 0]))
    out, start = [], 0
    for i in range(1, len(ang)):
        if np.isnan(ang[i]) or np.isnan(ang[i - 1]):
            continue
        turn = abs((ang[i] - ang[i - 1] + 180) % 360 - 180)
        if turn > turn_deg or idx[i + 1] - idx[i] != 1:
            if i - start >= min_run:
                out.append((start, i))
            start = i
    if len(ang) - start >= min_run:
        out.append((start, len(ang)))
    return out


def main() -> int:
    argv = list(sys.argv[1:])
    prefix, min_run, files = "", 15, []
    i = 0
    while i < len(argv):
        if argv[i] == "--prefix":
            prefix, i = argv[i + 1], i + 2
        elif argv[i] == "--min-run":
            min_run, i = int(argv[i + 1]), i + 2
        else:
            files.append(argv[i])
            i += 1

    for f in files:
        idx, p3 = centres(Path(f), prefix)
        xy = (p3 - p3.mean(0)) @ ground_plane(p3).T
        rr = runs(idx, xy, min_run)
        if not rr:
            print(f"{f}: no straight runs found")
            continue
        rows = []
        for a, b in rr:
            v = xy[b] - xy[a]
            head = np.degrees(np.arctan2(v[1], v[0])) % 90.0
            rows.append((idx[a], idx[b], b - a, head,
                         float(np.linalg.norm(v))))
        heads = np.array([r[3] for r in rows])
        lens = np.array([r[4] for r in rows])
        # circular mean on a 90 degree period, weighted by run length
        th = np.radians(heads * 4)
        mean = (np.degrees(np.arctan2((lens * np.sin(th)).sum(),
                                      (lens * np.cos(th)).sum())) / 4) % 90
        dev = (heads - mean + 45) % 90 - 45
        print(f"{f}")
        print(f"  {len(rows)} straight runs of >= {min_run} indices, "
              f"dominant heading {mean:.2f} deg (mod 90)")
        print(f"  deviation: median {np.median(np.abs(dev)):.2f} deg, "
              f"p90 {np.percentile(np.abs(dev), 90):.2f} deg, "
              f"max {np.abs(dev).max():.2f} deg")
        for (a, b, n, h, ln), dv in sorted(zip(rows, dev),
                                           key=lambda t: -abs(t[1]))[:6]:
            print(f"    index {a:>5}-{b:<5} ({n:>3} long, {ln:6.1f} across): "
                  f"heading {h:6.2f}, off by {dv:+6.2f} deg")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
