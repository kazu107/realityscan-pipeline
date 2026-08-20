"""Measure the angular coverage of each capture's views.

The Internal/External CSV only carries Euler angles, and these cameras sit at
pitch ~89 deg where yaw and roll are degenerate, so the heading cannot be read
off it. The OpenCV-compliant export carries the rotation matrix instead, from
which the optical axis follows directly.

Each set-0 component also holds the seeded 1-mid cameras - a known 8-view,
360-degree rig - so the same measurement on those doubles as a sanity check.
"""

from __future__ import annotations

import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"K:\realityscan")
from rspipe.cli import _settings                       # noqa: E402
from rspipe.config import PipelineConfig               # noqa: E402

CFG = PipelineConfig.load(r"K:\realityscan\presets\1-mid-1.json")
OPENCV = "{B5331837-609D-4B12-A931-2863653d19F7}"
WORK = Path(r"K:\realityscan\out\_fovcheck")
CREATE_NO_WINDOW = 0x08000000

TARGETS = {
    "1-mid-2-1": r"K:\realityscan\out\1-mid-2-1\1-mid-2-repaired_0000-0024\1-mid-2-repaired_0000-0024.rsalign",
    "1-mid-2b": r"K:\realityscan\out\1-mid-2b\1-mid-2b_0000-0024\1-mid-2b_0000-0024.rsalign",
}


def export(label: str, component: Path) -> Path:
    d = WORK / label
    (d / "crash").mkdir(parents=True, exist_ok=True)
    out = d / "opencv.csv"
    if out.exists():
        return out
    args = [CFG.run.exe, "-headless", "-stdConsole", "-silent", str(d / "crash")]
    for k, v in _settings(CFG, force_rematch=False):
        args += ["-set", f"{k}={v}"]
    args += ["-set", f"calexFileFormatId={OPENCV}",
             "-set", "calexExportImages=false",
             "-set", "calexUndistortImages=false",
             "-newScene", "-importComponent", str(component),
             "-selectMaximalComponent",
             "-exportRegistration", str(out), "-quit"]
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=3600, creationflags=CREATE_NO_WINDOW)
    if not out.exists():
        print(f"  export failed for {label} (exit {p.returncode})")
    return out


def analyse(label: str, csv: Path) -> None:
    cams = []
    for line in csv.open(encoding="utf-8", errors="replace"):
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split(",")
        name = c[0]
        t = [float(c[1]), float(c[2]), float(c[3])]
        R = [[float(c[4 + 3 * r + k]) for k in range(3)] for r in range(3)]
        # X_cam = R X_world + t  ->  centre = -R^T t, optical axis = R's 3rd row
        centre = [-sum(R[r][k] * t[r] for r in range(3)) for k in range(3)]
        axis = R[2]
        m = re.match(r"(.+)_(\d+)_(\d+)\.", name)
        if m:
            cams.append((m.group(1), int(m.group(2)), int(m.group(3)), centre, axis))

    groups = defaultdict(list)
    for pre, i, v, centre, axis in cams:
        groups[pre].append((i, v, centre, axis))

    for pre, rows in sorted(groups.items()):
        ctr = {}
        for i, v, c, a in rows:
            ctr.setdefault(i, []).append(c)
        ctr = {i: [statistics.fmean(p[k] for p in v) for k in range(3)]
               for i, v in ctr.items()}
        idx = sorted(ctr)
        # local travel direction, projected on the horizontal plane
        travel = {}
        for a, b in zip(idx, idx[1:]):
            dx, dy = ctr[b][0] - ctr[a][0], ctr[b][1] - ctr[a][1]
            if math.hypot(dx, dy) > 1e-9:
                travel[a] = math.atan2(dy, dx)

        per_view = defaultdict(list)
        for i, v, c, ax in rows:
            if i not in travel:
                continue
            az = math.atan2(ax[1], ax[0])
            rel = math.degrees((az - travel[i] + math.pi) % (2 * math.pi) - math.pi)
            per_view[v].append(rel)

        if not per_view:
            continue
        print(f"\n  {label} / {pre}: {len(rows)} cameras, {len(per_view)} views")
        meds = {}
        for v in sorted(per_view):
            s = sorted(per_view[v])
            meds[v] = s[len(s) // 2]
            print(f"     view {v:2d}: {s[len(s)//2]:7.1f} deg from the travel "
                  f"direction  (spread p10..p90 {s[len(s)//10]:7.1f} .. "
                  f"{s[len(s)*9//10]:7.1f})")
        vals = sorted(meds.values())
        gaps = [(vals[(k + 1) % len(vals)] - vals[k]) % 360 for k in range(len(vals))]
        print(f"     -> views span {max(vals) - min(vals):.0f} deg; "
              f"largest gap between neighbouring views {max(gaps):.0f} deg")


WORK.mkdir(parents=True, exist_ok=True)
for label, comp in TARGETS.items():
    p = Path(comp)
    if not p.is_file():
        print(f"MISSING {label}: {p}")
        continue
    print(f"exporting {label} ...", flush=True)
    csv = export(label, p)
    if csv.exists():
        analyse(label, csv)
