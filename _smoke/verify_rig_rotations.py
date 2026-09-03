"""Check the assumed rig against rotations measured from a real reconstruction.

colpipe/rig.py builds cam_from_rig for view k as Rx(-pitch) @ Ry(yaw), and the
1-mid-1 run generated the directions as eight evenly spaced yaws starting at 0.
Neither the axis signs nor the spacing was ever checked against the images, and
the mapper runs with ba_refine_sensor_from_rig off - so if the rig is wrong,
nothing in the pipeline can notice: every frame carries the same error, the
reconstruction stays self-consistent, reprojection error stays low, and the
world comes out deformed. Which is what the 1-mid-1 model looks like.

RealityScan solved these same images as independent cameras, so its export is
an independent measurement of the extraction geometry. Relative rotations
between the views of one frame are compared here, because a relative rotation's
angle does not depend on whether the exported matrix is world-from-camera or
camera-from-world - only its axis flips sign, and the axis is reported too.

The Euler convention for the CSV was derived in find_euler_convention.py:
intrinsic ZXY on (-yaw, pitch, roll), residual 0.0487 degrees.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colpipe.rig import ring, rotation_matrix                    # noqa: E402

NAME = re.compile(r"^(?P<prefix>.*)_(?P<frame>\d+)_(?P<view>\d+)\.jpg$")


def load(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """frame -> view -> rotation matrix, from a RealityScan camera export."""
    out: dict[int, dict[int, np.ndarray]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            mt = NAME.match(row[0])
            if not mt:
                continue
            try:
                yaw, pitch, roll = (float(row[4]), float(row[5]), float(row[6]))
            except (IndexError, ValueError):
                continue
            R = Rotation.from_euler("ZXY", [-yaw, pitch, roll],
                                    degrees=True).as_matrix()
            out.setdefault(int(mt.group("frame")), {})[int(mt.group("view"))] = R
    return out


def axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    r = Rotation.from_matrix(R)
    v = r.as_rotvec()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-12 else np.zeros(3)), np.degrees(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--assumed-step", type=float, default=45.0)
    args = ap.parse_args()

    frames = load(Path(args.csv))
    full = {f: v for f, v in frames.items() if len(v) == args.views}
    print(f"{len(frames)} frames in the export, {len(full)} with all "
          f"{args.views} views")
    if not full:
        return 1

    # ---- measured: view k relative to view 0, per frame -----------------
    print(f"\n=== measured rotation of view k relative to view 0 ===")
    print(f"{'view':>5} {'angle (deg)':>14} {'spread':>9} {'axis (median)':>26}"
          f" {'axis spread':>12}")
    measured: dict[int, float] = {}
    axes: dict[int, np.ndarray] = {}
    for k in range(args.views):
        angs, axs = [], []
        for v in full.values():
            ax, an = axis_angle(v[k] @ v[0].T)
            angs.append(an)
            if an > 1.0:
                axs.append(ax)
        a = np.array(angs)
        measured[k] = float(np.median(a))
        if axs:
            X = np.array(axs)
            X *= np.sign(X @ np.median(X, axis=0))[:, None]   # unify direction
            axes[k] = np.median(X, axis=0)
            spread = float(np.degrees(np.arccos(np.clip(
                X @ (axes[k] / np.linalg.norm(axes[k])), -1, 1))).max())
            astr = np.array2string(axes[k] / np.linalg.norm(axes[k]),
                                   precision=3, suppress_small=True)
        else:
            spread, astr = 0.0, "-"
        print(f"{k:>5} {np.median(a):>14.3f} {np.ptp(a):>9.3f} {astr:>26}"
              f" {spread:>11.2f}")

    # ---- consecutive views ----------------------------------------------
    print(f"\n=== measured angle between consecutive views ===")
    steps = []
    for k in range(args.views):
        j = (k + 1) % args.views
        angs = [axis_angle(v[j] @ v[k].T)[1] for v in full.values()]
        steps.append(float(np.median(angs)))
        print(f"  view {k} -> {j}: {np.median(angs):8.3f} deg "
              f"(spread {np.ptp(angs):.3f})")
    print(f"  median consecutive step: {np.median(steps):.3f} deg, "
          f"assumed {args.assumed_step:.3f}")

    # ---- what the pipeline assumed --------------------------------------
    dirs = ring(args.views, 0.0, 0.0)
    print(f"\n=== the rig colpipe generated (ring of {args.views}, pitch 0, "
          f"start yaw 0) ===")
    print(f"{'view':>5} {'yaw':>8} {'assumed angle vs view 0':>25} "
          f"{'measured':>10} {'error':>8}")
    worst = 0.0
    for d in dirs:
        R0 = np.asarray(rotation_matrix(dirs[0].yaw, dirs[0].pitch))
        Rk = np.asarray(rotation_matrix(d.yaw, d.pitch))
        _, assumed = axis_angle(Rk @ R0.T)
        err = abs(assumed - measured[d.index])
        worst = max(worst, err)
        print(f"{d.index:>5} {d.yaw:>8.1f} {assumed:>25.3f} "
              f"{measured[d.index]:>10.3f} {err:>8.3f}")
    print(f"\nworst angular disagreement: {worst:.3f} deg")
    if worst < 1.0:
        print("  -> the rig's inter-view angles match the images")
    else:
        print("  -> the rig does NOT match the images. With "
              "ba_refine_sensor_from_rig off, nothing in the pipeline can "
              "correct this, and every frame carries the same error.")

    # ---- is the ring even? ----------------------------------------------
    d = np.diff([measured[k] for k in range(args.views)])
    print(f"\nmeasured angle to view 0, differences between adjacent views: "
          f"{np.array2string(d, precision=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
