"""Render the stitched reconstruction, next to the merges it replaces.

The stitched result has no .rsproj to open - RealityScan never produced it - so
this draws what there is: the sparse cloud from above with the camera path on
top, and the same path from the two merges that failed, which is the quickest
way to see what was wrong with them.

    python plot_result.py <out.png>
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plyfile import PlyData  # noqa: E402

BASE = Path(r"K:\realityscan\out")
PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")
PREFIX = "1-mid-2"

STITCH = BASE / "1-mid-2-nolock" / "_stitched"
CASES = [
    ("stitched (final)", STITCH / "cameras.csv"),
    ("RealityScan merge, no lock", BASE / "1-mid-2-nolock" / "_merged" / "cameras.csv"),
    ("RealityScan 3-way join", BASE / "1-mid-2" / "_merged_join" / "cameras.csv"),
]


def path_of(p: Path):
    acc: dict[int, list[np.ndarray]] = {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            m = PAT.match(r["#name"])
            if not m or m.group(1) != PREFIX:
                continue
            acc.setdefault(int(m.group(2)), []).append(
                [float(r["x"]), float(r["y"]), float(r["alt"])])
    idx = sorted(acc)
    return np.array(idx), np.array([np.mean(acc[i], axis=0) for i in idx])


def ground_plane(pts: np.ndarray):
    """Basis of the plane the walk lies in.

    A merged component gets its axes from a ground plane RealityScan derives,
    so "up" is alt. An offline stitch inherits the first set's frame, where up
    can be anything. Plotting x against y therefore shows a top view of one and
    a side view of the other - the walk looks like a straight line and the
    result looks broken when it is not. Taking the two leading principal axes
    gives a top view of either.
    """
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[:2]


def project(pts: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (pts - pts.mean(0)) @ basis.T


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "result.png")

    fig = plt.figure(figsize=(16, 11), dpi=110)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1], hspace=0.22, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    idx, p = path_of(STITCH / "cameras.csv")
    basis = ground_plane(p)
    centre = p.mean(0)

    ply = PlyData.read(str(STITCH / "stitched_sparse.ply"))["vertex"].data
    step = max(1, len(ply) // 900_000)
    v = ply[::step]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    q = (xyz - centre) @ basis.T
    col = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float32) / 255
    lim = np.percentile(np.abs(project(p, basis)), 99.5) * 1.6
    keep = (np.abs(q) < lim).all(1)
    ax.scatter(q[keep, 0], q[keep, 1], s=0.06, c=col[keep], marker=".",
               linewidths=0, alpha=0.55)

    pp = project(p, basis)
    sc = ax.scatter(pp[:, 0], pp[:, 1], s=5, c=idx, cmap="turbo", zorder=3)
    ax.plot(pp[:, 0], pp[:, 1], lw=0.5, color="k", alpha=0.35, zorder=2)
    fig.colorbar(sc, ax=ax, label="index", fraction=0.025, pad=0.01)
    ax.set_title(f"1-mid-2 stitched: {len(ply):,} sparse points, "
                 f"{len(idx):,} rig positions (top view, every {step}th point)")
    ax.set_aspect("equal")

    for k, (label, path) in enumerate(CASES):
        a = fig.add_subplot(gs[1, k])
        if not path.exists():
            a.set_title(f"{label}\n(missing)")
            a.axis("off")
            continue
        i2, q3 = path_of(path)
        q = project(q3, ground_plane(q3))
        s = a.scatter(q[:, 0], q[:, 1], s=2, c=i2, cmap="turbo")
        a.plot(q[:, 0], q[:, 1], lw=0.4, color="k", alpha=0.3)
        w, h = q.max(0) - q.min(0)
        a.set_title(f"{label}\n{len(i2)} indices, {w:.0f} x {h:.0f}", fontsize=10)
        a.set_aspect("equal")
        a.tick_params(labelsize=7)
        if k == 2:
            fig.colorbar(s, ax=a, label="index", fraction=0.04, pad=0.02)

    fig.savefig(out, bbox_inches="tight")
    print(f"{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
