"""Assemble every set into one frame without -mergeComponents.

RealityScan could not do this for 1-mid-2: with the overlap poses locked the
sets carry internal scale jumps and the merge refuses to join them, and without
the lock the merge has only five near-collinear rig positions to fix the scale
from and blew the second half up by 9x. The information needed is all there
though - consecutive sets share their overlap images - so the transform can be
computed directly and checked.

Scale and translation come from the shared positions. Rotation does not: all
views of one index sit at the same rig centre, so the shared stretch is a short
near-straight line and the roll about it is unconstrained. The camera
orientations settle it - eight views per index point in eight known directions -
so the rotation is taken from those and the positions only fix scale and shift.

    python stitch_chain.py <set root> <out dir> [--prefix P] [--no-ply]
                           [--patches <dir>]

Writes cameras.csv and stitched_sparse.ply, and reports the residual at every
junction so a bad handoff cannot pass silently.

--patches takes a directory of stand-alone alignments straddling the weak
junctions (align_patches.py). Where a patch covers a junction, the chain routes
A -> patch -> B, which shares about fifteen indices with each side instead of
five, and the direct fit is used only as a fallback.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")
SPAN = re.compile(r"_(\d+)-(\d+)$")
SEQ = "ZXY"                      # R = Rz(-yaw) Rx(pitch) Ry(roll)
FIELDS = ["#name", "x", "y", "alt", "yaw", "pitch", "roll", "f_35mm",
          "px_norm", "py_norm", "k1", "k2", "k3", "k4", "t1", "t2"]


def to_matrix(a: np.ndarray) -> np.ndarray:
    """(yaw, pitch, roll) columns -> rotation matrices."""
    return Rotation.from_euler(
        SEQ, np.column_stack([-a[:, 0], a[:, 1], a[:, 2]]), degrees=True
    ).as_matrix()


def to_angles(m: np.ndarray) -> np.ndarray:
    e = Rotation.from_matrix(m).as_euler(SEQ, degrees=True)
    e[:, 0] *= -1
    return e


def read_set(p: Path):
    rows = []
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    names = [r["#name"] for r in rows]
    pos = np.array([[float(r["x"]), float(r["y"]), float(r["alt"])] for r in rows])
    ang = np.array([[float(r["yaw"]), float(r["pitch"]), float(r["roll"])]
                    for r in rows])
    return rows, names, pos, ang


def fit(pa: np.ndarray, ra: np.ndarray, pb: np.ndarray, rb: np.ndarray,
        idx: np.ndarray):
    """Similarity taking frame B to frame A: p_a = s Q p_b + t, R_a = Q R_b.

    Scale is the median ratio of the distance the rig travelled between
    consecutive shared indices. Spread about the centroid would do the same job
    in principle, but the shared stretch is a short near-straight line and the
    centroid measure is dominated by its ends, where a set's solve is least
    constrained - chaining that over 67 junctions drifted by 4.8x, while the
    per-step median does not.
    """
    q = np.einsum("nij,nkj->ik", ra, rb) / len(ra)      # mean of R_a R_b^T
    u, _, vt = np.linalg.svd(q)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    q = u @ np.diag(d) @ vt

    ca_i = {i: pa[idx == i].mean(0) for i in np.unique(idx)}
    cb_i = {i: pb[idx == i].mean(0) for i in np.unique(idx)}
    keys = sorted(ca_i)
    ratios = []
    for x, y in zip(keys, keys[1:]):
        if y - x != 1:
            continue
        da = float(np.linalg.norm(ca_i[y] - ca_i[x]))
        db = float(np.linalg.norm(cb_i[y] - cb_i[x]))
        if da > 0 and db > 0:
            ratios.append(da / db)
    s = float(np.median(ratios)) if ratios else 1.0

    ca, cb = pa.mean(0), pb.mean(0)
    t = ca - s * q @ cb
    res = np.linalg.norm((s * q @ pb.T).T + t - pa, axis=1)
    return s, q, t, res


def compose(a, b):
    """a after b:  p -> a(b(p))."""
    sa, qa, ta = a
    sb, qb, tb = b
    return sa * sb, qa @ qb, sa * qa @ tb + ta


def main() -> int:
    argv = list(sys.argv[1:])
    prefix, do_ply, patch_dir, pos_args = "", True, None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--prefix":
            prefix, i = argv[i + 1], i + 2
        elif argv[i] == "--no-ply":
            do_ply, i = False, i + 1
        elif argv[i] == "--patches":
            patch_dir, i = Path(argv[i + 1]), i + 2
        else:
            pos_args.append(argv[i])
            i += 1
    root, out = Path(pos_args[0]), Path(pos_args[1])
    out.mkdir(parents=True, exist_ok=True)

    dirs = sorted((d for d in root.iterdir()
                   if d.is_dir() and not d.name.startswith("_")
                   and (d / "cameras.csv").exists()),
                  key=lambda d: int(SPAN.search(d.name).group(1)))
    print(f"{len(dirs)} sets in {root}")

    data = {d.name: read_set(d / "cameras.csv") for d in dirs}
    mats = {n: to_matrix(v[3]) for n, v in data.items()}
    index = {n: {name: k for k, name in enumerate(v[1])} for n, v in data.items()}

    # how far apart are the views of one index? that is what the scale fit has
    # to work with beyond the motion along the path
    first = dirs[0].name
    by_idx: dict[int, list[int]] = defaultdict(list)
    for name, k in index[first].items():
        m = PAT.match(name)
        if m:
            by_idx[int(m.group(2))].append(k)
    spread = np.median([np.linalg.norm(data[first][2][v] - data[first][2][v].mean(0),
                                       axis=1).max() for v in by_idx.values()])
    step = np.median([np.linalg.norm(data[first][2][b].mean(0) -
                                     data[first][2][a].mean(0))
                      for a, b in zip(*[list(by_idx.values())[i:]
                                        for i in (0, 1)])])
    print(f"  views of one index are spread {spread:.4f}, "
          f"the step between indices is {step:.4f} "
          f"({spread / step * 100:.1f}% of it)")

    patches = {}
    if patch_dir and patch_dir.is_dir():
        for d in sorted(patch_dir.iterdir()):
            c = d / "cameras.csv"
            if d.is_dir() and c.exists():
                v = read_set(c)
                patches[d.name] = (v, to_matrix(v[3]),
                                   {n: k for k, n in enumerate(v[1])})
        print(f"  {len(patches)} patches loaded from {patch_dir}")

    def pair_fit(na: str, va, ma, ixa, nb: str, vb, mb, ixb):
        common = [n for n in vb[1] if n in ixa]
        if len(common) < 8:
            return None
        ia = np.array([ixa[n] for n in common])
        ib = np.array([ixb[n] for n in common])
        idx = np.array([int(PAT.match(n).group(2)) for n in common])
        s, q, t, res = fit(va[2][ia], ma[ia], vb[2][ib], mb[ib], idx)
        return s, q, t, float(np.median(res)), len(common)

    glob = {dirs[0].name: (1.0, np.eye(3), np.zeros(3))}
    print(f"\n{'junction':<30}{'shared':>7}{'scale':>10}{'residual':>12}  via")
    worst = 0.0
    for a, b in zip(dirs, dirs[1:]):
        na, nb = a.name, b.name
        va, ma, ixa = data[na], mats[na], index[na]
        vb, mb, ixb = data[nb], mats[nb], index[nb]
        direct = pair_fit(na, va, ma, ixa, nb, vb, mb, ixb)
        if direct is None:
            print(f"{na} -> {nb}: too few shared cameras")
            return 1
        best, via = direct, "direct"

        # a patch straddling this junction shares far more with both sides
        for pn, (pv, pm, pix) in patches.items():
            f1 = pair_fit(na, va, ma, ixa, pn, pv, pm, pix)
            f2 = pair_fit(pn, pv, pm, pix, nb, vb, mb, ixb)
            # only worth routing through a patch that really covers both sides,
            # which means more shared cameras than the direct hop has
            if not f1 or not f2 or min(f1[4], f2[4]) <= direct[4]:
                continue
            comb = compose((f1[0], f1[1], f1[2]), (f2[0], f2[1], f2[2]))
            # residual of the composed hop, in set A's units
            score = max(f1[3], f2[3] * f1[0])
            if score < best[3]:
                best, via = (comb[0], comb[1], comb[2], score,
                             min(f1[4], f2[4])), pn.split("_")[-1]

        s, q, t = best[0], best[1], best[2]
        glob[nb] = compose(glob[na], (s, q, t))
        rel = best[3] / step * glob[na][0] if step else 0.0
        worst = max(worst, rel)
        print(f"{na[-9:]} -> {nb[-9:]:<9}{best[4]:>7}{s:>10.4f}"
              f"{rel:>11.3f}x  {via}")
    print(f"worst junction residual: {worst:.3f} index steps")

    seen: dict[str, str] = {}
    rows_out = []
    for d in dirs:
        n = d.name
        s, q, t = glob[n]
        rows, names, pos, ang = data[n]
        p2 = (s * q @ pos.T).T + t
        a2 = to_angles(np.einsum("ij,njk->nik", q, mats[n]))
        for k, name in enumerate(names):
            if name in seen:
                continue
            seen[name] = n
            row = {f: rows[k].get(f, "") for f in FIELDS}
            row["x"], row["y"], row["alt"] = (f"{v:.12g}" for v in p2[k])
            row["yaw"], row["pitch"], row["roll"] = (f"{v:.12g}" for v in a2[k])
            rows_out.append(row)

    dst = out / "cameras.csv"
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n{dst}: {len(rows_out)} cameras")

    if do_ply:
        chunks = []
        total = 0
        for d in dirs:
            plys = list(d.glob("*_sparse.ply"))
            if not plys:
                continue
            s, q, t = glob[d.name]
            v = PlyData.read(str(plys[0]))["vertex"].data.copy()
            p = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
            p = (s * q @ p.T).T + t
            v["x"], v["y"], v["z"] = p[:, 0], p[:, 1], p[:, 2]
            chunks.append(v)
            total += len(v)
        if chunks:
            allv = np.concatenate(chunks)
            PlyData([PlyElement.describe(allv, "vertex")],
                    text=False).write(str(out / "stitched_sparse.ply"))
            print(f"{out / 'stitched_sparse.ply'}: {total} points "
                  f"from {len(chunks)} sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
