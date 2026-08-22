"""Will a patch actually fix a broken set join? Decide before re-merging.

A patch is one alignment straddling a boundary the merge failed to join. Testing
it by re-merging costs eleven hours, so test it on the camera lists instead.

Two things have to hold. The patch's own reconstruction must run smoothly
through the boundary - if it jumps there too, the data really is discontinuous
and no amount of overlap will help. And each neighbouring set must fit onto the
patch, because that is the link the merge would use: the patch shares a dozen
indices with each side instead of the three or five that failed.

Rotation comes from the camera orientations, not the positions: all views of one
index sit at the same rig centre, so a shared stretch is a short near-straight
line and the roll about it is unconstrained. Scale comes from the median ratio
of the distance travelled between consecutive shared indices.

    python diag_patch_effect.py <patch dir> <set dir> <set dir> [--merged CSV]

The two set dirs are the neighbours the boundary falls between. --merged points
at the merged camera list, to print what the join looks like today.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")
SEQ = "ZXY"                     # R = Rz(-yaw) Rx(pitch) Ry(roll)


def load(p: Path):
    names, pos, ang = [], [], []
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                pos.append([float(r["x"]), float(r["y"]), float(r["alt"])])
                ang.append([float(r["yaw"]), float(r["pitch"]), float(r["roll"])])
            except (KeyError, ValueError):
                continue
            names.append(r["#name"])
    return names, np.array(pos), np.array(ang)


def mats(ang: np.ndarray) -> np.ndarray:
    return Rotation.from_euler(
        SEQ, np.column_stack([-ang[:, 0], ang[:, 1], ang[:, 2]]),
        degrees=True).as_matrix()


def centres(names, pos):
    acc = defaultdict(list)
    for n, p in zip(names, pos):
        m = PAT.match(n)
        if m:
            acc[int(m.group(2))].append(p)
    return {i: np.mean(v, axis=0) for i, v in acc.items()}


def step_at(c: dict[int, np.ndarray], boundary: int):
    keys = sorted(c)
    d = [float(np.linalg.norm(c[b] - c[a]))
         for a, b in zip(keys, keys[1:]) if b - a == 1]
    if not d:
        return None, None
    med = float(np.median(d))
    here = (float(np.linalg.norm(c[boundary] - c[boundary - 1]))
            if boundary in c and boundary - 1 in c else None)
    return (here / med if here is not None else None), med


def fit(pa, ra, pb, rb, idx):
    """Similarity taking frame B to frame A, and the residual it leaves.

    Robust, because a component can contain a camera that is simply wrong: a
    patch over 25 indices came back with two cameras 27,000 and 88,000 units
    out while the other 191 sat within 15. Both were registered and both were
    in the exported component. A centroid and a least-squares residual are at
    the mercy of a single such point - the translation alone came out 887 units
    off - so the centre is taken as a median and the whole fit is repeated with
    the worst offenders dropped.
    """
    q = np.einsum("nij,nkj->ik", ra, rb) / len(ra)
    u, _, vt = np.linalg.svd(q)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    q = u @ np.diag(d) @ vt

    ca = {i: np.median(pa[idx == i], axis=0) for i in np.unique(idx)}
    cb = {i: np.median(pb[idx == i], axis=0) for i in np.unique(idx)}
    keys = sorted(ca)
    ratios = [float(np.linalg.norm(ca[y] - ca[x]))
              / float(np.linalg.norm(cb[y] - cb[x]))
              for x, y in zip(keys, keys[1:])
              if y - x == 1 and np.linalg.norm(cb[y] - cb[x]) > 0]
    s = float(np.median(ratios)) if ratios else 1.0

    keep = np.ones(len(pa), bool)
    for _ in range(3):
        t = np.median(pa[keep] - (s * q @ pb[keep].T).T, axis=0)
        res = np.linalg.norm((s * q @ pb.T).T + t - pa, axis=1)
        cut = np.median(res[keep]) * 5 + 1e-12
        new = res <= max(cut, np.percentile(res, 50))
        if new.sum() < 8 or (new == keep).all():
            break
        keep = new
    return s, q, t, res, keep


def main() -> int:
    argv = list(sys.argv[1:])
    merged = None
    if "--merged" in argv:
        i = argv.index("--merged")
        merged = Path(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) < 3:
        print(__doc__)
        return 2
    patch_dir, *set_dirs = [Path(a) for a in argv]

    pn, pp, pa = load(patch_dir / "cameras.csv")
    pc = centres(pn, pp)
    lo, hi = min(pc), max(pc)
    boundary = max(int(d.name.split("_")[1].split("-")[0]) for d in set_dirs)
    print(f"patch {patch_dir.name}: indices {lo}-{hi}, boundary at {boundary}")

    if merged and merged.is_file():
        mn, mp, _ = load(merged)
        pre = PAT.match(pn[0]).group(1)
        mc = centres([n for n in mn if n.startswith(pre + "_")],
                     np.array([q for n, q in zip(mn, mp)
                               if n.startswith(pre + "_")]))
        r, _ = step_at(mc, boundary)
        print(f"  today, in the merged result: {r:.1f}x the median step"
              if r else "  boundary not in the merged result")

    r, _ = step_at(pc, boundary)
    print(f"  inside the patch itself:     {r:.2f}x the median step"
          if r else "  the patch does not cover the boundary")

    print()
    pm = mats(pa)
    pidx = {n: k for k, n in enumerate(pn)}
    for d in set_dirs:
        sn, sp, sa = load(d / "cameras.csv")
        sm = mats(sa)
        common = [n for n in sn if n in pidx]
        if len(common) < 8:
            print(f"  {d.name}: only {len(common)} cameras shared with the patch")
            continue
        ia = np.array([pidx[n] for n in common])
        ib = np.array([sn.index(n) for n in common])
        idx = np.array([int(PAT.match(n).group(2)) for n in common])
        s, q, t, res, keep = fit(pp[ia], pm[ia], sp[ib], sm[ib], idx)
        _, med = step_at(centres(pn, pp), boundary)
        print(f"  {d.name} -> patch: {len(common)} shared cameras "
              f"({len(set(idx))} indices), scale {s:.4f}")
        print(f"      residual median {np.median(res[keep]) / med:.3f}, "
              f"max {res[keep].max() / med:.3f} index steps "
              f"over {keep.sum()}/{len(res)} cameras")
        if (~keep).any():
            worst = np.argsort(res)[::-1][:2]
            print(f"      rejected {int((~keep).sum())}: "
                  + ", ".join(f"{common[i]} at {res[i] / med:.0f} steps"
                              for i in worst if not keep[i]))
    print("\nA patch is usable when it runs smoothly through the boundary and "
          "both\nneighbours fit onto it to well under one index step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
