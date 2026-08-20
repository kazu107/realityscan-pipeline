"""Find which reference indices a set can actually be seeded with.

A seed only helps if the two captures really see the same place. Guessing costs
a full set alignment per guess, and a seed that fails to join leaves that
stretch with no constraint at all - which is exactly where the merged bands were
measured to drift.

The bands can be put in one frame using the seeds that did join, and then the
question is geometric: for each set of the target band, which reference indices
pass closest to it? Those are the candidates, ranked by distance.

    python find_seed_anchors.py <band cameras.csv> <reference cameras.csv>
                               <band prefix> <reference prefix>
                               [--sets 0090-0119,...] [--chunk 25 --overlap 3]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def centres(p: Path) -> dict[str, dict[int, np.ndarray]]:
    acc: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            m = PAT.match(r["#name"])
            if not m:
                continue
            acc[m.group(1)][int(m.group(2))].append(
                [float(r["x"]), float(r["y"]), float(r["alt"])])
    return {k: {i: np.mean(v, axis=0) for i, v in d.items()} for k, d in acc.items()}


def umeyama(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    e = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[-1] = -1
    r = u @ np.diag(e) @ vt
    var = (s ** 2).sum() / len(src)
    sc = float((sig * e).sum() / var) if var > 0 else 1.0
    return sc, r, mu_d - sc * r @ mu_s


def main() -> int:
    argv = list(sys.argv[1:])
    chunk, overlap, want = 25, 3, None
    for flag, conv in (("--chunk", int), ("--overlap", int), ("--sets", str)):
        if flag in argv:
            i = argv.index(flag)
            v = conv(argv[i + 1])
            if flag == "--chunk":
                chunk = v
            elif flag == "--overlap":
                overlap = v
            else:
                want = [x.strip() for x in v.split(",")]
            del argv[i:i + 2]
    band_csv, ref_csv, bpre, rpre = argv[:4]

    band = centres(Path(band_csv))
    ref = centres(Path(ref_csv))
    if bpre not in band or rpre not in band:
        print(f"{band_csv} has prefixes {sorted(band)} - need {bpre} and {rpre}")
        return 1

    shared = sorted(set(band[rpre]) & set(ref[rpre]))
    src = np.array([band[rpre][i] for i in shared])
    dst = np.array([ref[rpre][i] for i in shared])
    s, r, t = umeyama(src, dst)
    resid = np.linalg.norm((s * r @ src.T).T + t - dst, axis=1)
    rk = sorted(ref[rpre])
    unit = float(np.median([np.linalg.norm(ref[rpre][b] - ref[rpre][a])
                            for a, b in zip(rk, rk[1:]) if b - a == 1]))
    print(f"aligned on {len(shared)} shared {rpre} indices, scale {s:.4f}, "
          f"fit residual median {np.median(resid) / unit:.2f} index steps\n")

    # the band's own cameras, moved into the reference frame
    bidx = sorted(band[bpre])
    bpos = {i: (s * r @ band[bpre][i]) + t for i in bidx}
    ridx = np.array(rk)
    rpos = np.array([ref[rpre][i] for i in rk])

    stride = chunk - overlap
    starts = [i for i in range(bidx[0], bidx[-1] + 1, stride)]
    names = [f"{a:04d}-{min(a + chunk - 1, bidx[-1]):04d}" for a in starts]
    for a, name in zip(starts, names):
        if want and name not in want:
            continue
        mine = [bpos[i] for i in range(a, a + chunk) if i in bpos]
        if not mine:
            continue
        mine = np.array(mine)
        # closest approach of every reference index to this set's path
        d = np.linalg.norm(ridx.reshape(-1, 1, 1) * 0 + rpos[:, None, :]
                           - mine[None, :, :], axis=2).min(axis=1) / unit
        order = np.argsort(d)
        print(f"{bpre}_{name}: nearest {rpre} indices")
        shown = 0
        seen: list[int] = []
        for k in order:
            i = int(ridx[k])
            if any(abs(i - j) < chunk for j in seen):
                continue
            seen.append(i)
            lo = (i // stride) * stride
            print(f"   {rpre}_{i:04d} at {d[k]:6.1f} index steps  "
                  f"-> component {rpre}_{lo:04d}-{lo + chunk - 1:04d}")
            shown += 1
            if shown == 3:
                break
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
