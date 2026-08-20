"""Can one similarity transform put a band into another band's frame?

Seeding puts a stretch of 1-mid cameras inside a 1-high or 1-low set, so after
that band is merged its component holds 1-mid cameras of its own - the same
images 1-mid-1 also solved. Those shared cameras are the only thing tying the
bands together, and -mergeComponents can only apply a single similarity per
component. So the question is whether one similarity actually fits them.

Fit it on every shared camera and look at where the residual lands. Small
everywhere means a merge can work and any drift is a merge-settings problem.
Residual that grows with distance from one region means the bands disagree in
shape, and no setting fixes that - only re-solving the cameras together can.

    python diag_band_fit.py <reference cameras.csv> <band cameras.csv> ...
                            [--anchor PREFIX] [--groups N]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def load(p: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[r["#name"]] = np.array(
                    [float(r["x"]), float(r["y"]), float(r["alt"])])
            except (KeyError, ValueError):
                continue
    return out


def umeyama(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    e = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[-1] = -1
    r = u @ np.diag(e) @ vt
    var = (s ** 2).sum() / len(src)
    scale = float((sig * e).sum() / var) if var > 0 else 1.0
    return scale, r, mu_d - scale * r @ mu_s


def step_of(cams: dict[str, np.ndarray], prefix: str) -> float:
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for n, p in cams.items():
        m = PAT.match(n)
        if m and m.group(1) == prefix:
            acc[int(m.group(2))].append(p)
    k = sorted(acc)
    c = {i: np.mean(acc[i], axis=0) for i in k}
    d = [float(np.linalg.norm(c[b] - c[a])) for a, b in zip(k, k[1:]) if b - a == 1]
    return float(np.median(d)) if d else 1.0


def main() -> int:
    argv = list(sys.argv[1:])
    anchor, groups, files = "1-mid", 8, []
    i = 0
    while i < len(argv):
        if argv[i] == "--anchor":
            anchor, i = argv[i + 1], i + 2
        elif argv[i] == "--groups":
            groups, i = int(argv[i + 1]), i + 2
        else:
            files.append(argv[i])
            i += 1

    ref = load(Path(files[0]))
    unit = step_of(ref, anchor)
    print(f"reference {files[0]}: {len(ref)} cams, "
          f"median {anchor} index step {unit:.4f}\n")

    for f in files[1:]:
        band = load(Path(f))
        common = sorted(n for n in band if n in ref and n.startswith(anchor + "_"))
        if len(common) < 20:
            print(f"{f}: only {len(common)} shared {anchor} cameras - skipped\n")
            continue
        src = np.array([band[n] for n in common])
        dst = np.array([ref[n] for n in common])
        s, r, t = umeyama(src, dst)
        res = np.linalg.norm((s * r @ src.T).T + t - dst, axis=1) / unit

        idx = np.array([int(PAT.match(n).group(2)) for n in common])
        print(f"{f}")
        print(f"  {len(common)} shared {anchor} cameras, scale {s:.4f}")
        print(f"  residual in {anchor} index steps: median {np.median(res):.2f}, "
              f"p90 {np.percentile(res, 90):.2f}, max {res.max():.2f}")
        edges = np.linspace(idx.min(), idx.max() + 1, groups + 1).astype(int)
        print(f"  {'anchor index':<18}{'cams':>6}{'median':>9}{'max':>9}")
        for a, b in zip(edges, edges[1:]):
            m = (idx >= a) & (idx < b)
            if not m.any():
                continue
            print(f"  {f'{a}-{b - 1}':<18}{m.sum():>6}"
                  f"{np.median(res[m]):>9.2f}{res[m].max():>9.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
