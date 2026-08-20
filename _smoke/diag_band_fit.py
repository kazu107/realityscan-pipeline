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

Do not rank two runs by that residual alone. It is computed only where the runs
happen to share cameras, so a run whose seeds failed to join has no cameras in
the stretch it got wrong and scores *better* - which is how a run with three
dead seeds was first read as the best of four. --holdout fits on one half of the
route and measures the error on the other, over the cameras every run shares,
which is what "drifts away from the anchors" actually means.

    python diag_band_fit.py <reference cameras.csv> <band cameras.csv> ...
                            [--anchor PREFIX] [--groups N] [--holdout]
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


def cross_validate(files: list[str], ref: dict, unit: float, anchor: str) -> int:
    """Score every run on the same points, fitting on half and testing on the rest."""
    loaded = {}
    for f in files[1:]:
        parts = Path(f).parts
        loaded[parts[-3] if len(parts) > 2 else f] = load(Path(f))
    common = set(ref)
    for d in loaded.values():
        common &= set(d)
    common = sorted(n for n in common if n.startswith(anchor + "_"))
    if len(common) < 40:
        print(f"only {len(common)} cameras shared by all runs - cannot compare")
        return 1
    idx = np.array([int(PAT.match(n).group(2)) for n in common])
    dst = np.array([ref[n] for n in common])
    mid = float(np.median(idx))
    print(f"evaluated on the {len(common)} {anchor} cameras every run shares, "
          f"index {idx.min()}-{idx.max()}")
    print()
    print(f"{'run':<24}{'fit all':>9}{'fit early':>11}{'fit late':>10}"
          f"{'held-out':>10}")
    rows = []
    for name, d in loaded.items():
        src = np.array([d[n] for n in common])

        def err(train, test):
            s, r, t = umeyama(src[train], dst[train])
            e = np.linalg.norm((s * r @ src[test].T).T + t - dst[test], axis=1)
            return float(np.median(e)) / unit

        every = np.ones(len(common), bool)
        a = err(idx <= mid, idx > mid)
        b = err(idx > mid, idx <= mid)
        rows.append((max(a, b), name, err(every, every), a, b))
    for worst, name, allf, a, b in sorted(rows):
        print(f"{name:<24}{allf:>9.2f}{a:>11.2f}{b:>10.2f}{worst:>10.2f}")
    print()
    print("held-out is the column to compare: the error where the fit was given "
          "no cameras to work from.")
    return 0


def main() -> int:
    argv = list(sys.argv[1:])
    anchor, groups, holdout, files = "1-mid", 8, False, []
    i = 0
    while i < len(argv):
        if argv[i] == "--anchor":
            anchor, i = argv[i + 1], i + 2
        elif argv[i] == "--groups":
            groups, i = int(argv[i + 1]), i + 2
        elif argv[i] == "--holdout":
            holdout, i = True, i + 1
        else:
            files.append(argv[i])
            i += 1

    ref = load(Path(files[0]))
    unit = step_of(ref, anchor)
    print(f"reference {files[0]}: {len(ref)} cams, "
          f"median {anchor} index step {unit:.4f}\n")

    if holdout:
        return cross_validate(files, ref, unit, anchor)

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
