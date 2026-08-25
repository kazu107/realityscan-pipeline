"""Are a band's seed anchors mutually consistent with 1-mid-1?

Seeding puts real 1-mid images into several of a band's sets, so those cameras
exist in two reconstructions at once and give exact correspondences. If the band
is internally sound, one similarity maps all of them onto their 1-mid-1
positions at once. If the band drifted, each seed group pulls in a different
direction and no single transform satisfies them - which is what "adding more
seeds did not help" looks like from the inside.

Reported per seed group: how far its cameras land from 1-mid-1 after the joint
fit, in multiples of the local index spacing, plus what the best fit to that
group alone would have been. A group whose own fit is good but whose joint
residual is large is a group the rest of the band disagrees with.

    python diag_seed_fit.py <band cameras.csv> [...] [--ref <1-mid-1 csv>]
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")
REF = Path(r"K:\realityscan\out\1-mid-1\_merged\cameras.csv")
SEED_PREFIX = "1-mid"


def load(p: Path):
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
    t = mu_d - scale * r @ mu_s
    return scale, r, t


def groups(names: list[str], gap: int = 5) -> list[list[str]]:
    """Split seed camera names into runs of consecutive indices."""
    idx = sorted({int(PAT.match(n).group(2)) for n in names})
    runs, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        (cur.append(b) if b - a <= gap else (runs.append(cur), cur := [b]))
    runs.append(cur)
    return [[n for n in names if int(PAT.match(n).group(2)) in set(r)] for r in runs]


def main() -> int:
    argv = list(sys.argv[1:])
    ref_path = REF
    files = []
    i = 0
    while i < len(argv):
        if argv[i] == "--ref":
            ref_path, i = Path(argv[i + 1]), i + 2
        else:
            files.append(argv[i])
            i += 1
    ref = load(ref_path)
    ref_c = {}
    for n, p in ref.items():
        m = PAT.match(n)
        if m:
            ref_c.setdefault(int(m.group(2)), []).append(p)
    ref_idx = sorted(ref_c)
    ref_step = float(np.median([
        np.linalg.norm(np.mean(ref_c[b], 0) - np.mean(ref_c[a], 0))
        for a, b in zip(ref_idx, ref_idx[1:]) if b - a == 1]))
    print(f"reference {ref_path}: {len(ref)} cameras, "
          f"median index step {ref_step:.4f}\n")

    for f in files:
        band = load(Path(f))
        shared = [n for n in band if n in ref and n.startswith(SEED_PREFIX + "_")]
        if len(shared) < 8:
            print(f"{f}: only {len(shared)} seed cameras shared - skipped")
            continue
        src = np.array([band[n] for n in shared])
        dst = np.array([ref[n] for n in shared])
        s, r, t = umeyama(src, dst)
        res = np.linalg.norm((s * r @ src.T).T + t - dst, axis=1)
        gs = groups(shared)
        print(f"{f}")
        print(f"  {len(shared)} seed cameras in {len(gs)} group(s), "
              f"joint fit scale {s:.4f}")
        print(f"  joint residual: median {np.median(res) / ref_step:.2f} steps, "
              f"p90 {np.percentile(res, 90) / ref_step:.2f}, "
              f"max {res.max() / ref_step:.2f}")
        print(f"  {'seed index':<16}{'n':>5}{'joint':>10}{'alone':>10}"
              f"{'own scale':>11}")
        for g in gs:
            gi = sorted({int(PAT.match(n).group(2)) for n in g})
            m = np.array([n in g for n in shared])
            joint = float(np.median(res[m])) / ref_step
            gs_, gr, gt = umeyama(src[m], dst[m])
            alone = float(np.median(np.linalg.norm(
                (gs_ * gr @ src[m].T).T + gt - dst[m], axis=1))) / ref_step
            print(f"  {gi[0]:>5}-{gi[-1]:<10}{len(g):>5}{joint:>9.2f}x"
                  f"{alone:>9.2f}x{gs_ / s:>10.3f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
