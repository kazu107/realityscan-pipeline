"""Measure whether the two halves of a split merge actually disagree.

The bridge component is a single reconstruction spanning index 440-664, so it
crosses the break at 505. Each side of the split shares a long stretch with it:

    component 2  <- 440..504 ->  bridge  <- 505..664 ->  component 1

Fitting the bridge onto each side separately gives two similarity transforms. If
both fit well but the transforms differ, the two halves are genuinely
inconsistent and no merge setting will reconcile them - the scale ratio and
rotation angle between the transforms say by how much. If the transforms agree,
the halves are compatible and the merge simply failed to join them.

A similarity fit needs the points to span three dimensions, so the singular
values of the centred positions are printed too: a trajectory that is nearly a
straight line leaves the fit underdetermined and the numbers meaningless.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

BRIDGE = Path(r"K:\realityscan\out\1-mid-2\_bridges\bridge_0440-0664_cameras.csv")
MERGED = Path(r"K:\realityscan\out\1-mid-2\_merged_bridge")
PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")
PREFIX = "1-mid-2"


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
    t = mu_d - scale * r @ mu_s
    res = np.linalg.norm((scale * r @ src.T).T + t - dst, axis=1)
    return scale, r, t, res


def shape(pts: np.ndarray) -> str:
    sv = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
    sv = sv / sv[0]
    return f"singular values {sv[0]:.3f} / {sv[1]:.3f} / {sv[2]:.3f}"


def side(name: str, bridge: dict, other: dict, lo: int, hi: int):
    common = [k for k in bridge if k in other
              and (m := PAT.match(k)) and lo <= int(m.group(2)) <= hi]
    if len(common) < 20:
        print(f"{name}: only {len(common)} shared cameras in {lo}-{hi} - skipped")
        return None
    src = np.array([bridge[k] for k in common])
    dst = np.array([other[k] for k in common])
    scale, r, t, res = umeyama(src, dst)
    span = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    print(f"{name}: {len(common)} shared cameras, index {lo}-{hi}")
    print(f"  {shape(src)}")
    print(f"  scale {scale:.6f}   residual median {np.median(res) / span * 100:.3f}% "
          f"max {res.max() / span * 100:.3f}% of the span")
    return scale, r, t


def main() -> int:
    bridge = load(BRIDGE)
    c1 = load(MERGED / "cameras.csv")               # the exported maximal one
    p2 = MERGED / "Component_2.csv"
    if not p2.exists():
        print(f"{p2} not written yet")
        return 1
    c2 = load(p2)
    print(f"bridge {len(bridge)} cams, component1 {len(c1)}, component2 {len(c2)}\n")

    a = side("component 1 (post-break)", bridge, c1, 505, 664)
    print()
    b = side("component 2 (pre-break)", bridge, c2, 440, 504)
    if not (a and b):
        return 0

    s1, r1, t1 = a
    s2, r2, t2 = b

    # Each component carries its own arbitrary frame, so comparing the two fits
    # directly says nothing. Compose them instead: component 2 -> bridge ->
    # component 1 puts both halves in one frame, and the trajectory either runs
    # continuously through the break or it does not.
    def to_c1(p: np.ndarray) -> np.ndarray:
        via_bridge = (r2.T @ (p - t2).T).T / s2
        return (s1 * r1 @ via_bridge.T).T + t1

    # The seed cameras carry a different prefix but the same index numbers, so
    # they have to be filtered out or they land in the middle of the path.
    centres_1: dict[int, list[np.ndarray]] = {}
    centres_2: dict[int, list[np.ndarray]] = {}
    for src, dst in ((c1, centres_1), (c2, centres_2)):
        for n, p in src.items():
            m = PAT.match(n)
            if m and m.group(1) == PREFIX:
                dst.setdefault(int(m.group(2)), []).append(p)
    path = {i: (np.mean(v, axis=0), len(v)) for i, v in centres_1.items()}
    for i, v in centres_2.items():
        path.setdefault(i, (to_c1(np.array(v)).mean(axis=0), len(v)))

    keys = sorted(path)
    steps = {b_: float(np.linalg.norm(path[b_][0] - path[a_][0]))
             for a_, b_ in zip(keys, keys[1:]) if b_ - a_ == 1}
    med = float(np.median(list(steps.values())))
    print(f"\nboth halves placed in component 1's frame, via the bridge "
          f"(prefix {PREFIX} only)")
    print(f"  {len(keys)} indices, median step {med:.4f}")
    print(f"  step across the break (504 -> 505): "
          f"{steps.get(505, float('nan')) / med:.2f}x the median")
    print("  largest steps anywhere (with the view count at each index):")
    for i, d in sorted(steps.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {i - 1} -> {i}: {d / med:6.2f}x   "
              f"views {path[i - 1][1]} -> {path[i][1]}")
    big = sum(1 for d in steps.values() if d > 3 * med)
    print(f"  {big} of {len(steps)} steps exceed 3x the median")
    return 0


if __name__ == "__main__":
    sys.exit(main())
