"""Work out how RealityScan's yaw/pitch/roll map to a rotation matrix.

The CSV export gives Euler angles without saying which convention they use, and
guessing wrong silently produces cameras that point the wrong way. It can be
derived instead: the bridge component and component 1 contain the same physical
cameras in two frames related by a similarity whose rotation A is recoverable
from the positions alone. Whatever the convention is, it has to satisfy

    R(angles in frame 1) == R(angles in the bridge) . A^T          (or A . R, ...)

for every shared camera. Trying every plausible axis order, rotation direction,
and composition side and scoring the residual angle picks the right one out, and
the score of the runner-up shows how unambiguous the answer is.
"""

from __future__ import annotations

import csv
import itertools
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BRIDGE = Path(r"K:\realityscan\out\1-mid-2\_bridges\bridge_0440-0664_cameras.csv")
OTHER = Path(r"K:\realityscan\out\1-mid-2\_merged_bridge\cameras.csv")
PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def load(p: Path):
    pos: dict[str, np.ndarray] = {}
    ang: dict[str, np.ndarray] = {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                pos[r["#name"]] = np.array(
                    [float(r["x"]), float(r["y"]), float(r["alt"])])
                ang[r["#name"]] = np.array(
                    [float(r["yaw"]), float(r["pitch"]), float(r["roll"])])
            except (KeyError, ValueError):
                continue
    return pos, ang


def umeyama_rot(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    s, d = src - src.mean(0), dst - dst.mean(0)
    u, _, vt = np.linalg.svd(d.T @ s / len(src))
    e = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[-1] = -1
    return u @ np.diag(e) @ vt


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample rotation angle of a . b^T, in degrees."""
    m = np.einsum("nij,nkj->nik", a, b)
    tr = np.clip((np.trace(m, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def main() -> int:
    pb, ab = load(BRIDGE)
    po, ao = load(OTHER)
    common = sorted(set(pb) & set(po))
    if len(common) < 50:
        print(f"only {len(common)} shared cameras")
        return 1
    src = np.array([pb[k] for k in common])
    dst = np.array([po[k] for k in common])
    A = umeyama_rot(src, dst)
    print(f"{len(common)} shared cameras; frame rotation recovered from positions")

    eb = np.array([ab[k] for k in common])
    eo = np.array([ao[k] for k in common])

    results = []
    orders = ["".join(p) for p in itertools.permutations("xyz")]
    for order in orders:
        for upper in (False, True):            # intrinsic vs extrinsic
            seq = order.upper() if upper else order
            for signs in itertools.product((1, -1), repeat=3):
                s = np.array(signs, dtype=float)
                try:
                    rb = Rotation.from_euler(seq, eb * s, degrees=True).as_matrix()
                    ro = Rotation.from_euler(seq, eo * s, degrees=True).as_matrix()
                except ValueError:
                    continue
                for side, pred in (
                        ("R_o = R_b A^T", np.einsum("nij,kj->nik", rb, A)),
                        ("R_o = A R_b", np.einsum("ij,njk->nik", A, rb)),
                        ("R_o = A^T R_b", np.einsum("ji,njk->nik", A, rb)),
                        ("R_o = R_b A", np.einsum("nij,jk->nik", rb, A))):
                    err = float(np.median(angle_between(ro, pred)))
                    results.append((err, seq, signs, side))

    results.sort()
    print("\nbest candidates (median residual angle over the shared cameras):")
    for err, seq, signs, side in results[:6]:
        print(f"  {err:8.4f} deg   seq={seq!r} signs={signs} {side}")
    # The same rotation can be written several ways (negating the angles and
    # transposing the composition give an identical matrix), so those score
    # identically and are not competitors. Compare against the first candidate
    # that is actually a different answer.
    best = results[0][0]
    rival = next((e for e, *_ in results if e > best + 1e-6), float("inf"))
    print(f"\nbest {best:.4f} deg, next distinct answer {rival:.4f} deg "
          f"({rival / max(best, 1e-9):.0f}x worse) - "
          f"{'unambiguous' if rival > 10 * max(best, 1e-6) else 'NOT clearly separated'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
