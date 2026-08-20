"""Put a split merge back together outside RealityScan.

When -mergeComponents refuses to join two halves that are demonstrably
compatible, the join can be done directly: a component spanning the break shares
a long stretch with each half, which fixes the similarity from one half's frame
to the other. Cameras and sparse points are then transformed and concatenated.

What this does NOT do is a joint bundle adjustment, so whatever step exists at
the seam stays. Check it against the steps already present inside each half
(diag_step_profile.py) before deciding that matters.

    python stitch_components.py <out_dir>

Rotation convention (derived by find_euler_convention.py, residual 0.05 deg over
1261 shared cameras): R = Rz(-yaw) Rx(pitch) Ry(roll), intrinsic, and a frame
rotation A acts as R_new = A R_old.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

BASE = Path(r"K:\realityscan\out\1-mid-2")
BRIDGE_CSV = BASE / "_bridges" / "bridge_0440-0664_cameras.csv"
KEEP_CSV = BASE / "_merged_bridge" / "cameras.csv"           # component 1
KEEP_PLY = BASE / "_merged_bridge" / "merged_bridge_sparse.ply"
MOVE_CSV = BASE / "_merged_bridge" / "Component_2.csv"       # component 2
MOVE_PLY = BASE / "_merged_bridge" / "Component_2_sparse.ply"
KEEP_RANGE = (505, 664)      # indices shared between the bridge and component 1
MOVE_RANGE = (440, 504)      # indices shared between the bridge and component 2

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")
SEQ = "ZXY"
FIELDS = ["#name", "x", "y", "alt", "yaw", "pitch", "roll", "f_35mm",
          "px_norm", "py_norm", "k1", "k2", "k3", "k4", "t1", "t2"]


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def positions(rows: list[dict]) -> dict[str, np.ndarray]:
    return {r["#name"]: np.array([float(r["x"]), float(r["y"]), float(r["alt"])])
            for r in rows}


def to_matrix(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    return Rotation.from_euler(
        SEQ, np.column_stack([-yaw, pitch, roll]), degrees=True).as_matrix()


def to_angles(m: np.ndarray) -> np.ndarray:
    e = Rotation.from_matrix(m).as_euler(SEQ, degrees=True)
    e[:, 0] *= -1
    return e                                        # yaw, pitch, roll


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


def fit(bridge: dict, other: dict, lo: int, hi: int, label: str):
    common = [k for k in bridge if k in other
              and (m := PAT.match(k)) and lo <= int(m.group(2)) <= hi]
    src = np.array([bridge[k] for k in common])
    dst = np.array([other[k] for k in common])
    s, r, t = umeyama(src, dst)
    res = np.linalg.norm((s * r @ src.T).T + t - dst, axis=1)
    span = float(np.linalg.norm(dst.max(0) - dst.min(0)))
    print(f"  {label}: {len(common)} cameras, scale {s:.6f}, "
          f"residual median {np.median(res) / span * 100:.3f}% of the span")
    return s, r, t


def transform_ply(src: Path, dst: Path, s: float, a: np.ndarray, b: np.ndarray):
    ply = PlyData.read(str(src))
    el = ply["vertex"]
    data = el.data.copy()
    p = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    q = (s * a @ p.T).T + b
    data["x"], data["y"], data["z"] = q[:, 0], q[:, 1], q[:, 2]
    if all(n in data.dtype.names for n in ("nx", "ny", "nz")):
        n = np.column_stack([data["nx"], data["ny"], data["nz"]]).astype(np.float64)
        n = (a @ n.T).T
        data["nx"], data["ny"], data["nz"] = n[:, 0], n[:, 1], n[:, 2]
    PlyData([PlyElement.describe(data, "vertex")],
            text=False).write(str(dst))
    return len(data)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else BASE / "_stitched")
    out.mkdir(parents=True, exist_ok=True)

    bridge = positions(read_csv(BRIDGE_CSV))
    keep_rows, move_rows = read_csv(KEEP_CSV), read_csv(MOVE_CSV)
    keep, move = positions(keep_rows), positions(move_rows)
    print(f"component 1 {len(keep)} cams, component 2 {len(move)} cams, "
          f"bridge {len(bridge)} cams")

    s1, r1, t1 = fit(bridge, keep, *KEEP_RANGE, "bridge -> component 1")
    s2, r2, t2 = fit(bridge, move, *MOVE_RANGE, "bridge -> component 2")

    # component 2 -> bridge -> component 1, composed into one similarity
    s = s1 / s2
    a = r1 @ r2.T
    b = t1 - s * a @ t2
    print(f"  composed component 2 -> component 1: scale {s:.6f}")

    ang = np.array([[float(r["yaw"]), float(r["pitch"]), float(r["roll"])]
                    for r in move_rows])
    pos = np.array([[float(r["x"]), float(r["y"]), float(r["alt"])]
                    for r in move_rows])
    new_pos = (s * a @ pos.T).T + b
    new_ang = to_angles(a @ to_matrix(ang[:, 0], ang[:, 1], ang[:, 2]))

    # round-trip check: the parametrisation must reproduce itself exactly
    back = to_angles(to_matrix(ang[:, 0], ang[:, 1], ang[:, 2]))
    err = np.abs((back - ang + 180) % 360 - 180).max()
    print(f"  euler round-trip error {err:.6f} deg")

    dst = out / "cameras.csv"
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in keep_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        for i, r in enumerate(move_rows):
            row = {k: r.get(k, "") for k in FIELDS}
            row["x"], row["y"], row["alt"] = (f"{v:.12g}" for v in new_pos[i])
            row["yaw"], row["pitch"], row["roll"] = (f"{v:.12g}" for v in new_ang[i])
            w.writerow(row)
    print(f"\n{dst}: {len(keep_rows) + len(move_rows)} cameras")

    if KEEP_PLY.exists() and MOVE_PLY.exists():
        moved = out / "_moved.ply"
        n2 = transform_ply(MOVE_PLY, moved, s, a, b)
        p1 = PlyData.read(str(KEEP_PLY))["vertex"].data
        p2 = PlyData.read(str(moved))["vertex"].data
        if p1.dtype == p2.dtype:
            both = np.concatenate([p1, p2])
            PlyData([PlyElement.describe(both, "vertex")],
                    text=False).write(str(out / "stitched_sparse.ply"))
            moved.unlink()
            print(f"{out / 'stitched_sparse.ply'}: {len(both)} points "
                  f"({len(p1)} + {n2})")
        else:
            print(f"point formats differ ({p1.dtype} vs {p2.dtype}); "
                  f"left the transformed half at {moved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
