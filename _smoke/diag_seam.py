"""How well do two neighbouring set components actually agree?

Each component lives in its own similarity frame, so the only meaningful
comparison is: fit the best similarity transform on the cameras they share
(the overlap indices) and look at the residual. A small residual means the
two components are consistent and any kink in the merged result was
introduced by the merge; a large one means the chained hand-off itself
produced disagreeing geometry.
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"K:\realityscan\out\1-high")
PREFIX = "1-high"


def load(csv: Path) -> dict[str, tuple[float, float, float]]:
    out = {}
    for line in csv.open(encoding="utf-8", errors="replace"):
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split(",")
        out[c[0]] = (float(c[1]), float(c[2]), float(c[3]))
    return out


def umeyama(src: list, dst: list):
    """Least-squares similarity (scale, R, t) mapping src onto dst."""
    n = len(src)
    cs = [statistics.fmean(p[k] for p in src) for k in range(3)]
    cd = [statistics.fmean(p[k] for p in dst) for k in range(3)]
    X = [[p[k] - cs[k] for k in range(3)] for p in src]
    Y = [[p[k] - cd[k] for k in range(3)] for p in dst]
    # 3x3 covariance
    C = [[sum(Y[i][r] * X[i][c] for i in range(n)) / n for c in range(3)]
         for r in range(3)]
    U, S, Vt = svd3(C)
    d = 1.0 if det3(matmul(U, Vt)) > 0 else -1.0
    D = [[1, 0, 0], [0, 1, 0], [0, 0, d]]
    R = matmul(matmul(U, D), Vt)
    var = sum(sum(v * v for v in X[i]) for i in range(n)) / n
    s = (S[0] + S[1] + d * S[2]) / var if var else 1.0
    t = [cd[k] - s * sum(R[k][j] * cs[j] for j in range(3)) for k in range(3)]
    return s, R, t


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def det3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def svd3(A):
    """Jacobi SVD good enough for a 3x3 covariance."""
    import copy
    V = [[float(i == j) for j in range(3)] for i in range(3)]
    B = copy.deepcopy(A)
    for _ in range(60):
        off = sum(B[i][j] ** 2 for i in range(3) for j in range(3) if i != j)
        if off < 1e-24:
            break
        for p in range(2):
            for q in range(p + 1, 3):
                app = sum(B[i][p] ** 2 for i in range(3))
                aqq = sum(B[i][q] ** 2 for i in range(3))
                apq = sum(B[i][p] * B[i][q] for i in range(3))
                if abs(apq) < 1e-18:
                    continue
                theta = 0.5 * math.atan2(2 * apq, app - aqq)
                c, s = math.cos(theta), math.sin(theta)
                for i in range(3):
                    bp, bq = B[i][p], B[i][q]
                    B[i][p], B[i][q] = c * bp + s * bq, -s * bp + c * bq
                    vp, vq = V[i][p], V[i][q]
                    V[i][p], V[i][q] = c * vp + s * vq, -s * vp + c * vq
    S = [math.sqrt(sum(B[i][j] ** 2 for i in range(3))) for j in range(3)]
    U = [[(B[i][j] / S[j] if S[j] > 1e-15 else float(i == j)) for j in range(3)]
         for i in range(3)]
    order = sorted(range(3), key=lambda j: -S[j])
    U = [[U[i][j] for j in order] for i in range(3)]
    V = [[V[i][j] for j in order] for i in range(3)]
    S = [S[j] for j in order]
    Vt = [[V[j][i] for j in range(3)] for i in range(3)]
    return U, S, Vt


def seam(a_name: str, b_name: str) -> None:
    A = load(ROOT / a_name / "cameras.csv")
    B = load(ROOT / b_name / "cameras.csv")
    shared = sorted(set(A) & set(B))
    if len(shared) < 4:
        print(f"{a_name} | {b_name}: only {len(shared)} shared cameras")
        return
    src = [A[k] for k in shared]
    dst = [B[k] for k in shared]
    s, R, t = umeyama(src, dst)
    res = []
    for p, q in zip(src, dst):
        m = [s * sum(R[k][j] * p[j] for j in range(3)) + t[k] for k in range(3)]
        res.append(math.dist(m, q))
    # median step inside B, to express the residual in "how far the camera moves
    # between two indices" units
    idx = defaultdict(list)
    for k, v in B.items():
        mm = re.match(rf"{re.escape(PREFIX)}_(\d+)_", k)
        if mm:
            idx[int(mm.group(1))].append(v)
    ctr = {i: tuple(statistics.fmean(p[k] for p in v) for k in range(3))
           for i, v in idx.items()}
    ks = sorted(ctr)
    step = statistics.median([math.dist(ctr[a], ctr[b]) for a, b in zip(ks, ks[1:])])
    print(f"{a_name} -> {b_name}: {len(shared):3d} shared cameras, "
          f"scale {s:.4f}, residual median {statistics.median(res):.4f} "
          f"max {max(res):.4f}  ({statistics.median(res)/step*100:5.1f}% / "
          f"{max(res)/step*100:5.1f}% of one index step)")


names = sorted(p.name for p in ROOT.iterdir()
               if p.is_dir() and not p.name.startswith("_"))
print(f"{len(names)} sets\n")
for a, b in zip(names, names[1:]):
    seam(a, b)
