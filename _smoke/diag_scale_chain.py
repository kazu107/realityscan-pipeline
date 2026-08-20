"""Build a reference trajectory scale from the per-set alignments alone.

Each set is an independent reconstruction in its own arbitrary frame, but two
consecutive sets share their overlap indices, and the ratio of that shared
stretch's length in the two frames is the relative scale between them. Chaining
those ratios gives every set's scale relative to the first, and with it the step
size the merged result *should* show at each index.

That reference owes nothing to -mergeComponents, so comparing a merged camera
list against it shows where the merge stretched or collapsed the path.

    python diag_scale_chain.py <set dir> [merged.csv ...] [--prefix P]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")
SPAN = re.compile(r"_(\d+)-(\d+)$")


def centres(p: Path, prefix: str) -> dict[int, np.ndarray]:
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            m = PAT.match(r["#name"])
            if not m or (prefix and m.group(1) != prefix):
                continue
            acc[int(m.group(2))].append(
                np.array([float(r["x"]), float(r["y"]), float(r["alt"])]))
    return {i: np.mean(v, axis=0) for i, v in acc.items()}


def path_len(c: dict[int, np.ndarray], idx: list[int]) -> float:
    idx = [i for i in idx if i in c]
    return float(sum(np.linalg.norm(c[b] - c[a])
                     for a, b in zip(idx, idx[1:]) if b - a == 1))


def main() -> int:
    argv = list(sys.argv[1:])
    prefix, rest = "", []
    i = 0
    while i < len(argv):
        if argv[i] == "--prefix":
            prefix, i = argv[i + 1], i + 2
        else:
            rest.append(argv[i])
            i += 1
    root = Path(rest[0])
    merged_files = rest[1:]

    sets = sorted((d for d in root.iterdir()
                   if d.is_dir() and not d.name.startswith("_")
                   and (d / "cameras.csv").exists()),
                  key=lambda d: int(SPAN.search(d.name).group(1)))
    print(f"{len(sets)} sets in {root}")
    cams = {d.name: centres(d / "cameras.csv", prefix) for d in sets}

    scale = {sets[0].name: 1.0}
    weak = []
    for a, b in zip(sets, sets[1:]):
        ca, cb = cams[a.name], cams[b.name]
        shared = sorted(set(ca) & set(cb))
        la, lb = path_len(ca, shared), path_len(cb, shared)
        if la <= 0 or lb <= 0 or len(shared) < 3:
            scale[b.name] = scale[a.name]
            weak.append((a.name, b.name, len(shared)))
            continue
        scale[b.name] = scale[a.name] * (la / lb)
    if weak:
        print(f"  {len(weak)} handoffs had no usable shared stretch")

    # step size at each index, expressed in the first set's units
    ref: dict[int, float] = {}
    for d in sets:
        c, s = cams[d.name], scale[d.name]
        idx = sorted(c)
        for a, b in zip(idx, idx[1:]):
            if b - a == 1:
                ref[b] = float(np.linalg.norm(c[b] - c[a])) * s
    med = float(np.median(list(ref.values())))
    v = np.array([d / med for d in ref.values()])
    print(f"reference from the sets: {len(ref)} steps, "
          f"p50 {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} "
          f"p95 {np.percentile(v, 95):.2f} max {v.max():.2f}")

    bins = np.linspace(min(ref), max(ref) + 1, 9).astype(int)
    profs = [("reference", {i: d / med for i, d in ref.items()})]
    for f in merged_files:
        c = centres(Path(f), prefix)
        idx = sorted(c)
        st = {b: float(np.linalg.norm(c[b] - c[a]))
              for a, b in zip(idx, idx[1:]) if b - a == 1}
        m = float(np.median(list(st.values())))
        profs.append((Path(f).parent.name, {i: d / m for i, d in st.items()}))

    print("\nmedian normalised step per index range")
    print("  range".ljust(16) + "".join(f"{n:>16}" for n, _ in profs))
    for a, b in zip(bins, bins[1:]):
        cells = []
        for _, s in profs:
            vals = [d for i2, d in s.items() if a <= i2 < b]
            cells.append(f"{np.median(vals):16.2f}" if vals else f"{'-':>16}")
        print(f"  {a:>5}-{b - 1:<9}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
