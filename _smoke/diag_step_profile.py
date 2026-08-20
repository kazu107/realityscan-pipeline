"""Where along the trajectory does the reconstruction jump?

Walks the rig centres index by index and reports the step between consecutive
ones. A capture at constant speed gives an almost flat profile, so an outlier is
either a genuine discontinuity or an index whose views are mostly missing (the
centre of 2 cameras is not the centre of 8), which is why the view count is
carried along and thin indices can be excluded.

Steps are also bucketed by position within the set pattern. Sets start every
`step` indices and span `span`, so the first `overlap` indices of each set are
carried over from the previous one and the rest are new. If the handoff is where
error lands, the bucket at the overlap boundary stands out.

    python diag_step_profile.py <cameras.csv> [...] [--prefix P] [--min-views N]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")
CHUNK, OVERLAP = 25, 5            # index span per set, indices carried over
STEP = CHUNK - OVERLAP            # a new set starts every 20 indices


def centres(p: Path, prefix: str, min_views: int):
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            m = PAT.match(r["#name"])
            if not m or (prefix and m.group(1) != prefix):
                continue
            try:
                acc[int(m.group(2))].append(
                    np.array([float(r["x"]), float(r["y"]), float(r["alt"])]))
            except (KeyError, ValueError):
                continue
    return {i: np.mean(v, axis=0) for i, v in acc.items() if len(v) >= min_views}, \
           {i: len(v) for i, v in acc.items()}


def main() -> int:
    argv = sys.argv[1:]
    args: list[str] = []
    prefix = ""
    min_views = 8
    i = 0
    while i < len(argv):
        if argv[i] == "--prefix":
            prefix, i = argv[i + 1], i + 2
        elif argv[i] == "--min-views":
            min_views, i = int(argv[i + 1]), i + 2
        else:
            args.append(argv[i])
            i += 1

    for arg in args:
        p = Path(arg)
        c, counts = centres(p, prefix, min_views)
        keys = sorted(c)
        steps = {b: float(np.linalg.norm(c[b] - c[a]))
                 for a, b in zip(keys, keys[1:]) if b - a == 1}
        if not steps:
            print(f"{p}: no consecutive indices with >= {min_views} views")
            continue
        med = float(np.median(list(steps.values())))
        print(f"{p}")
        print(f"  {len(keys)} indices with >= {min_views} views, "
              f"{len(steps)} consecutive pairs, median step {med:.4f}")

        # position of the step's end index within the set pattern:
        # 0 .. OVERLAP-1 are carried over, OVERLAP is the first new index
        buckets: dict[int, list[float]] = defaultdict(list)
        for i, d in steps.items():
            buckets[i % STEP].append(d / med)
        print("  step size by position in the set pattern "
              f"(set every {STEP} indices, {OVERLAP} carried over):")
        for pos in sorted(buckets):
            v = np.array(buckets[pos])
            tag = "  <- first new index" if pos == OVERLAP else ""
            tag = "  <- set boundary" if pos == 0 else tag
            print(f"    index %% {STEP} == {pos:>2}: n={len(v):>4} "
                  f"median {np.median(v):5.2f}x  p95 {np.percentile(v, 95):6.2f}x "
                  f"max {v.max():7.2f}x{tag}")

        out = sorted(steps.items(), key=lambda kv: -kv[1])[:10]
        print("  largest steps:")
        for i, d in out:
            print(f"    {i - 1:>5} -> {i:<5} {d / med:7.2f}x   "
                  f"views {counts.get(i - 1)} -> {counts.get(i)}   "
                  f"index %% {STEP} = {i % STEP}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
