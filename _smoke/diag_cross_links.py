"""How well are two captures tied together in a merged database?

Counting verified crossing pairs is not enough. What decides whether they can
be reconstructed as one thing is where those pairs land:

*   coverage - a stretch of one walk with no crossing link is free to drift
    relative to the other, and no amount of links elsewhere fixes it.

*   coherence - if the crossing links are real, the frame each one points at
    should move smoothly along the other walk as the query frame advances.
    Two walks of the same route have a monotonic-ish correspondence between
    them, possibly offset or reversed. Links scattered uniformly across the
    other walk are retrieval noise that happened to survive verification.

Both are read off the database, so this can be run before spending hours on a
mapper that would have had nothing to work with.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict

import numpy as np

MAX = 2147483647
FRAME = re.compile(r"_(?P<frame>\d+)\.\w+$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("database")
    ap.add_argument("--prefix", default="low_",
                    help="folder prefix naming one of the two sets")
    ap.add_argument("--bin", type=int, default=25,
                    help="frames per bucket in the coverage report")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    side, frame = {}, {}
    for iid, name in db.execute("select image_id, name from images"):
        n = name.replace("\\", "/")
        side[iid] = n.startswith(args.prefix)
        if (mt := FRAME.search(n)):
            frame[iid] = int(mt.group("frame"))

    gen = ver = 0
    inl = 0
    links: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pid, rows in db.execute("select pair_id, rows from two_view_geometries"):
        a, b = pid // MAX, pid % MAX
        if a not in side or b not in side or side[a] == side[b]:
            continue
        gen += 1
        if rows <= 0 or a not in frame or b not in frame:
            continue
        ver += 1
        inl += rows
        q, h = (a, b) if not side[a] else (b, a)     # q is the non-prefix set
        links[frame[q]].append((frame[h], rows))
    db.close()

    print(f"crossing pairs: {gen} in the database, {ver} verified "
          f"({100 * ver / max(gen, 1):.1f}%), {inl:,} inliers")
    if not links:
        print("  nothing crosses - the two sets are separate components")
        return 1

    qf = sorted(links)
    print(f"  frames of the queried set with a verified crossing link: "
          f"{len(qf)} ({qf[0]}..{qf[-1]})")

    # ---- coverage ------------------------------------------------------
    lo, hi = qf[0], qf[-1]
    print(f"\n=== coverage, {args.bin} frames a bucket ===")
    print(f"  {'frames':>13} {'links':>7} {'inliers':>10}  partner frames")
    gaps = []
    for start in range(lo - lo % args.bin, hi + 1, args.bin):
        got = [(h, r) for f in range(start, start + args.bin)
               for h, r in links.get(f, [])]
        if not got:
            gaps.append(start)
            print(f"  {start:>5}-{start+args.bin-1:<7} {0:>7} {0:>10}   "
                  f"-- nothing crosses here --")
            continue
        hs = np.array([h for h, _ in got])
        print(f"  {start:>5}-{start+args.bin-1:<7} {len(got):>7} "
              f"{sum(r for _, r in got):>10}  "
              f"{hs.min()}..{hs.max()} (median {int(np.median(hs))})")
    print(f"  buckets with no crossing link: {len(gaps)}"
          + (f" -> {gaps[:10]}" if gaps else " (none)"))

    # ---- coherence -----------------------------------------------------
    # the partner frame should advance with the query frame; a real
    # correspondence correlates, noise does not
    x, y, w = [], [], []
    for f, hits in links.items():
        best = max(hits, key=lambda t: t[1])
        x.append(f)
        y.append(best[0])
        w.append(best[1])
    x, y, w = np.array(x), np.array(y), np.array(w)
    order = np.argsort(x)
    x, y, w = x[order], y[order], w[order]
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    print(f"\n=== coherence ===")
    print(f"  best-partner frame against query frame: correlation {r:+.3f}")
    d = np.diff(y)
    print(f"  step of the partner frame: median {np.median(d):+.1f}, "
          f"{100 * np.mean(np.abs(d) <= 20):.0f}% of steps within 20 frames")
    print("  " + ("a smooth correspondence - the two walks line up"
                  if abs(r) > 0.7 else
                  "NOT smooth - these links may be retrieval noise, or the "
                  "walks may cover the route in a different order"))
    print(f"\n  sample of the correspondence:")
    for i in range(0, len(x), max(1, len(x) // 12)):
        print(f"    query frame {x[i]:>4} -> partner {y[i]:>4} "
              f"({w[i]} inliers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
