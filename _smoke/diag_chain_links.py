"""Check every chunk-to-chunk handoff for geometric agreement.

Set N imports set N-1's component and aligns new images onto it, so both live in
the SAME coordinate frame - the shared overlap cameras should have identical
coordinates, and the pose lock is supposed to guarantee it. Comparing raw
positions is therefore the right test.

A similarity fit is not: the 40 shared cameras sit at only 5 rig positions (all
views of one index share a centre) along a near-straight path, which is rank-1
and leaves the scale free.

Distances are reported in units of the local step between consecutive indices,
so "3.2" means the two components disagree by three times the spacing the camera
actually moved - a break the merge cannot resolve from overlaps.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"K:\realityscan\out\1-mid-2")
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


def step_of(cams: dict[str, np.ndarray]) -> float:
    """Median distance between consecutive index centres."""
    centres: dict[int, list[np.ndarray]] = {}
    for n, p in cams.items():
        m = PAT.match(n)
        if m:
            centres.setdefault(int(m.group(2)), []).append(p)
    keys = sorted(centres)
    c = {k: np.mean(centres[k], axis=0) for k in keys}
    d = [float(np.linalg.norm(c[b] - c[a])) for a, b in zip(keys, keys[1:])]
    return float(np.median(d)) if d else 1.0


def main() -> int:
    sets = sorted(p for p in OUT.iterdir()
                  if p.is_dir() and not p.name.startswith("_")
                  and (p / "cameras.csv").exists())
    cams = {p.name: load(p / "cameras.csv") for p in sets}
    print(f"{len(sets)} sets with cameras.csv in {OUT}")
    print("displacement of the shared overlap cameras, in units of the "
          "local index step\n")
    print(f"{'boundary':<25} {'shared':>6} {'median':>9} {'max':>9}")

    rows = []
    for a, b in zip(sets, sets[1:]):
        na, nb = a.name, b.name
        common = sorted(set(cams[na]) & set(cams[nb]))
        if not common:
            print(f"{na[-9:]} -> {nb[-9:]:<9} {0:>6}   NO SHARED CAMERAS")
            rows.append((float("inf"), na, nb, 0))
            continue
        d = np.array([np.linalg.norm(cams[na][k] - cams[nb][k]) for k in common])
        s = step_of(cams[nb]) or 1.0
        med, mx = float(np.median(d)) / s, float(d.max()) / s
        flag = "  <<<" if mx > 0.5 else ""
        print(f"{na[-9:]} -> {nb[-9:]:<9} {len(common):>6} {med:>9.3f} "
              f"{mx:>9.3f}{flag}")
        rows.append((mx, na, nb, len(common)))

    rows.sort(reverse=True)
    print("\nweakest handoffs:")
    for mx, na, nb, n in rows[:10]:
        print(f"  {na} -> {nb}: shared={n} max={mx:.3f} steps")
    ok = sum(1 for mx, *_ in rows if mx <= 0.5)
    print(f"\n{ok}/{len(rows)} handoffs agree to within half an index step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
