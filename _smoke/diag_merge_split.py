"""Find where a merged component broke apart.

The merge of 1-mid-2 produced two components instead of one and only the maximal
component is exported, so the second one has to be reconstructed by subtraction:
every source image that is not in the exported camera list is either in the
other component or unregistered. That is enough to see the shape of the split.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(r"K:\realityscan\out\1-mid-2")
SRC = Path(r"K:\data\jpeg\1-mid-2")
PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def load(p: Path) -> set[str]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return {r["#name"] for r in csv.DictReader(f)}


def runs(idx: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def main() -> int:
    got = load(OUT / "_merged" / "cameras.csv")
    all_names = {p.name for p in SRC.glob("1-mid-2_*.jpg")}
    missing = sorted(all_names - got)
    print(f"source images {len(all_names)}, in the exported component {len(got)}, "
          f"elsewhere {len(missing)}")

    by_view: dict[int, list[int]] = defaultdict(list)
    for n in missing:
        m = PAT.match(n)
        if m:
            by_view[int(m.group(3))].append(int(m.group(2)))

    print("\nimages outside the exported component, by view direction")
    for v in sorted(by_view):
        idx = sorted(by_view[v])
        r = runs(idx)
        shown = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in r[:6])
        more = f"  (+{len(r) - 6} more runs)" if len(r) > 6 else ""
        print(f"  view {v:02d}: {len(idx):>5} images  {shown}{more}")

    have_view: dict[int, list[int]] = defaultdict(list)
    for n in got:
        m = PAT.match(n)
        if m:
            have_view[int(m.group(3))].append(int(m.group(2)))
    print("\nimages inside the exported component, by view direction")
    for v in sorted(have_view):
        idx = sorted(have_view[v])
        print(f"  view {v:02d}: {len(idx):>5} images  {idx[0]}..{idx[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
