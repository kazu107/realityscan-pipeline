"""Per-view index coverage of an exported camera list.

    python diag_coverage.py <cameras.csv> [...]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

PAT = re.compile(r"^(.*)_(\d+)_(\d+)\.jpg$")


def main() -> int:
    for arg in sys.argv[1:]:
        p = Path(arg)
        by_view: dict[int, list[int]] = defaultdict(list)
        n = 0
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                n += 1
                m = PAT.match(r["#name"])
                if m:
                    by_view[int(m.group(3))].append(int(m.group(2)))
        print(f"{p}: {n} cams")
        for v in sorted(by_view):
            i = sorted(by_view[v])
            have = set(i)
            miss = [x for x in range(i[0], i[-1] + 1) if x not in have]
            shown = str(miss) if len(miss) < 12 else f"{len(miss)} indices"
            print(f"  view {v:02d}: {len(i):>5} imgs  {i[0]}..{i[-1]}  "
                  f"missing {shown}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
