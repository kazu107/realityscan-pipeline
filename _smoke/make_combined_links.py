"""Present two half-coverage captures of the same walk as one 8-view capture.

1-mid-2-1 holds the forward views and 1-mid-2-2 the rearward ones, both
numbered 00..03 over the same indices - they are the same camera positions,
split only because the sideways footage was unusable. Aligned separately each
sees only a narrow arc around the direction of travel, which is the weak case
for triangulation and lets scale drift. Renumbering the second set to 04..07
and hard-linking both into one folder gives back a wide angular spread per
position at no storage cost.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

FRONT = Path(r"K:\data\jpeg\1-mid-2-1")
REAR = Path(r"K:\data\jpeg\1-mid-2-2")
DST = Path(r"K:\data\jpeg\1-mid-2-all")
OLD = "1-mid-2-repaired"
NEW = "1-mid-2all"
REAR_VIEW_OFFSET = 4

if len({FRONT.resolve().drive.lower(), REAR.resolve().drive.lower(),
        DST.resolve().drive.lower()}) != 1:
    raise SystemExit("hard links need every path on the same volume")

DST.mkdir(parents=True, exist_ok=True)
PAT = re.compile(rf"^{re.escape(OLD)}_(\d+)_(\d+)(\.jpg)(\.mask\.png)?$")

made = skipped = 0
for src, offset in ((FRONT, 0), (REAR, REAR_VIEW_OFFSET)):
    for p in sorted(src.iterdir()):
        m = PAT.match(p.name)
        if not m:
            continue
        idx, view, ext, mask = m.groups()
        target = DST / f"{NEW}_{idx}_{int(view) + offset:02d}{ext}{mask or ''}"
        if target.exists():
            skipped += 1
            continue
        os.link(p, target)
        made += 1

jpg = sorted(DST.glob("*.jpg"))
png = list(DST.glob("*.mask.png"))
views = sorted({int(PAT.sub(r"\2", p.name)) if False else
                int(p.stem.rsplit("_", 1)[1]) for p in jpg})
idxs = sorted({int(p.stem.rsplit("_", 2)[1]) for p in jpg})
print(f"{FRONT.name} + {REAR.name} -> {DST}")
print(f"  links created {made}, already present {skipped}")
print(f"  {len(jpg)} jpg, {len(png)} masks")
print(f"  indices {idxs[0]}-{idxs[-1]} ({len(idxs)}), views {views}")
counts = {}
for p in jpg:
    counts[int(p.stem.rsplit('_', 2)[1])] = counts.get(int(p.stem.rsplit('_', 2)[1]), 0) + 1
bad = {i: c for i, c in counts.items() if c != len(views)}
print(f"  indices without all {len(views)} views: {len(bad)}"
      + (f" e.g. {list(bad.items())[:5]}" if bad else ""))
