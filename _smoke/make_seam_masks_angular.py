"""Mask the 360 stitch seam wherever it falls, in every view that contains it.

The seam sits at fixed world directions (+-90 deg from the front). With 8 views
at 45 deg spacing and a 100 deg field of view, each seam lands in three views at
once: the middle of one, and near the edge of its two neighbours. Masking only
the middle one leaves the ghosting in the others, so the band is specified as an
angle around the seam and converted to pixels per view.

The pixel width is not proportional to the angle: x = f * tan(angle from the
view centre), so the same angular band covers far more pixels near a view's edge
than at its middle.

    python make_seam_masks_angular.py [half_angle_deg] [out_dir]
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(r"K:\data\jpeg\1-mid-2")
PREFIX = "1-mid-2"
HALF = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
DST = Path(sys.argv[2] if len(sys.argv) > 2
           else rf"K:\data\jpeg\1-mid-2-masks{int(HALF)}")

N_VIEWS = 8
FOV = 100.0                      # horizontal, measured from the aligned result
SEAMS = (90.0, 270.0)            # where the two lenses are joined
PAT = re.compile(rf"^{re.escape(PREFIX)}_(\d+)_(\d+)\.jpg$")


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def band_for_view(view: int, w: int) -> list[tuple[int, int]]:
    """Pixel columns of this view covered by the seam band."""
    f = (w / 2) / math.tan(math.radians(FOV / 2))
    half_fov = FOV / 2 - 0.5                      # keep off the very edge
    centre = view * (360.0 / N_VIEWS)
    out = []
    for seam in SEAMS:
        d = wrap(seam - centre)
        lo, hi = d - HALF, d + HALF
        if hi <= -half_fov or lo >= half_fov:
            continue
        lo, hi = max(lo, -half_fov), min(hi, half_fov)
        x0 = int(w / 2 + f * math.tan(math.radians(lo)))
        x1 = int(w / 2 + f * math.tan(math.radians(hi)))
        out.append((max(0, x0), min(w, x1 + 1)))
    return out


def main() -> int:
    first = next(SRC.glob(f"{PREFIX}_*_00.jpg"))
    with Image.open(first) as im:
        w, h = im.size

    bands = {v: band_for_view(v, w) for v in range(N_VIEWS)}
    print(f"seam half-angle {HALF} deg, image {w}x{h}, fov {FOV} deg")
    for v in range(N_VIEWS):
        if bands[v]:
            cov = sum(b - a for a, b in bands[v]) / w * 100
            print(f"  view {v:02d}: columns {bands[v]}  ({cov:.1f}% of the width)")
        else:
            print(f"  view {v:02d}: clear of the seam")

    templates = {}
    for v, spans in bands.items():
        if not spans:
            continue
        m = np.full((h, w), 255, dtype=np.uint8)
        for a, b in spans:
            m[:, a:b] = 0
        templates[v] = m

    DST.mkdir(parents=True, exist_ok=True)
    made = linked = 0
    for p in sorted(SRC.glob(f"{PREFIX}_*.jpg")):
        mt = PAT.match(p.name)
        if not mt:
            continue
        view = int(mt.group(2))
        existing = p.with_name(p.name + ".mask.png")
        target = DST / (p.name + ".mask.png")
        if target.exists():
            continue
        if view not in templates:
            if existing.exists():
                os.link(existing, target)
                linked += 1
            continue
        out = templates[view]
        if existing.exists():
            with Image.open(existing) as im:
                prev = np.asarray(im.convert("L"), dtype=np.uint8)
            if prev.shape == out.shape:
                out = np.minimum(prev, out)
        Image.fromarray(out).save(target, optimize=True)
        made += 1

    print(f"\n{DST}")
    print(f"  seam masks written {made}, existing masks linked {linked}, "
          f"total {len(list(DST.glob('*.mask.png')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
