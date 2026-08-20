"""Mask out the 360 camera's stitch seam in the views that sit on it.

A dual-fisheye 360 camera joins its two lenses at +-90 deg from the front, which
lands in the middle of views 02 and 06. When the two lenses are not exposed at
the same instant, shake makes content there appear twice. The doubled pixels
still yield features, and some of the resulting false matches survive the
reprojection threshold, so they end up as noise in the reconstruction.

The seam's angular position is fixed even though its severity varies, so a
central vertical band in those two views covers it. This writes a complete mask
set: a composite (existing mask AND NOT seam band) for the seam views, and a
plain link to the existing mask for every other view.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(r"K:\data\jpeg\1-mid-2")
DST = Path(r"K:\data\jpeg\1-mid-2-masks")
PREFIX = "1-mid-2"
SEAM_VIEWS = {2, 6}
BAND = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30   # fraction of the width
PAT = re.compile(rf"^{re.escape(PREFIX)}_(\d+)_(\d+)\.jpg$")

DST.mkdir(parents=True, exist_ok=True)
band_mask: np.ndarray | None = None
made = linked = 0

for p in sorted(SRC.glob(f"{PREFIX}_*.jpg")):
    m = PAT.match(p.name)
    if not m:
        continue
    idx, view = m.group(1), int(m.group(2))
    existing = p.with_name(p.name + ".mask.png")
    target = DST / (p.name + ".mask.png")
    if target.exists():
        continue

    if view not in SEAM_VIEWS:
        if existing.exists():
            os.link(existing, target)
            linked += 1
        continue

    if band_mask is None:
        with Image.open(p) as im:
            w, h = im.size
        band_mask = np.full((h, w), 255, dtype=np.uint8)
        half = int(w * BAND / 2)
        band_mask[:, w // 2 - half: w // 2 + half] = 0

    out = band_mask
    if existing.exists():
        with Image.open(existing) as im:
            prev = np.asarray(im.convert("L"), dtype=np.uint8)
        if prev.shape == band_mask.shape:
            out = np.minimum(prev, band_mask)          # black wins: both excluded
    Image.fromarray(out, mode="L").save(target, optimize=True)
    made += 1

n = len(list(DST.glob("*.mask.png")))
print(f"{SRC} -> {DST}")
print(f"  seam masks written {made} (band {BAND*100:.0f}% of the width, "
      f"views {sorted(SEAM_VIEWS)})")
print(f"  existing masks linked {linked}")
print(f"  {n} mask files in total")
