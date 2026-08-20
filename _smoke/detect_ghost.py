"""Find the stitch-seam ghosting: where is the content doubled, and how badly?

A dual-fisheye 360 camera seams its two lenses at +-90 deg from the front, which
is exactly the middle of views 02 and 06. When the two lenses are not exposed at
the same instant, moving or shaken content lands twice at that seam, offset
horizontally - a doubled image.

Doubling shows up as a secondary peak in the horizontal self-similarity of the
seam band: the band correlates with itself shifted by the ghost offset. Views
that never contain a seam (00, 04) act as the control, so the measure is
calibrated on this very footage rather than on an arbitrary threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(r"K:\data\jpeg\1-mid-2")
PREFIX = "1-mid-2"
BAND_FRAC = 0.28          # width of the central strip, as a fraction of the image
SHIFTS = range(6, 80)     # ghost offsets worth looking for, in pixels
DOWNSCALE = 3             # plenty for a structure this large, and much faster


def score(path: Path) -> float:
    """Strength of the strongest horizontal self-echo in the central band."""
    im = Image.open(path).convert("L")
    w, h = im.size
    im = im.resize((w // DOWNSCALE, h // DOWNSCALE), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    H, W = a.shape
    bw = int(W * BAND_FRAC)
    band = a[H // 4: H * 7 // 8, (W - bw) // 2: (W + bw) // 2]

    # high-pass: keep edges, drop illumination
    k = 9
    pad = np.pad(band, ((0, 0), (k // 2, k // 2)), mode="edge")
    box = np.cumsum(pad, axis=1)
    box = (box[:, k:] - box[:, :-k]) / k
    hp = band[:, : box.shape[1]] - box
    hp -= hp.mean()
    denom = float(np.sqrt((hp * hp).sum()))
    if denom < 1e-6:
        return 0.0

    ncc = []
    for d in SHIFTS:
        if d >= hp.shape[1] - 8:
            break
        x, y = hp[:, d:], hp[:, :-d]
        n = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
        ncc.append(float((x * y).sum()) / n if n > 1e-6 else 0.0)
    if len(ncc) < 8:
        return 0.0
    v = np.array(ncc)
    # a clean image decays smoothly; a ghost adds a bump above that trend
    trend = np.poly1d(np.polyfit(np.arange(len(v)), v, 2))(np.arange(len(v)))
    return float((v - trend).max())


def main() -> int:
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    views = [0, 2, 4, 6]
    idxs = list(range(0, 1332, step))
    print(f"sampling {len(idxs)} indices (every {step}) x views {views}", flush=True)

    rows = {}
    for v in views:
        vals = []
        for i in idxs:
            p = SRC / f"{PREFIX}_{i:04d}_{v:02d}.jpg"
            vals.append(score(p) if p.exists() else float("nan"))
        rows[v] = np.array(vals, dtype=float)
        good = rows[v][~np.isnan(rows[v])]
        print(f"  view {v:02d}: median {np.median(good):.4f}  p90 {np.percentile(good,90):.4f}"
              f"  max {good.max():.4f}", flush=True)

    ctrl = np.concatenate([rows[0][~np.isnan(rows[0])], rows[4][~np.isnan(rows[4])]])
    thr = float(np.percentile(ctrl, 99))
    print(f"\ncontrol views (00, 04) 99th percentile = {thr:.4f}  -> threshold")

    print("\nindex   view02   view06   flagged")
    flagged = []
    for j, i in enumerate(idxs):
        s2, s6 = rows[2][j], rows[6][j]
        hit = [v for v, s in ((2, s2), (6, s6)) if s > thr]
        if hit:
            flagged.append(i)
        print(f"{i:5d}   {s2:6.4f}   {s6:6.4f}   {'yes ' + str(hit) if hit else ''}")

    print(f"\nflagged {len(flagged)}/{len(idxs)} sampled indices")
    if flagged:
        halves = [sum(1 for i in flagged if i < 666), sum(1 for i in flagged if i >= 666)]
        n = [sum(1 for i in idxs if i < 666), sum(1 for i in idxs if i >= 666)]
        print(f"  first half  (index <666): {halves[0]}/{n[0]}")
        print(f"  second half (index >=666): {halves[1]}/{n[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
