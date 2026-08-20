"""Is there real detail left at the current extraction resolution?

Extracting perspective views from an equirectangular frame resamples it, and
the rate is not uniform: a rectilinear image samples x = f*tan(theta), so the
middle of a view is coarser than the source and the edges are finer. Whether
raising the extraction resolution would recover anything depends on whether the
pixels that are there still carry detail up to their own Nyquist limit, or
whether video compression and the lens already ran out first.

The radially averaged power spectrum answers it. Energy reaching close to
Nyquist means the sampling is the limit and more pixels would help; energy
dying at half of it means the image is already soft and more pixels would only
interpolate.

Centre and edge crops are measured separately, because the theory says the edge
is oversampled by nearly two and should therefore look softer relative to its
own Nyquist.

    python diag_sharpness.py <image> [...] [--crop N]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def spectrum(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = a - a.mean()
    w = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    f = np.fft.fftshift(np.abs(np.fft.fft2(a * w)) ** 2)
    cy, cx = np.array(f.shape) // 2
    y, x = np.indices(f.shape)
    r = np.hypot(y - cy, x - cx).astype(int)
    n = np.bincount(r.ravel())
    s = np.bincount(r.ravel(), f.ravel()) / np.maximum(n, 1)
    k = np.arange(len(s)) / (min(f.shape) / 2)        # 1.0 = Nyquist
    keep = k <= 1.0
    return k[keep], s[keep]


def cutoff(k: np.ndarray, s: np.ndarray, frac: float) -> float:
    """Where the (normalised, cumulative-from-DC) energy passes frac of total."""
    e = np.cumsum(s * k)          # weight by annulus circumference
    e /= e[-1]
    return float(np.interp(frac, e, k))


def main() -> int:
    argv = list(sys.argv[1:])
    crop = 768
    if "--crop" in argv:
        i = argv.index("--crop")
        crop = int(argv[i + 1])
        del argv[i:i + 2]

    print(f"{'image':<34}{'region':<8}{'k(50%)':>8}{'k(90%)':>8}"
          f"{'k(99%)':>8}")
    for f in argv:
        p = Path(f)
        im = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
        h, w = im.shape
        regions = {
            "centre": im[h // 2 - crop // 2: h // 2 + crop // 2,
                         w // 2 - crop // 2: w // 2 + crop // 2],
            "edge": im[h // 2 - crop // 2: h // 2 + crop // 2, :crop],
        }
        for name, a in regions.items():
            k, s = spectrum(a)
            print(f"{p.name:<34}{name:<8}"
                  f"{cutoff(k, s, 0.50):>8.3f}{cutoff(k, s, 0.90):>8.3f}"
                  f"{cutoff(k, s, 0.99):>8.3f}")
    print("\nk is spatial frequency, 1.0 = Nyquist of the crop.")
    print("Detail that stops well below 1.0 cannot be recovered by extracting "
          "more pixels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
