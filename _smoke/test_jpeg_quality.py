"""How much does the JPEG step cost, and would PNG buy anything back?

The extraction resolution turned out to be at its limit already, so the next
place detail can be lost is the encode. This takes a losslessly extracted frame
as the reference, saves it at a range of JPEG qualities, and measures what each
one throws away - both as signal (high-frequency energy) and as something the
pipeline actually cares about: how many features a detector still finds, which
is what alignment runs on.

The quality the existing images were written at is estimated from their
quantisation tables, so the measurements can be read against where they sit.

    python test_jpeg_quality.py <reference.png> [sample.jpg ...]
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:                                     # noqa: BLE001
    cv2 = None

QUALITIES = [75, 80, 85, 90, 92, 95, 97, 100]
# the standard luma table, used to estimate an unknown file's quality
STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=float)


def estimate_quality(p: Path) -> float:
    """Invert libjpeg's table scaling.

    It builds the table as round((std * scale + 50) / 100) with
    scale = 200 - 2q for q >= 50 and 5000 / q below that, so recovering the
    median scale gives the quality back. Entries that saturated at 255 or
    bottomed out at 1 carry no information and are dropped.
    """
    q = np.array(Image.open(p).quantization[0], dtype=float)
    ok = (q > 1) & (q < 255)
    if not ok.any():
        return float("nan")
    scale = float(np.median((q[ok] * 100 - 50) / STD_LUMA[ok]))
    return (200 - scale) / 2 if scale <= 100 else 5000 / max(scale, 1e-9)


def hf_energy(a: np.ndarray, lo: float = 0.25) -> float:
    a = a - a.mean()
    w = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    f = np.fft.fftshift(np.abs(np.fft.fft2(a * w)) ** 2)
    cy, cx = np.array(f.shape) // 2
    y, x = np.indices(f.shape)
    r = np.hypot(y - cy, x - cx) / (min(f.shape) / 2)
    return float(f[r > lo].sum())


def features(a: np.ndarray) -> int:
    if cv2 is None:
        return -1
    sift = cv2.SIFT_create(nfeatures=0)
    return len(sift.detect(a.astype(np.uint8), None))


def main() -> int:
    ref_path = Path(sys.argv[1])
    ref_rgb = Image.open(ref_path).convert("RGB")
    ref = np.asarray(ref_rgb.convert("L"), dtype=np.float64)
    n = ref.shape[0] // 3
    c = slice(ref.shape[0] // 2 - n // 2, ref.shape[0] // 2 + n // 2)
    ref_c = ref[c, c]
    hf_ref = hf_energy(ref_c)
    f_ref = features(ref)
    png_bytes = ref_path.stat().st_size

    print(f"reference {ref_path.name}: {ref.shape[1]}x{ref.shape[0]}, "
          f"PNG {png_bytes / 1024:.0f} KB, {f_ref} features")
    print(f"\n{'quality':>8}{'KB':>9}{'vs PNG':>9}{'RMSE':>8}"
          f"{'HF kept':>10}{'features':>10}{'vs PNG':>9}")
    for q in QUALITIES:
        buf = io.BytesIO()
        ref_rgb.save(buf, "JPEG", quality=q, subsampling=0)
        size = buf.tell()
        a = np.asarray(Image.open(buf).convert("L"), dtype=np.float64)
        rmse = float(np.sqrt(((a - ref) ** 2).mean()))
        hf = hf_energy(a[c, c]) / hf_ref * 100
        nf = features(a)
        print(f"{q:>8}{size / 1024:>9.0f}{size / png_bytes * 100:>8.0f}%"
              f"{rmse:>8.2f}{hf:>9.1f}%{nf:>10}"
              f"{nf / f_ref * 100 if f_ref > 0 else 0:>8.0f}%")

    if len(sys.argv) > 2:
        print(f"\n{'existing file':<30}{'KB':>8}{'est. quality':>14}")
        for s in sys.argv[2:]:
            p = Path(s)
            print(f"{p.name:<30}{p.stat().st_size / 1024:>8.0f}"
                  f"{estimate_quality(p):>14.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
