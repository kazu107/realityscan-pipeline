"""Does extracting the perspective views larger actually capture more?

The equirectangular source carries 7680/360 = 21.3 px per degree at the equator,
while a 2133 px view at 100 deg samples its own centre at only 15.6 - so in
theory a bigger extraction should recover something there. Theory is not enough:
the fisheye optics and 200 Mbps of HEVC may already have taken that detail away.

So extract the same frame at several sizes and ask what each step added. A step
that adds nothing beyond what upscaling the smaller version predicts is a step
that recovers no detail. The smaller sizes act as the control: whatever
1556 -> 2133 gained is the scale against which 2133 -> 3200 has to be judged.

    python test_extract_res.py <video> <timestamp> <out dir> [yaw]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SIZES = [1067, 1556, 2133, 2900, 3200]
FOV = 100


def extract(video: str, ts: str, out: Path, size: int, yaw: int) -> Path:
    dst = out / f"view_{size:04d}.png"
    if dst.exists():
        return dst
    vf = (f"v360=e:rectilinear:h_fov={FOV}:v_fov={FOV}:"
          f"yaw={yaw}:pitch=0:w={size}:h={size}:interp=lanczos")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", ts, "-i", video,
           "-frames:v", "1", "-vf", vf, str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def hf_energy(a: np.ndarray, lo: float = 0.25) -> float:
    """Energy above lo x Nyquist."""
    a = a - a.mean()
    w = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    f = np.fft.fftshift(np.abs(np.fft.fft2(a * w)) ** 2)
    cy, cx = np.array(f.shape) // 2
    y, x = np.indices(f.shape)
    r = np.hypot(y - cy, x - cx) / (min(f.shape) / 2)
    return float(f[r > lo].sum())


def centre(a: np.ndarray, n: int) -> np.ndarray:
    h, w = a.shape
    return a[h // 2 - n // 2: h // 2 + n // 2, w // 2 - n // 2: w // 2 + n // 2]


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    video, ts, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    yaw = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    out.mkdir(parents=True, exist_ok=True)

    paths = {s: extract(video, ts, out, s, yaw) for s in SIZES}
    for s, p in paths.items():
        print(f"  {s:>5} px -> {p.name} ({p.stat().st_size / 1024:.0f} KB)")

    print(f"\nwhat each step in extraction size adds, over the middle "
          f"third of the view")
    print(f"{'step':<18}{'residual RMS':>14}{'as % of the':>14}"
          f"{'HF energy':>12}")
    print(f"{'':<18}{'(0-255)':>14}{'image RMS':>14}{'recovered':>12}")
    prev = None
    for s in SIZES:
        a = np.asarray(Image.open(paths[s]).convert("L"), dtype=np.float64)
        if prev is not None:
            small, big = prev, (s, a)
            up = np.asarray(
                Image.open(paths[small[0]]).convert("L")
                .resize((s, s), Image.LANCZOS), dtype=np.float64)
            n = s // 3
            res = centre(big[1], n) - centre(up, n)
            rms = float(np.sqrt((res ** 2).mean()))
            ref = float(centre(big[1], n).std())
            hf_big = hf_energy(centre(big[1], n))
            hf_up = hf_energy(centre(up, n))
            gain = (hf_big - hf_up) / hf_big * 100 if hf_big else 0.0
            print(f"{small[0]:>5} -> {s:<9}{rms:>14.2f}{rms / ref * 100:>13.1f}%"
                  f"{gain:>11.1f}%")
        prev = (s, a)
    print("\nA step whose residual is near zero recovered nothing that "
          "upscaling could not predict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
