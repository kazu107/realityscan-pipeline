"""Decisive mask test: black out ~96% of one image and watch its feature count.

-exportMasks needs (folder, params.xml) together, so it cannot be used as a
cheap read-back. Comparing detected feature counts is unambiguous instead.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

RS = r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
SRC = Path(r"K:\data\jpeg\1-mid-1")
W = Path(r"K:\realityscan\_smoke\p3")
# indices far from anything processed earlier, so the feature cache is cold
TARGET = "1-mid_0800_00.jpg"
FEAT = re.compile(r"Detected (\d+) features in image '([^']+)'")


def write_gray_png(path: Path, w: int, h: int, keep_box) -> None:
    """8-bit grayscale PNG: black everywhere except keep_box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = keep_box
    raw = bytearray()
    white_row = bytes([255]) * w
    black_row = bytes([0]) * w
    for y in range(h):
        raw.append(0)  # filter: none
        if y0 <= y < y1:
            row = bytearray(black_row)
            row[x0:x1] = bytes([255]) * (x1 - x0)
            raw += row
        else:
            raw += black_row
    del white_row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def run(label: str, extra: list[str]):
    args = [RS, "-headless", "-stdConsole", "-silent", str(W / "crash"),
            "-set", "appQuitOnError=true", "-set", "appAutoSaveMode=false",
            "-set", "sfmImageDownscaleFactor=1",
            "-set", "sfmMaxFeaturesPerImage=40000",
            "-set", "sfmMaxFeaturesPerMpx=10000",
            "-set", "sfmDetectorSensitivity=Medium",
            "-set", "sfmFeatureDetectionQuality=High",
            "-newScene", "-add", str(W / "small.imagelist")]
    args += extra
    # -detectFeatures on its own aborts here; -align reliably prints the
    # "Detected N features in image '...'" lines we want to compare.
    args += ["-align", "-quit"]
    p = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    (W / f"{label}.log").write_text(p.stdout or "", encoding="utf-8")
    counts = {m.group(2): int(m.group(1)) for m in FEAT.finditer(p.stdout or "")}
    bad = [ln.strip() for ln in (p.stdout or "").splitlines()
           if "failed" in ln.lower() or "Access denied" in ln]
    print(f"--- {label}: exit={p.returncode} images_with_features={len(counts)}")
    for ln in bad[:5]:
        print("    ! " + ln)
    return counts


if W.exists():
    shutil.rmtree(W)
W.mkdir(parents=True)
(W / "crash").mkdir()
(W / "masks").mkdir()

imgs = sorted(SRC.glob("1-mid_080[0-3]_0*.jpg"))
(W / "small.imagelist").write_text("\n".join(str(p) for p in imgs) + "\n", encoding="utf-8")
print(f"{len(imgs)} images, target = {TARGET}")

# keep only a 400x400 window -> ~3.5% of a 2133x2133 frame
mask = W / "masks" / f"{TARGET}.mask.png"
write_gray_png(mask, 2133, 2133, (900, 900, 1300, 1300))
print(f"mask written: {mask} ({mask.stat().st_size} B)")

before = run("00_nomask", [])

rscmd = W / "masks.rscmd"
target_path = SRC / TARGET
rscmd.write_text(
    f"-selectImage {target_path}\n"
    f"-setImagesLayer {mask} mask\n"
    f"-selectAllImages\n"
    f"-editInputSelection inpMaskOpts=1\n"
    f"-deselectAllImages\n",
    encoding="utf-8")
after = run("10_masked", ["-execRSCMD", str(rscmd)])

print("\nimage                    no-mask   masked   ratio")
for name in sorted(set(before) | set(after)):
    b, a = before.get(name, 0), after.get(name, 0)
    print(f"{name:24s} {b:8d} {a:8d}   {(a / b if b else 0):.3f}")
sys.exit(0)
