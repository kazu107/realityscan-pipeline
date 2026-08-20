"""Probe the pieces needed for masks, component hand-off and cross-set merging.

P1  importComponent -> add new images -> align   (does it stay one component?)
P2  importComponent -> disable old images -> add -> align -> export
    (does the exported component drop the disabled cameras?)
P3  two independently aligned overlapping sets -> mergeComponents
P4  per-image masks via an .rscmd file (selectImage + setImagesLayer),
    verified by exporting the masks the project actually uses
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

RS = r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
SRC = Path(r"K:\data\jpeg\1-mid-1")
W = Path(r"K:\realityscan\_smoke\p2")
STAT = re.compile(r"^\s*RSSTAT\|(.*)$")

REPORT = (
    "$IterateComponents(RSSTAT|name=$(componentName)|cams=$(componentCamerasCount)"
    "$ComponentInfo(componentGUID,|pts=$(componentPointsCount))"
    "$ComponentStats(componentGUID,|mean=$(componentMeanError:.4f))\n)"
)

PRIORS = [
    "-selectAllImages",
    "-editInputSelection", "inpCalibrationGroup=1",
    "-editInputSelection", "inpCalibration=1",
    "-editInputSelection", "inpFocal=15.103900",
    "-editInputSelection", "inpLensGroup=1",
    "-editInputSelection", "inpDistortion=2",
    "-editInputSelection", "inpDistortionModel=0",
    "-deselectAllImages",
]

BASE_SETS = [
    ("appQuitOnError", "true"),
    ("appAutoSaveMode", "false"),
    ("appIncSubdirs", "false"),
    ("sfmImageDownscaleFactor", "1"),
    ("sfmMaxFeaturesPerImage", "40000"),
    ("sfmMaxFeaturesPerMpx", "10000"),
    ("sfmImagesOverlap", "Medium"),
    ("sfmDetectorSensitivity", "Medium"),
    ("sfmFeatureDetectionQuality", "High"),
    ("sfmMaxFeatureReprojectionError", "2.0"),
    ("sfmPreselectorFeatures", "10000"),
    ("sfmDistortionModel", "Brown3"),
    ("sfmForceComponentRematch", "false"),
    ("sfmAutoReconRegionAfterAlignment", "false"),
]


def images(idx_from: int, idx_to: int) -> list[Path]:
    out = []
    for p in sorted(SRC.glob("*.jpg")):
        m = re.match(r"^(.+)_(\d+)_(\d+)$", p.stem)
        if m and idx_from <= int(m.group(2)) <= idx_to:
            out.append(p)
    return out


def imagelist(name: str, paths: list[Path]) -> Path:
    f = W / f"{name}.imagelist"
    f.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return f


def head(extra_sets=()) -> list[str]:
    args = [RS, "-headless", "-stdConsole", "-silent", str(W / "crash")]
    for k, v in list(BASE_SETS) + list(extra_sets):
        args += ["-set", f"{k}={v}"]
    return args


def run(label: str, args: list[str], timeout=1800):
    print(f"\n########## {label}")
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None
    (W / f"{label}.log").write_text(p.stdout or "", encoding="utf-8")
    for ln in (p.stdout or "").splitlines():
        if STAT.match(ln):
            print("  " + ln.strip())
        elif ("rror" in ln and "appQuitOnError" not in ln) or "failed" in ln.lower():
            print("  ! " + ln.strip())
        elif "Added" in ln or "Imported" in ln or "component" in ln.lower():
            print("  . " + ln.strip())
    print(f"  exit={p.returncode}")
    return p


def idx_regex(indices: list[int]) -> str:
    return "g/_(" + "|".join(f"{i:04d}" for i in indices) + ")_/"


# --------------------------------------------------------------------------
if W.exists():
    shutil.rmtree(W)
W.mkdir(parents=True)
(W / "crash").mkdir()

A = imagelist("A_0_4", images(0, 4))          # 40 images, idx 0-4
B_new = imagelist("B_5_9", images(5, 9))      # 40 images, idx 5-9
B_full = imagelist("B_3_9", images(3, 9))     # 56 images, idx 3-9 (overlap 3,4)

what = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---- base component A ----------------------------------------------------
if what in ("all", "p1", "p2", "p3"):
    run("00_alignA", head() + [
        "-newScene", "-add", str(A), *PRIORS,
        "-align", "-printReport", REPORT,
        "-selectMaximalComponent",
        "-exportSelectedComponentFile", str(W / "A.rsalign"),
        "-save", str(W / "A.rsproj"), "-quit"])

# ---- P1: import + add + align -------------------------------------------
if what in ("all", "p1"):
    run("10_import_add_align", head() + [
        "-newScene",
        "-importComponent", str(W / "A.rsalign"),
        "-printReport", REPORT,
        "-add", str(B_new), *PRIORS,
        "-align", "-printReport", REPORT,
        "-selectMaximalComponent",
        "-exportSelectedComponentFile", str(W / "P1.rsalign"),
        "-quit"])

# ---- P2: import + disable old + add + align -----------------------------
if what in ("all", "p2"):
    run("20_import_disable_add_align", head() + [
        "-newScene",
        "-importComponent", str(W / "A.rsalign"),
        "-deselectAllImages",
        "-selectImage", idx_regex([0, 1, 2]),
        "-editInputSelection", "inpEnabled=false",
        "-deselectAllImages",
        "-add", str(B_new), *PRIORS,
        "-align", "-printReport", REPORT,
        "-selectMaximalComponent",
        "-exportSelectedComponentFile", str(W / "P2.rsalign"),
        "-quit"])
    run("21_check_P2", head() + [
        "-newScene", "-importComponent", str(W / "P2.rsalign"),
        "-printReport", REPORT, "-quit"])

# ---- P3: independent sets + mergeComponents ------------------------------
if what in ("all", "p3"):
    run("30_alignB", head() + [
        "-newScene", "-add", str(B_full), *PRIORS,
        "-align", "-printReport", REPORT,
        "-selectMaximalComponent",
        "-exportSelectedComponentFile", str(W / "B.rsalign"), "-quit"])
    run("31_merge", head() + [
        "-newScene",
        "-importComponent", str(W / "A.rsalign"),
        "-importComponent", str(W / "B.rsalign"),
        "-printReport", REPORT,
        "-mergeComponents",
        "-printReport", REPORT,
        "-selectMaximalComponent",
        "-exportSelectedComponentFile", str(W / "AB_merged.rsalign"),
        "-quit"])

# ---- P4: per-image masks through an .rscmd -------------------------------
if what in ("all", "p4"):
    imgs = images(0, 4)
    lines = []
    n_masked = 0
    for p in imgs:
        mask = p.with_name(p.name + ".mask.png")
        if mask.exists():
            lines.append(f"-selectImage {p}")
            lines.append(f"-setImagesLayer {mask} mask")
            n_masked += 1
    lines.append("-selectAllImages")
    lines.append("-editInputSelection inpMaskOpts=1")
    lines.append("-deselectAllImages")
    rscmd = W / "masks.rscmd"
    rscmd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[masks] {n_masked}/{len(imgs)} images have a mask, rscmd has {len(lines)} lines")

    (W / "maskdump").mkdir(exist_ok=True)
    run("40_masks", head() + [
        "-newScene", "-add", str(A), *PRIORS,
        "-execRSCMD", str(rscmd),
        "-tag", "MASKS_SET",
        "-selectAllImages",
        "-exportMasks", str(W / "maskdump"),
        "-deselectAllImages",
        "-align", "-printReport", REPORT,
        "-quit"])
    dumped = list((W / "maskdump").glob("*"))
    print(f"  exported masks: {len(dumped)}")
