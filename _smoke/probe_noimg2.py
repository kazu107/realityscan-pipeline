"""Try candidate keys for turning OFF undistorted-image export (OpenCV CSV format)."""
import shutil
import subprocess
from pathlib import Path

RS = r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
W = Path(r"K:\realityscan\_smoke")
PROJ = W / "smoke.rsproj"
ROOT = W / "noimg2"
OPENCV = "{B5331837-609D-4B12-A931-2863653d19F7}"

CANDIDATES = [
    ("baseline", []),
    ("undistFormat_empty", [("calexUndistortImageFormat", "")]),
    ("undistFormat_none", [("calexUndistortImageFormat", "none")]),
    ("exportImages_false", [("calexExportImages", "false")]),
    ("undistortImages_false", [("calexUndistortImages", "false")]),
    ("hasImageExport0", [("calexHasImageExport", "0")]),
]


def fresh(name):
    d = ROOT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


for label, sets in CANDIDATES:
    d = fresh(label)
    args = [RS, "-headless", "-stdConsole",
            "-silent", str(W / "crash"),
            "-set", "appQuitOnError=true",
            "-set", "appAutoSaveMode=false",
            "-set", f"calexFileFormatId={OPENCV}"]
    for k, v in sets:
        args += ["-set", f"{k}={v}"]
    args += ["-load", str(PROJ), "-selectMaximalComponent",
             "-exportRegistration", str(d / "cameras.csv"), "-quit"]
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    files = [x for x in d.rglob("*") if x.is_file()]
    total = sum(x.stat().st_size for x in files)
    print(f"{label:24s} exit={rc} files={len(files):3d} bytes={total}")
