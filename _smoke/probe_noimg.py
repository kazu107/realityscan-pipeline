"""Check whether calexHasImageExport=false suppresses undistorted-image output."""
import shutil
import subprocess
from pathlib import Path

RS = r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
W = Path(r"K:\realityscan\_smoke")
PROJ = W / "smoke.rsproj"
ROOT = W / "noimg"

FORMATS = {
    "opencv_csv": ("{B5331837-609D-4B12-A931-2863653d19F7}", "cameras.csv"),
    "colmap_txt": ("{280B11A4-F9A3-47D1-AE58-C0DEA33487D8}", "colmap.txt"),
    "rf_json": ("{314B5F22-C39F-4050-AE19-2236584B6932}", "transforms.json"),
}


def fresh(name):
    d = ROOT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


for label, (guid, fname) in FORMATS.items():
    d = fresh(label)
    args = [RS, "-headless", "-stdConsole",
            "-silent", str(W / "crash"),
            "-set", "appQuitOnError=true",
            "-set", "appAutoSaveMode=false",
            "-set", f"calexFileFormatId={guid}",
            "-set", "calexHasImageExport=false",
            "-load", str(PROJ),
            "-selectMaximalComponent",
            "-exportRegistration", str(d / fname),
            "-quit"]
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    files = sorted(x.relative_to(d).as_posix() for x in d.rglob("*") if x.is_file())
    total = sum(x.stat().st_size for x in d.rglob("*") if x.is_file())
    print(f"--- {label}: exit={rc} files={len(files)} bytes={total}")
    for f in files[:8]:
        print("      " + f)
