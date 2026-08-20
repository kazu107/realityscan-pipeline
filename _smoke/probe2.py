"""Probe RealityScan CLI via argv list (same path the GUI runner will use).

Checks:
  1. -printReport variants -> machine-readable per-component stats
  2. -exportRegistration driven by a params.xml that only carries the format GUID
"""
import subprocess
import sys
from pathlib import Path

RS = r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
W = Path(r"K:\realityscan\_smoke")
OUT = W / "probe_out"
P = W / "params"
OUT.mkdir(exist_ok=True)

BODY = (
    "RSSTAT|$(componentName)|cams=$(componentCamerasCount)|imgs=$(imageCount)"
    "$ComponentInfo(componentGUID,"
    "|pts=$(componentPointsCount)"
    "|mean=$(componentMeanError:.4f)"
    "|median=$(componentMedianError:.4f)"
    "|max=$(componentMaximalError:.4f)"
    "|track=$(componentAverageTrackLength:.2f)"
    ")\n"
)

VARIANTS = {
    "A_plain_iterate": "$IterateComponents(" + BODY + ")",
    "B_projectinfo": "$ExportProjectInfo($IterateComponents(SEL=$If(actualComponentGUID==componentGUID,1,0)|" + BODY + "))",
    "C_selected_only": "$ExportProjectInfo($IterateComponents($If(actualComponentGUID==componentGUID," + BODY + ")))",
}

args = [
    RS,
    "-headless",
    "-stdConsole",
    "-silent", str(W / "crash"),
    "-set", "appQuitOnError=false",
    "-set", "appAutoSaveMode=false",
    "-load", str(W / "smoke.rsproj"),
    "-selectMaximalComponent",
]
for name, rep in VARIANTS.items():
    args += ["-tag", f"VARIANT_{name}", "-printReport", rep]

args += [
    "-tag", "REPORT_DONE",
    "-exportRegistration", str(OUT / "cameras_opencv.csv"), str(P / "reg_opencv_csv.xml"),
    "-tag", "OPENCV_CSV_DONE",
    "-exportRegistration", str(OUT / "cameras_intext.csv"), str(P / "reg_intext_csv.xml"),
    "-tag", "INTEXT_CSV_DONE",
    "-exportSparsePointCloud", str(OUT / "sparse_params.ply"), str(P / "sparse_ply.xml"),
    "-tag", "PLY_PARAMS_DONE",
    "-quit",
]

proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
sys.stdout.write(proc.stdout or "")
sys.stderr.write(proc.stderr or "")
print(f"\nEXITCODE={proc.returncode}")
