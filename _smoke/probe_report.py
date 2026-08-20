"""Verify the final -printReport string and the params.xml-driven exports."""
import subprocess
import sys
from pathlib import Path

RS = r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
W = Path(r"K:\realityscan\_smoke")
PROJ = W / "smoke.rsproj"
OUT = W / "probe_out"
P = W / "params"
OUT.mkdir(exist_ok=True)

REPORT = (
    "$IterateComponents("
    "RSSTAT|name=$(componentName)|cams=$(componentCamerasCount)"
    "$ComponentInfo(componentGUID,"
    "|pts=$(componentPointsCount)"
    "|cps=$(componentControlPointsCountUsed)"
    ")"
    "$ComponentStats(componentGUID,"
    "|proj=$(componentTotalProjection)"
    "|track=$(componentAverageTrackLength:.2f)"
    "|max=$(componentMaximalError:.4f)"
    "|median=$(componentMedianError:.4f)"
    "|mean=$(componentMeanError:.4f)"
    "|metric=$(componentMetric)"
    ")\n)"
)

args = [
    RS, "-headless", "-stdConsole",
    "-silent", str(W / "crash"),
    "-set", "appQuitOnError=true",
    "-set", "appAutoSaveMode=false",
    "-load", str(PROJ),
    "-selectMaximalComponent",
    "-printReport", REPORT,
    "-tag", "REPORT_DONE",
    "-exportRegistration", str(OUT / "cameras_opencv.csv"), str(P / "reg_opencv_csv.xml"),
    "-tag", "OPENCV_CSV_DONE",
    "-exportRegistration", str(OUT / "cameras_intext.csv"), str(P / "reg_intext_csv.xml"),
    "-tag", "INTEXT_CSV_DONE",
    "-exportSparsePointCloud", str(OUT / "sparse_params.ply"), str(P / "sparse_ply.xml"),
    "-tag", "PLY_PARAMS_DONE",
    "-quit",
]

p = subprocess.run(args, capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=300)
sys.stdout.write(p.stdout or "")
print(f"EXITCODE={p.returncode}")
