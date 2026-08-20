"""Find how to pin the registration-export format from the CLI.

Config key discovered in the project appConfig blob: calexFileFormatId
Try it via -set and via a params.xml <Configuration><entry .../></Configuration>.
"""
import shutil
import subprocess
import sys
from pathlib import Path

RS = r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe"
W = Path(r"K:\realityscan\_smoke")
PROJ = W / "smoke.rsproj"
ROOT = W / "fmt"

OPENCV = "{B5331837-609D-4B12-A931-2863653d19F7}"
INTEXT = "{0CA18733-1EBC-4254-9974-17197EB409BD}"


def fresh(name: str) -> Path:
    d = ROOT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def run(extra_pre, out_file, params=None, timeout=240):
    args = [RS, "-headless", "-stdConsole",
            "-silent", str(W / "crash"),
            "-set", "appQuitOnError=true",
            "-set", "appAutoSaveMode=false"]
    args += extra_pre
    args += ["-load", str(PROJ), "-selectMaximalComponent",
             "-exportRegistration", str(out_file)]
    if params:
        args.append(str(params))
    args.append("-quit")
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, p.stdout or ""
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""


def report(label, d, rc, out):
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    print(f"--- {label}: exit={rc} files={len(files)}")
    for f in files[:6]:
        print(f"      {f}")
    for ln in out.splitlines():
        if "error" in ln.lower() or "failed" in ln.lower():
            print("      ! " + ln)


# 1) -set calexFileFormatId
d = fresh("set_opencv")
rc, out = run(["-set", f"calexFileFormatId={OPENCV}"], d / "cameras.csv")
report("set calexFileFormatId=OpenCV", d, rc, out)

# 2) params.xml with entry, no Configuration id
d = fresh("params_opencv")
px = ROOT / "params_opencv.xml"
px.write_text(
    f'<Configuration>\n  <entry key="calexFileFormatId" value="{OPENCV}"/>\n</Configuration>\n',
    encoding="utf-8")
rc, out = run([], d / "cameras.csv", px)
report("params.xml entry calexFileFormatId=OpenCV", d, rc, out)

# 3) -set to Internal/External (sanity: different format, different content)
d = fresh("set_intext")
rc, out = run(["-set", f"calexFileFormatId={INTEXT}"], d / "cameras.csv")
report("set calexFileFormatId=Internal/External", d, rc, out)

for name in ("set_opencv", "params_opencv", "set_intext"):
    f = ROOT / name / "cameras.csv"
    if f.exists():
        print(f"\n===== {name}/cameras.csv ({f.stat().st_size} B)")
        with f.open(encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= 3:
                    break
                print("   " + line.rstrip()[:220])
sys.exit(0)
