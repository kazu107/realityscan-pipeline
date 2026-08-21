"""Rebuild a .rsproj from an exported component.

A project stores its image paths relative to itself, so one cannot be moved to
a different directory depth - it comes back up asking for files, through a
dialog that headless mode suppresses and then waits on forever. A component
does not have that problem: -importComponent resolved 112 of them across
mismatched depths in the band merge. So the way to relocate a project is to
throw it away and rebuild it from the component next to it.

    python rebuild_project.py <component.rsalign> [<project.rsproj>]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.config import PipelineConfig, find_realityscan  # noqa: E402


def main() -> int:
    # RealityScan's banner carries bytes that the console's cp932 codec
    # cannot encode, and printing them kills the run before it starts
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    comp = Path(sys.argv[1]).resolve()
    proj = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 \
        else comp.with_suffix(".rsproj")
    if not comp.is_file():
        print(f"not found: {comp}")
        return 1
    exe = PipelineConfig().run.exe or find_realityscan()
    crash = proj.parent / "crash_rebuild"
    crash.mkdir(parents=True, exist_ok=True)

    args = [exe, "-headless", "-stdConsole", "-printProgress",
            "-silent", str(crash),
            "-set", "appQuitOnError=true",
            "-set", "appAutoSaveMode=false",
            "-newScene",
            "-importComponent", str(comp),
            "-printReport", "RSSTAT|cams=$(componentCamerasCount)",
            "-save", str(proj),
            "-quit"]
    print(f"importing {comp.name} ({comp.stat().st_size / 2**30:.1f} GB)")
    print(f"saving    {proj}")
    t0 = time.time()
    log = proj.parent / "rebuild.log"
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            f.write(line)
            f.flush()
            if "#progress" not in line and line.strip():
                print("  " + line.rstrip()[:150], flush=True)
        p.wait()
    print(f"exit {p.returncode} in {int(time.time() - t0)}s")
    print(f"  {proj.name}: "
          f"{'ok, ' + str(round(proj.stat().st_size / 2**20, 1)) + ' MB' if proj.exists() else 'MISSING'}")
    return 0 if proj.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
