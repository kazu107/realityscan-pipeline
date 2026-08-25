"""Export just the camera list from saved projects.

export_components.py also writes the component and point cloud, which for a
merged band is several GB and several minutes each. When all that is wanted is
where the cameras ended up, this is the cheap version.

    python export_registration.py <project.rsproj> [...]

Writes cameras.csv next to each project. Paths are resolved to absolute first:
RealityScan resolves a relative path against its own working directory and then
sits there having silently loaded nothing.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.config import PipelineConfig  # noqa: E402
from rspipe.formats import REGISTRATION_FORMATS  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cfg = PipelineConfig()
    fmt = REGISTRATION_FORMATS[cfg.export.registration_format]
    for arg in sys.argv[1:]:
        proj = Path(arg).resolve()
        out = proj.parent / fmt.filename
        if out.exists():
            print(f"{out} already there")
            continue
        crash = proj.parent / "crash_export"
        crash.mkdir(parents=True, exist_ok=True)
        args = [cfg.run.exe, "-headless", "-stdConsole", "-printProgress",
                "-silent", str(crash),
                "-set", f"calexFileFormatId={fmt.guid}",
                "-set", "calexExportImages=false",
                "-set", "appQuitOnError=true",
                "-load", str(proj),
                "-selectMaximalComponent",
                "-exportRegistration", str(out),
                "-quit"]
        t0 = time.time()
        p = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        ok = out.exists()
        print(f"{proj.parent.name}: {'ok' if ok else 'FAILED'} "
              f"in {int(time.time() - t0)}s")
        if not ok:
            for ln in (p.stdout or "").splitlines()[-8:]:
                if "#progress" not in ln:
                    print("   " + ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
