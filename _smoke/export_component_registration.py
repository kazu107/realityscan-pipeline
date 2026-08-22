"""Write the camera list of a .rsalign, without needing a project.

export_registration.py needs a saved project; a patch is aligned without one to
save the disk. But a patch has to be checked before a merge is committed to, and
that check reads camera positions - so import the component into an empty scene
and export the registration from there. Cheap: seconds for a few hundred
cameras.

    python export_component_registration.py <component.rsalign> [...]

Writes cameras.csv beside each component. Only works while the component and
its images are on the same drive - a component written to another drive records
its images as "..\\..\\..\\K:\\..." and cannot be re-imported at all.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.config import PipelineConfig, find_realityscan  # noqa: E402
from rspipe.formats import REGISTRATION_FORMATS  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    cfg = PipelineConfig()
    exe = cfg.run.exe or find_realityscan()
    fmt = REGISTRATION_FORMATS[cfg.export.registration_format]
    bad = 0
    for arg in sys.argv[1:]:
        comp = Path(arg).resolve()
        if not comp.is_file():
            print(f"not found: {comp}")
            bad += 1
            continue
        out = comp.parent / "cameras.csv"
        crash = comp.parent / "crash_reg"
        crash.mkdir(parents=True, exist_ok=True)
        args = [exe, "-headless", "-stdConsole", "-printProgress",
                "-silent", str(crash),
                "-set", "appQuitOnError=true",
                "-set", f"calexFileFormatId={fmt.guid}",
                "-set", "calexExportImages=false",
                "-newScene",
                "-importComponent", str(comp),
                "-selectMaximalComponent",
                "-exportRegistration", str(out),
                "-quit"]
        print(f"=== {comp.parent.name}", flush=True)
        t0 = time.time()
        log = comp.parent / "export_registration.log"
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:
                f.write(line)
                if "Could not find file" in line or "Processing failed" in line:
                    print("    " + line.rstrip()[:160], flush=True)
            p.wait()
        ok = out.is_file()
        bad += not ok
        print(f"--- exit {p.returncode} in {int(time.time() - t0)}s, "
              f"cameras.csv {'ok' if ok else 'MISSING'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
