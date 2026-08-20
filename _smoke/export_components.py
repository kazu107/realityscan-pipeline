"""Export the camera list of every component in a saved project.

-selectMaximalComponent only ever gives the biggest one, so a merge that split
leaves the other side invisible. -importComponent names a component after its
file, and -selectComponent takes that name, so each one can be exported on its
own.

    python export_components.py <project.rsproj> <name> [<name> ...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.config import PipelineConfig  # noqa: E402
from rspipe.formats import REGISTRATION_FORMATS  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    # RealityScan resolves paths against its own working directory, not the
    # shell's - a relative -load silently finds nothing and the run then hangs
    # on the first command that needs a scene.
    proj = Path(sys.argv[1]).resolve()
    names = sys.argv[2:]
    cfg = PipelineConfig()
    fmt = REGISTRATION_FORMATS[cfg.export.registration_format]
    out = proj.parent
    (out / "crash_export").mkdir(parents=True, exist_ok=True)

    args = [cfg.run.exe or PipelineConfig().run.exe, "-headless", "-stdConsole",
            "-printProgress", "-silent", str(out / "crash_export"),
            "-set", f"calexFileFormatId={fmt.guid}",
            "-set", "calexExportImages=false",
            "-set", "appQuitOnError=true",
            "-load", str(proj)]
    for n in names:
        safe = n.replace(" ", "_").replace("(", "").replace(")", "")
        args += ["-selectComponent", n,
                 "-exportRegistration", str(out / f"{safe}.csv"),
                 "-exportSparsePointCloud", str(out / f"{safe}_sparse.ply"),
                 "-exportSelectedComponentFile", str(out / f"{safe}.rsalign")]
    args.append("-quit")

    print(" ".join(args), flush=True)
    log = out / "export_components.log"
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            f.write(line)
            if "#progress" not in line:
                print(line.rstrip(), flush=True)
        p.wait()
    print(f"exit {p.returncode}")
    for n in names:
        safe = n.replace(" ", "_").replace("(", "").replace(")", "")
        q = out / f"{safe}.csv"
        print(f"  {q.name}: {'ok' if q.exists() else 'MISSING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
