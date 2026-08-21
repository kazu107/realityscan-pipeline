"""Merge three bands at set level, with the merge's feature budget cut down.

Two things were established first. Feature source 0 links components only
through images they literally share, which across bands means only the seeded
1-mid cameras - so everything between the seeds is unconstrained and drifts.
Source 1 uses real features and does link them, but three bands at full feature
count ran a 96 GB machine out of memory.

Hence: source 1 with merge-time overrides that cut the budget, and every set's
component rather than three whole-band ones. -mergeComponents applies a single
similarity per component, so three rigid blocks cannot absorb a shape
difference the bands were measured to have, while a hundred small ones can
follow it piecewise.

    python merge_bands.py <out dir name> <component list file> [--fs 1]
                          [--downscale 2] [--per-mpx 8000] [--quality Normal]
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.cli import build_merge_command, command_as_batch, merge_paths  # noqa: E402
from rspipe.config import PipelineConfig  # noqa: E402

STAT = re.compile(r"RSSTAT\|phase=(\w+)\|name=([^|]+)\|cams=(\d+)\|pts=(\d+)")


def main() -> int:
    argv = list(sys.argv[1:])
    opts = {"--fs": 1, "--downscale": 2, "--per-mpx": 8000, "--quality": "Normal"}
    for k in list(opts):
        if k in argv:
            i = argv.index(k)
            opts[k] = argv[i + 1]
            del argv[i:i + 2]
    if len(argv) < 2:
        print(__doc__)
        return 2
    name, listfile = argv[0], Path(argv[1])
    comps = [Path(l.strip()).resolve() for l in listfile.read_text().splitlines()
             if l.strip()]
    missing = [c for c in comps if not c.is_file()]
    if missing:
        print(f"{len(missing)} component(s) missing, first: {missing[0]}")
        return 1

    cfg = PipelineConfig()
    cfg.run.exe = PipelineConfig().run.exe or ""
    from rspipe.config import find_realityscan
    cfg.run.exe = cfg.run.exe or find_realityscan()
    cfg.run.timeout_min = 1440
    # absolute: RealityScan resolves a relative path against its own working
    # directory, which is the install folder - an 8 hour merge wrote 21 GB of
    # exports into C:\Program Files before this was noticed
    cfg.export = replace(cfg.export, out_root=str((Path("out") / name).resolve()),
                         sparse=True, registration=True, project=True)
    cfg.merge = replace(cfg.merge, dir_name="_merged", feature_source=int(opts["--fs"]),
                        force_rematch=False, extra_components=[],
                        align_overrides={
                            "image_downscale": int(opts["--downscale"]),
                            "max_features_per_mpx": int(opts["--per-mpx"]),
                            "feature_detection_quality": str(opts["--quality"]),
                        })
    paths = merge_paths(cfg)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.crash.mkdir(parents=True, exist_ok=True)
    args = build_merge_command(cfg, comps, paths)
    (paths.root / "command.bat").write_text(command_as_batch(args), encoding="utf-8")
    print(f"merging {len(comps)} components into {paths.root}")
    print(f"  feature_source={cfg.merge.feature_source} "
          f"overrides={cfg.merge.align_overrides}", flush=True)

    t0 = time.time()
    with open(paths.log, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            log.write(line)
            log.flush()
            m = STAT.search(line)
            if m and (m.group(1) != "imported" or "Component" in m.group(2)):
                print(f"  [{m.group(1)}] {m.group(2)}: {m.group(3)} cams, "
                      f"{m.group(4)} pts", flush=True)
            elif "Processing failed" in line or "out of memory" in line.lower():
                print("  " + line.rstrip(), flush=True)
        p.wait()
    print(f"exit {p.returncode} in {int(time.time() - t0)}s")
    for q in (paths.component, paths.sparse, paths.registration):
        print(f"  {q.name}: {'ok' if q.exists() else 'MISSING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
