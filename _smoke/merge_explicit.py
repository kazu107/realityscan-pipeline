"""Merge an explicit list of .rsalign files.

The pipeline's own merge always takes every set's component. Repairing a merge
that split needs the opposite: feed it the two halves plus something that spans
the break, and let it work on three well-overlapped components instead of
sixty-eight thin ones.

    python merge_explicit.py <out_dir_name> <feature_source> <a.rsalign> <b.rsalign> ...
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

PRESET = Path(r"K:\realityscan\presets\1-mid-2.json")
STAT = re.compile(r"RSSTAT\|phase=(\w+)\|name=([^|]+)\|cams=(\d+)\|pts=(\d+)")


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    name, fs = sys.argv[1], int(sys.argv[2])
    comps = [Path(p).resolve() for p in sys.argv[3:]]
    missing = [p for p in comps if not p.exists()]
    if missing:
        for p in missing:
            print(f"not found: {p}")
        return 2

    cfg = PipelineConfig.load(PRESET)
    cfg.merge = replace(cfg.merge, dir_name=name, feature_source=fs,
                        force_rematch=False, extra_components=[])
    paths = merge_paths(cfg)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.crash.mkdir(parents=True, exist_ok=True)
    args = build_merge_command(cfg, comps, paths)
    (paths.root / "command.bat").write_text(command_as_batch(args), encoding="utf-8")

    print(f"merging {len(comps)} components with feature_source={fs} into "
          f"{paths.root}")
    for p in comps:
        print(f"  {p}")
    t0 = time.time()
    with open(paths.log, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            log.write(line)
            log.flush()
            m = STAT.search(line)
            if m:
                print(f"  [{m.group(1)}] {m.group(2)}: {m.group(3)} cams, "
                      f"{m.group(4)} pts", flush=True)
            elif "#progress" not in line and line.strip():
                print(line.rstrip()[:160], flush=True)
        p.wait()
    print(f"exit {p.returncode} in {int(time.time() - t0)}s")
    for q in (paths.component, paths.sparse, paths.registration):
        print(f"  {q.name}: {'ok' if q.exists() else 'MISSING'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
