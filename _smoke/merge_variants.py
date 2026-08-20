"""Reproduce the 1-mid-2 merge split on a small slice and test settings on it.

The full merge takes ~4.8 h, so trying fixes on it is not practical. The break
sits at index ~505, where the forward views run ahead as their own component and
the rear views stay with the previous block until ~604. Merging only the sets
that span that region reproduces the same decision on a fraction of the data.

    python merge_variants.py fs0rm0 fs1rm0 ...

Variants are named <feature source><force rematch>. Each writes into its own
folder at the same tree depth as the real _merged, because RealityScan resolves
a project's image paths relative to the project file.
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
OUT = Path(r"K:\realityscan\out\1-mid-2")
LO, HI = 440, 664                     # sets whose span falls inside this window
STAT = re.compile(r"RSSTAT\|phase=(\w+)\|name=([^|]+)\|cams=(\d+)\|pts=(\d+)")
SPAN = re.compile(r"_(\d+)-(\d+)$")

VARIANTS = {
    # name:      (feature_source, force_rematch)
    "fs0rm0": (0, False),             # what the production run used
    "fs1rm0": (1, False),             # RealityScan's own default source
    "fs0rm1": (0, True),
    "fs2rm0": (2, False),             # all image features - slowest, strongest
    "fs1rm1": (1, True),
}


def components() -> list[Path]:
    out = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        m = SPAN.search(d.name)
        c = d / f"{d.name}.rsalign"
        if m and c.exists() and int(m.group(1)) >= LO and int(m.group(2)) <= HI:
            out.append(c)
    return out


def run(name: str, fs: int, rm: bool, comps: list[Path]) -> dict:
    cfg = PipelineConfig.load(PRESET)
    cfg.merge = replace(cfg.merge, dir_name=f"_v_{name}", feature_source=fs,
                        force_rematch=rm)
    cfg.export = replace(cfg.export, sparse=False, project=False)
    paths = merge_paths(cfg)
    paths.root.mkdir(parents=True, exist_ok=True)
    args = build_merge_command(cfg, comps, paths)
    (paths.root / "command.bat").write_text(command_as_batch(args), encoding="utf-8")

    t0 = time.time()
    merged: list[tuple[str, int, int]] = []
    with open(paths.log, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             bufsize=1)
        for line in p.stdout:
            log.write(line)
            log.flush()
            m = STAT.search(line)
            if m and m.group(1) == "merged":
                merged.append((m.group(2), int(m.group(3)), int(m.group(4))))
        p.wait()
    secs = time.time() - t0

    comps_out = [(n, c, pt) for n, c, pt in merged if n.lower().startswith("component")]
    total = sum(c for _, c, _ in comps_out)
    biggest = max((c for _, c, _ in comps_out), default=0)
    return {"name": name, "fs": fs, "rematch": rm, "exit": p.returncode,
            "seconds": round(secs), "components": len(comps_out),
            "cams_total": total, "cams_largest": biggest,
            "ratio": round(biggest / total, 4) if total else 0.0,
            "detail": comps_out}


def main() -> int:
    want = sys.argv[1:] or ["fs0rm0"]
    comps = components()
    print(f"{len(comps)} components spanning indices {LO}-{HI}")
    for c in comps:
        print(f"  {c.parent.name}")
    print()
    for name in want:
        if name not in VARIANTS:
            print(f"unknown variant {name}; known: {', '.join(VARIANTS)}")
            return 2
        fs, rm = VARIANTS[name]
        print(f"=== {name}: feature_source={fs} force_rematch={rm}", flush=True)
        r = run(name, fs, rm, comps)
        print(f"--- {name}: exit={r['exit']} {r['seconds']}s "
              f"components={r['components']} "
              f"largest={r['cams_largest']}/{r['cams_total']} "
              f"({r['ratio'] * 100:.1f}%)", flush=True)
        for n, c, pt in r["detail"]:
            print(f"      {n}: {c} cams, {pt} pts", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
