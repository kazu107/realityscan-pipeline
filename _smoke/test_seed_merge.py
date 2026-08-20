"""Does seeding actually let two separate captures merge?

Merges the same four components twice:
    no seed : 1-mid sets 0-1  +  1-high sets 0-1 (aligned on their own)
    seeded  : 1-mid sets 0-1  +  1-high sets 0-1 (set 0 was given the mid component)

The two captures share no images, so without the seed the merge has nothing to
latch onto and should leave them in separate components.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"K:\realityscan")

from rspipe.cli import _settings, report_string          # noqa: E402
from rspipe.config import PipelineConfig                 # noqa: E402

CFG = PipelineConfig.load(r"K:\realityscan\presets\1-mid-1.json")
OUT = Path(r"K:\realityscan\out")
STAT = re.compile(r"^\s*RSSTAT\|(.*)$")
CREATE_NO_WINDOW = 0x08000000

MID = [OUT / "1-mid-1" / n / f"{n}.rsalign"
       for n in ("1-mid_0000-0024", "1-mid_0020-0044")]
CASES = {
    "noseed": [OUT / "_high-noseed" / n / f"{n}.rsalign"
               for n in ("1-high_0000-0024", "1-high_0020-0044")],
    "seed": [OUT / "_high-seed" / n / f"{n}.rsalign"
             for n in ("1-high_0000-0024", "1-high_0020-0044")],
}


def parse_stat(body: str) -> dict:
    return {k.strip(): v.strip() for k, v in
            (p.split("=", 1) for p in body.split("|") if "=" in p)}


def merge(label: str, comps: list[Path]) -> dict:
    root = OUT / f"_seedmerge_{label}"
    if root.exists():
        shutil.rmtree(root)
    (root / "crash").mkdir(parents=True)
    out_file = root / "merged.rsalign"

    args = [CFG.run.exe, "-headless", "-stdConsole", "-printProgress",
            "-silent", str(root / "crash")]
    for k, v in _settings(CFG, force_rematch=False):
        args += ["-set", f"{k}={v}"]
    args.append("-newScene")
    for c in comps:
        args += ["-importComponent", str(c)]
    args += ["-selectAllImages", "-setFeatureSource", "0", "-deselectAllImages",
             "-printReport", report_string("imported"),
             "-mergeComponents",
             "-printReport", report_string("merged"),
             "-selectMaximalComponent",
             "-exportRegistration", str(root / "cameras.csv"),
             "-exportSelectedComponentFile", str(out_file),
             "-quit"]

    t0 = time.time()
    phases: dict[str, list[dict]] = {}
    with (root / "merge.log").open("w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             bufsize=1, creationflags=CREATE_NO_WINDOW)
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            m = STAT.match(line)
            if m:
                s = parse_stat(m.group(1))
                phases.setdefault(s.get("phase", "?"), []).append(
                    {"name": s.get("name", ""), "cams": int(s.get("cams", 0))})
        p.wait()
    sec = time.time() - t0

    merged = {c["name"]: c["cams"] for c in phases.get("merged", [])}
    imported = {c["name"]: c["cams"] for c in phases.get("imported", [])}
    new = {n: c for n, c in merged.items() if n not in imported}
    best = max(new.values()) if new else max(merged.values(), default=0)

    counts: dict[str, int] = {}
    csv = root / "cameras.csv"
    if csv.exists():
        for line in csv.open(encoding="utf-8", errors="replace"):
            if line.startswith("#"):
                continue
            mm = re.match(r"([^_]+(?:-[^_]+)*)_\d+_\d+\.", line)
            if mm:
                counts[mm.group(1)] = counts.get(mm.group(1), 0) + 1

    return {"case": label, "seconds": round(sec, 1), "exit": p.returncode,
            "imported": imported, "merged_new": new, "best": best,
            "exported_by_capture": counts}


rows = []
for label, high in CASES.items():
    print(f"\n########## {label}", flush=True)
    r = merge(label, MID + high)
    rows.append(r)
    print(f"  imported : {r['imported']}")
    print(f"  new after merge: {r['merged_new']}")
    print(f"  exported component: {r['best']} cameras {r['exported_by_capture']}"
          f"  ({r['seconds']:.0f}s, exit {r['exit']})", flush=True)
Path(r"K:\realityscan\_smoke\seed_merge_results.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8")

print("\n================ SUMMARY")
for r in rows:
    joined = len(r["exported_by_capture"]) > 1
    print(f"{r['case']:8s} largest merged component = {r['best']:4d} cameras "
          f"{r['exported_by_capture']}  -> captures joined: {'YES' if joined else 'NO'}")
