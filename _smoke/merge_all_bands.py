"""Merge every band's already-merged component into a single scene.

Feeding the 250-odd per-set components in would create a dense web of cycles,
which is exactly what concentrated the loop-closure error into one seam on
1-high. Each band's merged component is a single validated rigid piece, so
merging those instead keeps the graph small: every band shares its seeded
1-mid cameras with 1-mid-1, and that is enough to connect all of them.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"K:\realityscan")

from rspipe.cli import _settings, report_string          # noqa: E402
from rspipe.config import PipelineConfig                 # noqa: E402

CFG = PipelineConfig.load(r"K:\realityscan\presets\1-mid-1.json")
CFG.merge.force_rematch = False
OUT = Path(r"K:\realityscan\out\_all")
STAT = re.compile(r"^\s*RSSTAT\|(.*)$")
CREATE_NO_WINDOW = 0x08000000
TIMEOUT_H = 24

BANDS = {
    "1-mid-1": r"K:\realityscan\out\1-mid-1\_merged_norematch\merged_norematch.rsalign",
    "1-high": r"K:\realityscan\out\1-high\_merged_noloop\merged_noloop.rsalign",
    "1-low": r"K:\realityscan\out\1-low\_merged\merged.rsalign",
    "1-mid-2-1": r"K:\realityscan\out\1-mid-2-1\_merged\merged.rsalign",
    "1-mid-2b": r"K:\realityscan\out\1-mid-2b\_merged\merged.rsalign",
}


def main() -> int:
    comps = []
    missing = False
    for name, p in BANDS.items():
        f = Path(p)
        if not f.is_file():
            print(f"MISSING {name}: {f}")
            missing = True
            continue
        print(f"  {name:12s} {f.stat().st_size / 1e9:6.2f} GB  {f}")
        comps.append(f)
    if missing:
        return 1
    if "--check" in sys.argv:
        print(f"\n{len(comps)} band components ready; re-run without --check to merge")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crash").mkdir(exist_ok=True)
    out_component = OUT / "all.rsalign"

    args = [CFG.run.exe, "-headless", "-stdConsole", "-printProgress",
            "-silent", str(OUT / "crash")]
    for k, v in _settings(CFG, force_rematch=False):
        args += ["-set", f"{k}={v}"]
    args.append("-newScene")
    for c in comps:
        args += ["-importComponent", str(c)]
    args += ["-selectAllImages", "-setFeatureSource", "0", "-deselectAllImages",
             "-printReport", report_string("imported"),
             "-tag", "RSPIPE_MERGE_BEGIN",
             "-mergeComponents",
             "-tag", "RSPIPE_MERGE_END",
             "-printReport", report_string("merged"),
             "-selectMaximalComponent",
             "-exportSparsePointCloud", str(OUT / "all_sparse.ply"),
             "-exportRegistration", str(OUT / "cameras.csv"),
             "-exportSelectedComponentFile", str(out_component),
             "-save", str(OUT / "all.rsproj"),
             "-quit"]
    (OUT / "command.bat").write_text(
        " ^\n     ".join(a if " " not in a else f'"{a}"' for a in args) + "\n",
        encoding="utf-8")

    print(f"\nmerging {len(comps)} band components, timeout {TIMEOUT_H} h", flush=True)
    t0 = time.time()
    phases: dict[str, list[dict]] = {}
    with (OUT / "merge.log").open("w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             bufsize=1, creationflags=CREATE_NO_WINDOW)
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            m = STAT.match(line)
            if m:
                s = {k.strip(): v.strip() for k, v in
                     (x.split("=", 1) for x in m.group(1).split("|") if "=" in x)}
                phases.setdefault(s.get("phase", "?"), []).append(
                    {"name": s.get("name", ""), "cams": int(s.get("cams", 0)),
                     "pts": int(s.get("pts", 0)), "mean": float(s.get("mean", 0) or 0)})
                print(f"  [{s.get('phase')}] {s.get('name')}: {s.get('cams')} cams",
                      flush=True)
            if time.time() - t0 > TIMEOUT_H * 3600:
                print("  !!! timeout, terminating", flush=True)
                p.terminate()
                break
        p.wait()
    sec = time.time() - t0

    imported = {c["name"] for c in phases.get("imported", [])}
    new = [c for c in phases.get("merged", []) if c["name"] not in imported]
    best = max(new, key=lambda c: c["cams"]) if new else None
    result = {"seconds": round(sec, 1), "exit": p.returncode,
              "imported": phases.get("imported", []),
              "merged_new": new, "best": best}
    (Path(r"K:\realityscan\_smoke") / "merge_all_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nexit={p.returncode}  {sec/3600:.2f} h")
    if best:
        print(f"largest merged component: {best['cams']} cameras, "
              f"{best['pts']} points, mean {best['mean']:.4f} px")
    return 0 if p.returncode == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
