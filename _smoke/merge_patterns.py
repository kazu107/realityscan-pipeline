"""Measure different ways of merging the same set of components.

Input is fixed: the 15 chained components covering index 0-304 (2,440 distinct
images) produced by the full 25/5 run. Only the merge procedure changes.

    one_shot        import all 15 -> mergeComponents
    two_stage_g3    3 groups of 5 -> merge each -> merge the 3 results
    two_stage_g5    5 groups of 3 -> merge each -> merge the 5 results
    sequential      merge 1+2, then add one component at a time (14 steps)
    one_shot_fs1    one shot with -setFeatureSource 1 (component features)
    one_shot_align  one shot followed by -align

Groups are contiguous, so neighbouring groups still share the overlap images
that the chain gave them - without that the second stage has nothing to merge on.
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

RS_CFG = PipelineConfig.load(r"K:\realityscan\presets\1-mid-1.json")
RS_CFG.merge.force_rematch = False
SRC = Path(r"K:\realityscan\out\1-mid-1")
ROOT = Path(r"K:\realityscan\out\_mergepat")
RESULTS = Path(r"K:\realityscan\_smoke\merge_patterns.json")
CREATE_NO_WINDOW = 0x08000000
STAT = re.compile(r"^\s*RSSTAT\|(.*)$")
MAX_INDEX = 304

rows: list[dict] = []


def components() -> list[Path]:
    out = []
    for d in sorted(SRC.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        m = re.match(r".+_(\d{4})-(\d{4})$", d.name)
        if not m or int(m.group(2)) > MAX_INDEX:
            continue
        c = d / f"{d.name}.rsalign"
        if c.exists():
            out.append(c)
    return out


def parse_stat(body: str) -> dict:
    d = {}
    for part in body.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def merge(label: str, comps: list[Path], out_file: Path,
          feature_source: int = 0, align_after: bool = False,
          timeout: int = 7200) -> dict:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    crash = out_file.parent / "crash"
    crash.mkdir(exist_ok=True)

    args = [RS_CFG.run.exe, "-headless", "-stdConsole", "-printProgress",
            "-silent", str(crash)]
    for k, v in _settings(RS_CFG, force_rematch=False):
        args += ["-set", f"{k}={v}"]
    args.append("-newScene")
    for c in comps:
        args += ["-importComponent", str(c)]
    if feature_source >= 0:
        args += ["-selectAllImages", "-setFeatureSource", str(feature_source),
                 "-deselectAllImages"]
    args += ["-mergeComponents"]
    if align_after:
        args += ["-align"]
    args += ["-printReport", report_string("merged"),
             "-selectMaximalComponent",
             "-exportSelectedComponentFile", str(out_file),
             "-quit"]

    t0 = time.time()
    log = out_file.with_suffix(".log")
    best = {}
    try:
        with log.open("w", encoding="utf-8", errors="replace") as lf:
            p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1,
                                 creationflags=CREATE_NO_WINDOW)
            for line in p.stdout:
                lf.write(line)
                lf.flush()
                m = STAT.match(line)
                if m:
                    s = parse_stat(m.group(1))
                    if int(s.get("cams", 0)) > int(best.get("cams", 0)):
                        best = s
                if time.time() - t0 > timeout:
                    p.terminate()
                    break
            p.wait()
        rc = p.returncode
    except Exception as exc:                                  # noqa: BLE001
        rc, best = -1, {"error": str(exc)}
    sec = time.time() - t0
    return {"label": label, "n_components": len(comps), "seconds": round(sec, 1),
            "exit": rc, "cams": int(best.get("cams", 0)),
            "points": int(best.get("pts", 0)),
            "mean_px": float(best.get("mean", 0) or 0),
            "output_mb": round(out_file.stat().st_size / 1e6, 1) if out_file.exists() else 0}


def record(row: dict, extra: dict | None = None) -> dict:
    if extra:
        row.update(extra)
    rows.append(row)
    RESULTS.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  >>> {row['label']}: {row['seconds']:.0f}s  cams={row['cams']} "
          f"pts={row['points']} mean={row['mean_px']:.4f} exit={row['exit']}",
          flush=True)
    return row


def groups(comps: list[Path], n: int) -> list[list[Path]]:
    size = -(-len(comps) // n)
    return [comps[i:i + size] for i in range(0, len(comps), size)]


def two_stage(comps: list[Path], n_groups: int) -> None:
    label = f"two_stage_g{n_groups}"
    print(f"\n########## {label}", flush=True)
    stage1_total = 0.0
    intermediates: list[Path] = []
    for gi, g in enumerate(groups(comps, n_groups)):
        out = ROOT / label / f"stage1_{gi}.rsalign"
        r = merge(f"{label}/stage1_{gi}", g, out)
        stage1_total += r["seconds"]
        record(r)
        if out.exists():
            intermediates.append(out)
    out = ROOT / label / "stage2.rsalign"
    r2 = merge(f"{label}/stage2", intermediates, out)
    record(r2)
    record({"label": label + " TOTAL", "n_components": len(comps),
            "seconds": round(stage1_total + r2["seconds"], 1), "exit": r2["exit"],
            "cams": r2["cams"], "points": r2["points"], "mean_px": r2["mean_px"],
            "output_mb": r2["output_mb"]},
           {"stage1_seconds": round(stage1_total, 1),
            "stage2_seconds": r2["seconds"], "groups": n_groups})


def sequential(comps: list[Path], budget: int = 3600) -> None:
    label = "sequential"
    print(f"\n########## {label}", flush=True)
    acc = ROOT / label / "acc_000.rsalign"
    r = merge(f"{label}/step_000", comps[:2], acc)
    record(r)
    total = r["seconds"]
    for i, c in enumerate(comps[2:], start=1):
        if total > budget:
            record({"label": label + " ABORTED", "n_components": i + 2,
                    "seconds": round(total, 1), "exit": 0, "cams": 0,
                    "points": 0, "mean_px": 0, "output_mb": 0})
            print(f"  !!! {label} aborted after {total:.0f}s ({i + 2} components)",
                  flush=True)
            return
        nxt = ROOT / label / f"acc_{i:03d}.rsalign"
        r = merge(f"{label}/step_{i:03d}", [acc, c], nxt)
        record(r)
        total += r["seconds"]
        if not nxt.exists():
            break
        acc = nxt
    record({"label": label + " TOTAL", "n_components": len(comps),
            "seconds": round(total, 1), "exit": r["exit"], "cams": r["cams"],
            "points": r["points"], "mean_px": r["mean_px"],
            "output_mb": r["output_mb"]})


def main() -> int:
    comps = components()
    print(f"{len(comps)} components, index <= {MAX_INDEX}")
    for c in comps:
        print("   ", c.parent.name)
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    plan = sys.argv[1:] or ["one_shot", "two_stage_g3", "two_stage_g5",
                            "sequential", "one_shot_fs1", "one_shot_align"]

    if "one_shot" in plan:
        print("\n########## one_shot", flush=True)
        record(merge("one_shot", comps, ROOT / "one_shot" / "merged.rsalign"))
    if "two_stage_g3" in plan:
        two_stage(comps, 3)
    if "two_stage_g5" in plan:
        two_stage(comps, 5)
    if "sequential" in plan:
        sequential(comps)
    if "one_shot_fs1" in plan:
        print("\n########## one_shot_fs1", flush=True)
        record(merge("one_shot_fs1", comps,
                     ROOT / "one_shot_fs1" / "merged.rsalign", feature_source=1))
    if "one_shot_align" in plan:
        print("\n########## one_shot_align", flush=True)
        record(merge("one_shot_align", comps,
                     ROOT / "one_shot_align" / "merged.rsalign", align_after=True))

    print("\n================ SUMMARY")
    print(f"{'pattern':30s} {'comps':>6s} {'sec':>8s} {'cams':>6s} "
          f"{'points':>10s} {'mean':>8s}")
    for r in rows:
        if "/" in r["label"]:
            continue
        print(f"{r['label']:30s} {r['n_components']:6d} {r['seconds']:8.0f} "
              f"{r['cams']:6d} {r['points']:10d} {r['mean_px']:8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
