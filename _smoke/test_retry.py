"""Does a follow-up -align recover the images a set failed to register?

Runs on copies of the real under-registered sets from the full 917-index run,
so the originals stay intact. Two modes per set:

    plain  : just -align again
    drop   : delete every component but the largest, then -align

Up to 3 extra passes each, stopping when the set reaches 100%.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"K:\realityscan")

from rspipe.cli import build_retry_command, chunk_paths, report_string  # noqa: E402
from rspipe.config import PipelineConfig                                # noqa: E402
from rspipe.dataset import make_chunks, scan_folder                     # noqa: E402

SRC = Path(r"K:\realityscan\out\1-mid-1")
#: .rsproj stores image paths RELATIVE to the project, so a copy has to sit at
#: the same directory depth as the original (out\<x>\<set>\) or RealityScan
#: stops on a "Locate file" dialog - invisible in headless mode.
WORK = Path(r"K:\realityscan\out")
RESULTS = Path(r"K:\realityscan\_smoke\retry_results.json")
CREATE_NO_WINDOW = 0x08000000
STAT = re.compile(r"^\s*RSSTAT\|(.*)$")
MAX_ATTEMPTS = 3

# name -> (registered, images) from the original run
CASES = [
    ("1-mid_0840-0864", 192, 200),   # 3 components  <- drop should help most
    ("1-mid_0000-0024", 198, 200),   # 2 components
    ("1-mid_0480-0504", 191, 200),   # 1 component, worst ratio
    ("1-mid_0580-0604", 194, 200),   # 1 component
    ("1-mid_0360-0384", 195, 200),   # 1 component
]

rows: list[dict] = []


def parse_stat(body: str) -> dict:
    return {k.strip(): v.strip() for k, v in
            (p.split("=", 1) for p in body.split("|") if "=" in p)}


def probe(cfg: PipelineConfig, paths) -> list[dict]:
    """List the components currently in the project."""
    args = [cfg.run.exe, "-headless", "-stdConsole", "-silent", str(paths.crash),
            "-set", "appQuitOnError=true", "-set", "appAutoSaveMode=false",
            "-load", str(paths.project),
            "-printReport", report_string("probe"), "-quit"]
    try:
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=600,
                           creationflags=CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        print("  !!! probe timed out - RealityScan is probably waiting on a dialog",
              flush=True)
        return []
    (paths.root / "probe.log").write_text(p.stdout or "", encoding="utf-8")
    out = []
    for line in (p.stdout or "").splitlines():
        m = STAT.match(line)
        if m:
            s = parse_stat(m.group(1))
            out.append({"name": s.get("name", ""), "cams": int(s.get("cams", 0))})
    if not out:
        print(f"  !!! probe found no components (exit {p.returncode}) - see probe.log",
              flush=True)
    return out


def run_attempt(cfg: PipelineConfig, paths, attempt: int, drop: list[str]) -> dict:
    args = build_retry_command(cfg, paths, attempt, drop)
    log = paths.root / f"retry{attempt}.log"
    t0 = time.time()
    comps: list[dict] = []
    with log.open("w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             bufsize=1, creationflags=CREATE_NO_WINDOW)
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            m = STAT.match(line)
            if m:
                s = parse_stat(m.group(1))
                comps.append({"name": s.get("name", ""), "cams": int(s.get("cams", 0)),
                              "pts": int(s.get("pts", 0)),
                              "mean": float(s.get("mean", 0) or 0)})
        p.wait()
    best = max(comps, key=lambda c: c["cams"]) if comps else {"cams": 0, "pts": 0, "mean": 0}
    return {"seconds": round(time.time() - t0, 1), "exit": p.returncode,
            "components": comps, "best": best}


def main() -> int:
    base = PipelineConfig.load(r"K:\realityscan\presets\1-mid-1.json")
    scan = scan_folder(base.dataset)
    chunks = {c.name: c for c in make_chunks(scan, base.dataset, base.chain)}

    for mode in ("plain", "drop"):
        d = WORK / f"_rt_{mode}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for name, reg0, n_img in CASES:
        for mode in ("plain", "drop"):
            cfg = copy.deepcopy(base)
            cfg.export.out_root = str(WORK / f"_rt_{mode}")
            cfg.retry.enabled = True
            cfg.retry.drop_minor_components = (mode == "drop")
            paths = chunk_paths(chunks[name], cfg)

            # copy the original set folder so the real outputs stay untouched
            paths.root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SRC / name, paths.root)
            paths.crash.mkdir(exist_ok=True)

            comps = probe(cfg, paths)
            fresh = [c for c in comps if not c["name"].startswith("1-mid_")]
            print(f"\n########## {name} [{mode}]  start {reg0}/{n_img} "
                  f"({reg0 / n_img:.1%})  components={[c['cams'] for c in fresh]}",
                  flush=True)

            cur = max((c["cams"] for c in fresh), default=0)
            total_sec = 0.0
            history = []
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if cur >= n_img:
                    print("  reached 100%, stopping", flush=True)
                    break
                drop = []
                if mode == "drop" and len(fresh) > 1:
                    keep = max(fresh, key=lambda c: c["cams"])
                    drop = [c["name"] for c in fresh if c["name"] != keep["name"]]
                r = run_attempt(cfg, paths, attempt, drop)
                total_sec += r["seconds"]
                after = r["best"]["cams"]
                fresh = [c for c in r["components"] if not c["name"].startswith("1-mid_")] \
                    or r["components"]
                print(f"  attempt {attempt}: {cur} -> {after} ({after - cur:+d}) "
                      f"in {r['seconds']:.0f}s  components={[c['cams'] for c in fresh]}"
                      f"  exit={r['exit']}", flush=True)
                history.append({"attempt": attempt, "before": cur, "after": after,
                                "dropped": len(drop), "seconds": r["seconds"],
                                "exit": r["exit"],
                                "components": [c["cams"] for c in fresh]})
                cur = after
                if r["exit"] != 0:
                    break

            rows.append({"set": name, "mode": mode, "images": n_img,
                         "start": reg0, "end": cur, "gain": cur - reg0,
                         "attempts": len(history), "seconds": round(total_sec, 1),
                         "history": history})
            RESULTS.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            shutil.rmtree(paths.root, ignore_errors=True)

    print("\n================ SUMMARY")
    print(f"{'set':22s} {'mode':6s} {'start':>9s} {'end':>9s} {'gain':>5s} "
          f"{'passes':>7s} {'sec':>7s}")
    for r in rows:
        print(f"{r['set']:22s} {r['mode']:6s} {r['start']:4d}/{r['images']:<4d} "
              f"{r['end']:4d}/{r['images']:<4d} {r['gain']:+5d} "
              f"{r['attempts']:7d} {r['seconds']:7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
