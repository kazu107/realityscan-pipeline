"""48 images x 10 sets: does locking the overlap pose help, and which
Features source should the merge use?

Pass 1  chain with lock_overlap_pose = False -> 10 components
Pass 2  merge those components with -setFeatureSource -1 / 0 / 1 / 2
Pass 3  chain with lock_overlap_pose = True  -> 10 components
Pass 4  merge those with -setFeatureSource -1 / 0

Sets are skipped on re-runs (skip_existing), so every merge variant re-uses
the same components.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"K:\realityscan")

from rspipe.config import PipelineConfig            # noqa: E402
from rspipe.dataset import make_chunks, scan_folder  # noqa: E402
from rspipe.runner import PipelineRunner            # noqa: E402

BASE = PipelineConfig.load(r"K:\realityscan\presets\_test-10sets.json")
OUT = Path(r"K:\realityscan\_smoke\t10_results.json")
records: list[dict] = []


def run(cfg: PipelineConfig, label: str) -> list:
    scan = scan_folder(cfg.dataset)
    chunks = make_chunks(scan, cfg.dataset)
    print(f"\n########## {label}  ({len(chunks)} sets)", flush=True)

    def emit(event: str, payload: dict) -> None:
        if event == "step_start":
            r = payload["result"]
            print(f"  === {r.name}: {r.n_images} img"
                  f"{' [chained]' if r.chained else ''}", flush=True)
        elif event == "step_end":
            r = payload["result"]
            b = r.best
            print(f"  --- {r.name}: {r.status} "
                  f"reg={b.cams if b else '-'}/{r.n_images} "
                  f"pts={b.pts if b else '-'} mean={b.mean if b else '-'} "
                  f"({r.seconds:.0f}s) {r.message}", flush=True)

    runner = PipelineRunner(cfg, chunks, emit)
    t0 = time.time()
    runner.start()
    while runner.running:
        runner.join(0.5)
    print(f"  total {time.time() - t0:.0f}s", flush=True)

    for r in runner.results:
        b = r.best
        records.append({
            "variant": label,
            "step": r.name,
            "kind": r.kind,
            "chained": r.chained,
            "images": r.n_images,
            "registered": b.cams if b else 0,
            "points": b.pts if b else 0,
            "mean_px": b.mean if b else 0,
            "median_px": b.median if b else 0,
            "max_px": b.max if b else 0,
            "track": b.track if b else 0,
            "seconds": round(r.seconds, 1),
            "status": r.status,
            "message": r.message,
        })
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return runner.results


def variant(lock: bool, feature_source: int, first: bool) -> PipelineConfig:
    cfg = copy.deepcopy(BASE)
    cfg.chain.lock_overlap_pose = lock
    cfg.export.out_root = (r"K:\realityscan\out\_t10_lock" if lock
                           else r"K:\realityscan\out\_t10_nolock")
    cfg.merge.feature_source = feature_source
    cfg.merge.dir_name = f"_merged_fs{feature_source}"
    cfg.merge.enabled = True
    return cfg


plan = [
    (False, -1, "nolock / merge fs=-1 (as imported)"),
    (False, 0, "nolock / merge fs=0 (merge using overlaps)"),
    (False, 1, "nolock / merge fs=1 (component features)"),
    (False, 2, "nolock / merge fs=2 (all image features)"),
    (True, -1, "lock / merge fs=-1 (as imported)"),
    (True, 0, "lock / merge fs=0 (merge using overlaps)"),
]

for i, (lock, fs, label) in enumerate(plan):
    run(variant(lock, fs, i == 0), label)

print("\n================ SUMMARY (merge steps)")
print(f"{'variant':44s} {'reg':>8s} {'imgs':>6s} {'points':>9s} {'mean':>7s} {'sec':>6s}  status")
for rec in records:
    if rec["kind"] == "merge":
        print(f"{rec['variant']:44s} {rec['registered']:8d} {rec['images']:6d} "
              f"{rec['points']:9d} {rec['mean_px']:7.4f} {rec['seconds']:6.0f}  "
              f"{rec['status']} {rec['message']}")

print("\n================ SUMMARY (per set)")
for lockname in ("nolock", "lock"):
    sets = [r for r in records if r["kind"] == "set"
            and r["variant"].startswith(lockname) and r["status"] == "ok"]
    if not sets:
        continue
    tot = sum(r["seconds"] for r in sets)
    reg = sum(r["registered"] for r in sets)
    img = sum(r["images"] for r in sets)
    mean = sum(r["mean_px"] for r in sets) / len(sets)
    print(f"{lockname:8s} sets_run={len(sets):3d} registered={reg}/{img} "
          f"avg_mean={mean:.4f} px total={tot:.0f}s")
