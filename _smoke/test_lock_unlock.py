"""Re-test the lock variant now that the pose lock is released before export.

Pass 1: 10 chained sets with lock_overlap_pose = True (lock for -align only)
Pass 2: merge those components with -setFeatureSource -1 and 0
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
OUT = Path(r"K:\realityscan\_smoke\t10_lockunlock.json")
records: list[dict] = []


def run(cfg: PipelineConfig, label: str):
    scan = scan_folder(cfg.dataset)
    chunks = make_chunks(scan, cfg.dataset)
    print(f"\n########## {label}  ({len(chunks)} sets)", flush=True)

    def emit(event: str, payload: dict) -> None:
        if event == "step_end":
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
            "variant": label, "step": r.name, "kind": r.kind,
            "images": r.n_images, "registered": b.cams if b else 0,
            "points": b.pts if b else 0, "mean_px": b.mean if b else 0,
            "seconds": round(r.seconds, 1), "status": r.status,
            "message": r.message,
        })
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")


def variant(fs: int) -> PipelineConfig:
    cfg = copy.deepcopy(BASE)
    cfg.chain.lock_overlap_pose = True
    cfg.export.out_root = r"K:\realityscan\out\_t10_lockfix"
    cfg.merge.enabled = True
    cfg.merge.feature_source = fs
    cfg.merge.dir_name = f"_merged_fs{fs}"
    return cfg


run(variant(-1), "lock+unlock / merge fs=-1")
run(variant(0), "lock+unlock / merge fs=0")

print("\n================ SUMMARY")
for rec in records:
    if rec["kind"] == "merge" or rec["status"] != "skipped":
        print(f"{rec['variant']:28s} {rec['step']:20s} {rec['kind']:6s} "
              f"reg={rec['registered']:4d}/{rec['images']:<4d} "
              f"pts={rec['points']:8d} mean={rec['mean_px']:.4f} "
              f"{rec['seconds']:6.0f}s {rec['status']} {rec['message']}")
sets = [r for r in records if r["kind"] == "set" and r["status"] == "ok"]
if sets:
    print(f"\nsets: registered={sum(r['registered'] for r in sets)}/"
          f"{sum(r['images'] for r in sets)} "
          f"total={sum(r['seconds'] for r in sets):.0f}s")
