"""Parameter sweep harness.

    python _smoke\sweep.py <plan-name>

Each variant gets its own out_root under K:\realityscan\out\_sweep\<name>, runs
the full pipeline (chained sets + merge) and appends one row to
_smoke\sweep_results.csv / .json. Results are flushed after every variant, so a
partial sweep is still usable.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, r"K:\realityscan")

from rspipe.config import PipelineConfig            # noqa: E402
from rspipe.dataset import make_chunks, scan_folder  # noqa: E402
from rspipe.runner import PipelineRunner            # noqa: E402

BASE_PRESET = r"K:\realityscan\presets\1-mid-1.json"
SWEEP_ROOT = Path(r"K:\realityscan\out\_sweep")
RESULTS_JSON = Path(r"K:\realityscan\_smoke\sweep_results.json")
RESULTS_CSV = Path(r"K:\realityscan\_smoke\sweep_results.csv")


@dataclass
class Variant:
    name: str
    chunk: int = 100
    sets: int = 3
    overlap: int = 10
    images_overlap: str = "Medium"
    sensitivity: str = "Medium"
    downscale: int = 1
    index_end: int = -1
    merge: bool = True

    def config(self) -> PipelineConfig:
        cfg = PipelineConfig.load(BASE_PRESET)
        cfg.dataset.chunk_indices = self.chunk
        cfg.dataset.overlap_indices = self.overlap
        cfg.dataset.max_chunks = self.sets
        cfg.dataset.index_end = self.index_end
        cfg.align.images_overlap = self.images_overlap
        cfg.align.detector_sensitivity = self.sensitivity
        cfg.align.image_downscale = self.downscale
        cfg.merge.enabled = self.merge and self.sets > 1
        cfg.export.out_root = str(SWEEP_ROOT / self.name)
        cfg.run.timeout_min = 0
        return cfg


def load_results() -> list[dict]:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return []


def save_results(rows: list[dict]) -> None:
    RESULTS_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    cols = ["name", "chunk", "sets", "overlap", "images_overlap", "sensitivity",
            "downscale", "indices", "n_indices", "images", "set_registered",
            "set_total", "set_ratio", "set_seconds", "merged_registered",
            "merged_images", "merged_ratio", "merged_points", "merged_mean_px",
            "merge_seconds", "total_seconds", "sec_per_index", "status"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    RESULTS_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_variant(v: Variant, rows: list[dict]) -> dict:
    cfg = v.config()
    scan = scan_folder(cfg.dataset)
    chunks = make_chunks(scan, cfg.dataset)
    n_indices = len({i for c in chunks for i in range(c.index_from, c.index_to + 1)})
    images = len({str(p) for c in chunks for p in c.images})
    print(f"\n########## {v.name}: {len(chunks)} sets, index "
          f"{chunks[0].index_from}-{chunks[-1].index_to} "
          f"({n_indices} idx, {images} distinct images)", flush=True)
    print(f"           chunk={v.chunk} overlap={v.overlap} "
          f"imgOverlap={v.images_overlap} sens={v.sensitivity} "
          f"downscale={v.downscale}", flush=True)

    def emit(event: str, payload: dict) -> None:
        if event == "step_end":
            r = payload["result"]
            b = r.best
            print(f"  --- {r.name}: {r.status} reg={b.cams if b else '-'}/{r.n_images} "
                  f"pts={b.pts if b else '-'} mean={b.mean if b else '-'} "
                  f"({r.seconds:.0f}s) {r.message}", flush=True)

    runner = PipelineRunner(cfg, chunks, emit)
    t0 = time.time()
    runner.start()
    while runner.running:
        runner.join(1.0)
    wall = time.time() - t0

    sets = [r for r in runner.results if r.kind == "set"]
    merges = [r for r in runner.results if r.kind == "merge"]
    reg = sum((r.best.cams if r.best else 0) for r in sets)
    tot = sum(r.n_images for r in sets)
    set_sec = sum(r.seconds for r in sets)
    m = merges[0] if merges else None
    mb = m.best if m else None

    row = {
        "name": v.name, "chunk": v.chunk, "sets": len(chunks), "overlap": v.overlap,
        "images_overlap": v.images_overlap, "sensitivity": v.sensitivity,
        "downscale": v.downscale,
        "indices": f"{chunks[0].index_from}-{chunks[-1].index_to}",
        "n_indices": n_indices, "images": images,
        "set_registered": reg, "set_total": tot,
        "set_ratio": round(reg / tot, 4) if tot else 0,
        "set_seconds": round(set_sec, 1),
        "merged_registered": mb.cams if mb else "",
        "merged_images": m.n_images if m else "",
        "merged_ratio": round(mb.cams / m.n_images, 4) if (m and mb and m.n_images) else "",
        "merged_points": mb.pts if mb else "",
        "merged_mean_px": round(mb.mean, 4) if mb else "",
        "merge_seconds": round(m.seconds, 1) if m else "",
        "total_seconds": round(wall, 1),
        "sec_per_index": round(wall / n_indices, 2) if n_indices else "",
        "status": ";".join(sorted({r.status for r in runner.results})),
    }
    print(f"  ==> sets {reg}/{tot} ({row['set_ratio']:.1%}) in {set_sec:.0f}s | "
          f"merged {row['merged_registered']}/{row['merged_images']} "
          f"in {row['merge_seconds']}s | wall {wall:.0f}s "
          f"({row['sec_per_index']}s per index)", flush=True)
    rows.append(row)
    save_results(rows)
    return row


# ---------------------------------------------------------------- plans ----
def plan_calibration() -> list[Variant]:
    """One 25-index set at both downscales - measures the unit cost."""
    return [
        Variant("cal_ds1", chunk=25, sets=1, downscale=1, merge=False),
        Variant("cal_ds2", chunk=25, sets=1, downscale=2, merge=False),
    ]


def plan_chunksize() -> list[Variant]:
    """The three set sizes the user asked for, same baseline settings."""
    return [
        Variant("size_100x3", chunk=100, sets=3),
        Variant("size_50x6", chunk=50, sets=6),
        Variant("size_25x12", chunk=25, sets=12),
    ]


def plan_params() -> list[Variant]:
    """One-factor-at-a-time screening around a 25-index x 6-set baseline,
    plus the interactions worth checking for downscale 2."""
    def V(name, **kw):
        return Variant(name, chunk=25, sets=6, **kw)

    return [
        V("p_base"),
        V("p_ov5", overlap=5),
        V("p_ov20", overlap=20),
        V("p_imgov_low", images_overlap="Low"),
        V("p_imgov_high", images_overlap="High"),
        V("p_sens_low", sensitivity="Low"),
        V("p_sens_high", sensitivity="High"),
        V("p_sens_ultra", sensitivity="Ultra"),
        V("p_ds2", downscale=2),
        V("p_ds2_sens_high", downscale=2, sensitivity="High"),
        V("p_ds2_sens_ultra", downscale=2, sensitivity="Ultra"),
        V("p_ds2_imgov_high", downscale=2, images_overlap="High"),
    ]


def plan_best() -> list[Variant]:
    """Confirm that the individually-best factors still win when combined.
    Same index range (0-124) as p_ov5, so the comparison is direct."""
    return [
        Variant("best_ov5_imgovhigh", chunk=25, sets=6, overlap=5,
                images_overlap="High"),
        Variant("best50_ov5_imgovhigh", chunk=50, sets=3, overlap=5,
                images_overlap="High"),
    ]


PLANS = {
    "calibration": plan_calibration,
    "chunksize": plan_chunksize,
    "params": plan_params,
    "best": plan_best,
}


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "calibration"
    if name not in PLANS:
        print(f"unknown plan {name}; have {', '.join(PLANS)}")
        return 1
    rows = load_results()
    done = {r["name"] for r in rows}
    for v in PLANS[name]():
        if v.name in done:
            print(f"skip {v.name} (already in results)")
            continue
        run_variant(v, rows)
    print("\n================ RESULTS SO FAR")
    for r in rows:
        print(f"{r['name']:16s} idx={r['indices']:10s} sets={r['sets']:3d} "
              f"reg={r['set_ratio']:.1%} merged={r['merged_ratio']} "
              f"wall={r['total_seconds']:.0f}s ({r['sec_per_index']}s/idx)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
