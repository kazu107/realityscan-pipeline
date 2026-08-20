"""Headless entry point: run the pipeline from a saved config, no GUI.

    python run_batch.py presets\1-mid-1.json [--max-sets 2] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rspipe.cli import build_command, chunk_paths, command_as_batch
from rspipe.config import PipelineConfig, parse_seed_spec
from rspipe.dataset import make_chunks, mask_pairs, scan_folder, seeds_for
from rspipe.runner import PipelineRunner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--max-sets", type=int, default=0, help="override dataset.max_chunks")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first two commands and exit")
    ap.add_argument("--quiet", action="store_true", help="only per-set lines")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    if args.max_sets:
        cfg.dataset.max_chunks = args.max_sets

    scan = scan_folder(cfg.dataset)
    print(scan.summary())
    chunks = make_chunks(scan, cfg.dataset, cfg.chain)
    print(f"{len(chunks)} image sets, chaining={cfg.chain.mode}, "
          f"close_loop={cfg.chain.close_loop}, "
          f"merge={'on' if cfg.merge.enabled else 'off'}")

    def describe(c) -> str:
        extra = f", {len(c.loop_images)} loop closure" if c.loop_images else ""
        return (f"  {c.number}: {c.name} idx {c.index_from}-{c.index_to} "
                f"{c.n_images} images ({len(c.new_images)} new, "
                f"{len(c.overlap_images)} carried over{extra})")

    for c in chunks[:4]:
        print(describe(c))
    if len(chunks) > 5:
        print(f"  ... {len(chunks) - 5} more")
    if len(chunks) > 4:
        print(describe(chunks[-1]))
    if not chunks:
        return 1

    if args.dry_run:
        first_c = chunk_paths(chunks[0], cfg).component
        first_n = chunks[0].name
        show = chunks[:2] + ([chunks[-1]] if len(chunks) > 2 else [])
        for c in show:
            paths = chunk_paths(c, cfg)
            prev_c, prev_n = None, None
            if c.number > 0:
                prev = chunks[c.number - 1]
                prev_c, prev_n = chunk_paths(prev, cfg).component, prev.name
            seeds = seeds_for(c, parse_seed_spec(cfg.seed.spec)) \
                if cfg.seed.enabled else []
            cmd = build_command(c, cfg, paths, prev_c, prev_n,
                                has_masks=bool(mask_pairs(c.images, cfg.masks)),
                                first_component=first_c, first_component_name=first_n,
                                seeds=seeds)
            print(f"\nREM ===== {c.name}\n{command_as_batch(cmd)}")
        return 0

    def emit(event: str, payload: dict) -> None:
        if event == "log":
            if not args.quiet:
                print(payload["line"])
        elif event == "stat":
            s = payload["stat"]
            print(f"    [{s.phase}] {s.name}: {s.cams} cams, {s.pts} pts, "
                  f"mean {s.mean:.3f} px")
        elif event == "step_start":
            r = payload["result"]
            seed_note = " [seed: " + ", ".join(r.seeds) + "]" if r.seeds else ""
            print(f"=== {r.name}: {r.n_images} images"
                  f"{' [chained]' if r.chained else ''}"
                  f"{f' [loop closed: {r.n_loop} images]' if r.looped else ''}"
                  f"{seed_note}"
                  f"{f', {r.n_masks} masks' if r.n_masks else ''}", flush=True)
        elif event == "step_end":
            r = payload["result"]
            b = r.best
            reg = f"{b.cams}/{r.n_images}" if b and r.n_images else (b.cams if b else "-")
            print(f"--- {r.name}: {r.status} registered={reg} "
                  f"({r.seconds:.0f}s) {r.message}", flush=True)
        elif event == "pipeline_end":
            print(f"finished {payload['ok']}/{payload['total']} in {payload['seconds']:.0f}s")

    runner = PipelineRunner(cfg, chunks, emit)
    runner.start()
    while runner.running:
        runner.join(0.5)
    return 0 if all(r.status in ("ok", "skipped") for r in runner.results) else 2


if __name__ == "__main__":
    sys.exit(main())
