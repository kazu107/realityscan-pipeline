"""Align a stand-alone set across a weak set-to-set junction.

Consecutive sets share only their overlap indices, and all views of one index
sit at the same rig centre, so that shared stretch is a handful of points on a
near-straight line. Where a set's solve is a little off at its own end, the two
sets disagree about that stretch and the transform between them is badly
determined - the worst junction in 1-mid-2 was out by 24 index steps.

A patch fixes it: one alignment centred on the boundary, overlapping each
neighbour by about fifteen indices instead of five. Chaining A -> patch -> B
then replaces the direct A -> B fit.

    python align_patches.py <preset> <out dir> <index> [<index> ...]

Each index is the first index of the later set (the boundary); the patch is
centred there.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.cli import (ChunkPaths, build_command, command_as_batch,  # noqa: E402
                        write_mask_rscmd)
from rspipe.config import PipelineConfig  # noqa: E402
from rspipe.dataset import (Chunk, mask_pairs, scan_folder,  # noqa: E402
                            write_imagelist)

STAT = re.compile(r"RSSTAT\|phase=(\w+)\|name=([^|]+)\|cams=(\d+)\|pts=(\d+)")
HALF = 12                        # indices either side of the boundary


def patch_chunk(scan, cfg, prefix: str, lo: int, hi: int, n: int) -> Chunk:
    by_index: dict[int, list[Path]] = {}
    for e in sorted(scan.entries, key=lambda x: (x.index, x.view)):
        if lo <= e.index <= hi:
            by_index.setdefault(e.index, []).append(e.path)
    idx = sorted(by_index)
    imgs = [p for i in idx for p in by_index[i]]
    return Chunk(number=n, name=f"{prefix}_patch_{idx[0]:04d}-{idx[-1]:04d}",
                 prefix=prefix, index_from=idx[0], index_to=idx[-1],
                 overlap_indices=[], overlap_images=[], new_images=imgs)


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    cfg = PipelineConfig.load(Path(sys.argv[1]))
    # RealityScan resolves paths against its own working directory, so every
    # path handed to it - imagelists included - has to be absolute.
    out = Path(sys.argv[2]).resolve()
    bounds = [int(a) for a in sys.argv[3:]]

    scan = scan_folder(cfg.dataset)
    prefix = scan.prefixes[0] if scan.prefixes else "set"
    cfg = replace(cfg, export=replace(cfg.export, project=False))

    for n, b in enumerate(bounds):
        chunk = patch_chunk(scan, cfg, prefix, b - HALF, b + HALF, n)
        root = out / chunk.name
        root.mkdir(parents=True, exist_ok=True)
        paths = ChunkPaths(
            root=root,
            imagelist_all=root / f"{chunk.name}_all.imagelist",
            imagelist_new=root / f"{chunk.name}_new.imagelist",
            masks_rscmd=root / f"{chunk.name}_masks.rscmd",
            sparse=root / f"{chunk.name}_sparse{cfg.export.sparse_ext}",
            registration=root / "cameras.csv",
            component=root / f"{chunk.name}.rsalign",
            project=root / f"{chunk.name}.rsproj",
            log=root / f"{chunk.name}.log",
            crash=root / "crash",
        )
        paths.crash.mkdir(parents=True, exist_ok=True)
        if paths.registration.exists():
            print(f"=== {chunk.name}: already done", flush=True)
            continue

        write_imagelist(chunk.images, paths.imagelist_all)
        write_imagelist(chunk.new_images, paths.imagelist_new)
        pairs = mask_pairs(chunk.images, cfg.masks)
        if pairs:
            write_mask_rscmd(pairs, cfg.masks.usage, paths.masks_rscmd)

        args = build_command(chunk, cfg, paths, None, None, has_masks=bool(pairs))
        (root / "command.bat").write_text(command_as_batch(args), encoding="utf-8")
        print(f"=== {chunk.name}: {chunk.n_images} images "
              f"(indices {chunk.index_from}-{chunk.index_to})", flush=True)
        t0 = time.time()
        with open(paths.log, "w", encoding="utf-8", errors="replace") as log:
            p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:
                log.write(line)
                m = STAT.search(line)
                if m:
                    print(f"    [{m.group(1)}] {m.group(2)}: {m.group(3)} cams, "
                          f"{m.group(4)} pts", flush=True)
                elif "Processing failed" in line:
                    print("    " + line.rstrip(), flush=True)
            p.wait()
        print(f"--- {chunk.name}: exit {p.returncode} in {int(time.time() - t0)}s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
