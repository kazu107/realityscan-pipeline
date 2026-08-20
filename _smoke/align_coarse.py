"""Align the whole capture at a coarse index stride, as one set.

Every estimate of the global shape so far has come from chaining 67 handoffs,
and a chain cannot check itself: the reference and the offline stitch drift
one way, the locked variant drifts the other, and nothing arbitrates. Taking
every Nth index and aligning that in a single pass gives a reconstruction with
no handoffs in it at all, so its step profile is the arbiter.

It doubles as a component that spans the entire capture, which is exactly what
-mergeComponents lacks when consecutive sets share only five near-collinear
indices.

    python align_coarse.py <preset> <out dir> [stride] [--views 0,2,4,6]

Neighbouring sampled positions are `stride` indices apart, so the camera moves
that much further between them - too coarse and the alignment falls apart. 10
was the intended starting point for 1-mid-2 (1332 indices -> 133 positions).
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


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    argv = sys.argv[1:]
    views = None
    if "--views" in argv:
        k = argv.index("--views")
        views = {int(v) for v in argv[k + 1].split(",")}
        del argv[k:k + 2]
    cfg = PipelineConfig.load(Path(argv[0]))
    out = Path(argv[1]).resolve()
    stride = int(argv[2]) if len(argv) > 2 else 10

    scan = scan_folder(cfg.dataset)
    prefix = scan.prefixes[0] if scan.prefixes else "set"
    by_index: dict[int, list[Path]] = {}
    for e in sorted(scan.entries, key=lambda x: (x.index, x.view)):
        if views is not None and e.view not in views:
            continue
        by_index.setdefault(e.index, []).append(e.path)
    idx = [i for i in sorted(by_index) if i % stride == 0]
    imgs = [p for i in idx for p in by_index[i]]

    name = f"{prefix}_coarse{stride}"
    chunk = Chunk(number=0, name=name, prefix=prefix,
                  index_from=idx[0], index_to=idx[-1],
                  overlap_indices=[], overlap_images=[], new_images=imgs)
    root = out / name
    root.mkdir(parents=True, exist_ok=True)
    paths = ChunkPaths(
        root=root,
        imagelist_all=root / f"{name}_all.imagelist",
        imagelist_new=root / f"{name}_new.imagelist",
        masks_rscmd=root / f"{name}_masks.rscmd",
        sparse=root / f"{name}_sparse{cfg.export.sparse_ext}",
        registration=root / "cameras.csv",
        component=root / f"{name}.rsalign",
        project=root / f"{name}.rsproj",
        log=root / f"{name}.log",
        crash=root / "crash",
    )
    paths.crash.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, export=replace(cfg.export, project=False))

    write_imagelist(chunk.images, paths.imagelist_all)
    write_imagelist(chunk.new_images, paths.imagelist_new)
    pairs = mask_pairs(chunk.images, cfg.masks)
    if pairs:
        write_mask_rscmd(pairs, cfg.masks.usage, paths.masks_rscmd)

    args = build_command(chunk, cfg, paths, None, None, has_masks=bool(pairs))
    (root / "command.bat").write_text(command_as_batch(args), encoding="utf-8")
    print(f"=== {name}: {len(idx)} indices every {stride}, "
          f"{len(imgs)} images"
          + (f", views {sorted(views)}" if views else ""), flush=True)

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
    print(f"--- {name}: exit {p.returncode} in {int(time.time() - t0)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
