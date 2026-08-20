"""Re-run one chained set without the overlap pose lock.

The lock freezes the carried-over cameras at the previous set's poses. If the
new images then solve at a different scale, the mismatch has nowhere to go and
lands as a step at the first new index. Running the same set with the lock
released lets the whole span adjust together, so comparing the two step profiles
says whether the lock is what creates the step.

    python test_lock_off.py <set name> [<set name> ...]
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.cli import (build_command, chunk_paths, command_as_batch,  # noqa: E402
                        write_mask_rscmd)
from rspipe.config import PipelineConfig, parse_seed_spec  # noqa: E402
from rspipe.dataset import (make_chunks, mask_pairs, scan_folder,  # noqa: E402
                            seeds_for, write_imagelist)

PRESET = Path(r"K:\realityscan\presets\1-mid-2.json")
STAT = re.compile(r"RSSTAT\|phase=(\w+)\|name=([^|]+)\|cams=(\d+)\|pts=(\d+)")


def main() -> int:
    wanted = sys.argv[1:]
    if not wanted:
        print(__doc__)
        return 2
    cfg = PipelineConfig.load(PRESET)
    scan = scan_folder(cfg.dataset)
    chunks = make_chunks(scan, cfg.dataset, cfg.chain)
    by_name = {c.name: (i, c) for i, c in enumerate(chunks)}

    for name in wanted:
        if name not in by_name:
            print(f"unknown set {name}")
            return 2
        i, chunk = by_name[name]
        prev = chunk_paths(chunks[i - 1], cfg).component if i else None
        prev_name = chunks[i - 1].name if i else None

        # keep the same tree depth: a project stores its image paths relative
        run = replace(cfg, export=replace(cfg.export, project=False))
        paths = chunk_paths(chunk, run)
        paths = replace(paths, root=paths.root.parent / f"_nolock_{name}")
        for f in ("imagelist_all", "imagelist_new", "masks_rscmd", "sparse",
                  "registration", "component", "project", "log", "crash"):
            p = getattr(paths, f, None)
            if isinstance(p, Path):
                paths = replace(paths, **{f: paths.root / p.name})
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.crash.mkdir(parents=True, exist_ok=True)

        write_imagelist(chunk.images, paths.imagelist_all)
        write_imagelist(chunk.new_images, paths.imagelist_new)
        pairs = mask_pairs(chunk.images, run.masks)
        if pairs:
            write_mask_rscmd(pairs, run.masks.usage, paths.masks_rscmd)
        seeds = seeds_for(chunk, parse_seed_spec(run.seed.spec)) \
            if run.seed.enabled else []

        args = build_command(chunk, run, paths, prev, prev_name,
                             has_masks=bool(pairs), seeds=seeds,
                             lock_overlap=False)
        (paths.root / "command.bat").write_text(command_as_batch(args),
                                                encoding="utf-8")
        print(f"=== {name} without the pose lock -> {paths.root}", flush=True)
        t0 = time.time()
        with open(paths.log, "w", encoding="utf-8", errors="replace") as log:
            p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:
                log.write(line)
                m = STAT.search(line)
                if m:
                    print(f"  [{m.group(1)}] {m.group(2)}: {m.group(3)} cams, "
                          f"{m.group(4)} pts", flush=True)
                elif "Processing failed" in line:
                    print("  " + line.rstrip(), flush=True)
            p.wait()
        print(f"--- exit {p.returncode} in {int(time.time() - t0)}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
