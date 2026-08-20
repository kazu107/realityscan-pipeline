"""Show exactly what resuming a stopped run would do, without running anything.

A resume leans on two things: the outputs already on disk, and what the earlier
run recorded about them. This walks a preset's sets and reports, per set,
whether it would be skipped, whether its measurements can be carried into the
new summary, and why anything is being redone.

    python test_resume.py <preset.json>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.cli import chunk_paths  # noqa: E402
from rspipe.config import PipelineConfig  # noqa: E402
from rspipe.dataset import make_chunks, scan_folder  # noqa: E402
from rspipe.runner import PipelineRunner  # noqa: E402


def main() -> int:
    cfg = PipelineConfig.load(Path(sys.argv[1]))
    scan = scan_folder(cfg.dataset)
    chunks = make_chunks(scan, cfg.dataset, cfg.chain)
    lines: list[str] = []
    runner = PipelineRunner(cfg, chunks, lambda e, p: lines.append(str(p)))
    out_root = Path(cfg.export.out_root)
    runner._load_previous(out_root)

    print(f"{len(chunks)} sets, out_root {out_root}")
    print(f"records carried over from summary.json: {len(runner._previous)}")
    skip = redo = restored = 0
    first_redo = None
    for c in chunks:
        paths = chunk_paths(c, cfg)
        expected = paths.expected_outputs(cfg)
        present = bool(expected) and all(p.exists() for p in expected)
        usable = present and runner._reusable(expected)
        if usable:
            skip += 1
            if c.name in runner._previous:
                restored += 1
            else:
                print(f"  {c.name}: skipped, but no record to restore")
        else:
            redo += 1
            if first_redo is None:
                first_redo = c.name
                missing = [p.name for p in expected if not p.exists()]
                print(f"  {c.name}: will run"
                      + (f" (missing {', '.join(missing)})" if missing else
                         " (outputs present but too small)"))
    print(f"\nskip {skip}, of which {restored} keep their measurements")
    print(f"run  {redo}, starting at {first_redo}")
    for ln in lines:
        print(f"  [emit] {ln}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
