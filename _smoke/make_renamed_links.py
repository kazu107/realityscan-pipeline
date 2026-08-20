"""Give a capture a distinct file-name prefix without copying any data.

1-mid-2-1 and 1-mid-2-2 hold different renders under identical file names, so a
merged camera list cannot tell them apart. NTFS hard links on the same volume
let us present one of them under a new prefix at no storage cost - the existing
components keep pointing at the original paths and stay valid.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else r"K:\data\jpeg\1-mid-2-2")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else r"K:\data\jpeg\1-mid-2b")
OLD = sys.argv[3] if len(sys.argv) > 3 else "1-mid-2-repaired"
NEW = sys.argv[4] if len(sys.argv) > 4 else "1-mid-2b"

if SRC.resolve().drive.lower() != DST.resolve().drive.lower():
    raise SystemExit("hard links need both paths on the same volume")

DST.mkdir(parents=True, exist_ok=True)
made = skipped = 0
for p in sorted(SRC.iterdir()):
    if not p.is_file() or not p.name.startswith(OLD + "_"):
        continue
    target = DST / (NEW + p.name[len(OLD):])
    if target.exists():
        skipped += 1
        continue
    os.link(p, target)
    made += 1

jpg = len(list(DST.glob("*.jpg")))
png = len(list(DST.glob("*.mask.png")))
print(f"{SRC} -> {DST}")
print(f"  links created {made}, already present {skipped}")
print(f"  {jpg} jpg, {png} masks")
sample = next(iter(sorted(DST.glob('*.jpg'))), None)
if sample:
    st = sample.stat()
    print(f"  sample {sample.name}  {st.st_size:,} B  (hard link, no extra storage)")
