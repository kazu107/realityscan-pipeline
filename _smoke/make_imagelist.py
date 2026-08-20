"""Build a .imagelist for the smoke test: N consecutive frames x all views."""
import re
import sys
from pathlib import Path

SRC = Path(r"K:\data\jpeg\1-mid-1")
OUT = Path(__file__).with_name("test.imagelist")

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
count = int(sys.argv[2]) if len(sys.argv) > 2 else 10

pat = re.compile(r"^(?P<prefix>.+)_(?P<frame>\d+)_(?P<view>\d+)$")
rows = []
for p in sorted(SRC.glob("*.jpg")):
    m = pat.match(p.stem)
    if not m:
        continue
    f = int(m.group("frame"))
    if start <= f < start + count:
        rows.append((f, int(m.group("view")), p))

rows.sort()
OUT.write_text("\n".join(str(p) for _, _, p in rows) + "\n", encoding="utf-8")
print(f"{len(rows)} images -> {OUT}")
