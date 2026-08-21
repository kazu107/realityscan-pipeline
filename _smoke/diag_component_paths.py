"""Show how a .rsalign records the paths to its images.

Relocating a component turned out to fail with

    Could not find file "..\\..\\..\\K:\\data\\jpeg\\1-low\\1-low_0242_04.jpg"

which is a relative prefix glued to an absolute path on another drive - a form
that cannot resolve from anywhere. Reading the stored strings back shows what
RealityScan actually wrote, and therefore whether a component can be moved at
all or has to be produced in the right place to begin with.

    python diag_component_paths.py <component.rsalign> [--mb 40]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

PATH_RE = re.compile(r"[A-Za-z0-9_.: \\/-]{0,80}\.jpg", re.IGNORECASE)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    comp = Path(sys.argv[1])
    mb = 40
    if "--mb" in sys.argv:
        mb = int(sys.argv[sys.argv.index("--mb") + 1])

    buf = comp.read_bytes()[: mb * 2 ** 20]
    text = buf.decode("utf-16-le", errors="ignore")
    hits = PATH_RE.findall(text)
    if not hits:
        text = buf.decode("latin-1", errors="ignore")
        hits = PATH_RE.findall(text)
    print(f"{comp} ({comp.stat().st_size / 2**30:.1f} GB), "
          f"scanned the first {mb} MB")
    print(f"{len(hits)} path-like strings, {len(set(hits))} distinct\n")

    prefixes = Counter()
    for h in set(hits):
        head = h.rsplit("\\", 1)[0] if "\\" in h else ""
        prefixes[head] += 1
    print("directories referenced:")
    for head, n in prefixes.most_common(8):
        print(f"  {n:>6}  {head!r}")

    rel = [h for h in set(hits) if h.startswith("..")]
    absol = [h for h in set(hits) if re.match(r"^[A-Za-z]:", h)]
    print(f"\nrelative form: {len(rel)}, absolute form: {len(absol)}")
    if rel:
        print("  example relative:", rel[0])
        depth = rel[0].count("..")
        print(f"  climbs {depth} level(s); what follows is "
              f"{'ABSOLUTE - cannot resolve from any directory' if re.search(r'[A-Za-z]:', rel[0]) else 'relative'}")
    if absol:
        print("  example absolute:", absol[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
