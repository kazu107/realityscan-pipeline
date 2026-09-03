"""Write the image pairs the sequential matcher cannot see.

The sequential matcher links frames by index. That covers a walk while it goes
forward and nothing else, which leaves two holes measured on 1-mid-1:

* the loop never closes. The walk returns to its start, but index 903 is 869
  steps from index 34, so no pair was ever generated between them - zero points
  tied frame <60 to frame >860, and the last 26 frames ran away into a volume
  larger than the other 839 put together.
* the views inside one frame never meet. At 100 degrees of field on 45 degree
  spacing adjacent views overlap by 55 degrees, the largest overlap anywhere in
  the set, and skipping those pairs leaves each view's track chain separate
  from its neighbour's. Median track span came out at 3 frames, p90 at 10, and
  no point at all spanned more than 500 frames - so scale drifted freely along
  an 865-frame chain.

Both are cheap to repair because the features are already in the database:
``matches_importer --match_type pairs`` matches and verifies exactly the pairs
listed and writes them alongside the existing ones. At the rate this dataset
matched (35 ms a pair) the two lists below cost minutes, not the five hours the
full pass took.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

NAME = re.compile(r"^(?P<view>.+)/(?P<stem>.*?_(?P<frame>\d+))\.\w+$")


@dataclass
class PairPlan:
    path: Path
    loop: int = 0
    same_frame: int = 0
    dense: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.loop + self.same_frame + self.dense


def read_layout(database: Path) -> dict[str, dict[int, str]]:
    """view folder -> frame -> image name, as COLMAP has them recorded.

    Taken from the database rather than the disk so the names match byte for
    byte; ``matches_importer`` looks each one up and a near-miss is silently a
    missing pair.
    """
    out: dict[str, dict[int, str]] = {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        for (name,) in db.execute("select name from images"):
            mt = NAME.match(name.replace("\\", "/"))
            if mt:
                out.setdefault(mt.group("view"), {})[int(mt.group("frame"))] = name
    return out


def _sep(a: int, b: int, n: int) -> int:
    """Separation between two view indices on a ring of n."""
    d = abs(a - b) % n
    return min(d, n - d)


def write_pairs(database: Path, out_path: Path, *, loop_window: int = 40,
                same_frame: bool = True, max_view_sep: int = 2,
                loop_max_view_sep: int = 0, dense_window: int = 0,
                dense_max_view_sep: int = 1,
                dense_skip_offsets: "set[int] | tuple[int, ...]" = ()) -> PairPlan:
    """Write the pair list, and say what went into it.

    ``loop_window`` pairs the last N frames against the first N. ``max_view_sep``
    limits the same-frame pairs by ring separation - 1 is the 45 degree
    neighbour, 2 the 90 degree one, and past that a 100 degree field does not
    overlap at all, so pairing them is pure cost. ``loop_max_view_sep`` of 0
    means every view combination at the seam, which is the honest default: the
    walk can come back on any heading, and which views face the same way is not
    known in advance.
    """
    layout = read_layout(Path(database))
    plan = PairPlan(path=Path(out_path))
    if not layout:
        plan.messages.append("no images in the database, or none matched the "
                             "cam<NN>/<prefix>_<frame> layout")
        return plan

    views = sorted(layout)
    nv = len(views)
    frames = sorted({f for v in layout.values() for f in v})
    plan.messages.append(f"{len(frames)} frames x {nv} views in the database "
                         f"({frames[0]}..{frames[-1]})")

    lines: list[str] = []

    # ---- the loop seam -------------------------------------------------
    if loop_window > 0:
        head = frames[:loop_window]
        tail = frames[-loop_window:]
        if set(head) & set(tail):
            plan.messages.append(
                f"loop window {loop_window} covers more than half of "
                f"{len(frames)} frames, so head and tail overlap - skipped")
        else:
            for fa in tail:
                for fb in head:
                    for ia, va in enumerate(views):
                        na = layout[va].get(fa)
                        if not na:
                            continue
                        for ib, vb in enumerate(views):
                            if loop_max_view_sep and \
                                    _sep(ia, ib, nv) > loop_max_view_sep:
                                continue
                            nb = layout[vb].get(fb)
                            if nb:
                                lines.append(f"{na} {nb}")
                                plan.loop += 1
            plan.messages.append(
                f"loop seam: frames {tail[0]}..{tail[-1]} against "
                f"{head[0]}..{head[-1]}, {plan.loop} pairs")

    # ---- inside one frame ----------------------------------------------
    if same_frame and max_view_sep > 0:
        for f in frames:
            for ia in range(nv):
                na = layout[views[ia]].get(f)
                if not na:
                    continue
                for ib in range(ia + 1, nv):
                    if _sep(ia, ib, nv) > max_view_sep:
                        continue
                    nb = layout[views[ib]].get(f)
                    if nb:
                        lines.append(f"{na} {nb}")
                        plan.same_frame += 1
        plan.messages.append(
            f"inside one frame: view separation up to {max_view_sep} "
            f"({45 * max_view_sep} deg on an 8-view ring), "
            f"{plan.same_frame} pairs")

    # ---- the offsets the quadratic matcher never generated -------------
    if dense_window > 0:
        skip = set(dense_skip_offsets)
        added = sorted(d for d in range(1, dense_window + 1) if d not in skip)
        for f in frames:
            for d in added:
                g = f + d
                for ia, va in enumerate(views):
                    na = layout[va].get(f)
                    if not na:
                        continue
                    for ib, vb in enumerate(views):
                        if _sep(ia, ib, nv) > dense_max_view_sep:
                            continue
                        nb = layout[vb].get(g)
                        if nb:
                            lines.append(f"{na} {nb}")
                            plan.dense += 1
        plan.messages.append(
            f"dense local window: offsets {added} (skipping "
            f"{sorted(skip) or 'nothing'} as already matched), view "
            f"separation up to {dense_max_view_sep}, {plan.dense} pairs")

    plan.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.write_text("\n".join(lines) + ("\n" if lines else ""),
                         encoding="utf-8")
    plan.messages.append(f"{plan.total} pairs written to {plan.path.name}")
    return plan
