"""Drop views that sit away from the rest of their frame.

Every view of a frame is one physical position, so a view far from its frame's
others is wrong, and saying so needs no model of the scene - just the spread.

It matters because the failure is concentrated. On the repaired 904-frame
1-mid-1 model the spread was 0.03-0.12 index-steps almost everywhere and 16 to
125 steps across frames 59-69, where cam00, cam01 and cam02 had drifted off
together. That is 53 images of 6,213 - 0.85% - and they were the whole
difference between a trajectory with eleven breaks over 3x the median step and
one with none:

    against RealityScan     residual median    step p90    breaks
    as the mapper left it              5.92        9.87        11
    with those 53 dropped              0.04        1.12         0
    RealityScan itself                 0.00        1.12         0

The threshold is in index-steps because that is the only scale the model comes
with; a step is the distance between neighbouring frames.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import read_model

FRAME = re.compile(r"_(?P<frame>\d+)\.\w+$")


@dataclass
class CleanResult:
    dropped: list[str] = field(default_factory=list)
    frames_touched: list[int] = field(default_factory=list)
    step: float = 0.0
    messages: list[str] = field(default_factory=list)


def find_stray(model_dir: Path, threshold: float = 1.0) -> CleanResult:
    """Name the views more than ``threshold`` index-steps off their frame."""
    m = read_model(Path(model_dir))
    by_frame: dict[int, list] = {}
    for im in m.images.values():
        _, _, stem = im.name.replace("\\", "/").rpartition("/")
        if (mt := FRAME.search(stem)):
            by_frame.setdefault(int(mt.group("frame")), []).append(im)

    res = CleanResult()
    if len(by_frame) < 3:
        res.messages.append("too few frames to measure a step")
        return res

    frames = sorted(by_frame)
    # the median is the right centre here: it survives the very outliers we
    # are looking for, where a mean would be dragged towards them
    centre = {f: np.median([im.centre for im in by_frame[f]], axis=0)
              for f in frames}
    res.step = float(np.median(np.linalg.norm(
        np.diff(np.array([centre[f] for f in frames]), axis=0), axis=1)))
    if res.step <= 0:
        res.messages.append("the frames have no measurable step")
        return res

    hit: set[int] = set()
    for f in frames:
        for im in by_frame[f]:
            if np.linalg.norm(im.centre - centre[f]) / res.step > threshold:
                res.dropped.append(im.name)
                hit.add(f)
    res.frames_touched = sorted(hit)
    res.messages.append(
        f"{len(res.dropped)} of {len(m.images)} views over {threshold} "
        f"index-steps from their frame "
        f"({100 * len(res.dropped) / max(len(m.images), 1):.2f}%), "
        f"across {len(hit)} frames")
    return res


def clean_model(exe: str, model_dir: Path, out_dir: Path, list_path: Path,
                threshold: float = 1.0) -> CleanResult:
    """Find the stray views and write a model without them."""
    res = find_stray(model_dir, threshold)
    if not res.dropped:
        res.messages.append("nothing to drop")
        return res
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(res.dropped) + "\n", encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        [exe, "image_deleter",
         "--input_path", str(Path(model_dir).resolve()),
         "--output_path", str(Path(out_dir).resolve()),
         "--image_names_path", str(Path(list_path).resolve())],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        res.messages.append(f"image_deleter failed: {p.stdout[-400:]}")
    else:
        res.messages.append(f"cleaned model written to {out_dir.name}")
    return res
