"""Report on a finished COLMAP rig model: coverage, rig rigidity, trajectory.

The questions that matter for a walked 360 capture are not the ones COLMAP
prints. It says how many images it registered; it does not say whether the
walk came out as one continuous path, which is the failure that wrecked the
RealityScan runs. So: what got left out, whether the rig actually stayed
rigid, and where the per-frame step jumps.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colpipe.model import read_model                      # noqa: E402


def ground_plane(pts: np.ndarray) -> np.ndarray:
    """Project onto the plane the walk actually lies in, not onto x-y."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return (pts - c) @ vt[:2].T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--expect-frames", type=int, default=0)
    ap.add_argument("--expect-views", type=int, default=0)
    ap.add_argument("--break-factor", type=float, default=3.0)
    args = ap.parse_args()

    m = read_model(Path(args.model))
    print(f"model     {args.model}")
    print(f"          {m.summary()}")

    # ---- cameras -------------------------------------------------------
    print("\n=== sensors ===")
    for cid, cam in sorted(m.cameras.items()):
        print(f"  cam {cid}: {cam.model} {cam.width}x{cam.height} "
              f"{' '.join(f'{p:.2f}' for p in cam.params)}")

    # ---- group by frame ------------------------------------------------
    rx = re.compile(r"_(?P<frame>\d+)\.\w+$")
    by_frame: dict[int, list[np.ndarray]] = {}
    by_view: dict[str, set[int]] = {}
    for im in m.images.values():
        folder, _, stem = im.name.replace("\\", "/").rpartition("/")
        mt = rx.search(stem)
        if not mt:
            continue
        f = int(mt.group("frame"))
        by_frame.setdefault(f, []).append(im.centre)
        by_view.setdefault(folder, set()).add(f)

    views = sorted(by_view)
    frames = sorted(by_frame)
    print(f"\n=== coverage ===")
    print(f"  views     {len(views)}: {', '.join(views)}")
    print(f"  frames    {len(frames)}  ({frames[0]} .. {frames[-1]})")
    if args.expect_frames:
        missing = args.expect_frames - len(frames)
        print(f"  expected  {args.expect_frames} -> {missing} missing "
              f"({100 * missing / args.expect_frames:.1f}%)")
    nv = args.expect_views or len(views)
    partial = [f for f in frames if len(by_frame[f]) < nv]
    print(f"  complete  {len(frames) - len(partial)} frames have all {nv} views, "
          f"{len(partial)} partial")
    if partial[:20]:
        print(f"  partial   {partial[:20]}{' ...' if len(partial) > 20 else ''}")
    for v in views:
        miss = sorted(set(frames) - by_view[v])
        if miss:
            print(f"    {v}: missing {len(miss)} -> {miss[:12]}"
                  f"{' ...' if len(miss) > 12 else ''}")

    # ---- rig rigidity --------------------------------------------------
    # every view of one frame is one physical position, so the spread of the
    # centres inside a frame is pure error - and with the rig held rigid it
    # should be zero to float precision
    spread = np.array([np.ptp(np.array(by_frame[f]), axis=0).max()
                       for f in frames if len(by_frame[f]) > 1])
    print(f"\n=== rig rigidity ===")
    print(f"  intra-frame centre spread: median {np.median(spread):.3e}, "
          f"max {spread.max():.3e}")

    # ---- trajectory ----------------------------------------------------
    pos = np.array([np.array(by_frame[f]).mean(0) for f in frames])
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    gap = np.diff(frames)                    # frames skipped between entries
    norm = step / np.maximum(gap, 1)         # step per frame index
    med = np.median(norm)
    print(f"\n=== trajectory ===")
    print(f"  frame-to-frame step: median {med:.4f}, "
          f"p90 {np.percentile(norm, 90):.4f}, max {norm.max():.4f}")
    breaks = [(frames[i], frames[i + 1], norm[i] / med)
              for i in range(len(norm)) if norm[i] > args.break_factor * med]
    print(f"  breaks over {args.break_factor}x median: {len(breaks)}")
    for a, b, r in sorted(breaks, key=lambda t: -t[2])[:20]:
        print(f"    {a:>5} -> {b:<5}  {r:6.2f}x")

    xy = ground_plane(pos)
    print(f"  extent in the walk plane: "
          f"{np.ptp(xy[:, 0]):.2f} x {np.ptp(xy[:, 1]):.2f}")

    # ---- reprojection --------------------------------------------------
    if len(m.errors):
        e = m.errors
        print(f"\n=== reprojection error ===")
        print(f"  mean {e.mean():.4f} px, median {np.median(e):.4f}, "
              f"p99 {np.percentile(e, 99):.4f}, max {e.max():.4f}")
    obs = np.array([im.num_points for im in m.images.values()])
    print(f"  keypoints per image: median {np.median(obs):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
