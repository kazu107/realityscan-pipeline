"""Write the reconstruction as a flat COLMAP dataset - images/, masks/, sparse/0.

The layout this produces is the one K:\\data\\col is in, which is what the
gaussian-splatting side of the work reads:

    images/00000.jpg  00001.jpg  ...     flat, five digits, sequential
    masks/00000.jpg   00002.jpg  ...     the same name as the image it masks
    sparse/0/cameras.bin images.bin points3D.bin

Three things differ from what the pipeline keeps internally and have to be
converted rather than copied:

*   names. The workspace stores ``camNN/<prefix>_<frame>.jpg``, because that is
    what makes COLMAP group a frame and give each direction its own camera.
    Here they are renumbered in (frame, view) order, so the numbering follows
    the walk.

*   masks. COLMAP wants ``masks/<image name>.png`` and reads black as "ignore";
    this layout wants ``masks/<image name>`` with the image's own extension. The
    source PNGs are re-encoded as JPEG to match.

*   the untriangulated keypoints. The workspace model carries every detected
    feature - about 25,000 an image, which is why its images.bin is 4.2 GB
    against the 540 MB of the reference dataset. Only the observations that
    became 3D points mean anything downstream, so the rest are dropped and the
    point tracks are renumbered to match. Pass ``keep_all_keypoints`` to skip
    that if the model is going back into COLMAP for re-triangulation.
"""

from __future__ import annotations

import re
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NAME = re.compile(r"^(?P<view>[^/]+)/(?P<stem>.+)\.(?P<ext>\w+)$")
FRAME = re.compile(r"_(?P<frame>\d+)$")


@dataclass
class FlattenResult:
    images: int = 0
    linked: int = 0
    copied: int = 0
    masks: int = 0
    masks_missing: int = 0
    points: int = 0
    observations: int = 0
    dropped_keypoints: int = 0
    messages: list[str] = field(default_factory=list)


def _read_images_header(f):
    """Yield (image_id, pose_bytes, camera_id, name, points2D_raw) in order."""
    (n,) = struct.unpack("<Q", f.read(8))
    for _ in range(n):
        head = f.read(64)                       # id, qvec, tvec, camera_id
        image_id = struct.unpack_from("<i", head, 0)[0]
        camera_id = struct.unpack_from("<i", head, 60)[0]
        name = bytearray()
        while (c := f.read(1)) != b"\x00":
            name += c
        (npts,) = struct.unpack("<Q", f.read(8))
        raw = f.read(npts * 24)                 # x, y (double), point3D_id (int64)
        yield image_id, head, camera_id, name.decode("utf-8"), raw


VIEW = re.compile(r"(\d+)\s*$")


def source_name(name: str) -> str:
    """The flat extraction file a workspace image came from.

    The layout splits ``<prefix>_<frame>_<view>.jpg`` into
    ``cam<view>/<prefix>_<frame>.jpg`` so that COLMAP sees one camera per
    folder and one frame per file name. The masks sit beside the originals and
    are named after them, so the view number has to be put back.
    """
    mt = NAME.match(name.replace("\\", "/"))
    if not mt:
        return name
    v = VIEW.search(mt.group("view"))
    if not v:
        return f"{mt.group('stem')}.{mt.group('ext')}"
    return f"{mt.group('stem')}_{v.group(1)}.{mt.group('ext')}"


PARAM_COUNT = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5,
               10: 12, 11: 5}


def _read_cameras(path: Path) -> dict[int, tuple[int, int, int, bytes]]:
    """camera_id -> (model_id, width, height, raw parameter bytes)."""
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            k = PARAM_COUNT.get(model_id, 4)
            out[cid] = (model_id, w, h, f.read(k * 8))
    return out


def _order_key(name: str) -> tuple:
    """(frame, view) so the numbering follows the walk, not the folder listing."""
    mt = NAME.match(name.replace("\\", "/"))
    if not mt:
        return (1 << 30, name)
    fr = FRAME.search(mt.group("stem"))
    return (int(fr.group("frame")) if fr else 1 << 30, mt.group("view"))


def flatten(model_dir: Path, out_dir: Path, image_root: Path, *,
            source_dir: Path | None = None,
            mask_pattern: str = "{name}.mask.png",
            mask_dir: str = "", write_masks: bool = True,
            mask_quality: int = 92, digits: int = 5,
            keep_all_keypoints: bool = False,
            per_image_cameras: bool = False) -> FlattenResult:
    """Write ``model_dir`` and its images out as a flat COLMAP dataset.

    ``image_root`` is the workspace's ``images`` folder, whose files are already
    hard links to the extraction; linking again costs nothing on the same
    volume. ``source_dir`` is where the original extraction lives, which is
    where the masks are - the workspace's own mask tree is COLMAP-shaped and
    partly filled with blank stand-ins, so the originals are the better source.
    """
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    image_root = Path(image_root)
    res = FlattenResult()

    img_out = out_dir / "images"
    mask_out = out_dir / "masks"
    sparse_out = out_dir / "sparse" / "0"
    for d in (img_out, sparse_out):
        d.mkdir(parents=True, exist_ok=True)
    if write_masks:
        mask_out.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: decide the new names ----------------------------------
    with open(model_dir / "images.bin", "rb") as f:
        entries = [(i, h, c, n, r) for i, h, c, n, r in _read_images_header(f)]
    entries.sort(key=lambda e: _order_key(e[3]))
    res.images = len(entries)
    res.messages.append(f"{len(entries)} registered images")

    new_name = {}
    for i, (image_id, *_rest) in enumerate(entries):
        new_name[image_id] = f"{i:0{digits}d}.jpg"

    # ---- images and masks ----------------------------------------------
    from PIL import Image

    for image_id, _h, _c, name, _r in entries:
        dst = img_out / new_name[image_id]
        src = image_root / name
        if not dst.exists():
            try:
                dst.hardlink_to(src)
                res.linked += 1
            except OSError:
                shutil.copy2(src, dst)
                res.copied += 1
        if not write_masks:
            continue
        base = source_dir if source_dir else image_root
        msrc = Path(mask_dir or base) / mask_pattern.format(
            name=source_name(name))
        mdst = mask_out / new_name[image_id]
        if mdst.exists():
            res.masks += 1
        elif msrc.is_file():
            with Image.open(msrc) as im:
                im.convert("L").save(mdst, "JPEG", quality=mask_quality)
            res.masks += 1
        else:
            res.masks_missing += 1
    res.messages.append(f"images: {res.linked} linked, {res.copied} copied")
    if write_masks:
        res.messages.append(f"masks: {res.masks} written, "
                            f"{res.masks_missing} had no source")

    # ---- cameras ---------------------------------------------------------
    # Sharing one camera per direction is the correct description of this rig
    # and every COLMAP reader handles it, so it is the default. K:\data\col
    # has one camera per image instead, because it came out of RealityScan's
    # per-image undistortion; per_image_cameras reproduces that for a reader
    # that assumes it.
    cam_of = {}
    if per_image_cameras:
        src_cams = _read_cameras(model_dir / "cameras.bin")
        rows = []
        for i, (image_id, head, cam_id, _n, _r) in enumerate(entries):
            new_id = i + 1
            cam_of[image_id] = new_id
            rows.append((new_id, src_cams[cam_id]))
        with open(sparse_out / "cameras.bin", "wb") as out:
            out.write(struct.pack("<Q", len(rows)))
            for new_id, (model_id, w, h, params) in rows:
                out.write(struct.pack("<iiQQ", new_id, model_id, w, h))
                out.write(params)
        res.messages.append(f"cameras: {len(rows)}, one per image")
    else:
        shutil.copyfile(model_dir / "cameras.bin", sparse_out / "cameras.bin")
        res.messages.append("cameras: copied unchanged, shared per direction")

    # ---- images.bin, with the untriangulated keypoints dropped ----------
    keep_idx: dict[int, np.ndarray] = {}
    with open(sparse_out / "images.bin", "wb") as out:
        out.write(struct.pack("<Q", len(entries)))
        for image_id, head, _cam, _name, raw in entries:
            npts = len(raw) // 24
            if keep_all_keypoints:
                kept_raw, nkeep = raw, npts
            else:
                ids = np.frombuffer(raw, dtype="<i8")[2::3] \
                    if npts else np.zeros(0, dtype="<i8")
                sel = np.flatnonzero(ids != -1)
                keep_idx[image_id] = sel
                res.dropped_keypoints += npts - len(sel)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(npts, 24) \
                    if npts else np.zeros((0, 24), np.uint8)
                kept_raw = arr[sel].tobytes()
                nkeep = len(sel)
            res.observations += nkeep
            if per_image_cameras:
                head = head[:60] + struct.pack("<i", cam_of[image_id])
            out.write(head)
            out.write(new_name[image_id].encode("utf-8") + b"\x00")
            out.write(struct.pack("<Q", nkeep))
            out.write(kept_raw)

    # ---- points3D.bin, with the tracks renumbered -----------------------
    with open(model_dir / "points3D.bin", "rb") as f, \
            open(sparse_out / "points3D.bin", "wb") as out:
        (n,) = struct.unpack("<Q", f.read(8))
        out.write(struct.pack("<Q", n))
        res.points = n
        for _ in range(n):
            body = f.read(43)                   # id, xyz, rgb, error
            (t,) = struct.unpack("<Q", f.read(8))
            track = f.read(t * 8)
            if keep_all_keypoints:
                out.write(body)
                out.write(struct.pack("<Q", t))
                out.write(track)
                continue
            pairs = np.frombuffer(track, dtype="<i4").reshape(-1, 2)
            fixed = bytearray()
            kept = 0
            for image_id, point2d_idx in pairs:
                sel = keep_idx.get(int(image_id))
                if sel is None:
                    continue
                pos = int(np.searchsorted(sel, point2d_idx))
                if pos >= len(sel) or sel[pos] != point2d_idx:
                    continue                    # observation was dropped
                fixed += struct.pack("<ii", int(image_id), pos)
                kept += 1
            out.write(body)
            out.write(struct.pack("<Q", kept))
            out.write(bytes(fixed))
    res.messages.append(
        f"sparse: {res.points} points, {res.observations} observations kept"
        + ("" if keep_all_keypoints
           else f", {res.dropped_keypoints} untriangulated keypoints dropped"))
    res.messages.append(f"written to {out_dir}")
    return res


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="a COLMAP model directory")
    ap.add_argument("out", help="where to write images/, masks/, sparse/0")
    ap.add_argument("--image-root", required=True,
                    help="the workspace's images/ folder")
    ap.add_argument("--source-dir", default="",
                    help="the flat extraction, where the masks are")
    ap.add_argument("--mask-pattern", default="{name}.mask.png")
    ap.add_argument("--mask-dir", default="")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--keep-all-keypoints", action="store_true")
    ap.add_argument("--per-image-cameras", action="store_true",
                    help="one camera per image, as K:\\data\\col has")
    a = ap.parse_args(argv)
    r = flatten(Path(a.model), Path(a.out), Path(a.image_root),
                source_dir=Path(a.source_dir) if a.source_dir else None,
                mask_pattern=a.mask_pattern, mask_dir=a.mask_dir,
                write_masks=not a.no_masks,
                keep_all_keypoints=a.keep_all_keypoints,
                per_image_cameras=a.per_image_cameras)
    for m in r.messages:
        print(" ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
