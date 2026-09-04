"""Lay the extracted views out the way COLMAP wants to see a rig.

The extractor writes everything into one folder as ``<name>_<frame>_<view>.jpg``.
COLMAP forms frames from *identical filenames across per-camera folders*, so it
needs::

    images/cam00/<name>_<frame>.jpg
    images/cam01/<name>_<frame>.jpg
    ...

which is the same 7,336 files under different names. Copying them would cost
3.4 GB and the time to write it, so they are hard-linked instead: on one NTFS
volume that is a directory entry and nothing else. Falls back to copying when
the link cannot be made (different volume, or a filesystem without links).

Masks go into a parallel tree of their own: COLMAP looks for them at the same
sub-path below ``--ImageReader.mask_path`` as the image has below
``--image_path``, named ``<image file name>.png``.
"""

from __future__ import annotations

import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATTERN = r"^(?P<prefix>.+)_(?P<frame>\d+)_(?P<view>\d+)$"


@dataclass
class LayoutResult:
    linked: int = 0
    copied: int = 0
    skipped: int = 0
    frames: int = 0
    masked: int = 0
    filled: int = 0
    views: list[int] = field(default_factory=list)
    incomplete: list[int] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.linked + self.copied + self.skipped


def scan_flat(image_dir: Path, pattern: str = DEFAULT_PATTERN,
              extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) \
        -> dict[int, dict[int, Path]]:
    """Group ``<name>_<frame>_<view>`` files by frame, then by view."""
    rx = re.compile(pattern)
    out: dict[int, dict[int, Path]] = defaultdict(dict)
    for p in sorted(Path(image_dir).iterdir()):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        m = rx.match(p.stem)
        if not m:
            continue
        out[int(m.group("frame"))][int(m.group("view"))] = p
    return dict(out)


def build(image_dir: Path, out_dir: Path, views: list[int] | None = None,
          pattern: str = DEFAULT_PATTERN, frame_from: int = 0,
          frame_to: int = -1, frame_step: int = 1,
          require_all_views: bool = True,
          mask_pattern: str = "", mask_dir: str = "",
          mask_out: Path | None = None,
          fill_missing_masks: bool = True,
          folder_prefix: str = "") -> LayoutResult:
    """Populate ``out_dir/<prefix>camNN/<name>_<frame>.jpg`` from a flat folder.

    ``folder_prefix`` keeps two captures apart in one workspace. Folder names
    are what COLMAP turns into cameras, so sets with different direction rings
    - 1-mid's eight against 1-low's ten - must not land in the same cam00.
    """
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    frames = scan_flat(image_dir, pattern)
    res = LayoutResult()
    if not frames:
        res.messages.append(f"no files matching {pattern} in {image_dir}")
        return res

    all_views = sorted({v for f in frames.values() for v in f})
    want = sorted(views) if views else all_views
    res.views = want

    keep = [f for f in sorted(frames)
            if f >= frame_from and (frame_to < 0 or f <= frame_to)
            and (f - frame_from) % max(1, frame_step) == 0]
    for v in want:
        (out_dir / f"{folder_prefix}cam{v:02d}").mkdir(
            parents=True, exist_ok=True)

    rx = re.compile(pattern)
    for f in keep:
        have = frames[f]
        if require_all_views and any(v not in have for v in want):
            res.incomplete.append(f)
            continue
        res.frames += 1
        for v in want:
            src = have.get(v)
            if src is None:
                continue
            m = rx.match(src.stem)
            stem = f"{m.group('prefix')}_{m.group('frame')}"
            dst = out_dir / f"{folder_prefix}cam{v:02d}" / f"{stem}{src.suffix}"
            _place(src, dst, res)
            if mask_pattern and mask_out is not None:
                msrc = _mask_for(src, mask_pattern, mask_dir)
                # COLMAP looks for the mask at the same sub-path below
                # mask_path as the image has below image_path, named
                # <image file name>.png - not beside the image
                mdst = mask_out / f"{folder_prefix}cam{v:02d}" \
                    / f"{stem}{src.suffix}.png"
                if msrc and msrc.is_file():
                    _place(msrc, mdst, res)
                    res.masked += 1
                elif fill_missing_masks:
                    # An image whose mask cannot be read is DROPPED by COLMAP,
                    # not merely unmasked. 1-mid-1 has masks for 70% of its
                    # images, and turning masks on without this left one camera
                    # folder empty and the rig unbuildable. A white mask means
                    # "use every pixel", so filling the gaps changes nothing
                    # except that the image survives.
                    _place(_white_mask(src, mask_out), mdst, res)
                    res.filled += 1
    if res.incomplete:
        res.messages.append(
            f"{len(res.incomplete)} frame(s) skipped for missing views, "
            f"first {res.incomplete[:5]}")
    return res


def _white_mask(image: Path, root: Path) -> Path:
    """One all-white PNG per image size, hard-linked wherever a mask is missing."""
    from PIL import Image as PILImage

    with PILImage.open(image) as im:
        size = im.size
    cache = root / f"_all_white_{size[0]}x{size[1]}.png"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        PILImage.new("L", size, 255).save(cache, optimize=True)
    return cache


def _mask_for(image: Path, pattern: str, mask_dir: str) -> Path | None:
    folder = Path(mask_dir) if mask_dir else image.parent
    name = pattern.format(name=image.name, stem=image.stem, ext=image.suffix)
    return folder / name


def _place(src: Path, dst: Path, res: LayoutResult) -> None:
    if dst.exists():
        res.skipped += 1
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        res.linked += 1
        return
    except OSError as exc:                                   # noqa: BLE001
        if not res.copied:
            res.messages.append(f"hard link unavailable ({exc.strerror}), copying")
    shutil.copy2(src, dst)
    res.copied += 1
