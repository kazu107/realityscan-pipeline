"""Scan an image folder and slice it into overlapping image sets (chunks).

File names are expected to look like ``<name>_<index>_<view>``; splitting and
overlap are always counted in **index** units, never in image counts, so every
view of an index stays together (all views of one index share a camera centre,
so they carry no baseline on their own).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ChainConfig, DatasetConfig, MaskConfig


@dataclass(frozen=True)
class ImageEntry:
    path: Path
    prefix: str
    index: int
    view: int


@dataclass
class Scan:
    entries: list[ImageEntry] = field(default_factory=list)
    unmatched: list[Path] = field(default_factory=list)

    @property
    def indices(self) -> list[int]:
        return sorted({e.index for e in self.entries})

    @property
    def views(self) -> list[int]:
        return sorted({e.view for e in self.entries})

    @property
    def prefixes(self) -> list[str]:
        return sorted({e.prefix for e in self.entries})

    def summary(self) -> str:
        if not self.entries:
            return "no images matched"
        ix = self.indices
        return (
            f"{len(self.entries)} images | prefix: {', '.join(self.prefixes)} | "
            f"index {ix[0]}-{ix[-1]} ({len(ix)}) | views {len(self.views)}: "
            f"{','.join(str(v) for v in self.views)}"
            + (f" | {len(self.unmatched)} unmatched" if self.unmatched else "")
        )


@dataclass
class Chunk:
    number: int
    name: str
    prefix: str
    index_from: int
    index_to: int
    #: indices carried over from the previous set (empty for the first set)
    overlap_indices: list[int]
    #: images of the overlap indices - already in the previous component
    overlap_images: list[Path]
    #: images of the indices that this set introduces
    new_images: list[Path]
    #: loop closure: indices taken from the head of the very first set, so the
    #: last set of a loop capture links back to the beginning
    loop_indices: list[int] = field(default_factory=list)
    loop_images: list[Path] = field(default_factory=list)
    #: which tiling this set came from. Sets chain within a pass, never across
    #: one: an offset tiling is a second sweep of the same images and has no
    #: predecessor in the first.
    pass_name: str = ""

    @property
    def images(self) -> list[Path]:
        return self.overlap_images + self.loop_images + self.new_images

    @property
    def n_images(self) -> int:
        return len(self.overlap_images) + len(self.loop_images) + len(self.new_images)

    @property
    def kept_indices(self) -> list[int]:
        """Indices that stay active when a component is imported."""
        return self.overlap_indices + self.loop_indices


def scan_folder(cfg: DatasetConfig) -> Scan:
    """Match every image in ``cfg.image_dir`` against ``cfg.pattern``.

    The pattern needs an ``index`` group (``frame`` is accepted as an alias);
    ``prefix`` and ``view`` are optional.
    """
    scan = Scan()
    root = Path(cfg.image_dir)
    if not root.is_dir():
        return scan

    exts = {e.strip().lower() for e in cfg.extensions.split(",") if e.strip()}
    rx = re.compile(cfg.pattern)

    for p in sorted(root.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        # skip sidecars such as foo.jpg.mask.png
        if any(s.lower() in exts for s in p.suffixes[:-1]):
            continue
        m = rx.match(p.stem)
        if not m:
            scan.unmatched.append(p)
            continue
        groups = m.groupdict()
        raw = groups.get("index") if groups.get("index") is not None else groups.get("frame")
        try:
            index = int(raw)
        except (TypeError, ValueError):
            scan.unmatched.append(p)
            continue
        view_raw = groups.get("view")
        try:
            view = int(view_raw) if view_raw is not None else 0
        except ValueError:
            view = 0
        scan.entries.append(
            ImageEntry(p, groups.get("prefix") or root.name, index, view)
        )
    return scan


def _parse_views(spec: str) -> set[int] | None:
    spec = (spec or "").strip()
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out or None


def parse_chunk_sizes(spec: str, default_overlap: int,
                      default_passes: int) -> list[tuple[int, int, int]]:
    """Parse ``extra_chunk_sizes`` into (size, overlap, extra passes).

    Written as ``size:overlap:passes``, with the tail optional, separated by
    commas - ``"15, 35:12, 40:16:1"``. A size that leaves them out inherits the
    main Overlap indices and Extra offset passes, which is rarely what is
    wanted: an overlap tuned for 25 indices is a different fraction of 15 or 40.

    Anything unparseable is dropped rather than guessed at, so a typo costs a
    missing tiling instead of a wrong one. Sizes below two are dropped as well:
    a set of one index cannot be aligned at all - every view of an index shares
    the rig centre, so there is no baseline - and a stray "...,1" from a comma
    typed for a colon quietly asked for 1807 single-index sets.
    """
    out: list[tuple[int, int, int]] = []
    for token in (spec or "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(":")]
        try:
            size = int(parts[0])
            overlap = int(parts[1]) if len(parts) > 1 and parts[1] else default_overlap
            passes = int(parts[2]) if len(parts) > 2 and parts[2] else default_passes
        except ValueError:
            continue
        if size >= 2:
            out.append((size, max(0, overlap), max(0, passes)))
    return out


def make_chunks(scan: Scan, cfg: DatasetConfig,
                chain: "ChainConfig | None" = None) -> list[Chunk]:
    """Slice the scan into overlapping index windows.

    With ``chunk_indices=100`` and ``overlap_indices=10`` the sets are
    ``0-99``, ``90-189``, ``180-279`` ... - each set re-uses the last 10
    indices of the previous one.

    With ``chain.close_loop`` the last set additionally gets the first few
    indices of the whole range, for captures that walk back to where they
    started.
    """
    if not scan.entries:
        return []

    wanted_views = _parse_views(cfg.views)
    all_indices = scan.indices
    start = cfg.index_start if cfg.index_start >= 0 else all_indices[0]
    end = cfg.index_end if cfg.index_end >= 0 else all_indices[-1]
    step = max(1, cfg.index_step)

    indices = [i for i in all_indices if start <= i <= end and (i - start) % step == 0]
    if not indices:
        return []

    by_index: dict[int, list[Path]] = {}
    for e in sorted(scan.entries, key=lambda x: (x.index, x.view)):
        if wanted_views is not None and e.view not in wanted_views:
            continue
        by_index.setdefault(e.index, []).append(e.path)

    prefix = scan.prefixes[0] if scan.prefixes else "set"

    def tile(size: int, offset: int, pass_name: str,
             overlap_want: int | None = None) -> list[Chunk]:
        size = max(1, size)
        want = cfg.overlap_indices if overlap_want is None else overlap_want
        overlap = min(max(0, want), size - 1)
        stride = size - overlap
        out: list[Chunk] = []
        pos = offset
        previous_end: int | None = None
        while pos < len(indices):
            window = indices[pos: pos + size]
            if previous_end is None:
                ov, new = [], window
            else:
                ov = [i for i in window if i <= previous_end]
                new = [i for i in window if i > previous_end]
            if not new:
                break
            tag = f"_{pass_name}" if pass_name else ""
            out.append(
                Chunk(
                    number=len(out),
                    name=f"{prefix}_{window[0]:04d}-{window[-1]:04d}{tag}",
                    prefix=prefix,
                    index_from=window[0],
                    index_to=window[-1],
                    overlap_indices=ov,
                    overlap_images=[p for i in ov for p in by_index.get(i, [])],
                    new_images=[p for i in new for p in by_index.get(i, [])],
                    pass_name=pass_name,
                )
            )
            previous_end = window[-1]
            if pos + size >= len(indices):
                break
            pos += stride

        if cfg.max_chunks > 0:
            out = out[: cfg.max_chunks]

        # Every tiling closes its own loop. The head comes from the start of the
        # capture, not the start of the pass: what the loop is for is tying the
        # end of the walk back to its beginning, and an offset pass ends at the
        # same place the base one does. Leaving offset passes open left both
        # their ends unconstrained, at exactly the boundary they were added to
        # cover.
        if chain is not None and chain.close_loop and len(out) > 1:
            n = chain.loop_overlap_indices or overlap
            last = out[-1]
            taken = set(range(last.index_from, last.index_to + 1))
            loop = [i for i in indices[:n] if i not in taken]
            if loop:
                last.loop_indices = loop
                last.loop_images = [p for i in loop for p in by_index.get(i, [])]
        return out

    def with_offsets(size: int, overlap_want: int, passes: int,
                     tag: str) -> list[Chunk]:
        """One tiling plus its offset copies.

        The offsets exist so a boundary the merge failed to join across falls
        inside one of their sets: the merge then has a link half a set wide
        instead of only the overlap.
        """
        got = tile(size, 0, tag, overlap_want)
        eff = min(max(0, overlap_want), size - 1)
        stride = size - eff
        for k in range(1, max(0, passes) + 1):
            off = round(stride * k / (passes + 1))
            if off <= 0:
                continue
            got += tile(size, off, f"{tag}p{off}" if tag else f"p{off}",
                        overlap_want)
        return got

    base = max(1, cfg.chunk_indices)
    chunks = with_offsets(base, cfg.overlap_indices, cfg.extra_passes, "")

    # A size named twice would tile twice under the same pass name, and two sets
    # sharing a name share an output folder: they overwrite each other, and
    # skip_existing reads the second as already done.
    seen_sizes = {base}
    for size, overlap_want, passes in parse_chunk_sizes(
            cfg.extra_chunk_sizes, cfg.overlap_indices, cfg.extra_passes):
        if size in seen_sizes:
            continue
        seen_sizes.add(size)
        chunks += with_offsets(size, overlap_want, passes, f"s{size}")

    for n, c in enumerate(chunks):
        c.number = n
    return chunks


def write_imagelist(paths: list[Path], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return path


def seeds_for(chunk: Chunk, seed_map: dict[str, list[str]]) -> list[Path]:
    """Components the user pinned to this set, in spec order."""
    keys = [str(chunk.number), chunk.name]
    if chunk.number == 0:
        keys.append("first")
    out: list[Path] = []
    for key in keys:
        for raw in seed_map.get(key, []):
            p = Path(raw)
            if p.is_file() and p not in out:
                out.append(p)
    return out


def mask_for(image: Path, cfg: MaskConfig) -> Path | None:
    """Locate the mask that belongs to ``image``, or None."""
    if not cfg.enabled:
        return None
    folder = Path(cfg.directory) if cfg.directory else image.parent
    name = cfg.pattern.format(name=image.name, stem=image.stem, ext=image.suffix)
    candidate = folder / name
    return candidate if candidate.is_file() else None


def mask_pairs(images: list[Path], cfg: MaskConfig) -> list[tuple[Path, Path]]:
    pairs = []
    for img in images:
        m = mask_for(img, cfg)
        if m:
            pairs.append((img, m))
    return pairs
