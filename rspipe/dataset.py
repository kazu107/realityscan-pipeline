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

    size = max(1, cfg.chunk_indices)
    overlap = min(max(0, cfg.overlap_indices), size - 1)
    stride = size - overlap

    prefix = scan.prefixes[0] if scan.prefixes else "set"
    chunks: list[Chunk] = []
    pos = 0
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
        chunks.append(
            Chunk(
                number=len(chunks),
                name=f"{prefix}_{window[0]:04d}-{window[-1]:04d}",
                prefix=prefix,
                index_from=window[0],
                index_to=window[-1],
                overlap_indices=ov,
                overlap_images=[p for i in ov for p in by_index.get(i, [])],
                new_images=[p for i in new for p in by_index.get(i, [])],
            )
        )
        previous_end = window[-1]
        if pos + size >= len(indices):
            break
        pos += stride

    if cfg.max_chunks > 0:
        chunks = chunks[: cfg.max_chunks]

    if chain is not None and chain.close_loop and len(chunks) > 1:
        n = chain.loop_overlap_indices or overlap
        last = chunks[-1]
        taken = set(range(last.index_from, last.index_to + 1))
        loop = [i for i in indices[:n] if i not in taken]
        if loop:
            last.loop_indices = loop
            last.loop_images = [p for i in loop for p in by_index.get(i, [])]

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
