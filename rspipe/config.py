"""Configuration objects for the RealityScan batch pipeline (JSON-serialisable)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from .formats import DEFAULT_REGISTRATION_FORMAT

EXE_CANDIDATES = [
    r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe",
    r"C:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe",
    r"C:\Program Files\Epic Games\RealityScan\RealityScan.exe",
    r"C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe",
]

#: <name>_<index>_<view>; "frame" is accepted as an alias for "index"
DEFAULT_PATTERN = r"^(?P<prefix>.+)_(?P<index>\d+)_(?P<view>\d+)$"


def find_realityscan() -> str:
    for c in EXE_CANDIDATES:
        if Path(c).is_file():
            return c
    return ""


def focal35_from_fov(fov_deg: float) -> float:
    """35mm-equivalent focal length for a horizontal FOV over the long image side."""
    return 36.0 / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def fov_from_focal35(focal35: float) -> float:
    return math.degrees(2.0 * math.atan(36.0 / (2.0 * focal35)))


@dataclass
class DatasetConfig:
    image_dir: str = ""
    pattern: str = DEFAULT_PATTERN
    extensions: str = ".jpg,.jpeg,.png,.tif,.tiff"
    index_start: int = 0
    index_end: int = -1          # -1 = through the last index found
    index_step: int = 1
    views: str = ""              # "" = every view, else e.g. "0,1,2,3"
    chunk_indices: int = 100     # indices per image set
    #: Indices re-used from the previous set. This is what ties consecutive
    #: components together, and it needs to be long enough to fix the scale
    #: between them on its own. All views of one index share a rig centre
    #: (0.7% of the distance to the next index on 1-mid-2), so N overlap
    #: indices give N points on a near-straight line, not N x views of them:
    #: 5 was too few, and the merge either had to lean on the pose lock or got
    #: the scale badly wrong. Prefer 10 or more, and at least ~40% of
    #: chunk_indices on captures that move in a straight line.
    overlap_indices: int = 10
    max_chunks: int = 0          # 0 = no limit (debug aid)


@dataclass
class MaskConfig:
    """Per-image masks, attached with -setImagesLayer through an .rscmd file."""

    enabled: bool = False
    #: {name} = file name with extension, {stem} = without, {ext} = ".jpg"
    pattern: str = "{name}.mask.png"
    directory: str = ""          # "" = next to the image
    usage: int = 1               # inpMaskOpts: 0 off, 1 alignment, 2 meshing, 3 both


@dataclass
class ChainConfig:
    """How set N re-uses set N-1.

    ``component``: import the previous set's component, keep only the overlap
    indices active, then add the new images on top - the overlap cameras come
    in already posed.
    ``images``: every set is aligned from scratch; the overlap indices are just
    ordinary images that both sets contain.
    """

    mode: str = "component"      # component | images
    #: -lockPoseForContinue on the overlap cameras for the duration of -align
    #: only. The lock is released again before the component is exported,
    #: because -mergeComponents aborts on locked cameras.
    #:
    #: Measured on 10 chained sets: 477/480 -> 480/480 registered and ~20%
    #: faster.
    #:
    #: The lock has a second, larger job that only showed up on 1-mid-2. All
    #: views of one index sit at the same rig centre - 0.7% of the distance to
    #: the next index - so an overlap of 5 indices is really five points on a
    #: near-straight line, and the scale between two components is barely
    #: determined by it. The lock hides that: the carried-over cameras keep the
    #: previous set's poses, so the two components agree exactly. Released, the
    #: merge has to estimate the scale from the degenerate stretch, and on
    #: 1-mid-2 it stretched everything past index 832 by 9x.
    #:
    #: The cost is that a scale disagreement between the frozen cameras and the
    #: newly solved ones has nowhere to go and lands as a step at the first new
    #: index - 11.9x the median spacing at index 85, 8.2x at 225. 1-mid-1,
    #: 1-high and 1-low showed no such steps (max 1.2x, 1.3x, 2.1x) and all
    #: three merged faithfully, so on ordinary data the lock is the safer side
    #: of the trade and it stays on.
    #:
    #: The real fix is a longer overlap - see overlap_indices - which makes the
    #: shared stretch long enough to fix the scale on its own. With that in
    #: place the lock can come off and neither problem appears. That has not
    #: been measured yet, so the default has not been changed on a guess.
    lock_overlap_pose: bool = True
    #: For a capture that walks a loop: give the last set the head of the very
    #: first component as a second overlap, so the sequence closes on itself.
    close_loop: bool = False
    #: how many indices from the start to hand to the last set (0 = same as
    #: dataset.overlap_indices)
    loop_overlap_indices: int = 0


@dataclass
class SeedConfig:
    """Already-built components handed to specific sets before alignment.

    ``spec`` holds one ``<set> = <path to .rsalign>`` per line, where ``<set>``
    is the set number (0, 1, ...), the set name, or ``first``. Lines starting
    with ``#`` are ignored.

    The point is to tie two captures together. Two passes over the same area
    (say a normal-height and a high walk) share no images, so merging them
    afterwards has nothing to latch onto. Give the first set of the second pass
    the first component of the first pass and they align into one component,
    which the final merge can then follow.
    """

    enabled: bool = False
    spec: str = ""


def parse_seed_spec(spec: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in (spec or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        sel, path = line.split("=", 1)
        sel, path = sel.strip(), path.strip().strip('"')
        if sel and path:
            out.setdefault(sel, []).append(path)
    return out


@dataclass
class RetryConfig:
    """Extra -align passes for sets that did not register enough images.

    Each attempt is its own RealityScan process that re-loads the set's saved
    project, so the runner can stop as soon as the ratio is good enough.
    """

    enabled: bool = False
    #: retry while registered / images < this (1.0 = require every image)
    min_ratio: float = 1.0
    #: extra passes after the first alignment
    max_attempts: int = 2
    #: delete every component except the largest before re-aligning, so the
    #: stranded cameras get another chance to join the main one
    drop_minor_components: bool = True
    #: sfmForceComponentRematch during the retry passes
    force_rematch: bool = False


@dataclass
class MergeConfig:
    enabled: bool = True
    dir_name: str = "_merged"
    #: sfmForceComponentRematch during the merge run. Measured on the full
    #: 46-component / 7336-image merge: true and false register exactly the
    #: same 7287 cameras and agree to 0.03% (median) on pairwise camera
    #: distances, but false is ~6% faster - so there is nothing to buy here.
    force_rematch: bool = False
    align_after_merge: bool = False
    #: -setFeatureSource on every input before -mergeComponents
    #: -1 leave untouched, 0 merge using overlaps, 1 component features,
    #: 2 all image features.  1 is RealityScan's own default and the safe
    #: choice. 0 is an optimisation that pays off on easy data - on 1-mid-1 it
    #: merged the same cameras in half the time with ~10% more tie points - but
    #: it only looks at the shared overlap images, and on 1-mid-2 it split the
    #: result into 6212 + 4605 cameras twice, once even with a bridge component
    #: sharing 1776 cameras across the break. 1 joined the same data.
    #: 2 re-detects features on every image and ran out of memory at 96 GB on
    #: 10656 images.
    feature_source: int = 1
    #: extra .rsalign files imported alongside the per-set components.
    #: Consecutive sets share only the overlap indices, and a merge at full
    #: scale can still fail to join two of them - 1-mid-2 split into 6212 +
    #: 4605 cameras at one such boundary even though the same sets merged
    #: cleanly in isolation. A "bridge" component covering the break with a
    #: much wider overlap on both sides gives the merge a well-conditioned
    #: link to fit. Build one by aligning the span as a single set, or by
    #: merging just the sets around it.
    extra_components: list[str] = field(default_factory=list)


@dataclass
class PriorConfig:
    """Prior calibration pushed onto every input with -editInputSelection."""

    enabled: bool = True
    focal35: float = 15.1039     # 100 deg horizontal FOV on a square frame
    calibration_prior: int = 1   # 0 unknown / 1 approximate / 2 fixed
    calibration_group: int = 1   # -1 = groupless
    distortion_prior: int = 2    # rendered pinhole views have no distortion
    distortion_model: int = 0    # 0 = no lens distortion
    lens_group: int = 1


@dataclass
class AlignConfig:
    image_downscale: int = 1
    max_features_per_image: int = 160000
    max_features_per_mpx: int = 40000
    images_overlap: str = "Medium"           # Low | Medium | High
    detector_sensitivity: str = "Medium"     # Low | Medium | High | Ultra
    feature_detection_quality: str = "High"  # High | Normal
    max_feature_reprojection_error: float = 2.0
    preselector_features: int = 40000
    distortion_model: str = "Brown3"
    force_component_rematch: bool = False
    auto_recon_region: bool = False


@dataclass
class ExportConfig:
    out_root: str = ""
    sparse: bool = True
    sparse_ext: str = ".ply"
    registration: bool = True
    registration_format: str = DEFAULT_REGISTRATION_FORMAT
    registration_export_images: bool = False
    component: bool = True
    project: bool = True
    min_component_size: int = 5


@dataclass
class RunConfig:
    exe: str = field(default_factory=find_realityscan)
    headless: bool = True
    quit_on_error: bool = True
    autosave: bool = False
    clear_cache_after_chunk: bool = False
    skip_existing: bool = True
    #: Cancel a step that makes no CPU progress for this long (0 = never).
    #: This is a hang detector, not a deadline: the timer re-arms as long as
    #: the process keeps burning CPU, so a merge is free to run for as long as
    #: it needs. What it catches is the headless build blocking on a dialog,
    #: which sits at essentially zero CPU forever.
    timeout_min: int = 120


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    masks: MaskConfig = field(default_factory=MaskConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ---- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        cfg = cls()
        for f in fields(cls):
            section = data.get(f.name)
            if not isinstance(section, dict):
                continue
            target = getattr(cfg, f.name)
            for sf in fields(target):
                if sf.name in section:
                    setattr(target, sf.name, section[sf.name])
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


assert is_dataclass(PipelineConfig)
