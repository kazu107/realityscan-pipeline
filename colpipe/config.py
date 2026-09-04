"""Configuration for the COLMAP rig pipeline (JSON-serialisable)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

COLMAP_CANDIDATES = [
    r"D:\h3dgs-work\tools\colmap420\bin\colmap.exe",
    r"C:\Program Files\COLMAP\bin\colmap.exe",
]


def find_colmap() -> str:
    for c in COLMAP_CANDIDATES:
        if Path(c).is_file():
            return c
    return ""


@dataclass
class DatasetConfig:
    """Where the flat extraction lives and which part of it to use."""

    image_dir: str = ""
    pattern: str = r"^(?P<prefix>.+)_(?P<frame>\d+)_(?P<view>\d+)$"
    views: str = ""              # "" = every view found, else "0,1,2"
    frame_from: int = 0
    frame_to: int = -1           # -1 = through the last frame
    frame_step: int = 1
    #: Frames missing a view break the rig's "one frame, all sensors" model, so
    #: they are left out rather than half-populated.
    require_all_views: bool = True
    use_masks: bool = False
    #: COLMAP drops an image whose mask it cannot read, so a partly-masked set
    #: loses those images entirely. A white mask uses every pixel, which is the
    #: same as having none, and keeps the image in.
    fill_missing_masks: bool = True
    mask_pattern: str = "{name}.mask.png"
    mask_dir: str = ""


@dataclass
class RigConfig:
    """The extraction geometry, which is the rig."""

    #: Directions as "yaw:pitch" per entry, in view-index order.
    directions: str = ""
    #: Or generate them: N evenly spaced yaws at one pitch (0 = use `directions`)
    ring_count: int = 0
    ring_pitch: float = 0.0
    ring_start_yaw: float = 0.0
    fov: float = 100.0
    width: int = 2133
    height: int = 2133
    ref_view: int = 0
    camera_model: str = "PINHOLE"
    #: Tie the directions into one rig, so a frame is a single 6-DOF pose.
    #: Off, and it should stay off: measured on 150 frames of 1-mid-1 with
    #: identical matches, coupling collapsed runs of 40+ consecutive frames
    #: onto one point (residual against RealityScan 8.81 index-steps, step p90
    #: 17.84) where independent rigs tracked it to 2.55 with step p90 1.61,
    #: 2.4x the points and a lower reprojection error. See RigSpec.to_config.
    coupled: bool = False
    #: Import these from the extractor GUI's settings file.
    settings_path: str = ""
    settings_set_name: str = ""


@dataclass
class FeatureConfig:
    use_gpu: bool = True
    gpu_index: str = "-1"        # -1 = all available
    max_image_size: int = 3200
    max_num_features: int = 8192
    estimate_affine_shape: bool = False
    domain_size_pooling: bool = False


@dataclass
class MatchConfig:
    #: sequential is the right matcher for a walked capture; the others are here
    #: because a short test sometimes wants exhaustive.
    method: str = "sequential"   # sequential | exhaustive | vocab_tree
    overlap: int = 10
    quadratic_overlap: bool = True
    loop_detection: bool = False
    loop_detection_period: int = 10
    loop_detection_num_images: int = 50
    vocab_tree_path: str = ""
    use_gpu: bool = True
    gpu_index: str = "-1"
    max_num_matches: int = 32768
    #: Views of one frame share the optical centre, so such a pair cannot be
    #: triangulated from on its own. Skipping it anyway was a mistake: with the
    #: rig declared the relative pose is known, `rig_verification` checks the
    #: matches against it, and the pair merges two per-view track chains that
    #: otherwise never meet. On a 100 deg extraction at 45 deg spacing the
    #: adjacent views overlap by 55 deg - the largest overlap in the set. With
    #: these skipped on 1-mid-1 the median track span was 3 frames and no point
    #: spanned more than 500, so scale drifted freely along an 865-frame chain.
    #: COLMAP's own default is to keep them.
    skip_pairs_in_same_frame: bool = False
    #: Verify pairs against the declared rig (COLMAP 4.x).
    rig_verification: bool = True


@dataclass
class ExtraPairsConfig:
    """Pairs the frame index cannot express, matched into an existing database.

    Runs `matches_importer` over a hand-written pair list. The features are
    already extracted, so this repairs a finished workspace in minutes instead
    of repeating the whole matching pass.
    """

    enabled: bool = False
    #: Pair the last N frames against the first N, to close a walk that returns
    #: to where it started. 0 = leave the loop open.
    loop_window: int = 40
    #: Cap the view separation at the seam (0 = every combination). The walk can
    #: come back on any heading, so which views face the same way is not known.
    loop_max_view_sep: int = 0
    #: Also pair the views inside one frame, when the matching pass skipped them.
    same_frame: bool = True
    #: Ring separation to pair up to: 1 is the 45 deg neighbour, 2 the 90 deg
    #: one. A 100 deg field does not reach 135 deg, so past 2 it is pure cost.
    max_view_sep: int = 2
    #: Fill in the frame offsets the quadratic sequential matcher never makes.
    #: With overlap 5 and quadratic on it pairs offsets 1, 2, 4, 8, 16 and
    #: nothing between - so 3, 5, 6, 7 are simply absent. On 1-mid-1 offset 1
    #: carried 51% of every inlier in the reconstruction, offset 2 carried 25%
    #: and offset 4 carried 13%, so the gaps in between are not noise, they are
    #: most of what a dense window would add. RealityScan preselects candidates
    #: across a whole set instead of walking a chain, and over the same frames
    #: its trajectory is flat (p10..p90 = 0.61..1.11 of median step) where a
    #: chain-matched COLMAP model bends by a factor of 20. 0 = do not fill.
    dense_window: int = 0
    #: Views to pair for the dense window. 1 keeps it to the same camera and
    #: its 45 deg neighbours; the cost grows fast past that.
    dense_max_view_sep: int = 1


@dataclass
class MapperConfig:
    #: Only meaningful with rig.coupled on. Leave it off there: turning it on
    #: crashes COLMAP 4.2.0 (0xC0000409) 62 frames into 1-mid-1, so a coupled
    #: rig can be held rigid, which collapses the walk, or refined, which
    #: aborts. Neither is a usable setting - use independent rigs instead.
    refine_sensor_from_rig: bool = False
    refine_focal_length: bool = False
    refine_principal_point: bool = False
    refine_extra_params: bool = False
    #: Bundle adjustment backend: CERES or CASPAR. Caspar is COLMAP's own GPU
    #: backend, added in 4.1.0, and it is selected HERE - not by `ba_use_gpu`,
    #: which only asks Ceres for its CUDA linear solvers. Measured on the
    #: official Windows 4.2.0 build: CASPAR refuses with "COLMAP was built
    #: without CASPAR_ENABLED; rebuild with -DCASPAR_ENABLED=ON", and
    #: ba_use_gpu warns "Ceres was compiled without CUDA support" and falls
    #: back to the CPU. So on that binary bundle adjustment is CPU-only
    #: whatever these say, and a 1-mid-1 map ran 2.6 hours on the CPU.
    ba_global_backend: str = "CERES"
    ba_local_backend: str = "CERES"
    #: Ceres' own CUDA solvers. Needs a Ceres built with CUDA and cuDSS.
    ba_use_gpu: bool = False
    ba_gpu_index: str = "-1"
    min_num_matches: int = 15
    init_min_num_inliers: int = 100
    multiple_models: bool = False
    global_mapper: bool = False


@dataclass
class ExportConfig:
    work_root: str = ""
    #: also write the sparse model as text, which is what other tools read.
    #: Off by default: images.txt for the 904-frame 1-mid-1 model came out at
    #: 7.5 GB, because it carries every keypoint of every image.
    export_text: bool = False
    export_ply: bool = True
    #: Drop views sitting further than this many index-steps from the rest of
    #: their frame, before exporting. 0 = keep everything. On 1-mid-1 this was
    #: 53 views of 6,213, and it took the trajectory from eleven breaks over 3x
    #: the median step to none - see colpipe/clean.py.
    drop_stray_views: float = 1.0
    #: Also write a flat COLMAP dataset here - images/NNNNN.jpg, masks/NNNNN.jpg,
    #: sparse/0 - which is the layout K:\data\col is in and what the gaussian
    #: splatting side reads. Empty = do not write one.
    flat_dataset_dir: str = ""
    #: One camera per image in that dataset rather than one per direction.
    #: Sharing is correct for this rig and every COLMAP reader handles it;
    #: K:\data\col has per-image cameras because RealityScan undistorted each
    #: image separately, so turn this on for a reader that assumes that.
    flat_per_image_cameras: bool = False


@dataclass
class RunConfig:
    exe: str = field(default_factory=find_colmap)
    #: Cancel a step that makes no CPU progress for this long (0 = never).
    #: A hang detector, not a deadline - the timer re-arms while it works.
    timeout_min: int = 120
    skip_existing: bool = True
    #: Which stages to run, comma separated; empty means all of them. Repairing
    #: a finished workspace means running "pairs,map,export" over the database
    #: that is already there - matching 1-mid-1 took five hours and there is no
    #: version of "add the loop pairs" that should pay that again.
    stages: str = ""


@dataclass
class ColmapPipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rig: RigConfig = field(default_factory=RigConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    pairs: ExtraPairsConfig = field(default_factory=ExtraPairsConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ColmapPipelineConfig":
        cfg = cls()
        for f in fields(cls):
            section = data.get(f.name)
            if not isinstance(section, dict):
                continue
            obj = getattr(cfg, f.name)
            for g in fields(obj):
                if g.name in section:
                    setattr(obj, g.name, section[g.name])
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ColmapPipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
