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
    #: Views of one frame share the optical centre, so a pair drawn from the
    #: same frame has no baseline - matching them costs time and can only feed
    #: the reconstruction bad two-view geometries.
    skip_pairs_in_same_frame: bool = True
    #: Verify pairs against the declared rig (COLMAP 4.x).
    rig_verification: bool = True


@dataclass
class MapperConfig:
    #: Hold the rig rigid. The whole point of declaring it is that the views of
    #: one frame have no baseline to solve from - measured at 0.54% of the
    #: frame-to-frame step on 1-mid-1.
    refine_sensor_from_rig: bool = False
    refine_focal_length: bool = False
    refine_principal_point: bool = False
    refine_extra_params: bool = False
    #: Caspar, the GPU bundle adjustment backend added in COLMAP 4.1.0.
    ba_use_gpu: bool = True
    ba_gpu_index: str = "-1"
    min_num_matches: int = 15
    init_min_num_inliers: int = 100
    multiple_models: bool = False
    global_mapper: bool = False


@dataclass
class ExportConfig:
    work_root: str = ""
    #: also write the sparse model as text, which is what other tools read
    export_text: bool = True
    export_ply: bool = True


@dataclass
class RunConfig:
    exe: str = field(default_factory=find_colmap)
    #: Cancel a step that makes no CPU progress for this long (0 = never).
    #: A hang detector, not a deadline - the timer re-arms while it works.
    timeout_min: int = 120
    skip_existing: bool = True


@dataclass
class ColmapPipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rig: RigConfig = field(default_factory=RigConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
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
