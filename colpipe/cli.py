"""Build the COLMAP command line for each stage of the rig pipeline.

Every path is made absolute before it is handed over. COLMAP is less
treacherous about this than RealityScan, which resolves a relative path against
its own install folder and then waits forever on a dialog nobody can see, but
there is no reason to find out where the line is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ColmapPipelineConfig


@dataclass
class Paths:
    root: Path
    images: Path
    database: Path
    rig_config: Path
    pair_list: Path
    sparse: Path
    sparse_clean: Path
    sparse_text: Path
    ply: Path
    masks: Path
    logs: Path

    def ensure(self) -> None:
        for p in (self.root, self.images, self.sparse, self.sparse_text,
                  self.logs):
            p.mkdir(parents=True, exist_ok=True)


def paths_for(cfg: ColmapPipelineConfig) -> Paths:
    root = Path(cfg.export.work_root).resolve()
    return Paths(
        root=root,
        images=root / "images",
        database=root / "database.db",
        rig_config=root / "rig_config.json",
        pair_list=root / "extra_pairs.txt",
        sparse=root / "sparse",
        sparse_clean=root / "sparse_clean",
        sparse_text=root / "sparse_text",
        ply=root / "sparse.ply",
        masks=root / "masks",
        logs=root / "logs",
    )


def _flag(value: bool) -> str:
    return "1" if value else "0"


def feature_extractor(cfg: ColmapPipelineConfig, p: Paths) -> list[str]:
    f = cfg.feature
    args = [
        cfg.run.exe, "feature_extractor",
        "--database_path", str(p.database),
        "--image_path", str(p.images),
        # one camera per folder is what makes each direction its own sensor,
        # and identical file names across those folders what makes a frame
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_model", cfg.rig.camera_model,
        # 4.x moved the shared knobs to FeatureExtraction and left the
        # SIFT-only ones behind; 3.x names are silently rejected
        "--FeatureExtraction.use_gpu", _flag(f.use_gpu),
        "--FeatureExtraction.gpu_index", str(f.gpu_index),
        "--FeatureExtraction.max_image_size", str(f.max_image_size),
        "--SiftExtraction.max_num_features", str(f.max_num_features),
        "--SiftExtraction.estimate_affine_shape", _flag(f.estimate_affine_shape),
        "--SiftExtraction.domain_size_pooling", _flag(f.domain_size_pooling),
    ]
    if cfg.dataset.use_masks:
        # COLMAP wants masks in a parallel tree, named <image name>.png
        args += ["--ImageReader.mask_path", str(p.masks)]
    return args


def rig_configurator(cfg: ColmapPipelineConfig, p: Paths) -> list[str]:
    return [
        cfg.run.exe, "rig_configurator",
        "--database_path", str(p.database),
        "--rig_config_path", str(p.rig_config),
    ]


def matcher(cfg: ColmapPipelineConfig, p: Paths) -> list[str]:
    m = cfg.match
    common = [
        "--database_path", str(p.database),
        "--FeatureMatching.use_gpu", _flag(m.use_gpu),
        "--FeatureMatching.gpu_index", str(m.gpu_index),
        "--FeatureMatching.max_num_matches", str(m.max_num_matches),
        # the views of one frame share an optical centre, so a pair from the
        # same frame has no baseline to verify against - matching them can only
        # add bad two-view geometries
        "--FeatureMatching.skip_image_pairs_in_same_frame",
        _flag(m.skip_pairs_in_same_frame),
        "--FeatureMatching.rig_verification", _flag(m.rig_verification),
    ]
    if m.method == "exhaustive":
        return [cfg.run.exe, "exhaustive_matcher", *common]
    if m.method == "vocab_tree":
        return [cfg.run.exe, "vocab_tree_matcher", *common,
                "--VocabTreeMatching.vocab_tree_path", str(m.vocab_tree_path)]
    args = [cfg.run.exe, "sequential_matcher", *common,
            "--SequentialMatching.overlap", str(m.overlap),
            "--SequentialMatching.quadratic_overlap",
            _flag(m.quadratic_overlap),
            "--SequentialMatching.loop_detection", _flag(m.loop_detection)]
    if m.loop_detection:
        args += ["--SequentialMatching.loop_detection_period",
                 str(m.loop_detection_period),
                 "--SequentialMatching.loop_detection_num_images",
                 str(m.loop_detection_num_images)]
        if m.vocab_tree_path:
            args += ["--SequentialMatching.vocab_tree_path",
                     str(m.vocab_tree_path)]
    return args


def matches_importer(cfg: ColmapPipelineConfig, p: Paths) -> list[str]:
    """Match and verify exactly the pairs in the list, into the same database."""
    m = cfg.match
    return [
        cfg.run.exe, "matches_importer",
        "--database_path", str(p.database),
        "--match_list_path", str(p.pair_list),
        "--match_type", "pairs",
        "--FeatureMatching.use_gpu", _flag(m.use_gpu),
        "--FeatureMatching.gpu_index", str(m.gpu_index),
        "--FeatureMatching.max_num_matches", str(m.max_num_matches),
        # the point of these pairs is the ones inside a frame, so do not let
        # the importer drop them again
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.rig_verification", _flag(m.rig_verification),
    ]


def mapper(cfg: ColmapPipelineConfig, p: Paths) -> list[str]:
    m = cfg.mapper
    cmd = "global_mapper" if m.global_mapper else "mapper"
    args = [
        cfg.run.exe, cmd,
        "--database_path", str(p.database),
        "--image_path", str(p.images),
        "--output_path", str(p.sparse),
    ]
    if m.global_mapper:
        return args
    return args + [
        "--Mapper.ba_refine_sensor_from_rig", _flag(m.refine_sensor_from_rig),
        "--Mapper.ba_refine_focal_length", _flag(m.refine_focal_length),
        "--Mapper.ba_refine_principal_point", _flag(m.refine_principal_point),
        "--Mapper.ba_refine_extra_params", _flag(m.refine_extra_params),
        "--Mapper.ba_global_backend", m.ba_global_backend,
        "--Mapper.ba_local_backend", m.ba_local_backend,
        "--Mapper.ba_use_gpu", _flag(m.ba_use_gpu),
        "--Mapper.ba_gpu_index", str(m.ba_gpu_index),
        "--Mapper.min_num_matches", str(m.min_num_matches),
        "--Mapper.init_min_num_inliers", str(m.init_min_num_inliers),
        "--Mapper.multiple_models", _flag(m.multiple_models),
    ]


def model_converter(cfg: ColmapPipelineConfig, p: Paths, model: Path,
                    output: Path, fmt: str) -> list[str]:
    return [
        cfg.run.exe, "model_converter",
        "--input_path", str(model),
        "--output_path", str(output),
        "--output_type", fmt,
    ]


def as_batch(args: list[str]) -> str:
    """Render an argv list as a readable .bat line, for copy/paste debugging."""
    def q(s: str) -> str:
        return f'"{s}"' if (" " in s or "\t" in s) else s

    lines = [q(args[0])]
    i = 1
    while i < len(args):
        group = [args[i]]
        i += 1
        while i < len(args) and not args[i].startswith("--"):
            group.append(args[i])
            i += 1
        lines.append("     " + " ".join(q(g) for g in group))
    return " ^\n".join(lines) + "\n"
