"""Build the RealityScan.exe command lines for one image set and for the merge."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from pathlib import Path

from .config import PipelineConfig
from .dataset import Chunk
from .formats import REGISTRATION_FORMATS

#: Emitted by -printReport, one line per component. Parsed by runner.parse_stat.
#: componentName/componentCamerasCount live in $IterateComponents, the point
#: count in $ComponentInfo and the error stats in $ComponentStats - using a
#: variable outside its own block makes RealityScan abort the whole run.
STAT_MARKER = "RSSTAT"


def report_string(phase: str) -> str:
    return (
        "$IterateComponents("
        f"{STAT_MARKER}|phase={phase}"
        "|name=$(componentName)"
        "|cams=$(componentCamerasCount)"
        "$ComponentInfo(componentGUID,"
        "|pts=$(componentPointsCount)"
        "|cps=$(componentControlPointsCountUsed)"
        ")"
        "$ComponentStats(componentGUID,"
        "|proj=$(componentTotalProjection)"
        "|track=$(componentAverageTrackLength:.3f)"
        "|max=$(componentMaximalError:.4f)"
        "|median=$(componentMedianError:.4f)"
        "|mean=$(componentMeanError:.4f)"
        "|metric=$(componentMetric)"
        ")\n)"
    )


def index_regex(prefix: str, indices: list[int], width: int = 4) -> str:
    """-selectImage pattern matching `<prefix>_<index>_` for the given indices.

    The prefix matters as soon as a set holds images from more than one
    capture (a seeded set, for instance) - two captures happily share index
    numbers, and a bare `_0000_` would select both.
    """
    alts = "|".join(f"{i:0{width}d}" for i in indices)
    esc = re.sub(r"([.^$*+?()\[\]{}|\\/])", r"\\\1", prefix)
    return f"g/{esc}_({alts})_/"


@dataclass
class ChunkPaths:
    root: Path
    imagelist_all: Path
    imagelist_new: Path
    masks_rscmd: Path
    sparse: Path
    registration: Path
    component: Path
    project: Path
    log: Path
    crash: Path

    def expected_outputs(self, cfg: PipelineConfig) -> list[Path]:
        out: list[Path] = []
        if cfg.export.sparse:
            out.append(self.sparse)
        # the component is always written in chain mode - the next set needs it
        if cfg.export.component or cfg.chain.mode == "component" or cfg.merge.enabled:
            out.append(self.component)
        if cfg.export.project:
            out.append(self.project)
        return out


def chunk_paths(chunk: Chunk, cfg: PipelineConfig) -> ChunkPaths:
    root = Path(cfg.export.out_root) / chunk.name
    fmt = REGISTRATION_FORMATS[cfg.export.registration_format]
    return ChunkPaths(
        root=root,
        imagelist_all=root / f"{chunk.name}_all.imagelist",
        imagelist_new=root / f"{chunk.name}_new.imagelist",
        masks_rscmd=root / f"{chunk.name}_masks.rscmd",
        sparse=root / f"{chunk.name}_sparse{cfg.export.sparse_ext}",
        registration=root / fmt.filename,
        component=root / f"{chunk.name}.rsalign",
        project=root / f"{chunk.name}.rsproj",
        log=root / f"{chunk.name}.log",
        crash=root / "crash",
    )


@dataclass
class MergePaths:
    root: Path
    sparse: Path
    registration: Path
    component: Path
    project: Path
    log: Path
    crash: Path
    imports_rscmd: Path


def merge_paths(cfg: PipelineConfig) -> MergePaths:
    root = Path(cfg.export.out_root) / cfg.merge.dir_name
    fmt = REGISTRATION_FORMATS[cfg.export.registration_format]
    name = cfg.merge.dir_name.strip("_") or "merged"
    return MergePaths(
        root=root,
        sparse=root / f"{name}_sparse{cfg.export.sparse_ext}",
        registration=root / fmt.filename,
        component=root / f"{name}.rsalign",
        project=root / f"{name}.rsproj",
        log=root / f"{name}.log",
        crash=root / "crash",
        imports_rscmd=root / f"{name}_imports.rscmd",
    )


def _settings(cfg: PipelineConfig, force_rematch: bool | None = None,
              align_overrides: dict | None = None) -> list[tuple[str, str]]:
    """Every -set applied per run.

    RealityScan persists -set values in its application config, so a key left
    over from an earlier run would silently change the next one. Everything the
    pipeline depends on is therefore set explicitly on every invocation.
    """
    a, e = cfg.align, cfg.export
    if align_overrides:
        unknown = set(align_overrides) - {f.name for f in fields(a)}
        if unknown:
            raise ValueError(f"unknown align override(s): {sorted(unknown)}")
        a = replace(a, **align_overrides)
    fmt = REGISTRATION_FORMATS[e.registration_format]
    rematch = a.force_component_rematch if force_rematch is None else force_rematch
    return [
        ("appQuitOnError", "true" if cfg.run.quit_on_error else "false"),
        ("appAutoSaveMode", "true" if cfg.run.autosave else "false"),
        ("appIncSubdirs", "false"),
        ("appGroupCalibrationByExif", "false"),
        # alignment
        ("sfmImageDownscaleFactor", str(a.image_downscale)),
        ("sfmMaxFeaturesPerImage", str(a.max_features_per_image)),
        ("sfmMaxFeaturesPerMpx", str(a.max_features_per_mpx)),
        ("sfmImagesOverlap", a.images_overlap),
        ("sfmDetectorSensitivity", a.detector_sensitivity),
        ("sfmFeatureDetectionQuality", a.feature_detection_quality),
        ("sfmMaxFeatureReprojectionError", str(a.max_feature_reprojection_error)),
        ("sfmPreselectorFeatures", str(a.preselector_features)),
        ("sfmDistortionModel", a.distortion_model),
        ("sfmForceComponentRematch", "true" if rematch else "false"),
        ("sfmAutoReconRegionAfterAlignment", "true" if a.auto_recon_region else "false"),
        # registration export
        ("calexFileFormatId", fmt.guid),
        ("calexExportImages", "true" if e.registration_export_images else "false"),
        ("calexUndistortImages", "true" if e.registration_export_images else "false"),
        ("calexUndistortImageFormat", "jpg"),
        # COLMAP format only: 0 = sparse/0/ subfolder, 1 = flat next to the file
        ("colmapDirStructure", "0"),
    ]


def _priors(cfg: PipelineConfig) -> list[str]:
    p = cfg.priors
    if not p.enabled:
        return []
    pairs = [
        ("inpCalibrationGroup", str(p.calibration_group)),
        ("inpCalibration", str(p.calibration_prior)),
        ("inpFocal", f"{p.focal35:.6f}"),
        ("inpLensGroup", str(p.lens_group)),
        ("inpDistortion", str(p.distortion_prior)),
        ("inpDistortionModel", str(p.distortion_model)),
    ]
    args = ["-selectAllImages"]
    for k, v in pairs:
        args += ["-editInputSelection", f"{k}={v}"]
    args.append("-deselectAllImages")
    return args


def write_mask_rscmd(pairs: list[tuple[Path, Path]], usage: int, path: Path) -> Path:
    """Per-image mask assignment as an .rscmd script.

    -setImagesLayer applies to the current selection, so each image is selected
    by its own path first. This has to go through a file: a few hundred images
    would blow past the Windows command-line length limit.
    """
    lines = ["// generated by rspipe - per image mask layers"]
    for img, mask in pairs:
        # quoting is honoured by the .rscmd parser and survives spaces in paths
        lines.append(f'-selectImage "{img}"')
        lines.append(f'-setImagesLayer "{mask}" mask')
    lines.append("-selectAllImages")
    lines.append(f"-editInputSelection inpMaskOpts={usage}")
    lines.append("-deselectAllImages")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_import_rscmd(components: list[Path], path: Path) -> Path:
    """The merge's component imports as an .rscmd script.

    Same reason the mask assignments go through a file, and it took a 655-set
    run to notice the merge needed it too: Windows caps a command line at
    32,767 characters, and 655 components came to 54,365 in
    "-importComponent <path>" arguments alone. The process never started -
    FileNotFoundError, WinError 206, exit code None, zero seconds - which
    reads like a missing file rather than an over-long command.
    """
    lines = ["// generated by rspipe - components to merge"]
    lines += [f'-importComponent "{c}"' for c in components]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_command(
    chunk: Chunk,
    cfg: PipelineConfig,
    paths: ChunkPaths,
    prev_component: Path | None = None,
    prev_component_name: str | None = None,
    has_masks: bool = False,
    first_component: Path | None = None,
    first_component_name: str | None = None,
    seeds: list[Path] | None = None,
    lock_overlap: bool | None = None,
) -> list[str]:
    """Full argv for one image set. Passed to subprocess as a list - no shell."""
    exe = cfg.run.exe
    if not exe:
        raise ValueError("RealityScan.exe path is not set")

    chained = bool(prev_component and cfg.chain.mode == "component"
                   and chunk.overlap_indices)
    # loop closure: the last set also imports the head of the first component
    loop = bool(chained and chunk.loop_indices and first_component
                and first_component_name != prev_component_name)

    args: list[str] = [exe]
    if cfg.run.headless:
        args.append("-headless")
    # -stdConsole mirrors the app console to stdout; -printProgress adds
    # "algId fraction elapsed eta #event" lines that drive the per-set bar.
    args += ["-stdConsole", "-printProgress", "-silent", str(paths.crash)]

    for k, v in _settings(cfg):
        args += ["-set", f"{k}={v}"]

    args.append("-newScene")

    if chained:
        # Bring in the previous set's component, then keep only the overlap
        # indices active. Cameras disabled for alignment do not end up in the
        # component this run produces, which is what keeps each set bounded.
        args += ["-importComponent", str(prev_component)]
        if loop:
            args += ["-importComponent", str(first_component)]
        args += [
            "-deselectAllImages",
            "-selectImage", index_regex(chunk.prefix, chunk.kept_indices),
            "-invertImageSelection",
            "-editInputSelection", "inpEnabled=false",
            "-deselectAllImages",
            "-add", str(paths.imagelist_new),
        ]
    else:
        args += ["-add", str(paths.imagelist_all)]

    # Seeds go in after the "disable everything but the overlap" step above, so
    # they keep all of their cameras active and take part in this alignment.
    for s in seeds or []:
        args += ["-importComponent", str(s)]

    args += _priors(cfg)

    if has_masks:
        args += ["-execRSCMD", str(paths.masks_rscmd)]

    want_lock = cfg.chain.lock_overlap_pose if lock_overlap is None else lock_overlap
    locked = chained and want_lock
    if locked:
        # Only the chain overlap, never the loop-closure cameras: those come
        # from a different component and locking both groups would nail the
        # loop shut at whatever misalignment they happen to have.
        args += [
            "-deselectAllImages",
            "-selectImage", index_regex(chunk.prefix, chunk.overlap_indices),
            "-lockPoseForContinue", "true",
            "-deselectAllImages",
        ]

    args += [
        "-tag", "RSPIPE_ALIGN_BEGIN",
        "-align",
        "-tag", "RSPIPE_ALIGN_END",
        "-printReport", report_string("aligned"),
    ]

    if locked:
        # The lock is stored in the component, and -mergeComponents aborts with
        # "Invalid or corrupted input data" when it imports locked cameras.
        # Hold it only for the alignment, then clear it before exporting.
        # Caution: -lockPoseForContinue rejects a selection that contains any
        # unregistered input, so this step fails whenever -align dropped one of
        # the overlap cameras. The runner falls back to a lock-free rerun.
        # Only the images that were locked: -lockPoseForContinue rejects a
        # selection containing unregistered inputs (the ones this set disabled).
        args += [
            "-deselectAllImages",
            "-selectImage", index_regex(chunk.prefix, chunk.overlap_indices),
            "-lockPoseForContinue", "false",
            "-deselectAllImages",
            "-tag", "RSPIPE_POSE_UNLOCKED",
        ]

    # The imported components survive as their own components; drop them so
    # -selectMaximalComponent cannot pick one over the fresh one.
    dropped = []
    if chained and prev_component_name:
        dropped.append(prev_component_name)
        if loop and first_component_name:
            dropped.append(first_component_name)
    dropped += [s.stem for s in (seeds or [])]
    for name in dropped:
        args += ["-selectComponent", name, "-deleteSelectedComponent"]
    if dropped:
        args += ["-tag", "RSPIPE_IMPORTED_COMPONENT_DROPPED"]

    args += ["-selectMaximalComponent", "-tag", "RSPIPE_COMPONENT_SELECTED"]

    if cfg.export.sparse:
        args += ["-exportSparsePointCloud", str(paths.sparse),
                 "-tag", "RSPIPE_SPARSE_DONE"]
    if cfg.export.registration:
        args += ["-exportRegistration", str(paths.registration),
                 "-tag", "RSPIPE_REGISTRATION_DONE"]
    if cfg.export.component or cfg.chain.mode == "component" or cfg.merge.enabled:
        args += ["-exportSelectedComponentFile", str(paths.component),
                 "-tag", "RSPIPE_COMPONENT_DONE"]
    if cfg.export.project:
        args += ["-save", str(paths.project), "-tag", "RSPIPE_PROJECT_DONE"]
    if cfg.run.clear_cache_after_chunk and cfg.export.project:
        args += ["-clearCache"]

    args.append("-quit")
    return args


def build_retry_command(cfg: PipelineConfig, paths: ChunkPaths, attempt: int,
                        drop_components: list[str] | None = None) -> list[str]:
    """One extra -align pass on a set that did not register enough images.

    Re-loads the set's saved project, so the disabled carried-over images stay
    disabled and the masks are already attached. ``drop_components`` names the
    components to delete first - dropping everything but the largest gives the
    stranded cameras a fresh chance to join it.
    """
    exe = cfg.run.exe
    args: list[str] = [exe]
    if cfg.run.headless:
        args.append("-headless")
    args += ["-stdConsole", "-printProgress", "-silent", str(paths.crash)]
    for k, v in _settings(cfg, force_rematch=cfg.retry.force_rematch):
        args += ["-set", f"{k}={v}"]

    args += ["-load", str(paths.project)]
    for name in drop_components or []:
        args += ["-selectComponent", name, "-deleteSelectedComponent"]
    args += [
        "-tag", f"RSPIPE_RETRY_{attempt}_BEGIN",
        "-align",
        "-tag", f"RSPIPE_RETRY_{attempt}_END",
        "-printReport", report_string(f"retry{attempt}"),
        "-selectMaximalComponent",
    ]
    if cfg.export.sparse:
        args += ["-exportSparsePointCloud", str(paths.sparse)]
    if cfg.export.registration:
        args += ["-exportRegistration", str(paths.registration)]
    args += ["-exportSelectedComponentFile", str(paths.component)]
    args += ["-save", str(paths.project), "-quit"]
    return args


def build_merge_command(cfg: PipelineConfig, components: list[Path],
                        paths: MergePaths) -> list[str]:
    """Import every set's component into one scene and merge them."""
    exe = cfg.run.exe
    args: list[str] = [exe]
    if cfg.run.headless:
        args.append("-headless")
    args += ["-stdConsole", "-printProgress", "-silent", str(paths.crash)]
    for k, v in _settings(cfg, force_rematch=cfg.merge.force_rematch,
                          align_overrides=cfg.merge.align_overrides):
        args += ["-set", f"{k}={v}"]

    args.append("-newScene")
    # through a file: see write_import_rscmd. The caller writes it.
    args += ["-execRSCMD", str(paths.imports_rscmd)]
    if cfg.merge.feature_source >= 0:
        args += ["-selectAllImages",
                 "-setFeatureSource", str(cfg.merge.feature_source),
                 "-deselectAllImages"]
    args += ["-printReport", report_string("imported")]
    args += ["-tag", "RSPIPE_MERGE_BEGIN", "-mergeComponents", "-tag", "RSPIPE_MERGE_END"]
    if cfg.merge.align_after_merge:
        args += ["-align", "-tag", "RSPIPE_MERGE_ALIGNED"]
    args += ["-printReport", report_string("merged")]
    args += ["-selectMaximalComponent"]

    if cfg.export.sparse:
        args += ["-exportSparsePointCloud", str(paths.sparse),
                 "-tag", "RSPIPE_SPARSE_DONE"]
    if cfg.export.registration:
        args += ["-exportRegistration", str(paths.registration),
                 "-tag", "RSPIPE_REGISTRATION_DONE"]
    args += ["-exportSelectedComponentFile", str(paths.component),
             "-tag", "RSPIPE_COMPONENT_DONE"]
    if cfg.export.project:
        args += ["-save", str(paths.project), "-tag", "RSPIPE_PROJECT_DONE"]
    args.append("-quit")
    return args


#: Windows caps a command line at 32,767 characters. CreateProcess fails
#: before the program starts, and Python reports it as FileNotFoundError
#: WinError 206 - which reads like a missing executable, not an over-long
#: argument list. A 655-set merge hit it at 54,365 characters.
COMMAND_LIMIT = 32767


def command_length(args: list[str]) -> int:
    """How long the rendered command line will be, quoting included."""
    return sum(len(a) + (3 if " " in a else 1) for a in args)


def check_command_length(args: list[str], what: str = "command") -> str:
    """Empty if it fits, else an explanation of what to do about it."""
    n = command_length(args)
    if n < COMMAND_LIMIT * 0.9:
        return ""
    return (f"the {what} is {n:,} characters against a Windows limit of "
            f"{COMMAND_LIMIT:,}; it will fail to start with WinError 206. "
            f"Move the repeated arguments into an .rscmd script and pass "
            f"-execRSCMD instead")


def command_as_batch(args: list[str]) -> str:
    """Render an argv list as a readable .bat line (for copy/paste debugging)."""
    def q(s: str) -> str:
        return f'"{s}"' if (" " in s or "\t" in s) else s

    head, tail = args[0], args[1:]
    lines = [q(head)]
    i = 0
    while i < len(tail):
        tok = tail[i]
        group = [tok]
        i += 1
        while i < len(tail) and not tail[i].startswith("-"):
            group.append(tail[i])
            i += 1
        lines.append("     " + " ".join(q(g) for g in group))
    return " ^\n".join(lines) + "\n"
