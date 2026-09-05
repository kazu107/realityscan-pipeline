"""Run the COLMAP stages in order, one process per stage.

Stages are: lay the images out, extract features, declare the rig, match,
map, export. Each writes its own log next to the workspace so a failure can be
read afterwards, and each is skipped when its output is already there - a
mapper run is long enough that repeating the feature extraction to reach it
would be its own problem.

The timeout is a hang detector, not a deadline: it re-arms while the process
keeps burning CPU. A wall-clock limit killed a healthy RealityScan merge
seconds after it finished the work and before it wrote anything, and there is
no reason to repeat that here.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import cli
from .config import ColmapPipelineConfig
from .clean import clean_model
from .flatten import flatten
from .rsexport import import_into_realityscan, write_text_model
from .layout import build as build_layout
from .pairs import write_pairs
from .rig import Direction, RigSpec, from_extractor_settings, ring

Emit = Callable[[str, dict], None]
CREATE_NO_WINDOW = 0x08000000


def _cpu_seconds(proc: "subprocess.Popen | None") -> float | None:
    if proc is None or proc.poll() is not None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        h = ctypes.windll.kernel32.OpenProcess(0x0400, False, proc.pid)
        if not h:
            return None
        try:
            c, e, k, u = (wintypes.FILETIME() for _ in range(4))
            if not ctypes.windll.kernel32.GetProcessTimes(
                    h, ctypes.byref(c), ctypes.byref(e),
                    ctypes.byref(k), ctypes.byref(u)):
                return None

            def secs(ft):
                return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7

            return secs(k) + secs(u)
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:                                        # noqa: BLE001
        return None


@dataclass
class StageResult:
    name: str
    status: str = "pending"      # pending|running|ok|skipped|failed|cancelled
    seconds: float = 0.0
    exit_code: int | None = None
    message: str = ""
    detail: dict = field(default_factory=dict)


def directions_from(cfg: ColmapPipelineConfig) -> tuple[list[Direction], list[str]]:
    """Work out the rig's directions, and anything the user should know."""
    notes: list[str] = []
    r = cfg.rig
    if r.settings_path:
        spec, raw = from_extractor_settings(
            Path(r.settings_path), set_name=r.settings_set_name or None)
        if raw.get("reverse_direction_index_on_odd"):
            notes.append(
                "the extractor's reverse_direction_index_on_odd is on: odd "
                "frames have their view indices reversed, so a view number "
                "does not name one direction and the rig would be wrong for "
                "half the frames")
        return spec.directions, notes
    if r.ring_count > 0:
        return ring(r.ring_count, r.ring_pitch, r.ring_start_yaw), notes
    dirs: list[Direction] = []
    for i, token in enumerate((r.directions or "").replace(";", ",").split(",")):
        token = token.strip()
        if not token:
            continue
        yaw, _, pitch = token.partition(":")
        try:
            dirs.append(Direction(index=i, yaw=float(yaw),
                                  pitch=float(pitch or 0.0)))
        except ValueError:
            notes.append(f"could not read direction {token!r}")
    return dirs, notes


class PipelineRunner:
    """Sequential runner. Call :meth:`start` from the GUI thread."""

    STAGES = ("layout", "features", "rig", "match", "pairs", "map",
              "export")

    def __init__(self, cfg: ColmapPipelineConfig, emit: Emit):
        self.cfg = cfg
        self.emit = emit
        self.results: list[StageResult] = []
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run_all, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        p = self._proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ---- stages ----------------------------------------------------------
    def _run_all(self) -> None:
        p = cli.paths_for(self.cfg)
        p.ensure()
        wanted = {x.strip() for x in self.cfg.run.stages.split(",") if x.strip()}
        unknown = wanted - set(self.STAGES)
        if unknown:
            self.emit("log", {"line": f"[colpipe] unknown stage(s) "
                                      f"{sorted(unknown)}, expected "
                                      f"{list(self.STAGES)}"})
        self.results = [StageResult(name=s) for s in self.STAGES]
        for r in self.results:
            if wanted and r.name not in wanted:
                r.status = "off"
                r.message = "not in run.stages"
        self.emit("pipeline_start", {"total": len(self.results)})
        t0 = time.time()
        for res in self.results:
            if res.status == "off":
                self.emit("stage_end", {"result": res})
                continue
            if self._cancel.is_set():
                res.status = "cancelled"
                self.emit("stage_end", {"result": res})
                continue
            self.emit("stage_start", {"result": res})
            try:
                getattr(self, f"_stage_{res.name}")(res, p)
            except Exception as exc:                          # noqa: BLE001
                import traceback
                res.status = "failed"
                res.message = f"{type(exc).__name__}: {exc}"
                self.emit("log", {"line": traceback.format_exc()})
            self.emit("stage_end", {"result": res})
            if res.status == "failed":
                break
        ran = [r for r in self.results if r.status != "off"]
        self.emit("pipeline_end", {
            "seconds": time.time() - t0,
            "ok": sum(r.status in ("ok", "skipped") for r in ran),
            "total": len(ran)})

    def _stage_layout(self, res: StageResult, p: cli.Paths) -> None:
        d = self.cfg.dataset
        dirs, notes = directions_from(self.cfg)
        for n in notes:
            self.emit("log", {"line": f"[colpipe] {n}"})
        if any("reverse_direction_index_on_odd" in n for n in notes):
            res.status = "failed"
            res.message = "view indices are not stable across frames"
            return
        views = [x.index for x in dirs] if dirs else None
        if d.views.strip():
            views = [int(v) for v in d.views.replace(" ", "").split(",") if v]
        t0 = time.time()
        out = build_layout(
            Path(d.image_dir), p.images, views=views, pattern=d.pattern,
            frame_from=d.frame_from, frame_to=d.frame_to,
            frame_step=d.frame_step, require_all_views=d.require_all_views,
            mask_pattern=d.mask_pattern if d.use_masks else "",
            mask_dir=d.mask_dir,
            mask_out=p.masks if d.use_masks else None,
            fill_missing_masks=d.fill_missing_masks,
            folder_prefix=d.folder_prefix)
        res.seconds = time.time() - t0
        res.detail = {"frames": out.frames, "views": out.views,
                      "linked": out.linked, "copied": out.copied,
                      "skipped": out.skipped, "masked": out.masked,
                      "filled": out.filled}
        for m in out.messages:
            self.emit("log", {"line": f"[colpipe] {m}"})
        if out.frames == 0:
            res.status = "failed"
            res.message = "no complete frames to lay out"
            return
        res.status = "ok"
        res.message = (f"{out.frames} frames x {len(out.views)} views, "
                       f"{out.linked} linked, {out.copied} copied")

        # the rig describes the layout, so write it in the same breath
        spec = RigSpec(directions=dirs, fov=self.cfg.rig.fov,
                       width=self.cfg.rig.width, height=self.cfg.rig.height,
                       ref_index=self.cfg.rig.ref_view,
                       camera_model=self.cfg.rig.camera_model,
                       coupled=self.cfg.rig.coupled,
                       folder_prefix=self.cfg.dataset.folder_prefix)
        spec.write(p.rig_config)
        self.emit("log", {"line": f"[colpipe] rig config written with "
                                  f"{len(dirs)} sensors, "
                                  + ("one coupled rig - measured to collapse "
                                     "runs of frames onto one point"
                                     if self.cfg.rig.coupled
                                     else "one rig each (not coupled)")})

    def _stage_features(self, res: StageResult, p: cli.Paths) -> None:
        if self.cfg.run.skip_existing and p.database.is_file() \
                and p.database.stat().st_size > 1024:
            res.status = "skipped"
            res.message = "database already present"
            return
        self._spawn(cli.feature_extractor(self.cfg, p), p.logs / "features.log", res)

    def _stage_rig(self, res: StageResult, p: cli.Paths) -> None:
        if not p.rig_config.is_file():
            res.status = "skipped"
            res.message = "no rig config"
            return
        self._spawn(cli.rig_configurator(self.cfg, p), p.logs / "rig.log", res)

    def _stage_match(self, res: StageResult, p: cli.Paths) -> None:
        # matching is the long pole - five hours on 1-mid-1 - so never repeat it
        # by accident just because the run was started again
        if self.cfg.run.skip_existing and p.database.is_file():
            try:
                import sqlite3
                with sqlite3.connect(f"file:{p.database}?mode=ro",
                                     uri=True) as db:
                    (n,) = db.execute("select count(*) from two_view_geometries "
                                      "where rows > 0").fetchone()
            except Exception:                                 # noqa: BLE001
                n = 0
            if n:
                res.status = "skipped"
                res.message = f"{n} verified pairs already in the database"
                return
        self._spawn(cli.matcher(self.cfg, p), p.logs / "match.log", res)

    def _stage_pairs(self, res: StageResult, p: cli.Paths) -> None:
        c = self.cfg.pairs
        if not c.enabled:
            res.status = "skipped"
            res.message = "extra pairs not requested"
            return
        t0 = time.time()
        # do not pay to re-match what the sequential pass already did
        m = self.cfg.match
        if m.method == "sequential":
            covered = {2 ** i for i in range(m.overlap)} if m.quadratic_overlap \
                else set(range(1, m.overlap + 1))
        else:
            covered = set()
        plan = write_pairs(
            p.database, p.pair_list, loop_window=c.loop_window,
            same_frame=c.same_frame, max_view_sep=c.max_view_sep,
            loop_max_view_sep=c.loop_max_view_sep,
            dense_window=c.dense_window,
            dense_max_view_sep=c.dense_max_view_sep,
            dense_skip_offsets=covered)
        for m in plan.messages:
            self.emit("log", {"line": f"[colpipe] {m}"})
        res.detail = {"loop": plan.loop, "same_frame": plan.same_frame,
                      "dense": plan.dense, "total": plan.total}
        res.seconds = time.time() - t0
        if plan.total == 0:
            res.status = "skipped"
            res.message = "nothing to add"
            return
        self._spawn(cli.matches_importer(self.cfg, p), p.logs / "pairs.log", res)
        if res.status == "ok":
            res.message = (f"{plan.loop} loop-seam + {plan.same_frame} "
                           f"same-frame + {plan.dense} dense pairs matched")

    def _stage_map(self, res: StageResult, p: cli.Paths) -> None:
        models = [d for d in p.sparse.iterdir() if d.is_dir()] \
            if p.sparse.is_dir() else []
        if self.cfg.run.skip_existing and models:
            res.status = "skipped"
            res.message = f"{len(models)} model(s) already present"
            return
        if models:
            # the mapper writes straight into this folder, so the result of the
            # last run would be gone - and a run this long is worth keeping to
            # compare against
            n = 1
            while (alt := p.sparse.with_name(
                    f"{p.sparse.name}_prev{n:02d}")).exists():
                n += 1
            p.sparse.rename(alt)
            p.sparse.mkdir(parents=True, exist_ok=True)
            self.emit("log", {"line": f"[colpipe] kept the previous model as "
                                      f"{alt.name}"})
            res.detail["previous"] = str(alt)
        self._spawn(cli.mapper(self.cfg, p), p.logs / "map.log", res)

    def _stage_export(self, res: StageResult, p: cli.Paths) -> None:
        models = sorted(d for d in p.sparse.iterdir() if d.is_dir()) \
            if p.sparse.is_dir() else []
        if not models:
            res.status = "failed"
            res.message = "the mapper produced no model"
            return
        best = max(models, key=lambda d: sum(f.stat().st_size
                                             for f in d.iterdir() if f.is_file()))
        res.detail["model"] = str(best)
        if self.cfg.export.drop_stray_views > 0:
            c = clean_model(self.cfg.run.exe, best, p.sparse_clean,
                            p.root / "stray_views.txt",
                            self.cfg.export.drop_stray_views)
            for msg in c.messages:
                self.emit("log", {"line": f"[colpipe] {msg}"})
            res.detail["dropped_views"] = len(c.dropped)
            res.detail["frames_touched"] = c.frames_touched
            if c.dropped and p.sparse_clean.is_dir():
                best = p.sparse_clean
                res.detail["model"] = str(best)
        if self.cfg.export.export_text:
            p.sparse_text.mkdir(parents=True, exist_ok=True)
            self._spawn(cli.model_converter(self.cfg, p, best, p.sparse_text, "TXT"),
                        p.logs / "export_text.log", res)
        if self.cfg.export.export_ply and res.status != "failed":
            self._spawn(cli.model_converter(self.cfg, p, best, p.ply, "PLY"),
                        p.logs / "export_ply.log", res)
        if self.cfg.export.flat_dataset_dir and res.status != "failed":
            d = self.cfg.dataset
            fr = flatten(
                best, Path(self.cfg.export.flat_dataset_dir), p.images,
                source_dir=Path(d.image_dir) if d.image_dir else None,
                mask_pattern=d.mask_pattern, mask_dir=d.mask_dir,
                write_masks=d.use_masks,
                per_image_cameras=self.cfg.export.flat_per_image_cameras)
            for msg in fr.messages:
                self.emit("log", {"line": f"[colpipe] {msg}"})
            res.detail["flat_dataset"] = self.cfg.export.flat_dataset_dir
        if self.cfg.export.realityscan_project and res.status != "failed":
            d = self.cfg.dataset
            rs_exe = self.cfg.export.realityscan_exe or (
                r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe")
            out = Path(self.cfg.export.realityscan_project)
            # the prefix the layout used maps this set's images to where they
            # actually live; a workspace holding two captures needs the CLI,
            # which takes the mapping directly
            where = ({d.folder_prefix: Path(d.image_dir)} if d.folder_prefix
                     else Path(d.image_dir))
            tr = write_text_model(self.cfg.run.exe, best, out.parent, where)
            for msg in tr.messages:
                self.emit("log", {"line": f"[colpipe] {msg}"})
            if tr.text_dir:
                ir = import_into_realityscan(rs_exe, tr.text_dir, out,
                                             log_path=p.logs / "realityscan.log")
                for msg in ir.messages:
                    self.emit("log", {"line": f"[colpipe] {msg}"})
                res.detail["realityscan_project"] = str(out)
        if res.status != "failed":
            res.status = "ok"
            res.message = f"exported from {best.name} of {len(models)} model(s)"

    # ---- process ---------------------------------------------------------
    def _spawn(self, args: list[str], log_path: Path, res: StageResult) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        (log_path.parent / (log_path.stem + ".bat")).write_text(
            cli.as_batch(args), encoding="utf-8")
        res.status = "running"
        timeout = self.cfg.run.timeout_min * 60 or None
        watchdog: threading.Timer | None = None
        t0 = time.time()
        with log_path.open("w", encoding="utf-8", errors="replace") as f:
            self._proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=CREATE_NO_WINDOW)
            if timeout:
                busy = 0.05 * timeout

                def fire() -> None:
                    nonlocal watchdog
                    used = _cpu_seconds(self._proc)
                    if used is not None and used - fire.last > busy:  # type: ignore[attr-defined]
                        fire.last = used                              # type: ignore[attr-defined]
                        watchdog = threading.Timer(timeout, fire)
                        watchdog.daemon = True
                        watchdog.start()
                        return
                    res.message = (f"no CPU progress for "
                                   f"{self.cfg.run.timeout_min} min")
                    self.cancel()

                fire.last = _cpu_seconds(self._proc) or 0.0            # type: ignore[attr-defined]
                watchdog = threading.Timer(timeout, fire)
                watchdog.daemon = True
                watchdog.start()
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                f.write(line)
                f.flush()
                self.emit("log", {"line": line.rstrip()})
            self._proc.wait()
            if watchdog:
                watchdog.cancel()
        res.exit_code = self._proc.returncode
        res.seconds += time.time() - t0
        self._proc = None
        if self._cancel.is_set():
            res.status = "cancelled"
        elif res.exit_code == 0:
            res.status = "ok"
        else:
            res.status = "failed"
            if not res.message:
                res.message = f"exit code {res.exit_code}"

    # ---- summary ---------------------------------------------------------
    def write_summary(self, path: Path) -> None:
        path.write_text(json.dumps(
            [{"name": r.name, "status": r.status, "seconds": round(r.seconds, 2),
              "exit_code": r.exit_code, "message": r.message, "detail": r.detail}
             for r in self.results], indent=2, ensure_ascii=False),
            encoding="utf-8")
