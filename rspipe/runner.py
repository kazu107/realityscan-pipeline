"""Run the pipeline over a list of image sets, one RealityScan process per set.

Sets run in order because set N can hand its component to set N+1
(``chain.mode == "component"``). A final merge run imports every set's
component into one scene and calls ``-mergeComponents``.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .cli import (
    STAT_MARKER,
    build_command,
    build_merge_command,
    build_retry_command,
    chunk_paths,
    command_as_batch,
    merge_paths,
    write_mask_rscmd,
)
from .config import PipelineConfig, parse_seed_spec
from .dataset import Chunk, mask_pairs, seeds_for, write_imagelist

CREATE_NO_WINDOW = 0x08000000


def _cpu_seconds(proc: "subprocess.Popen | None") -> float | None:
    """Total CPU time the process has used, or None if it cannot be read.

    Used to tell a long computation apart from a hung one. psutil is not a
    dependency, so this asks Windows directly and gives up quietly anywhere
    else - the caller then falls back to a plain deadline.
    """
    if proc is None or proc.poll() is not None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        handle = ctypes.windll.kernel32.OpenProcess(
            0x0400, False, proc.pid)          # PROCESS_QUERY_INFORMATION
        if not handle:
            return None
        try:
            creation = wintypes.FILETIME()
            exit_ = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            ok = ctypes.windll.kernel32.GetProcessTimes(
                handle, ctypes.byref(creation), ctypes.byref(exit_),
                ctypes.byref(kernel), ctypes.byref(user))
            if not ok:
                return None

            def secs(ft: "wintypes.FILETIME") -> float:
                return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7

            return secs(kernel) + secs(user)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:                                        # noqa: BLE001
        return None

Emit = Callable[[str, dict], None]  # (event, payload)

_STAT_RE = re.compile(rf"^\s*{STAT_MARKER}\|(.*)$")
_PROGRESS_RE = re.compile(
    r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+#(\w+)\s*$"
)


@dataclass
class ComponentStat:
    phase: str = ""
    name: str = ""
    cams: int = 0
    pts: int = 0
    cps: int = 0
    proj: int = 0
    track: float = 0.0
    max: float = 0.0
    median: float = 0.0
    mean: float = 0.0
    metric: str = ""

    @classmethod
    def parse(cls, body: str) -> "ComponentStat":
        st = cls()
        for part in body.split("|"):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            if not hasattr(st, k):
                continue
            cur = getattr(st, k)
            try:
                setattr(st, k, type(cur)(v) if not isinstance(cur, str) else v)
            except ValueError:
                pass
        return st


@dataclass
class StepResult:
    number: int
    name: str
    kind: str = "set"            # set | merge
    index_from: int = 0
    index_to: int = 0
    n_images: int = 0
    n_new: int = 0
    n_overlap: int = 0
    n_masks: int = 0
    n_loop: int = 0
    #: components pinned to this set by the user, and how many of their
    #: cameras ended up in this set's own component
    seeds: list[str] = field(default_factory=list)
    seed_joined: int = 0
    chained: bool = False
    looped: bool = False
    #: extra -align passes actually run, and what each of them gained
    retries: int = 0
    retry_gain: list[int] = field(default_factory=list)
    #: names of the components carried in (previous set, plus the first set
    #: when the loop is closed) - still listed by -printReport, but not this
    #: set's own result
    imported_names: list[str] = field(default_factory=list)
    status: str = "pending"      # pending|running|ok|failed|skipped|cancelled
    exit_code: int | None = None
    seconds: float = 0.0
    components: list[ComponentStat] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    message: str = ""

    def _phase(self, phase: str) -> list[ComponentStat]:
        return [c for c in self.components if c.phase == phase]

    @property
    def best(self) -> ComponentStat | None:
        """Largest component of the last reported phase, ignoring the import."""
        if not self.components:
            return None
        last_phase = self.components[-1].phase
        pool = self._phase(last_phase) or self.components
        if self.imported_names:
            fresh = [c for c in pool if c.name not in self.imported_names]
            pool = fresh or pool
        return max(pool, key=lambda c: c.cams)

    def to_dict(self) -> dict:
        d = {
            "number": self.number,
            "name": self.name,
            "kind": self.kind,
            "index_from": self.index_from,
            "index_to": self.index_to,
            "n_images": self.n_images,
            "n_new": self.n_new,
            "n_overlap": self.n_overlap,
            "n_masks": self.n_masks,
            "n_loop": self.n_loop,
            "seeds": self.seeds,
            "seed_joined": self.seed_joined,
            "chained": self.chained,
            "looped": self.looped,
            "retries": self.retries,
            "retry_gain": self.retry_gain,
            "imported_names": self.imported_names,
            "status": self.status,
            "exit_code": self.exit_code,
            "seconds": round(self.seconds, 2),
            "outputs": self.outputs,
            "message": self.message,
            "components": [c.__dict__ for c in self.components],
        }
        b = self.best
        if b:
            d["registered"] = b.cams
            d["registered_ratio"] = round(b.cams / self.n_images, 4) if self.n_images else 0
            d["points"] = b.pts
            d["mean_error_px"] = b.mean
        return d

    def restore(self, data: dict) -> None:
        """Adopt the measurements a previous run recorded for this step.

        Resuming skips whatever is already on disk, and without this the
        summary would be rewritten with zeros for every skipped set - the
        record of the first run destroyed by the run that continued it. Only
        what was measured is taken; the set's shape is recomputed from the
        current configuration, so a preset edited between runs is not papered
        over.
        """
        self.components = [ComponentStat(**c) for c in data.get("components", [])]
        self.imported_names = list(data.get("imported_names", []))
        self.seeds = list(data.get("seeds", []))
        self.seed_joined = int(data.get("seed_joined", 0))
        self.retries = int(data.get("retries", 0))
        self.retry_gain = list(data.get("retry_gain", []))
        self.chained = bool(data.get("chained", False))
        self.looped = bool(data.get("looped", False))
        self.exit_code = data.get("exit_code")
        self.seconds = float(data.get("seconds", 0.0))
        self.outputs = list(data.get("outputs", []))
        self.n_masks = int(data.get("n_masks", self.n_masks))


class PipelineRunner:
    """Sequential runner. Call :meth:`start` from the GUI thread."""

    def __init__(self, cfg: PipelineConfig, chunks: Iterable[Chunk], emit: Emit):
        self.cfg = cfg
        self.chunks = list(chunks)
        self.emit = emit
        self.results: list[StepResult] = []
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()
        self._previous: dict[str, dict] = {}

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run_all, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ---- worker ----------------------------------------------------------
    def _load_previous(self, out_root: Path) -> None:
        """Read what an earlier run recorded, so a resume can carry it forward."""
        self._previous = {}
        path = out_root / "summary.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:            # noqa: BLE001
            self.emit("log", {"line": f"[rspipe] could not read {path.name}: {exc}"})
            return
        self._previous = {r["name"]: r for r in data
                          if isinstance(r, dict) and r.get("name")
                          and r.get("status") == "ok"}
        if self._previous:
            self.emit("log", {"line": f"[rspipe] {len(self._previous)} finished "
                                      f"step(s) found in {path.name}"})

    def _reusable(self, expected: list[Path]) -> bool:
        """Are the outputs on disk complete enough to skip the step?

        Existence alone is not enough. A run stopped while RealityScan was
        writing leaves a file that is present but truncated, and skipping on
        that would hand the next set a broken component. Anything suspiciously
        small is treated as unfinished and redone.
        """
        if not expected:
            return False
        for p in expected:
            try:
                if p.stat().st_size < 1024:
                    self.emit("log", {"line": f"[rspipe] {p.name} is only "
                                              f"{p.stat().st_size} bytes - redoing"})
                    return False
            except OSError:
                return False
        return True

    def _run_all(self) -> None:
        out_root = Path(self.cfg.export.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        if self.cfg.run.skip_existing:
            self._load_previous(out_root)

        self.results = [
            StepResult(
                number=c.number, name=c.name, kind="set",
                index_from=c.index_from, index_to=c.index_to,
                n_images=c.n_images, n_new=len(c.new_images),
                n_overlap=len(c.overlap_images), n_loop=len(c.loop_images),
            )
            for c in self.chunks
        ]
        merge_result: StepResult | None = None
        if self.cfg.merge.enabled and len(self.chunks) > 1:
            merge_result = StepResult(number=len(self.chunks),
                                      name=self.cfg.merge.dir_name, kind="merge")
            self.results.append(merge_result)

        self.emit("pipeline_start", {"total": len(self.results)})
        t0 = time.time()

        # Each tiling chains on its own. An offset or differently-sized pass is
        # a second sweep of the same images: its first set has no predecessor in
        # the base pass, and handing it one would carry the base pass's frame
        # into a set that is meant to be an independent check on it.
        prev: dict[str, tuple[Path, str] | None] = {}
        first: dict[str, tuple[Path, str] | None] = {}
        for chunk, result in zip(self.chunks, self.results):
            if self._cancel.is_set():
                result.status = "cancelled"
                self.emit("step_end", {"result": result})
                continue
            key = chunk.pass_name
            pc, pn = prev.get(key) or (None, None)
            fc, fn = first.get(key) or (None, None)
            paths = chunk_paths(chunk, self.cfg)
            try:
                self._run_chunk(chunk, result, paths, pc, pn, fc, fn)
            except Exception as exc:                      # noqa: BLE001
                result.status = "failed"
                result.message = f"{type(exc).__name__}: {exc}"
                self.emit("log", {"line": f"[rspipe] {result.name}: {result.message}\n"
                                          + traceback.format_exc()})
                self.emit("step_end", {"result": result})
            if paths.component.exists():
                prev[key] = (paths.component, chunk.name)
                first.setdefault(key, (paths.component, chunk.name))
            elif self.cfg.chain.mode == "component":
                self.emit("log", {"line": f"[rspipe] {chunk.name}: no component on disk, "
                                          f"the next set starts from scratch"})
                prev[key] = None
            self._write_summary(out_root)

        if merge_result is not None and not self._cancel.is_set():
            try:
                self._run_merge(merge_result)
            except Exception as exc:                      # noqa: BLE001
                merge_result.status = "failed"
                merge_result.message = f"{type(exc).__name__}: {exc}"
                self.emit("step_end", {"result": merge_result})
            self._write_summary(out_root)

        self.emit("pipeline_end", {
            "seconds": time.time() - t0,
            "ok": sum(r.status in ("ok", "skipped") for r in self.results),
            "total": len(self.results),
        })

    # ---- one image set ---------------------------------------------------
    def _run_chunk(self, chunk: Chunk, result: StepResult, paths,
                   prev_component: Path | None, prev_name: str | None,
                   first_component: Path | None = None,
                   first_name: str | None = None) -> None:
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.crash.mkdir(parents=True, exist_ok=True)

        expected = paths.expected_outputs(self.cfg)
        if (self.cfg.run.skip_existing and expected
                and all(p.exists() for p in expected) and self._reusable(expected)):
            result.status = "skipped"
            result.outputs = [str(p) for p in expected]
            prior = self._previous.get(result.name)
            if prior:
                result.restore(prior)
                result.message = "done in an earlier run"
            else:
                result.message = "outputs already present"
            self.emit("step_end", {"result": result})
            return

        write_imagelist(chunk.images, paths.imagelist_all)
        write_imagelist(chunk.new_images, paths.imagelist_new)

        pairs = mask_pairs(chunk.images, self.cfg.masks)
        result.n_masks = len(pairs)
        if pairs:
            write_mask_rscmd(pairs, self.cfg.masks.usage, paths.masks_rscmd)

        result.chained = bool(prev_component and self.cfg.chain.mode == "component"
                              and chunk.overlap_indices)
        result.looped = bool(result.chained and chunk.loop_indices
                             and first_component and first_name != prev_name)
        seeds = seeds_for(chunk, parse_seed_spec(self.cfg.seed.spec)) \
            if self.cfg.seed.enabled else []
        result.seeds = [s.stem for s in seeds]
        result.imported_names = [n for n in
                                 ([prev_name] if result.chained else [])
                                 + ([first_name] if result.looped else [])
                                 + result.seeds
                                 if n]
        args = build_command(chunk, self.cfg, paths, prev_component, prev_name,
                             has_masks=bool(pairs),
                             first_component=first_component,
                             first_component_name=first_name,
                             seeds=seeds)
        (paths.root / "command.bat").write_text(command_as_batch(args), encoding="utf-8")

        result.status = "running"
        self.emit("step_start", {"result": result, "chunk": chunk})
        self._spawn(args, paths.log, result)

        found = [p for p in expected if p.exists()]
        result.outputs = [str(p) for p in found]
        self._finish(result, expected, found)

        # Releasing the overlap pose lock fails if -align dropped any of those
        # cameras: they are no longer registered, and -lockPoseForContinue
        # rejects such a selection, which takes the whole run down before it
        # exports anything. Redo the set without the lock - it costs a little
        # registration but always produces a component for the chain.
        if (result.status == "failed" and result.chained
                and self.cfg.chain.lock_overlap_pose and not self._cancel.is_set()):
            self.emit("log", {"line": f"[rspipe] {result.name}: failed with the "
                                      f"overlap pose lock, retrying without it"})
            args = build_command(chunk, self.cfg, paths, prev_component, prev_name,
                                 has_masks=bool(pairs),
                                 first_component=first_component,
                                 first_component_name=first_name,
                                 seeds=seeds, lock_overlap=False)
            (paths.root / "command_nolock.bat").write_text(
                command_as_batch(args), encoding="utf-8")
            before = result.seconds
            result.components.clear()
            self._spawn(args, paths.root / f"{chunk.name}_nolock.log", result)
            result.seconds += before
            found = [p for p in expected if p.exists()]
            result.outputs = [str(p) for p in found]
            self._finish(result, expected, found)
            if result.status == "ok":
                result.message = ("recovered without the overlap pose lock"
                                  + (f"; {result.message}" if result.message else ""))

        # cameras beyond this set's own images can only have come from a seed
        if result.seeds and result.best:
            result.seed_joined = max(0, result.best.cams - result.n_images)
            note = (f"seed joined {result.seed_joined} cameras"
                    if result.seed_joined else
                    f"seed did NOT join ({', '.join(result.seeds)})")
            result.message = f"{result.message}; {note}" if result.message else note

        if result.status == "ok":
            self._retry_align(result, paths, expected)

        # A loop that does not close is not an error: the loop-closure images
        # simply form their own component and drop out of the export. Say so,
        # because the camera count alone looks like a normal partial failure.
        if result.looped and result.best:
            chain_only = result.n_images - result.n_loop
            if result.best.cams <= chain_only:
                note = (f"loop did NOT close: {result.n_loop} loop-closure images "
                        f"stayed outside the component")
            else:
                note = (f"loop closed: {result.best.cams - chain_only} of "
                        f"{result.n_loop} loop-closure images joined")
            result.message = f"{result.message}; {note}" if result.message else note

        self.emit("step_end", {"result": result})

    # ---- follow-up alignment --------------------------------------------
    def _fresh_components(self, result: StepResult) -> list[ComponentStat]:
        """Components this set produced, i.e. not the ones it imported.

        RealityScan sometimes echoes the report twice, so collapse by name -
        otherwise the retry would try to delete the same component twice and
        the second -selectComponent would abort the run.
        """
        phases = [c.phase for c in result.components]
        last = phases[-1] if phases else ""
        by_name: dict[str, ComponentStat] = {}
        for c in result.components:
            if c.phase != last or c.name in result.imported_names:
                continue
            keep = by_name.get(c.name)
            if keep is None or c.cams > keep.cams:
                by_name[c.name] = c
        return list(by_name.values())

    def _retry_align(self, result: StepResult, paths, expected: list[Path]) -> None:
        cfg = self.cfg.retry
        if not cfg.enabled or not self.cfg.export.project or not paths.project.exists():
            return

        # A seeded set also holds the seed's cameras, so measuring against its
        # own image count alone would put the ratio above 1 and never retry,
        # even with its own images missing.
        seed_cams = sum(c.cams for c in result.components
                        if c.name in result.seeds and c.phase == "aligned")
        expected_cams = result.n_images + seed_cams

        for attempt in range(1, cfg.max_attempts + 1):
            if self._cancel.is_set():
                return
            best = result.best
            if not best or not expected_cams:
                return
            ratio = best.cams / expected_cams
            if ratio >= cfg.min_ratio:
                return

            fresh = self._fresh_components(result)
            drop: list[str] = []
            if cfg.drop_minor_components and len(fresh) > 1:
                keep = max(fresh, key=lambda c: c.cams)
                drop = [c.name for c in fresh if c.name != keep.name]

            before = best.cams
            self.emit("log", {"line": f"[rspipe] {result.name}: {before}/{expected_cams}"
                                      f" ({ratio:.1%}) < {cfg.min_ratio:.0%}, retry"
                                      f" {attempt}/{cfg.max_attempts}"
                                      + (f", dropping {len(drop)} minor component(s)"
                                         if drop else "")})
            args = build_retry_command(self.cfg, paths, attempt, drop)
            (paths.root / f"command_retry{attempt}.bat").write_text(
                command_as_batch(args), encoding="utf-8")

            log = paths.root / f"{paths.project.stem}_retry{attempt}.log"
            t_before = result.seconds
            self._spawn(args, log, result)
            result.seconds += t_before          # _spawn overwrites, keep the total
            result.retries = attempt

            after = result.best.cams if result.best else before
            result.retry_gain.append(after - before)
            self.emit("log", {"line": f"[rspipe] {result.name}: retry {attempt} -> "
                                      f"{after}/{expected_cams} ({after - before:+d})"})
            if result.exit_code != 0:
                result.message = (result.message + "; " if result.message else "") + \
                    f"retry {attempt} exit {result.exit_code}"
                return

        result.outputs = [str(p) for p in expected if p.exists()]

    # ---- merge -----------------------------------------------------------
    def _run_merge(self, result: StepResult) -> None:
        paths = merge_paths(self.cfg)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.crash.mkdir(parents=True, exist_ok=True)

        components: list[Path] = []
        for chunk in self.chunks:
            c = chunk_paths(chunk, self.cfg).component
            if c.exists():
                components.append(c)
        for extra in self.cfg.merge.extra_components:
            p = Path(extra)
            if not p.exists():
                self.emit("log", {"line": f"[rspipe] bridge component not found, "
                                          f"skipping: {p}"})
                continue
            components.append(p)
            self.emit("log", {"line": f"[rspipe] bridge component: {p.name}"})
        # distinct images across all sets - the overlap must not be counted twice
        result.n_images = len({str(p) for c in self.chunks for p in c.images})
        if len(components) < 2:
            result.status = "skipped"
            result.message = f"only {len(components)} component(s) to merge"
            self.emit("step_end", {"result": result})
            return

        expected = [paths.component]
        if self.cfg.export.sparse:
            expected.append(paths.sparse)
        if (self.cfg.run.skip_existing and all(p.exists() for p in expected)
                and self._reusable(expected)):
            result.status = "skipped"
            result.outputs = [str(p) for p in expected]
            prior = self._previous.get(result.name)
            if prior:
                result.restore(prior)
                result.message = "done in an earlier run"
            else:
                result.message = "outputs already present"
            self.emit("step_end", {"result": result})
            return

        args = build_merge_command(self.cfg, components, paths)
        (paths.root / "command.bat").write_text(command_as_batch(args), encoding="utf-8")

        result.status = "running"
        result.message = f"{len(components)} components"
        self.emit("step_start", {"result": result, "chunk": None})
        self._spawn(args, paths.log, result)

        found = [p for p in expected if p.exists()]
        result.outputs = [str(p) for p in found]
        self._finish(result, expected, found)

        # a merge that did not actually join anything looks successful otherwise
        merged = [c for c in result.components if c.phase == "merged"]
        imported = [c for c in result.components if c.phase == "imported"]
        if merged and imported:
            best_merged = max(c.cams for c in merged)
            best_input = max(c.cams for c in imported)
            if best_merged <= best_input:
                result.status = "failed" if result.status == "ok" else result.status
                result.message = (f"merge did not join: largest component still "
                                  f"{best_merged} cameras (inputs up to {best_input})")
        self.emit("step_end", {"result": result})

    # ---- process plumbing ------------------------------------------------
    def _spawn(self, args: list[str], log_path: Path, result: StepResult) -> None:
        t0 = time.time()
        timeout = self.cfg.run.timeout_min * 60 or None
        watchdog: threading.Timer | None = None
        with log_path.open("w", encoding="utf-8", errors="replace") as logf:
            self._proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CREATE_NO_WINDOW,
            )
            assert self._proc.stdout is not None
            if timeout:
                # RealityScan can work for hours without writing a single line
                # (a big -mergeComponents does), so the deadline cannot be
                # checked from inside the read loop alone.
                #
                # It also must not be a plain deadline. What this is meant to
                # catch is a hung process - the headless build blocks forever on
                # a dialog nobody can see - and a hung process burns no CPU. A
                # merge that is still working is killed by a wall-clock limit
                # for no reason: one was cut off at 6 hours seconds after it had
                # finished merging but before it had written anything. So the
                # timer re-arms as long as the process keeps consuming CPU.
                # Working and hung are far apart: the stuck run burned 0.008
                # cores while a live merge held 8. Anything above a twentieth of
                # one core counts as progress, which no blocked process reaches
                # and no real phase - even a disk-bound project save - misses.
                busy = 0.05 * timeout

                def fire() -> None:
                    nonlocal watchdog
                    used = _cpu_seconds(self._proc)
                    if used is not None and used - fire.last_cpu > busy:  # type: ignore[attr-defined]
                        fire.last_cpu = used                              # type: ignore[attr-defined]
                        watchdog = threading.Timer(timeout, fire)
                        watchdog.daemon = True
                        watchdog.start()
                        return
                    result.message = (f"no CPU progress for "
                                      f"{self.cfg.run.timeout_min} min")
                    self.cancel()

                fire.last_cpu = _cpu_seconds(self._proc) or 0.0          # type: ignore[attr-defined]
                watchdog = threading.Timer(timeout, fire)
                watchdog.daemon = True
                watchdog.start()
            for line in self._proc.stdout:
                line = line.rstrip("\r\n")
                logf.write(line + "\n")
                logf.flush()
                self._handle_line(line, result)
            self._proc.wait()
            if watchdog:
                watchdog.cancel()
        result.exit_code = self._proc.returncode
        result.seconds = time.time() - t0
        self._proc = None

    def _finish(self, result: StepResult, expected: list[Path], found: list[Path]) -> None:
        if self._cancel.is_set():
            result.status = "cancelled"
        elif result.exit_code == 0 and (not expected or found):
            result.status = "ok"
            if expected and len(found) != len(expected):
                missing = [p.name for p in expected if not p.exists()]
                result.message = "missing: " + ", ".join(missing)
        else:
            result.status = "failed"
            if not result.message:
                result.message = f"exit code {result.exit_code}"

    def _handle_line(self, line: str, result: StepResult) -> None:
        m = _STAT_RE.match(line)
        if m:
            stat = ComponentStat.parse(m.group(1))
            result.components.append(stat)
            self.emit("stat", {"result": result, "stat": stat})
            return
        p = _PROGRESS_RE.match(line)
        if p:
            self.emit("progress", {
                "result": result,
                "alg": p.group(1),
                "fraction": float(p.group(2)),
                "elapsed": float(p.group(3)),
                "eta": float(p.group(4)),
                "event": p.group(5),
            })
            return
        if line.strip():
            self.emit("log", {"line": line, "result": result})

    # ---- summaries -------------------------------------------------------
    def _write_summary(self, out_root: Path) -> None:
        data = [r.to_dict() for r in self.results]
        (out_root / "summary.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rows = ["name,kind,indices,images,new,overlap,loop,masks,chained,retries,"
                "seed_joined,registered,ratio,points,mean_error_px,seconds,"
                "status,message"]
        for r in self.results:
            b = r.best
            ratio = f"{b.cams / r.n_images:.4f}" if b and r.n_images else "0"
            rows.append(",".join([
                r.name, r.kind,
                f"{r.index_from}-{r.index_to}" if r.kind == "set" else "",
                str(r.n_images), str(r.n_new), str(r.n_overlap), str(r.n_loop),
                str(r.n_masks),
                ("loop" if r.looped else "yes") if r.chained else "no",
                str(r.retries), str(r.seed_joined),
                str(b.cams if b else 0), ratio,
                str(b.pts if b else 0),
                f"{b.mean:.4f}" if b else "0",
                f"{r.seconds:.1f}", r.status,
                '"' + r.message.replace('"', "'") + '"' if r.message else "",
            ]))
        (out_root / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
