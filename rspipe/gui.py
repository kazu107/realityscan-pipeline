"""Tkinter front end for the RealityScan headless batch pipeline."""

from __future__ import annotations

import queue
import tkinter as tk
import webbrowser
from dataclasses import fields
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .cli import build_command, chunk_paths, command_as_batch
from .config import (
    PipelineConfig,
    find_realityscan,
    focal35_from_fov,
    fov_from_focal35,
    parse_seed_spec,
)
from .dataset import Chunk, Scan, make_chunks, mask_pairs, scan_folder, seeds_for
from .formats import REGISTRATION_FORMATS, SPARSE_EXTENSIONS
from .runner import PipelineRunner

PAD = dict(padx=6, pady=3)


def _row(parent, r, label, widget, hint="", hint_col=2):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="e", **PAD)
    widget.grid(row=r, column=1, sticky="w", **PAD)
    if hint:
        ttk.Label(parent, text=hint, foreground="#777").grid(
            row=r, column=hint_col, sticky="w", **PAD
        )
    return widget


class App(ttk.Frame):
    _KIND_ZERO = {tk.IntVar: 0, tk.DoubleVar: 0.0, tk.BooleanVar: False}

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.cfg = PipelineConfig()
        if not self.cfg.run.exe:
            self.cfg.run.exe = find_realityscan()
        self.scan: Scan | None = None
        self.chunks: list[Chunk] = []
        self.runner: PipelineRunner | None = None
        self.events: queue.Queue = queue.Queue()
        self.vars: dict[str, tk.Variable] = {}
        self._done = 0

        self._build()
        self._config_to_ui()
        self.after(100, self._drain)

    # ---- variable plumbing ----------------------------------------------
    def V(self, key, kind=tk.StringVar, default=""):
        if key not in self.vars:
            if default == "" and kind in self._KIND_ZERO:
                default = self._KIND_ZERO[kind]
            self.vars[key] = kind(value=default)
        return self.vars[key]

    def _config_to_ui(self):
        for section in fields(PipelineConfig):
            obj = getattr(self.cfg, section.name)
            for f in fields(obj):
                key = f"{section.name}.{f.name}"
                if key in self.vars:
                    self.vars[key].set(getattr(obj, f.name))
        fmt = REGISTRATION_FORMATS.get(self.cfg.export.registration_format)
        if fmt:
            self.V("_reg_label").set(fmt.label)
        self.txt_seed.delete("1.0", "end")
        self.txt_seed.insert("1.0", self.cfg.seed.spec)
        self.txt_bridge.delete("1.0", "end")
        self.txt_bridge.insert("1.0", "\n".join(self.cfg.merge.extra_components))
        self._sync_fov()
        self._sync_format_note()

    def _ui_to_config(self):
        for section in fields(PipelineConfig):
            obj = getattr(self.cfg, section.name)
            for f in fields(obj):
                key = f"{section.name}.{f.name}"
                if key not in self.vars:
                    continue
                raw = self.vars[key].get()
                cur = getattr(obj, f.name)
                try:
                    if isinstance(cur, bool):
                        val = bool(raw)
                    elif isinstance(cur, int):
                        val = int(raw)
                    elif isinstance(cur, float):
                        val = float(raw)
                    else:
                        val = str(raw)
                except (TypeError, ValueError):
                    raise ValueError(f"{key}: invalid value {raw!r}")
                setattr(obj, f.name, val)
        label = self.V("_reg_label").get()
        for k, f in REGISTRATION_FORMATS.items():
            if f.label == label:
                self.cfg.export.registration_format = k
                break
        # a Text widget cannot be backed by a Variable
        self.cfg.seed.spec = self.txt_seed.get("1.0", "end").rstrip("\n")
        self.cfg.merge.extra_components = [
            ln.strip() for ln in self.txt_bridge.get("1.0", "end").splitlines()
            if ln.strip()]

    # ---- layout ----------------------------------------------------------
    def _build(self):
        self.master.title("RealityScan batch pipeline")
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="Load config", command=self.on_load).pack(side="left", **PAD)
        ttk.Button(top, text="Save config", command=self.on_save).pack(side="left", **PAD)
        ttk.Button(top, text="Show command", command=self.on_show_command).pack(side="left", **PAD)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=4)
        nb.add(self._tab_dataset(nb), text="1. Images")
        nb.add(self._tab_align(nb), text="2. Alignment")
        nb.add(self._tab_export(nb), text="3. Export & Merge")
        nb.add(self._tab_run(nb), text="4. Run")
        self.nb = nb

    # -- tab 1 -------------------------------------------------------------
    def _tab_dataset(self, parent):
        f = ttk.Frame(parent)
        g = ttk.LabelFrame(f, text="Source")
        g.pack(fill="x", padx=6, pady=4)
        r = 0
        _row(g, r, "Image folder",
             ttk.Entry(g, textvariable=self.V("dataset.image_dir"), width=70))
        ttk.Button(g, text="...", width=3,
                   command=lambda: self._pick_dir("dataset.image_dir")).grid(
            row=r, column=2, sticky="w", **PAD)
        r += 1
        _row(g, r, "File name pattern",
             ttk.Entry(g, textvariable=self.V("dataset.pattern"), width=70),
             "regex over the stem; needs (?P<index>...), optional prefix/view", 3)
        r += 1
        _row(g, r, "Extensions",
             ttk.Entry(g, textvariable=self.V("dataset.extensions"), width=30))
        r += 1
        ttk.Button(g, text="Scan folder", command=self.on_scan).grid(
            row=r, column=0, sticky="e", **PAD)
        self.lbl_scan = ttk.Label(g, text="(not scanned)")
        self.lbl_scan.grid(row=r, column=1, columnspan=3, sticky="w", **PAD)

        g2 = ttk.LabelFrame(f, text="Image sets (everything is counted in index units)")
        g2.pack(fill="x", padx=6, pady=4)
        r = 0
        for label, key, hint in (
            ("Index from", "dataset.index_start", ""),
            ("Index to", "dataset.index_end", "-1 = last index"),
            ("Index step", "dataset.index_step", "1 = every index"),
            ("Views", "dataset.views", "empty = all, e.g. 0,1,2 or 0-3"),
            ("Indices per set", "dataset.chunk_indices", "e.g. 100 -> 0-99, 90-189, ..."),
            ("Overlap indices", "dataset.overlap_indices", "re-used from the previous set"),
            ("Max sets", "dataset.max_chunks", "0 = no limit (use 1-2 for a trial)"),
        ):
            kind = tk.StringVar if key.endswith("views") else tk.IntVar
            _row(g2, r, label, ttk.Entry(g2, textvariable=self.V(key, kind), width=12), hint)
            r += 1

        g6 = ttk.LabelFrame(g2, text="Cover the set boundaries (each option costs "
                                     "one more alignment pass over everything)")
        g6.grid(row=r, column=0, columnspan=4, sticky="we", padx=6, pady=(8, 4))
        g6.columnconfigure(2, weight=1)
        _row(g6, 0, "Extra offset passes",
             ttk.Entry(g6, textvariable=self.V("dataset.extra_passes", tk.IntVar),
                       width=12),
             "0 = off. 1 tiles the images a second time, shifted half a stride,\n"
             "so every boundary of the first tiling lands mid-set in the second.\n"
             "A merge joins two sets through their overlap alone, and a few of\n"
             "those joins fail - 4 of 109 broke by 2-8x the camera spacing.",
             2)
        _row(g6, 1, "Extra set sizes",
             ttk.Entry(g6, textvariable=self.V("dataset.extra_chunk_sizes"),
                       width=22),
             'empty = off. Written size:overlap:passes, the tail optional:\n'
             '   "15, 35"          both inherit the settings above\n'
             '   "15:5, 40:16"     each size gets its own overlap\n'
             '   "40:16:1"         and its own offset pass\n'
             "An overlap tuned for 25 indices is a different fraction of 15 or\n"
             "40, so setting it per size is usually what you want.",
             2)
        r += 1
        _row(g2, r, "Set chaining",
             self._combo_str(g2, "chain.mode", ["component", "images"], width=14),
             "component: import the previous set's component, keep only the "
             "overlap indices, then add the new images\n"
             "images: every set aligned from scratch, overlap indices are plain images")
        r += 1
        ttk.Checkbutton(g2, text="lock the overlap cameras' pose during alignment",
                        variable=self.V("chain.lock_overlap_pose", tk.BooleanVar, False)
                        ).grid(row=r, column=1, columnspan=2, sticky="w", **PAD)
        r += 1
        ttk.Checkbutton(g2, text="close the loop (give the last set the head of the "
                                 "first component as a second overlap)",
                        variable=self.V("chain.close_loop", tk.BooleanVar, False)
                        ).grid(row=r, column=1, columnspan=3, sticky="w", **PAD)
        r += 1
        _row(g2, r, "Loop overlap indices",
             ttk.Entry(g2, textvariable=self.V("chain.loop_overlap_indices", tk.IntVar),
                       width=12),
             "0 = same as Overlap indices")
        r += 1
        ttk.Button(g2, text="Build set list", command=self.on_build_chunks).grid(
            row=r, column=0, sticky="e", **PAD)
        self.lbl_chunks = ttk.Label(g2, text="")
        self.lbl_chunks.grid(row=r, column=1, columnspan=3, sticky="w", **PAD)

        g5 = ttk.LabelFrame(f, text="Seed sets with existing components")
        g5.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g5, text="import the components listed below",
                        variable=self.V("seed.enabled", tk.BooleanVar, False)
                        ).grid(row=0, column=0, sticky="w", **PAD)
        ttk.Label(g5, text="one \"<set> = <path to .rsalign>\" per line; "
                           "<set> is the number, the name, or \"first\"",
                  foreground="#777").grid(row=0, column=1, sticky="w", **PAD)
        self.txt_seed = tk.Text(g5, height=4, wrap="none")
        self.txt_seed.grid(row=1, column=0, columnspan=2, sticky="we", **PAD)
        g5.columnconfigure(1, weight=1)
        ttk.Button(g5, text="Add component...", command=self.on_add_seed).grid(
            row=2, column=0, sticky="w", **PAD)

        g3 = ttk.LabelFrame(f, text="Masks")
        g3.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g3, text="attach a mask to each image",
                        variable=self.V("masks.enabled", tk.BooleanVar, False)
                        ).grid(row=0, column=1, sticky="w", **PAD)
        _row(g3, 1, "Mask name",
             ttk.Entry(g3, textvariable=self.V("masks.pattern"), width=30),
             "{name} = image file name, {stem} = without extension, {ext} = .jpg")
        _row(g3, 2, "Mask folder",
             ttk.Entry(g3, textvariable=self.V("masks.directory"), width=50),
             "empty = next to the image", 3)
        ttk.Button(g3, text="...", width=3,
                   command=lambda: self._pick_dir("masks.directory")).grid(
            row=2, column=2, sticky="w", **PAD)
        _row(g3, 3, "Used for",
             self._combo(g3, "masks.usage",
                         {0: "0 not used", 1: "1 alignment only",
                          2: "2 meshing only", 3: "3 alignment and meshing"}))

        cols = ("#", "name", "indices", "images", "new", "overlap", "loop", "masks")
        self.tv_chunks = ttk.Treeview(f, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (40, 190, 105, 65, 55, 65, 50, 55)):
            self.tv_chunks.heading(c, text=c)
            self.tv_chunks.column(c, width=w, anchor="w")
        self.tv_chunks.pack(fill="both", expand=True, padx=6, pady=4)
        return f

    # -- tab 2 -------------------------------------------------------------
    def _tab_align(self, parent):
        f = ttk.Frame(parent)

        g = ttk.LabelFrame(f, text="RealityScan")
        g.pack(fill="x", padx=6, pady=4)
        _row(g, 0, "RealityScan.exe",
             ttk.Entry(g, textvariable=self.V("run.exe"), width=70))
        ttk.Button(g, text="...", width=3,
                   command=lambda: self._pick_file("run.exe")).grid(row=0, column=2)
        ttk.Checkbutton(g, text="headless (no UI)",
                        variable=self.V("run.headless", tk.BooleanVar, True)
                        ).grid(row=1, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g, text="quit on error",
                        variable=self.V("run.quit_on_error", tk.BooleanVar, True)
                        ).grid(row=2, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g, text="clear cache after each set",
                        variable=self.V("run.clear_cache_after_chunk", tk.BooleanVar, False)
                        ).grid(row=3, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g, text="skip sets whose outputs already exist",
                        variable=self.V("run.skip_existing", tk.BooleanVar, True)
                        ).grid(row=4, column=1, sticky="w", **PAD)
        _row(g, 5, "Timeout per set (min)",
             ttk.Entry(g, textvariable=self.V("run.timeout_min", tk.IntVar), width=8),
             "0 = wait forever")

        g4 = ttk.LabelFrame(f, text="Follow-up alignment for under-registered sets")
        g4.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g4, text="re-align sets that fall short",
                        variable=self.V("retry.enabled", tk.BooleanVar, False)
                        ).grid(row=0, column=1, sticky="w", **PAD)
        _row(g4, 1, "Registration threshold",
             ttk.Entry(g4, textvariable=self.V("retry.min_ratio", tk.DoubleVar), width=8),
             "1.0 = require every image, 0.98 = accept 98%")
        _row(g4, 2, "Max extra passes",
             ttk.Entry(g4, textvariable=self.V("retry.max_attempts", tk.IntVar), width=8))
        ttk.Checkbutton(g4, text="delete all but the largest component before re-aligning",
                        variable=self.V("retry.drop_minor_components", tk.BooleanVar, True)
                        ).grid(row=3, column=1, columnspan=2, sticky="w", **PAD)
        ttk.Checkbutton(g4, text="force component rematch during the retries",
                        variable=self.V("retry.force_rematch", tk.BooleanVar, False)
                        ).grid(row=4, column=1, columnspan=2, sticky="w", **PAD)

        g2 = ttk.LabelFrame(f, text="Prior calibration (applied to every image)")
        g2.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g2, text="apply priors",
                        variable=self.V("priors.enabled", tk.BooleanVar, True)
                        ).grid(row=0, column=1, sticky="w", **PAD)
        _row(g2, 1, "Focal length 35mm",
             ttk.Entry(g2, textvariable=self.V("priors.focal35", tk.DoubleVar), width=12))
        self.V("priors.focal35").trace_add("write", lambda *_: self._sync_fov())
        self.lbl_fov = ttk.Label(g2, text="")
        self.lbl_fov.grid(row=1, column=2, sticky="w", **PAD)
        fovbox = ttk.Frame(g2)
        fovbox.grid(row=2, column=1, sticky="w", **PAD)
        self.ent_fov = ttk.Entry(fovbox, width=8)
        self.ent_fov.pack(side="left")
        ttk.Button(fovbox, text="FOV(deg) -> focal", command=self._fov_to_focal
                   ).pack(side="left", padx=4)
        _row(g2, 3, "Calibration prior",
             self._combo(g2, "priors.calibration_prior",
                         {0: "0 unknown", 1: "1 approximate", 2: "2 fixed"}))
        _row(g2, 4, "Calibration group",
             ttk.Entry(g2, textvariable=self.V("priors.calibration_group", tk.IntVar), width=8),
             "-1 = groupless, same number = shared intrinsics")
        _row(g2, 5, "Distortion prior",
             self._combo(g2, "priors.distortion_prior",
                         {0: "0 unknown", 1: "1 approximate", 2: "2 fixed"}))
        _row(g2, 6, "Distortion model",
             self._combo(g2, "priors.distortion_model",
                         {0: "0 none", 1: "1 division", 2: "2 Brown3", 3: "3 Brown4",
                          4: "4 Brown3+tangential", 5: "5 Brown4+tangential"}),
             "rendered pinhole views -> 0 none")
        _row(g2, 7, "Lens group",
             ttk.Entry(g2, textvariable=self.V("priors.lens_group", tk.IntVar), width=8))

        g3 = ttk.LabelFrame(f, text="Alignment settings")
        g3.pack(fill="x", padx=6, pady=4)
        r = 0
        for label, key, kind, hint in (
            ("Image downscale", "align.image_downscale", tk.IntVar, "1 = full resolution"),
            ("Max features / image", "align.max_features_per_image", tk.IntVar, ""),
            ("Max features / Mpx", "align.max_features_per_mpx", tk.IntVar, ""),
            ("Max reprojection error", "align.max_feature_reprojection_error", tk.DoubleVar, "px"),
            ("Preselector features", "align.preselector_features", tk.IntVar, ""),
        ):
            _row(g3, r, label, ttk.Entry(g3, textvariable=self.V(key, kind), width=12), hint)
            r += 1
        _row(g3, r, "Images overlap",
             self._combo_str(g3, "align.images_overlap", ["Low", "Medium", "High"]))
        r += 1
        _row(g3, r, "Detector sensitivity",
             self._combo_str(g3, "align.detector_sensitivity",
                             ["Low", "Medium", "High", "Ultra"]))
        r += 1
        _row(g3, r, "Feature detection quality",
             self._combo_str(g3, "align.feature_detection_quality", ["High", "Normal"]))
        r += 1
        _row(g3, r, "Distortion model (solver)",
             self._combo_str(g3, "align.distortion_model",
                             ["Division", "Brown3", "Brown4",
                              "Brown3WithTangential2", "Brown4WithTangential2",
                              "KplusBrown3WithTangential2", "KplusBrown4WithTangential2"]))
        r += 1
        ttk.Checkbutton(g3, text="force component rematch",
                        variable=self.V("align.force_component_rematch", tk.BooleanVar, False)
                        ).grid(row=r, column=1, sticky="w", **PAD)
        r += 1
        ttk.Checkbutton(g3, text="add reconstruction region after alignment",
                        variable=self.V("align.auto_recon_region", tk.BooleanVar, False)
                        ).grid(row=r, column=1, sticky="w", **PAD)
        return f

    # -- tab 3 -------------------------------------------------------------
    def _tab_export(self, parent):
        f = ttk.Frame(parent)
        g = ttk.LabelFrame(f, text="Output")
        g.pack(fill="x", padx=6, pady=4)
        _row(g, 0, "Output root",
             ttk.Entry(g, textvariable=self.V("export.out_root"), width=70),
             "one subfolder per image set", 3)
        ttk.Button(g, text="...", width=3,
                   command=lambda: self._pick_dir("export.out_root")).grid(
            row=0, column=2, sticky="w", **PAD)

        g2 = ttk.LabelFrame(f, text="What to export from the maximal component")
        g2.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g2, text="sparse point cloud",
                        variable=self.V("export.sparse", tk.BooleanVar, True)
                        ).grid(row=0, column=0, sticky="w", **PAD)
        self._combo_str(g2, "export.sparse_ext", SPARSE_EXTENSIONS).grid(
            row=0, column=1, sticky="w", **PAD)

        ttk.Checkbutton(g2, text="camera parameters (registration)",
                        variable=self.V("export.registration", tk.BooleanVar, True)
                        ).grid(row=1, column=0, sticky="w", **PAD)
        labels = [fm.label for fm in REGISTRATION_FORMATS.values()]
        cb = ttk.Combobox(g2, values=labels, width=48, state="readonly",
                          textvariable=self.V("_reg_label", tk.StringVar, labels[0]))
        cb.grid(row=1, column=1, sticky="w", **PAD)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._sync_format_note())
        self.lbl_fmt = ttk.Label(g2, text="", foreground="#777", wraplength=520,
                                 justify="left")
        self.lbl_fmt.grid(row=2, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g2, text="also write undistorted images (large!)",
                        variable=self.V("export.registration_export_images",
                                        tk.BooleanVar, False)
                        ).grid(row=3, column=1, sticky="w", **PAD)

        ttk.Checkbutton(g2, text="component (.rsalign)",
                        variable=self.V("export.component", tk.BooleanVar, True)
                        ).grid(row=4, column=0, sticky="w", **PAD)
        ttk.Label(g2, text="always written when chaining or merging is on",
                  foreground="#777").grid(row=4, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g2, text="project (.rsproj)",
                        variable=self.V("export.project", tk.BooleanVar, True)
                        ).grid(row=5, column=0, sticky="w", **PAD)

        g3 = ttk.LabelFrame(f, text="Merge all sets when the run finishes")
        g3.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g3, text="import every set's component and run -mergeComponents",
                        variable=self.V("merge.enabled", tk.BooleanVar, True)
                        ).grid(row=0, column=1, sticky="w", **PAD)
        _row(g3, 1, "Output subfolder",
             ttk.Entry(g3, textvariable=self.V("merge.dir_name"), width=20))
        ttk.Checkbutton(g3, text="force component rematch during the merge",
                        variable=self.V("merge.force_rematch", tk.BooleanVar, True)
                        ).grid(row=2, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g3, text="run -align after merging (slow, refines everything)",
                        variable=self.V("merge.align_after_merge", tk.BooleanVar, False)
                        ).grid(row=3, column=1, sticky="w", **PAD)
        _row(g3, 4, "Features source",
             self._combo(g3, "merge.feature_source",
                         {-1: "-1 leave as imported", 0: "0 merge using overlaps",
                          1: "1 use component features", 2: "2 use all image features"}),
             "applied to every input right before -mergeComponents", 3)

        ttk.Label(g3, text="Bridge components").grid(row=5, column=0, sticky="ne", **PAD)
        self.txt_bridge = tk.Text(g3, height=3, width=64, wrap="none")
        self.txt_bridge.grid(row=5, column=1, columnspan=2, sticky="we", **PAD)
        ttk.Label(g3, text="one .rsalign per line, imported alongside the sets.\n"
                           "Use one spanning a boundary the merge splits at.",
                  foreground="#666").grid(row=6, column=1, sticky="w", **PAD)
        g3.columnconfigure(1, weight=1)
        self._sync_format_note()
        return f

    # -- tab 4 -------------------------------------------------------------
    def _tab_run(self, parent):
        f = ttk.Frame(parent)
        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=6, pady=4)
        self.btn_start = ttk.Button(bar, text="Start", command=self.on_start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(bar, text="Stop", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(bar, text="Open output folder", command=self.on_open_out).pack(side="left", padx=4)
        self.lbl_status = ttk.Label(bar, text="idle")
        self.lbl_status.pack(side="left", padx=12)

        self.pb_overall = ttk.Progressbar(f, mode="determinate")
        self.pb_overall.pack(fill="x", padx=6)
        self.pb_chunk = ttk.Progressbar(f, mode="determinate", maximum=1.0)
        self.pb_chunk.pack(fill="x", padx=6, pady=2)

        cols = ("step", "images", "new", "chain", "registered", "points",
                "mean px", "sec", "status")
        self.tv_res = ttk.Treeview(f, columns=cols, show="headings", height=9)
        for c, w in zip(cols, (190, 60, 55, 55, 100, 85, 70, 60, 220)):
            self.tv_res.heading(c, text=c)
            self.tv_res.column(c, width=w, anchor="w")
        self.tv_res.pack(fill="x", padx=6, pady=4)

        self.txt = tk.Text(f, height=14, wrap="none", bg="#111", fg="#ddd",
                           insertbackground="#ddd")
        self.txt.pack(fill="both", expand=True, padx=6, pady=4)
        return f

    # ---- small helpers ---------------------------------------------------
    def _combo(self, parent, key, mapping: dict[int, str]):
        var = self.V(key, tk.IntVar)
        disp = tk.StringVar(value=mapping.get(var.get(), ""))
        cb = ttk.Combobox(parent, values=list(mapping.values()), width=24,
                          state="readonly", textvariable=disp)

        def on_pick(_e=None):
            for k, v in mapping.items():
                if v == disp.get():
                    var.set(k)

        def on_var(*_a):
            disp.set(mapping.get(var.get(), ""))

        cb.bind("<<ComboboxSelected>>", on_pick)
        var.trace_add("write", on_var)
        return cb

    def _combo_str(self, parent, key, values, width=24):
        return ttk.Combobox(parent, values=list(values), width=width, state="readonly",
                            textvariable=self.V(key, tk.StringVar, values[0]))

    def _pick_dir(self, key):
        d = filedialog.askdirectory(initialdir=self.V(key).get() or ".")
        if d:
            self.V(key).set(str(Path(d)))

    def _pick_file(self, key):
        p = filedialog.askopenfilename(initialdir=str(Path(self.V(key).get() or ".").parent))
        if p:
            self.V(key).set(str(Path(p)))

    def _sync_fov(self):
        try:
            fov = fov_from_focal35(float(self.V("priors.focal35").get()))
            self.lbl_fov.config(text=f"= {fov:.2f} deg horizontal FOV")
        except (ValueError, ZeroDivisionError, tk.TclError):
            self.lbl_fov.config(text="")

    def _fov_to_focal(self):
        try:
            self.V("priors.focal35").set(round(focal35_from_fov(float(self.ent_fov.get())), 4))
        except ValueError:
            messagebox.showerror("FOV", "enter a number, e.g. 100")

    def _sync_format_note(self):
        label = self.V("_reg_label").get()
        for fm in REGISTRATION_FORMATS.values():
            if fm.label == label:
                note = fm.note or ""
                if fm.writes_images:
                    note += ("\n(this format normally dumps undistorted images; "
                             "the pipeline turns that off unless ticked)")
                self.lbl_fmt.config(text=note)
                return

    def log(self, text: str):
        self.txt.insert("end", text + "\n")
        if int(self.txt.index("end-1c").split(".")[0]) > 4000:
            self.txt.delete("1.0", "1000.0")
        self.txt.see("end")

    # ---- actions ---------------------------------------------------------
    def on_scan(self):
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        self.scan = scan_folder(self.cfg.dataset)
        self.lbl_scan.config(text=self.scan.summary())
        if self.scan.entries:
            self.V("dataset.index_start").set(self.scan.indices[0])
        self.on_build_chunks()

    def on_build_chunks(self):
        if not self.scan:
            return
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        self.chunks = make_chunks(self.scan, self.cfg.dataset, self.cfg.chain)
        self.tv_chunks.delete(*self.tv_chunks.get_children())
        total_masks = 0
        for c in self.chunks:
            n_masks = len(mask_pairs(c.images, self.cfg.masks))
            total_masks += n_masks
            self.tv_chunks.insert("", "end", values=(
                c.number, c.name, f"{c.index_from}-{c.index_to}", c.n_images,
                len(c.new_images), len(c.overlap_images), len(c.loop_images),
                n_masks))
        total = sum(c.n_images for c in self.chunks)
        extra = f", {total_masks} masks found" if self.cfg.masks.enabled else ""
        # Spell out the tilings. A comma typed where a colon belonged turned
        # "91:38:1" into a size-1 tiling and quietly added 1807 sets, which the
        # set count alone does not make obvious.
        by_pass: dict[str, int] = {}
        for c in self.chunks:
            by_pass[c.pass_name] = by_pass.get(c.pass_name, 0) + 1
        passes = ""
        if len(by_pass) > 1:
            passes = "  |  " + ", ".join(
                f"{k or 'base'} {v}" for k, v in by_pass.items())
        self.lbl_chunks.config(
            text=f"{len(self.chunks)} sets, {total} image slots{extra}{passes}")

    def on_add_seed(self):
        p = filedialog.askopenfilename(
            title="Component to hand to a set",
            filetypes=[("RealityScan component", "*.rsalign"), ("All", "*.*")])
        if not p:
            return
        sets = [c.name for c in self.chunks]
        win = tk.Toplevel(self)
        win.title("Which set?")
        ttk.Label(win, text="Give this component to:").pack(**PAD)
        var = tk.StringVar(value=sets[0] if sets else "first")
        cb = ttk.Combobox(win, values=sets or ["first"], width=40,
                          state="readonly" if sets else "normal", textvariable=var)
        cb.pack(**PAD)

        def add():
            line = f"{var.get()} = {Path(p)}"
            cur = self.txt_seed.get("1.0", "end").rstrip("\n")
            self.txt_seed.delete("1.0", "end")
            self.txt_seed.insert("1.0", (cur + "\n" + line).strip("\n"))
            self.V("seed.enabled").set(True)
            win.destroy()

        ttk.Button(win, text="Add", command=add).pack(**PAD)

    def on_show_command(self):
        if not self.chunks:
            messagebox.showinfo("Command", "Scan a folder and build the set list first.")
            return
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        if not self.cfg.export.out_root:
            messagebox.showerror("Command", "Set the output root on the Export tab.")
            return
        win = tk.Toplevel(self)
        win.title("Commands")
        t = tk.Text(win, wrap="none", width=120, height=40)
        t.pack(fill="both", expand=True)
        first = chunk_paths(self.chunks[0], self.cfg).component, self.chunks[0].name
        show = self.chunks[:2] + ([self.chunks[-1]] if len(self.chunks) > 2 else [])
        for chunk in show:
            paths = chunk_paths(chunk, self.cfg)
            prev = None, None
            if chunk.number > 0:
                pc = chunk_paths(self.chunks[chunk.number - 1], self.cfg)
                prev = pc.component, self.chunks[chunk.number - 1].name
            seeds = seeds_for(chunk, parse_seed_spec(self.cfg.seed.spec)) \
                if self.cfg.seed.enabled else []
            args = build_command(chunk, self.cfg, paths, prev[0], prev[1],
                                 has_masks=bool(mask_pairs(chunk.images, self.cfg.masks)),
                                 first_component=first[0], first_component_name=first[1],
                                 seeds=seeds)
            t.insert("end", f"REM ===== {chunk.name}\n{command_as_batch(args)}\n\n")

    def on_save(self):
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         initialdir=str(Path(__file__).parent.parent / "presets"))
        if p:
            self.cfg.save(p)
            self.log(f"[rspipe] saved config -> {p}")

    def on_load(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")],
                                       initialdir=str(Path(__file__).parent.parent / "presets"))
        if not p:
            return
        self.cfg = PipelineConfig.load(p)
        self._config_to_ui()
        self.log(f"[rspipe] loaded config <- {p}")

    def on_open_out(self):
        root = self.V("export.out_root").get()
        if root and Path(root).is_dir():
            webbrowser.open(f"file:///{root}")

    def on_start(self):
        if self.runner and self.runner.running:
            return
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        if not self.chunks:
            messagebox.showerror("Run", "No image sets. Scan a folder first.")
            return
        if not self.cfg.run.exe or not Path(self.cfg.run.exe).is_file():
            messagebox.showerror("Run", "RealityScan.exe not found.")
            return
        if not self.cfg.export.out_root:
            messagebox.showerror("Run", "Set the output root on the Export tab.")
            return

        self.tv_res.delete(*self.tv_res.get_children())
        self.runner = PipelineRunner(self.cfg, self.chunks, self._emit)
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.runner.start()

    def on_stop(self):
        if self.runner:
            self.log("[rspipe] stopping ...")
            self.runner.cancel()

    # ---- event pump (worker thread -> tk thread) -------------------------
    def _emit(self, event: str, payload: dict):
        self.events.put((event, payload))

    def _drain(self):
        try:
            while True:
                event, payload = self.events.get_nowait()
                self._on_event(event, payload)
        except queue.Empty:
            pass
        self.after(80, self._drain)

    def _row_values(self, r):
        b = r.best
        return (
            r.name if r.kind == "set" else f"[merge] {r.name}",
            r.n_images or "",
            r.n_new or "",
            "loop" if r.looped else ("yes" if r.chained else ""),
            f"{b.cams} ({b.cams / r.n_images:.0%})" if b and r.n_images else (b.cams if b else ""),
            b.pts if b else "",
            f"{b.mean:.3f}" if b else "",
            f"{r.seconds:.0f}" if r.seconds else "",
            (f"[retry x{r.retries}] " if r.retries else "")
            + r.status + (f" - {r.message}" if r.message else ""),
        )

    def _on_event(self, event: str, payload: dict):
        if event == "log":
            self.log(payload["line"])
        elif event == "progress":
            self.pb_chunk.config(value=payload["fraction"])
        elif event == "stat":
            s = payload["stat"]
            self.log(f"    [{s.phase}] {s.name}: {s.cams} cams, {s.pts} pts, "
                     f"mean {s.mean:.3f} px, track {s.track:.2f}")
        elif event == "pipeline_start":
            self._done = 0
            self.pb_overall.config(maximum=payload["total"], value=0)
            self.lbl_status.config(text=f"running 0/{payload['total']}")
            for r in self.runner.results:
                self.tv_res.insert("", "end", iid=str(r.number),
                                   values=self._row_values(r))
        elif event == "step_start":
            r = payload["result"]
            self.pb_chunk.config(value=0)
            self.lbl_status.config(text=f"{r.name} ({r.n_images} images)")
            self.tv_res.item(str(r.number), values=self._row_values(r))
            self.log(f"=== {r.name}: {r.n_images} images"
                     + (f", {r.n_new} new, {r.n_overlap} carried over" if r.kind == "set" else "")
                     + (f", {r.n_loop} loop closure" if r.n_loop else "")
                     + (f", {r.n_masks} masks" if r.n_masks else "")
                     + (" [chained]" if r.chained else "")
                     + (" [loop closed]" if r.looped else ""))
        elif event == "step_end":
            r = payload["result"]
            self.tv_res.item(str(r.number), values=self._row_values(r))
            self._done += 1
            self.pb_overall.config(value=self._done)
            self.log(f"--- {r.name}: {r.status} ({r.seconds:.0f}s) {r.message}")
        elif event == "pipeline_end":
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.lbl_status.config(
                text=f"done {payload['ok']}/{payload['total']} in {payload['seconds']:.0f}s")
            self.log(f"[rspipe] finished: {payload['ok']}/{payload['total']} ok")


def main():
    root = tk.Tk()
    root.geometry("1180x820")
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
