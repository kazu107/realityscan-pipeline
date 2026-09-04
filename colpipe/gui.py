"""tkinter front end for the COLMAP rig pipeline.

Five tabs, in the order the work happens: describe the images and the rig,
set feature extraction and matching, set the mapper, run it, look at what came
out. The last one matters as much as the rest - a reconstruction that is
plausible in numbers can still be visibly bent, and the only way to know is to
draw it.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import ColmapPipelineConfig, find_colmap
from .layout import scan_flat
from .rig import RigSpec, from_extractor_settings
from .runner import PipelineRunner, directions_from

PAD = {"padx": 6, "pady": 3}


def _row(parent, r, label, widget, hint="", hint_col=2):
    ttk.Label(parent, text=label).grid(row=r, column=0, sticky="e", **PAD)
    widget.grid(row=r, column=1, sticky="w", **PAD)
    if hint:
        ttk.Label(parent, text=hint, foreground="#777").grid(
            row=r, column=hint_col, sticky="w", **PAD)
    return widget


class App(ttk.Frame):
    _KIND_ZERO = {tk.IntVar: 0, tk.DoubleVar: 0.0, tk.BooleanVar: False}

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.cfg = ColmapPipelineConfig()
        if not self.cfg.run.exe:
            self.cfg.run.exe = find_colmap()
        self.vars: dict[str, tk.Variable] = {}
        self.events: queue.Queue = queue.Queue()
        self.runner: PipelineRunner | None = None
        self.model = None
        self._done = 0
        self._build()
        self._config_to_ui()
        self.after(100, self._drain)

    # ---- variables -------------------------------------------------------
    def V(self, key, kind=tk.StringVar, default=""):
        if key not in self.vars:
            if default == "" and kind in self._KIND_ZERO:
                default = self._KIND_ZERO[kind]
            self.vars[key] = kind(value=default)
        return self.vars[key]

    def _config_to_ui(self):
        from dataclasses import fields
        for section in fields(ColmapPipelineConfig):
            obj = getattr(self.cfg, section.name)
            for f in fields(obj):
                key = f"{section.name}.{f.name}"
                if key in self.vars:
                    self.vars[key].set(getattr(obj, f.name))
        # the stage boxes are one field spread over several widgets
        want = {x.strip() for x in self.cfg.run.stages.split(",") if x.strip()}
        for st, v in getattr(self, "_stage_vars", {}).items():
            v.set(st in want if want else True)

    def _ui_to_config(self):
        from dataclasses import fields
        for section in fields(ColmapPipelineConfig):
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
        boxes = getattr(self, "_stage_vars", {})
        if boxes:
            on = [st for st, v in boxes.items() if v.get()]
            if not on:
                raise ValueError("no stages are ticked, so there is nothing to run")
            self.cfg.run.stages = "" if len(on) == len(boxes) else ",".join(on)

    # ---- layout ----------------------------------------------------------
    def _build(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Load preset", command=self.on_load).pack(side="left")
        ttk.Button(bar, text="Save preset", command=self.on_save).pack(side="left", padx=4)
        ttk.Label(bar, text="COLMAP").pack(side="left", padx=(16, 4))
        ttk.Entry(bar, textvariable=self.V("run.exe"), width=54).pack(side="left")
        ttk.Button(bar, text="...", width=3,
                   command=lambda: self._pick_file("run.exe")).pack(side="left")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=4)
        nb.add(self._tab_images(nb), text="1. Images & Rig")
        nb.add(self._tab_match(nb), text="2. Feature & Match")
        nb.add(self._tab_mapper(nb), text="3. Mapper")
        nb.add(self._tab_run(nb), text="4. Run")
        nb.add(self._tab_view(nb), text="5. View")

    # -- tab 1 -------------------------------------------------------------
    def _tab_images(self, parent):
        f = ttk.Frame(parent)
        g = ttk.LabelFrame(f, text="Extracted views (one flat folder, "
                                   "name_frame_view.jpg)")
        g.pack(fill="x", padx=6, pady=4)
        _row(g, 0, "Image folder",
             ttk.Entry(g, textvariable=self.V("dataset.image_dir"), width=60), "", 3)
        ttk.Button(g, text="...", width=3,
                   command=lambda: self._pick_dir("dataset.image_dir")).grid(
            row=0, column=2, sticky="w", **PAD)
        for r, (label, key, hint) in enumerate((
            ("Frame from", "dataset.frame_from", ""),
            ("Frame to", "dataset.frame_to", "-1 = last frame"),
            ("Frame step", "dataset.frame_step", "1 = every frame"),
            ("Views", "dataset.views", "empty = every view found"),
        ), start=1):
            kind = tk.StringVar if key.endswith("views") else tk.IntVar
            _row(g, r, label, ttk.Entry(g, textvariable=self.V(key, kind), width=12), hint)
        ttk.Checkbutton(g, text="skip frames that are missing a view",
                        variable=self.V("dataset.require_all_views", tk.BooleanVar, True)
                        ).grid(row=5, column=1, columnspan=2, sticky="w", **PAD)
        ttk.Checkbutton(g, text="use masks",
                        variable=self.V("dataset.use_masks", tk.BooleanVar, False)
                        ).grid(row=6, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g, text="fill a white mask where one is missing",
                        variable=self.V("dataset.fill_missing_masks", tk.BooleanVar, True)
                        ).grid(row=6, column=2, sticky="w", **PAD)
        ttk.Label(g, text="COLMAP DROPS an image whose mask it cannot read, so a\n"
                          "partly-masked set silently loses those images.",
                  foreground="#777").grid(row=7, column=1, columnspan=3,
                                          sticky="w", **PAD)
        _row(g, 8, "Mask name",
             ttk.Entry(g, textvariable=self.V("dataset.mask_pattern"), width=24),
             "{name} = image file name")

        g2 = ttk.LabelFrame(f, text="Rig - the extraction directions are the rig")
        g2.pack(fill="x", padx=6, pady=4)
        _row(g2, 0, "Extractor settings",
             ttk.Entry(g2, textvariable=self.V("rig.settings_path"), width=60),
             "", 3)
        ttk.Button(g2, text="...", width=3,
                   command=lambda: self._pick_file("rig.settings_path")).grid(
            row=0, column=2, sticky="w", **PAD)
        _row(g2, 1, "Direction set",
             ttk.Entry(g2, textvariable=self.V("rig.settings_set_name"), width=24),
             'name as saved, e.g. "セット4" (names do not match their position)')
        _row(g2, 2, "Ring count",
             ttk.Entry(g2, textvariable=self.V("rig.ring_count", tk.IntVar), width=12),
             "or generate: N yaws evenly spaced, 0 = use the settings/list above")
        _row(g2, 3, "Ring pitch",
             ttk.Entry(g2, textvariable=self.V("rig.ring_pitch", tk.DoubleVar), width=12))
        _row(g2, 4, "Directions",
             ttk.Entry(g2, textvariable=self.V("rig.directions"), width=60),
             'or list them: "0:0, 45:0, 90:0"', 3)
        for r, (label, key, kind, hint) in enumerate((
            ("Horizontal FOV", "rig.fov", tk.DoubleVar, "degrees, as extracted"),
            ("Width", "rig.width", tk.IntVar, ""),
            ("Height", "rig.height", tk.IntVar, ""),
            ("Reference view", "rig.ref_view", tk.IntVar, "sensor with identity pose"),
        ), start=5):
            _row(g2, r, label, ttk.Entry(g2, textvariable=self.V(key, kind), width=12), hint)
        ttk.Checkbutton(g2, text="tie the directions into one rig",
                        variable=self.V("rig.coupled", tk.BooleanVar, False)
                        ).grid(row=9, column=1, sticky="w", **PAD)
        ttk.Label(g2, text="leave OFF. Measured on 150 frames of 1-mid-1 with the\n"
                           "same matches: coupled, runs of 40+ frames collapse onto\n"
                           "one point (residual 8.81 index-steps against RealityScan,\n"
                           "step p90 17.84); independent, 2.55 and 1.61, with 2.4x the\n"
                           "points. Refining the rig instead crashes COLMAP 4.2.0.\n"
                           "Independent still gives each direction its own camera",
                  foreground="#777").grid(row=9, column=2, sticky="w", **PAD)

        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Scan folder", command=self.on_scan).pack(side="left")
        ttk.Button(bar, text="Import directions",
                   command=self.on_import_directions).pack(side="left", padx=4)
        self.lbl_scan = ttk.Label(bar, text="")
        self.lbl_scan.pack(side="left", padx=10)

        cols = ("view", "yaw", "pitch", "rotation")
        self.tv_dirs = ttk.Treeview(f, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (60, 90, 90, 140)):
            self.tv_dirs.heading(c, text=c)
            self.tv_dirs.column(c, width=w, anchor="w")
        self.tv_dirs.pack(fill="both", expand=True, padx=6, pady=4)
        return f

    # -- tab 2 -------------------------------------------------------------
    def _tab_match(self, parent):
        f = ttk.Frame(parent)
        g = ttk.LabelFrame(f, text="SIFT feature extraction")
        g.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g, text="use GPU",
                        variable=self.V("feature.use_gpu", tk.BooleanVar, True)
                        ).grid(row=0, column=1, sticky="w", **PAD)
        _row(g, 1, "GPU index", ttk.Entry(g, textvariable=self.V("feature.gpu_index"),
                                          width=12), "-1 = all")
        _row(g, 2, "Max image size",
             ttk.Entry(g, textvariable=self.V("feature.max_image_size", tk.IntVar),
                       width=12), "-1 = full resolution")
        _row(g, 3, "Max features",
             ttk.Entry(g, textvariable=self.V("feature.max_num_features", tk.IntVar),
                       width=12), "per image")

        g2 = ttk.LabelFrame(f, text="Matching")
        g2.pack(fill="x", padx=6, pady=4)
        _row(g2, 0, "Method",
             self._combo_str(g2, "match.method",
                             ["sequential", "exhaustive", "vocab_tree"], 14),
             "sequential is right for a walked capture")
        _row(g2, 1, "Overlap",
             ttk.Entry(g2, textvariable=self.V("match.overlap", tk.IntVar), width=12),
             "how many neighbouring frames to compare against")
        ttk.Checkbutton(g2, text="quadratic overlap",
                        variable=self.V("match.quadratic_overlap", tk.BooleanVar, True)
                        ).grid(row=2, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g2, text="loop detection (needs a vocabulary tree)",
                        variable=self.V("match.loop_detection", tk.BooleanVar, False)
                        ).grid(row=3, column=1, columnspan=2, sticky="w", **PAD)
        _row(g2, 4, "Vocab tree",
             ttk.Entry(g2, textvariable=self.V("match.vocab_tree_path"), width=60),
             "", 3)
        ttk.Button(g2, text="...", width=3,
                   command=lambda: self._pick_file("match.vocab_tree_path")).grid(
            row=4, column=2, sticky="w", **PAD)
        ttk.Checkbutton(g2, text="skip pairs inside one frame",
                        variable=self.V("match.skip_pairs_in_same_frame",
                                        tk.BooleanVar, False)
                        ).grid(row=5, column=1, sticky="w", **PAD)
        ttk.Label(g2, text="leave this OFF. Adjacent views of a 100deg extraction on\n"
                           "45deg spacing overlap by 55deg, the largest overlap in the\n"
                           "set, and with the rig declared they are verified against\n"
                           "the known pose. Skipped, they left the median track span\n"
                           "at 3 frames on 1-mid-1 and scale drifted along the walk",
                  foreground="#777").grid(row=5, column=2, sticky="w", **PAD)
        ttk.Checkbutton(g2, text="verify pairs against the rig",
                        variable=self.V("match.rig_verification", tk.BooleanVar, True)
                        ).grid(row=6, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g2, text="use GPU for matching",
                        variable=self.V("match.use_gpu", tk.BooleanVar, True)
                        ).grid(row=7, column=1, sticky="w", **PAD)

        g3 = ttk.LabelFrame(f, text="Close the loop - pairs the frame index "
                                    "cannot reach")
        g3.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g3, text="add extra pairs to the database",
                        variable=self.V("pairs.enabled", tk.BooleanVar, False)
                        ).grid(row=0, column=1, columnspan=2, sticky="w", **PAD)
        ttk.Label(g3, text="pairs the last N frames against the first N, for a walk\n"
                           "that returns to where it started. The features are already\n"
                           "extracted, so this is a matches_importer pass - minutes,\n"
                           "not another whole match. On 1-mid-1 nothing at all tied\n"
                           "frame <60 to frame >860, and the last 26 frames ran away",
                  foreground="#777").grid(row=1, column=2, sticky="w", **PAD)
        _row(g3, 1, "Loop window",
             ttk.Entry(g3, textvariable=self.V("pairs.loop_window", tk.IntVar),
                       width=12), "", 3)
        _row(g3, 2, "Seam view sep",
             ttk.Entry(g3, textvariable=self.V("pairs.loop_max_view_sep",
                                               tk.IntVar), width=12),
             "0 = every view combination at the seam; the walk can come\n"
             "back on any heading, so which views agree is not known")
        ttk.Checkbutton(g3, text="also pair the views inside one frame",
                        variable=self.V("pairs.same_frame", tk.BooleanVar, True)
                        ).grid(row=3, column=1, sticky="w", **PAD)
        _row(g3, 4, "Frame view sep",
             ttk.Entry(g3, textvariable=self.V("pairs.max_view_sep", tk.IntVar),
                       width=12),
             "1 = the 45deg neighbour, 2 = also 90deg. A 100deg field\n"
             "does not reach 135deg, so past 2 is pure cost")
        _row(g3, 5, "Dense window",
             ttk.Entry(g3, textvariable=self.V("pairs.dense_window", tk.IntVar),
                       width=12),
             "fills the frame offsets quadratic overlap never makes -\n"
             "with overlap 5 it pairs 1,2,4,8,16 and nothing between.\n"
             "Offsets 1,2,4 carried 51/25/13% of every inlier on\n"
             "1-mid-1, so the gaps are most of what is missing. 0 = off")
        _row(g3, 6, "Dense view sep",
             ttk.Entry(g3, textvariable=self.V("pairs.dense_max_view_sep",
                                               tk.IntVar), width=12),
             "1 = same camera and its 45deg neighbours")
        return f

    # -- tab 3 -------------------------------------------------------------
    def _tab_mapper(self, parent):
        f = ttk.Frame(parent)
        g = ttk.LabelFrame(f, text="What bundle adjustment is allowed to move")
        g.pack(fill="x", padx=6, pady=4)
        for r, (text, key, hint) in enumerate((
            ("refine the rig (sensor_from_rig)", "mapper.refine_sensor_from_rig",
             "off = hold the rig rigid, which is the point of declaring it"),
            ("refine focal length", "mapper.refine_focal_length",
             "off = keep the PINHOLE intrinsics as extracted"),
            ("refine principal point", "mapper.refine_principal_point", ""),
            ("refine extra params", "mapper.refine_extra_params", ""),
        )):
            ttk.Checkbutton(g, text=text,
                            variable=self.V(key, tk.BooleanVar, False)
                            ).grid(row=r, column=1, sticky="w", **PAD)
            if hint:
                ttk.Label(g, text=hint, foreground="#777").grid(
                    row=r, column=2, sticky="w", **PAD)

        g2 = ttk.LabelFrame(f, text="Solver")
        g2.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(g2, text="GPU bundle adjustment (Caspar, COLMAP 4.1+)",
                        variable=self.V("mapper.ba_use_gpu", tk.BooleanVar, True)
                        ).grid(row=0, column=1, columnspan=2, sticky="w", **PAD)
        _row(g2, 0, "BA backend",
             self._combo_str(g2, "mapper.ba_global_backend",
                             ["CERES", "CASPAR"], 14),
             "CASPAR is COLMAP's GPU backend - and the official Windows\n"
             "4.2.0 build refuses it: built without CASPAR_ENABLED. Its\n"
             "Ceres has no CUDA either, so BA is CPU-only on that binary")
        _row(g2, 1, "BA GPU index",
             ttk.Entry(g2, textvariable=self.V("mapper.ba_gpu_index"), width=12),
             "-1 = pick automatically")
        _row(g2, 2, "Min matches",
             ttk.Entry(g2, textvariable=self.V("mapper.min_num_matches", tk.IntVar),
                       width=12))
        _row(g2, 3, "Init min inliers",
             ttk.Entry(g2, textvariable=self.V("mapper.init_min_num_inliers", tk.IntVar),
                       width=12))
        ttk.Checkbutton(g2, text="allow multiple models",
                        variable=self.V("mapper.multiple_models", tk.BooleanVar, False)
                        ).grid(row=4, column=1, sticky="w", **PAD)
        ttk.Checkbutton(g2, text="use the global mapper instead of incremental",
                        variable=self.V("mapper.global_mapper", tk.BooleanVar, False)
                        ).grid(row=5, column=1, columnspan=2, sticky="w", **PAD)
        return f

    # -- tab 4 -------------------------------------------------------------
    def _tab_run(self, parent):
        f = ttk.Frame(parent)
        g = ttk.LabelFrame(f, text="Workspace")
        g.pack(fill="x", padx=6, pady=4)
        _row(g, 0, "Work folder",
             ttk.Entry(g, textvariable=self.V("export.work_root"), width=60), "", 3)
        ttk.Button(g, text="...", width=3,
                   command=lambda: self._pick_dir("export.work_root")).grid(
            row=0, column=2, sticky="w", **PAD)
        ttk.Checkbutton(g, text="skip stages whose output already exists",
                        variable=self.V("run.skip_existing", tk.BooleanVar, True)
                        ).grid(row=1, column=1, columnspan=2, sticky="w", **PAD)
        g4 = ttk.LabelFrame(f, text="Stages to run")
        g4.pack(fill="x", padx=6, pady=4)
        self._stage_vars = {}
        for i, st in enumerate(PipelineRunner.STAGES):
            v = tk.BooleanVar(value=True)
            self._stage_vars[st] = v
            ttk.Checkbutton(g4, text=st, variable=v).grid(
                row=0, column=i, sticky="w", **PAD)
        ttk.Label(g4, text="all of them for a fresh run. To repair a finished\n"
                           "workspace - add the loop pairs and map again - tick\n"
                           "only pairs, map and export: the features and the\n"
                           "matches are already in the database",
                  foreground="#777").grid(row=1, column=0,
                                          columnspan=len(PipelineRunner.STAGES),
                                          sticky="w", **PAD)

        _row(g, 1, "Drop stray views",
             ttk.Entry(g, textvariable=self.V("export.drop_stray_views",
                                              tk.DoubleVar), width=12),
             "index-steps a view may sit from the rest of its frame\n"
             "before it is dropped; 0 keeps everything. On 1-mid-1 this\n"
             "removed 53 views of 6,213 and took the trajectory from\n"
             "eleven breaks over 3x the median step to none")
        _row(g, 3, "Flat dataset",
             ttk.Entry(g, textvariable=self.V("export.flat_dataset_dir"),
                       width=60), "", 3)
        ttk.Button(g, text="...", width=3,
                   command=lambda: self._pick_dir("export.flat_dataset_dir")
                   ).grid(row=3, column=2, sticky="w", **PAD)
        ttk.Checkbutton(g, text="one camera per image in it",
                        variable=self.V("export.flat_per_image_cameras",
                                        tk.BooleanVar, False)
                        ).grid(row=4, column=1, sticky="w", **PAD)
        ttk.Label(g, text="also writes images/NNNNN.jpg, masks/NNNNN.jpg and\n"
                          "sparse/0 there - the layout K:/data/col is in.\n"
                          "Sharing a camera per direction is correct and any\n"
                          "COLMAP reader takes it; tick the box only for a\n"
                          "reader that expects one camera per image",
                  foreground="#777").grid(row=4, column=2, sticky="w", **PAD)
        _row(g, 2, "Hang timeout (min)",
             ttk.Entry(g, textvariable=self.V("run.timeout_min", tk.IntVar), width=12),
             "0 = never. Re-arms while the process keeps using CPU,\n"
             "so it cuts off a hang and not a long computation.")

        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=6, pady=4)
        self.btn_start = ttk.Button(bar, text="Start", command=self.on_start)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(bar, text="Cancel", command=self.on_cancel,
                                     state="disabled")
        self.btn_cancel.pack(side="left", padx=4)
        self.prog = ttk.Progressbar(bar, length=260, mode="determinate")
        self.prog.pack(side="left", padx=10)
        self.lbl_run = ttk.Label(bar, text="")
        self.lbl_run.pack(side="left")

        cols = ("stage", "status", "seconds", "message")
        self.tv_stages = ttk.Treeview(f, columns=cols, show="headings", height=7)
        for c, w in zip(cols, (110, 90, 80, 520)):
            self.tv_stages.heading(c, text=c)
            self.tv_stages.column(c, width=w, anchor="w")
        self.tv_stages.pack(fill="x", padx=6, pady=4)

        self.txt_log = tk.Text(f, height=14, wrap="none")
        self.txt_log.pack(fill="both", expand=True, padx=6, pady=4)
        return f

    # -- tab 5 -------------------------------------------------------------
    def _tab_view(self, parent):
        f = ttk.Frame(parent)

        bar = ttk.Frame(f)
        bar.pack(fill="x", padx=6, pady=(4, 0))
        ttk.Button(bar, text="Load model",
                   command=self.on_load_model).pack(side="left")
        for text, cmd in (("Fit", lambda: self._view_preset("fit")),
                          ("Top", lambda: self._view_preset("top")),
                          ("Front", lambda: self._view_preset("front")),
                          ("Side", lambda: self._view_preset("side"))):
            ttk.Button(bar, text=text, width=6,
                       command=cmd).pack(side="left", padx=(6, 0))
        self.lbl_model = ttk.Label(bar, text="no model loaded")
        self.lbl_model.pack(side="left", padx=12)

        bar2 = ttk.Frame(f)
        bar2.pack(fill="x", padx=6, pady=(2, 0))
        for text, key, default in (("points", "_v_points", True),
                                   ("cameras", "_v_cams", True),
                                   ("path", "_v_path", True),
                                   ("grid", "_v_grid", True),
                                   ("colour", "_v_colour", True),
                                   ("on top", "_v_ontop", True)):
            ttk.Checkbutton(bar2, text=text,
                            variable=self.V(key, tk.BooleanVar, default),
                            command=self.on_draw).pack(side="left", padx=(0, 8))
        ttk.Label(bar2, text="cameras by").pack(side="left", padx=(8, 3))
        cb = self._combo_str(bar2, "_v_camcol", ["frame", "sensor"], 8)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self.on_draw())
        for label, key, default, width in (("size", "_v_psize", 1, 4),
                                           ("bright", "_v_bright", 1.8, 5),
                                           ("frusta", "_v_frusta", 400, 6)):
            ttk.Label(bar2, text=label).pack(side="left", padx=(10, 3))
            kind = tk.DoubleVar if isinstance(default, float) else tk.IntVar
            e = ttk.Entry(bar2, textvariable=self.V(key, kind, default),
                          width=width)
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self.on_draw())
        ttk.Button(bar2, text="Redraw", width=8,
                   command=self.on_draw).pack(side="left", padx=8)

        self.view_canvas = tk.Canvas(f, bg="#16181c", highlightthickness=0,
                                     cursor="fleur")
        self.view_canvas.pack(fill="both", expand=True, padx=6, pady=4)
        self.lbl_view = ttk.Label(
            f, text="drag to turn, right-drag or shift-drag to pan, "
                    "wheel to zoom", foreground="#777")
        self.lbl_view.pack(fill="x", padx=8, pady=(0, 4))

        c = self.view_canvas
        c.bind("<ButtonPress-1>", lambda e: self._view_down(e, "orbit"))
        c.bind("<B1-Motion>", self._view_drag)
        c.bind("<ButtonRelease-1>", self._view_up)
        c.bind("<Shift-ButtonPress-1>", lambda e: self._view_down(e, "pan"))
        c.bind("<Shift-B1-Motion>", self._view_drag)
        c.bind("<ButtonPress-3>", lambda e: self._view_down(e, "pan"))
        c.bind("<B3-Motion>", self._view_drag)
        c.bind("<ButtonRelease-3>", self._view_up)
        c.bind("<MouseWheel>", self._view_wheel)
        c.bind("<Button-4>", lambda e: self._view_wheel(e, 1))
        c.bind("<Button-5>", lambda e: self._view_wheel(e, -1))
        c.bind("<Configure>", self._view_resize)

        self._scene = None
        self._turn = None
        self._photo = None
        self._drag = None
        self._resize_job = None
        return f

    # ---- viewer ----------------------------------------------------------
    def _view_options(self):
        from .viewer import RenderOptions
        g = lambda k, kind, d: self.V(k, kind, d).get()          # noqa: E731
        return RenderOptions(
            show_points=bool(g("_v_points", tk.BooleanVar, True)),
            show_cameras=bool(g("_v_cams", tk.BooleanVar, True)),
            show_trajectory=bool(g("_v_path", tk.BooleanVar, True)),
            show_grid=bool(g("_v_grid", tk.BooleanVar, True)),
            colour_points=bool(g("_v_colour", tk.BooleanVar, True)),
            cameras_on_top=bool(g("_v_ontop", tk.BooleanVar, True)),
            brightness=float(g("_v_bright", tk.DoubleVar, 1.8)),
            camera_colour=str(self.V("_v_camcol").get() or "frame"),
            point_size=max(1, int(g("_v_psize", tk.IntVar, 1))),
            max_frusta=max(0, int(g("_v_frusta", tk.IntVar, 400))))

    def _view_preset(self, which: str):
        from .viewer import fit
        if self._scene is None or self._turn is None:
            return
        if which == "fit":
            fit(self._scene, self._turn)
        elif which == "top":
            self._turn.elevation = 89.0
        elif which == "front":
            self._turn.azimuth, self._turn.elevation = 0.0, 8.0
        elif which == "side":
            self._turn.azimuth, self._turn.elevation = 90.0, 8.0
        self.on_draw()

    def _view_down(self, event, mode):
        self._drag = (mode, event.x, event.y)
        self.view_canvas.config(cursor="tcross" if mode == "pan" else "fleur")

    def _view_drag(self, event):
        if not self._drag or self._turn is None:
            return
        mode, x, y = self._drag
        dx, dy = event.x - x, event.y - y
        if mode == "orbit":
            self._turn.orbit(dx, dy)
        else:
            self._turn.pan(dx, dy, self.view_canvas.winfo_height())
        self._drag = (mode, event.x, event.y)
        self.on_draw(budget=120_000)

    def _view_up(self, _event):
        self._drag = None
        self.view_canvas.config(cursor="fleur")
        self.on_draw()

    def _view_wheel(self, event, direction=None):
        if self._turn is None:
            return
        d = direction if direction is not None else (event.delta / 120.0)
        self._turn.dolly(d)
        self.on_draw(budget=200_000)

    def _view_resize(self, _event):
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self.on_draw)

    def on_draw(self, budget: int | None = None):
        self._resize_job = None
        if self._scene is None or self._turn is None:
            return
        import time

        from .viewer import render
        w = max(self.view_canvas.winfo_width(), 32)
        h = max(self.view_canvas.winfo_height(), 32)
        t0 = time.time()
        img = render(self._scene, self._turn, w, h, self._view_options(),
                     point_budget=budget)
        self._show(img)
        if budget is None:
            self.lbl_view.config(
                text=f"{len(self._scene.points):,} points, "
                     f"{len(self._scene.cam_centres):,} cameras, "
                     f"{len(self._scene.trajectory):,} frames   |   "
                     f"azimuth {self._turn.azimuth:.0f} deg, "
                     f"elevation {self._turn.elevation:.0f} deg   |   "
                     f"{1000*(time.time()-t0):.0f} ms   |   "
                     f"drag to turn, right-drag or shift-drag to pan, "
                     f"wheel to zoom")

    def _show(self, img):
        """Put an (H, W, 3) uint8 array on the canvas."""
        h, w = img.shape[:2]
        try:
            from PIL import Image, ImageTk
            self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        except Exception:                                     # noqa: BLE001
            # Tk reads a raw PPM straight from bytes, so PIL is optional here
            header = f"P6 {w} {h} 255 ".encode()
            self._photo = tk.PhotoImage(data=header + img.tobytes())
        self.view_canvas.delete("all")
        self.view_canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # ---- helpers ---------------------------------------------------------
    def _combo_str(self, parent, key, values, width=12):
        var = self.V(key, tk.StringVar, values[0])
        return ttk.Combobox(parent, textvariable=var, values=values,
                            state="readonly", width=width)

    def _pick_dir(self, key):
        p = filedialog.askdirectory()
        if p:
            self.V(key).set(str(Path(p)))

    def _pick_file(self, key):
        p = filedialog.askopenfilename()
        if p:
            self.V(key).set(str(Path(p)))

    # ---- actions ---------------------------------------------------------
    def on_load(self):
        p = filedialog.askopenfilename(filetypes=[("preset", "*.json")])
        if not p:
            return
        self.cfg = ColmapPipelineConfig.load(p)
        self._config_to_ui()

    def on_save(self):
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         filetypes=[("preset", "*.json")])
        if not p:
            return
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        self.cfg.save(p)

    def on_scan(self):
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        d = self.cfg.dataset
        if not d.image_dir:
            return
        frames = scan_flat(Path(d.image_dir), d.pattern)
        if not frames:
            self.lbl_scan.config(text="nothing matched the naming pattern")
            return
        views = sorted({v for f in frames.values() for v in f})
        short = sum(1 for v in frames.values() if len(v) != len(views))
        idx = sorted(frames)
        self.lbl_scan.config(
            text=f"{len(frames)} frames ({idx[0]}-{idx[-1]}), views {views}"
                 + (f", {short} frame(s) missing a view" if short else ""))
        self.on_import_directions()

    def on_import_directions(self):
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        dirs, notes = directions_from(self.cfg)
        self.tv_dirs.delete(*self.tv_dirs.get_children())
        spec = RigSpec(directions=dirs, fov=self.cfg.rig.fov,
                       width=self.cfg.rig.width, height=self.cfg.rig.height,
                       ref_index=self.cfg.rig.ref_view)
        cams = spec.to_config()[0]["cameras"] if dirs else []
        from .rig import quaternion_angle
        for d, c in zip(dirs, cams):
            rot = ("reference" if c.get("ref_sensor")
                   else f"{quaternion_angle(c['cam_from_rig_rotation']):.2f} deg")
            self.tv_dirs.insert("", "end",
                                values=(f"{d.index:02d}", f"{d.yaw:g}",
                                        f"{d.pitch:g}", rot))
        for n in notes:
            self._log(f"[colpipe] {n}")

    def on_start(self):
        try:
            self._ui_to_config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        if not self.cfg.export.work_root:
            messagebox.showerror("Run", "pick a work folder first")
            return
        self.tv_stages.delete(*self.tv_stages.get_children())
        self.txt_log.delete("1.0", "end")
        self._done = 0
        self.runner = PipelineRunner(self.cfg, lambda e, p: self.events.put((e, p)))
        self.runner.start()
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")

    def on_cancel(self):
        if self.runner:
            self.runner.cancel()

    def on_load_model(self):
        from .model import read_model
        start = self.cfg.export.work_root or ""
        p = filedialog.askdirectory(title="pick a sparse model folder (sparse/0)",
                                    initialdir=start)
        if not p:
            return
        try:
            self.model = read_model(Path(p))
        except Exception as e:                                # noqa: BLE001
            messagebox.showerror("Model", str(e))
            return
        from .viewer import Turntable, build_scene, fit
        self._scene = build_scene(self.model)
        self._turn = Turntable(target=self._scene.centre.copy(), distance=1.0,
                               up=self._scene.up)
        fit(self._scene, self._turn)
        self.lbl_model.config(text=self.model.summary())
        self.on_draw()

    # ---- events ----------------------------------------------------------
    def _log(self, line: str):
        self.txt_log.insert("end", line + "\n")
        self.txt_log.see("end")

    def _drain(self):
        try:
            while True:
                event, payload = self.events.get_nowait()
                self._handle(event, payload)
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _handle(self, event: str, payload: dict):
        if event == "pipeline_start":
            self.prog.config(maximum=payload["total"], value=0)
        elif event == "stage_start":
            r = payload["result"]
            self.tv_stages.insert("", "end", iid=r.name,
                                  values=(r.name, "running", "", ""))
        elif event == "stage_end":
            r = payload["result"]
            if self.tv_stages.exists(r.name):
                self.tv_stages.item(r.name, values=(
                    r.name, r.status, f"{r.seconds:.0f}", r.message))
            self._done += 1
            self.prog.config(value=self._done)
        elif event == "log":
            self._log(payload["line"])
        elif event == "pipeline_end":
            self.lbl_run.config(
                text=f"{payload['ok']}/{payload['total']} in "
                     f"{payload['seconds']:.0f}s")
            self.btn_start.config(state="normal")
            self.btn_cancel.config(state="disabled")


def main() -> int:
    root = tk.Tk()
    root.title("colpipe - COLMAP rig pipeline")
    root.geometry("1180x820")
    App(root)
    root.mainloop()
    return 0
