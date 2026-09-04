"""An interactive 3D view of a sparse reconstruction, rendered with numpy.

The old view projected everything onto the walk's PCA plane and drew it with
matplotlib - readable, but a single fixed angle, which is how a tilted block
went unnoticed in the RealityScan work for two rounds. What is wanted is the
thing RealityScan shows: the point cloud in perspective, the cameras as frusta
along the path, and a mouse that turns it.

There is no 3D toolkit here on purpose. A turntable camera, a projection and a
z-buffer are twenty lines of numpy each, and rasterising into an array is fast
enough for a million points - a full frame is about 120 ms, and dragging
decimates to stay interactive. Adding open3d or pyvista to draw dots would cost
more than it saves.

Coordinates: COLMAP's world axes are arbitrary, so "up" is taken from the walk
itself - the normal of the plane its camera centres lie in. Orbiting about a
guessed axis is what makes a viewer feel wrong, and for a capture walked on a
floor that plane is the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- scene ----
@dataclass
class Scene:
    """Everything drawable, in world coordinates."""

    points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.float32))
    colors: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.uint8))
    cam_centres: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.float32))
    #: world-from-camera rotation per camera, so a corner in camera
    #: coordinates becomes ``centre + R @ corner``
    cam_rot: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3, 3), np.float32))
    #: half-angles (tan) per camera, for the frustum shape
    cam_tan: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), np.float32))
    #: what to colour a camera by - frame number, usually
    cam_key: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    #: which of the eight directions each camera is
    cam_sensor: np.ndarray = field(
        default_factory=lambda: np.zeros(0, np.int64))
    #: the walk, in order
    trajectory: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1], np.float32))
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    extent: float = 1.0

    @property
    def step(self) -> float:
        """Median distance between consecutive path points - the scale unit."""
        if len(self.trajectory) < 2:
            return self.extent / 100.0 or 1.0
        d = np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1)
        return float(np.median(d)) or self.extent / 100.0


def ramp(t: np.ndarray) -> np.ndarray:
    """A blue-cyan-green-yellow-red ramp for t in [0, 1], as uint8 RGB."""
    t = np.clip(np.asarray(t, np.float32), 0.0, 1.0)
    stops = np.array([[48, 60, 200], [0, 190, 200], [40, 200, 70],
                      [235, 205, 40], [225, 60, 45]], np.float32)
    x = t * (len(stops) - 1)
    i = np.clip(x.astype(np.int32), 0, len(stops) - 2)
    f = (x - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def _key_colour(key: np.ndarray) -> np.ndarray:
    k = np.asarray(key, np.float32)
    if not len(k):
        return np.zeros((0, 3), np.uint8)
    lo, hi = float(k.min()), float(k.max())
    return ramp((k - lo) / max(hi - lo, 1.0))


def build_scene(model, max_points: int = 600_000, seed: int = 0) -> Scene:
    """Turn a :class:`colpipe.model.Model` into something drawable."""
    import re

    rx = re.compile(r"_(?P<frame>\d+)\.\w+$")
    s = Scene()

    pts = np.asarray(model.points, np.float32)
    cols = np.asarray(model.colors, np.uint8)
    if len(pts) > max_points:
        # a permutation rather than a stride, so that taking a prefix while
        # the mouse is down still samples the whole cloud
        idx = np.random.default_rng(seed).permutation(len(pts))[:max_points]
        pts, cols = pts[idx], cols[idx]
    s.points, s.colors = pts, cols

    imgs = list(model.images.values())
    if imgs:
        s.cam_centres = np.array([im.centre for im in imgs], np.float32)
        s.cam_rot = np.array([im.rotation.T for im in imgs], np.float32)
        tans = []
        for im in imgs:
            cam = model.cameras.get(im.camera_id)
            if cam is None or len(cam.params) < 2:
                tans.append([0.6, 0.6])
            else:
                fx, fy = float(cam.params[0]), float(cam.params[1])
                tans.append([cam.width / (2 * fx), cam.height / (2 * fy)])
        s.cam_tan = np.array(tans, np.float32)
        keys = []
        for im in imgs:
            _, _, stem = im.name.replace("\\", "/").rpartition("/")
            mt = rx.search(stem)
            keys.append(int(mt.group("frame")) if mt else -1)
        s.cam_key = np.array(keys, np.int64)
        folders = sorted({im.name.replace("\\", "/").rpartition("/")[0]
                          for im in imgs})
        order = {n: i for i, n in enumerate(folders)}
        s.cam_sensor = np.array(
            [order.get(im.name.replace("\\", "/").rpartition("/")[0], 0)
             for im in imgs], np.int64)

        # the walk: one point per frame, in frame order
        by = {}
        for c, k in zip(s.cam_centres, s.cam_key):
            if k >= 0:
                by.setdefault(k, []).append(c)
        if by:
            order = sorted(by)
            s.trajectory = np.array(
                [np.mean(by[k], axis=0) for k in order], np.float32)

    ref = s.trajectory if len(s.trajectory) >= 3 else s.cam_centres
    if len(ref) >= 3:
        d = ref - ref.mean(0)
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        s.up = vt[2].astype(np.float32)       # normal of the walk's plane
        s.centre = ref.mean(0).astype(np.float32)
        s.extent = float(np.linalg.norm(ref.max(0) - ref.min(0))) or 1.0
    elif len(pts):
        s.centre = pts.mean(0)
        s.extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    return s


# --------------------------------------------------------------- camera ----
@dataclass
class Turntable:
    target: np.ndarray
    distance: float
    azimuth: float = 45.0
    elevation: float = 25.0
    fov: float = 55.0
    up: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 1], np.float32))

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.up / max(np.linalg.norm(self.up), 1e-9)
        seed = np.array([1.0, 0.0, 0.0], np.float32)
        if abs(float(w @ seed)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0], np.float32)
        u = seed - w * float(w @ seed)
        u /= max(np.linalg.norm(u), 1e-9)
        return u, np.cross(w, u), w

    def eye(self) -> np.ndarray:
        u, v, w = self.basis()
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        d = (math.cos(e) * math.cos(a) * u + math.cos(e) * math.sin(a) * v
             + math.sin(e) * w)
        return (self.target + self.distance * d).astype(np.float32)

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        """(eye, R) where ``(P - eye) @ R.T`` is camera space: +Z ahead, +Y down."""
        eye = self.eye()
        _, _, w = self.basis()
        fwd = self.target - eye
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        right = np.cross(fwd, w)
        n = np.linalg.norm(right)
        if n < 1e-6:                                # looking straight down
            right = np.array([1.0, 0.0, 0.0], np.float32)
        else:
            right = right / n
        down = np.cross(fwd, right)
        return eye, np.stack([right, down, fwd]).astype(np.float32)

    def orbit(self, dx: float, dy: float) -> None:
        self.azimuth = (self.azimuth - dx * 0.4) % 360.0
        self.elevation = float(np.clip(self.elevation + dy * 0.4, -89.0, 89.0))

    def pan(self, dx: float, dy: float, height: int) -> None:
        _, R = self.view()
        scale = 2 * self.distance * math.tan(math.radians(self.fov) / 2) / max(height, 1)
        self.target = self.target - R[0] * dx * scale + R[1] * dy * scale

    def dolly(self, steps: float) -> None:
        self.distance = float(np.clip(self.distance * (0.88 ** steps),
                                      1e-4, 1e9))


# -------------------------------------------------------------- renderer ----
@dataclass
class RenderOptions:
    show_points: bool = True
    show_cameras: bool = True
    show_trajectory: bool = True
    show_grid: bool = True
    colour_points: bool = True          # else a flat grey
    #: Multiply point colour by this. These captures are dim - the 1-mid-1
    #: cloud sits around a quarter brightness - and a viewer that shows them
    #: raw looks like an empty scene with a coloured path through it.
    brightness: float = 1.8
    #: "frame" walks the ramp along the capture, so the path reads in order;
    #: "sensor" gives each of the eight directions its own colour.
    camera_colour: str = "frame"
    point_size: int = 1
    max_frusta: int = 400
    #: Draw the path and the frusta over the cloud rather than inside it. They
    #: are physically among the points, so a correct depth test buries them -
    #: and the path is the thing a viewer is usually opened to look at.
    cameras_on_top: bool = True
    frustum_scale: float = 1.5          # in walk steps
    background: tuple[int, int, int] = (22, 24, 28)
    near: float = 1e-4


def _project(P: np.ndarray, eye: np.ndarray, R: np.ndarray, f: float,
             cx: float, cy: float, near: float):
    pc = (P - eye) @ R.T
    z = pc[:, 2]
    ok = z > near
    x = np.empty(len(P), np.float32)
    y = np.empty(len(P), np.float32)
    x[ok] = f * pc[ok, 0] / z[ok] + cx
    y[ok] = f * pc[ok, 1] / z[ok] + cy
    return x, y, z, ok


def _plot(buf, depth, x, y, z, ok, colour, size, W, H, on_top=False):
    """Paint points with a z-buffer. ``colour`` is (N,3) uint8 or one RGB."""
    m = ok & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not m.any():
        return
    xi = x[m].astype(np.int32)
    yi = y[m].astype(np.int32)
    zi = z[m]
    col = colour[m] if getattr(colour, "ndim", 1) == 2 else colour
    # against what earlier calls already put there - without this the grid
    # would paint over the cloud, or the cloud over the frusta, by draw order
    vis = np.ones(len(zi), bool) if on_top else zi < depth[yi, xi]
    if not vis.any():
        return
    xi, yi, zi = xi[vis], yi[vis], zi[vis]
    col = col[vis] if getattr(col, "ndim", 1) == 2 else col
    # far to near, so within this call the nearest write is the one that stays
    order = np.argsort(-zi)
    xi, yi, zi = xi[order], yi[order], zi[order]
    col = col[order] if getattr(col, "ndim", 1) == 2 else col
    r = size // 2
    for oy in range(-r, r + 1):
        for ox in range(-r, r + 1):
            px, py = xi + ox, yi + oy
            inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            buf[py[inb], px[inb]] = col[inb] if getattr(col, "ndim", 1) == 2 else col
            depth[py[inb], px[inb]] = zi[inb]


def _segments(buf, depth, A, B, colour, eye, R, f, cx, cy, W, H, near,
              samples: int = 0, on_top: bool = False):
    """Draw 3D segments by sampling along them - enough for wireframes."""
    if not len(A):
        return
    xa, ya, za, oka = _project(A, eye, R, f, cx, cy, near)
    xb, yb, zb, okb = _project(B, eye, R, f, cx, cy, near)
    keep = oka & okb
    if not keep.any():
        return
    xa, ya, za = xa[keep], ya[keep], za[keep]
    xb, yb, zb = xb[keep], yb[keep], zb[keep]
    col = colour[keep] if getattr(colour, "ndim", 1) == 2 else colour
    n = samples or int(np.clip(
        np.percentile(np.hypot(xb - xa, yb - ya), 90), 2, 96))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    px = (xa[:, None] * (1 - t) + xb[:, None] * t).ravel()
    py = (ya[:, None] * (1 - t) + yb[:, None] * t).ravel()
    pz = (za[:, None] * (1 - t) + zb[:, None] * t).ravel()
    if getattr(col, "ndim", 1) == 2:
        pc = np.repeat(col, n, axis=0)
    else:
        pc = col
    _plot(buf, depth, px, py, pz, np.ones(len(px), bool), pc, 1, W, H,
          on_top)


def _grid_segments(scene: Scene, cam: Turntable, lines: int = 21):
    """A square grid in the walk's own plane, through its centre."""
    u, v, _ = cam.basis()
    span = max(scene.extent, 1e-6) * 0.75
    step = span * 2 / (lines - 1)
    o = scene.centre
    t = (np.arange(lines, dtype=np.float32) - (lines - 1) / 2) * step
    A, B = [], []
    for s in t:
        A.append(o + u * s - v * span)
        B.append(o + u * s + v * span)
        A.append(o + v * s - u * span)
        B.append(o + v * s + u * span)
    return np.array(A, np.float32), np.array(B, np.float32)


def _frusta(scene: Scene, idx: np.ndarray, size: float):
    """Wireframe pyramids: apex to four corners, plus the rim."""
    C = scene.cam_centres[idx]
    R = scene.cam_rot[idx]
    tan = scene.cam_tan[idx]
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
    # corner k of camera i, in world
    local = np.empty((len(idx), 4, 3), np.float32)
    local[:, :, 0] = signs[None, :, 0] * tan[:, None, 0] * size
    local[:, :, 1] = signs[None, :, 1] * tan[:, None, 1] * size
    local[:, :, 2] = size
    corners = C[:, None, :] + np.einsum("nij,nkj->nki", R, local)
    A = np.concatenate([np.repeat(C, 4, axis=0),
                        corners.reshape(-1, 3)])
    B = np.concatenate([corners.reshape(-1, 3),
                        corners[:, [1, 2, 3, 0], :].reshape(-1, 3)])
    return A, B


def render(scene: Scene, cam: Turntable, width: int, height: int,
           opt: RenderOptions, point_budget: int | None = None) -> np.ndarray:
    """Draw the scene into an (H, W, 3) uint8 array."""
    W, H = max(int(width), 16), max(int(height), 16)
    buf = np.empty((H, W, 3), np.uint8)
    buf[:] = np.array(opt.background, np.uint8)
    depth = np.full((H, W), np.inf, np.float32)

    eye, R = cam.view()
    f = (H / 2) / math.tan(math.radians(cam.fov) / 2)
    cx, cy = W / 2, H / 2
    step = scene.step

    if opt.show_grid:
        A, B = _grid_segments(scene, cam)
        _segments(buf, depth, A, B, np.array([44, 48, 56], np.uint8),
                  eye, R, f, cx, cy, W, H, opt.near, samples=64)

    if opt.show_points and len(scene.points):
        P = scene.points
        C = scene.colors
        if point_budget and len(P) > point_budget:
            P, C = P[:point_budget], C[:point_budget]
        x, y, z, ok = _project(P, eye, R, f, cx, cy, opt.near)
        if opt.colour_points:
            col = C
            if opt.brightness != 1.0:
                col = np.clip(C.astype(np.float32) * opt.brightness,
                              0, 255).astype(np.uint8)
        else:
            col = np.array([170, 172, 178], np.uint8)
        _plot(buf, depth, x, y, z, ok, col, max(1, opt.point_size), W, H)

    if opt.show_trajectory and len(scene.trajectory) > 1:
        T = scene.trajectory
        _segments(buf, depth, T[:-1], T[1:],
                  np.array([250, 250, 250], np.uint8),
                  eye, R, f, cx, cy, W, H, opt.near,
                  on_top=opt.cameras_on_top)

    if opt.show_cameras and len(scene.cam_centres):
        n = len(scene.cam_centres)
        idx = np.arange(n)
        if opt.max_frusta and n > opt.max_frusta:
            idx = np.linspace(0, n - 1, opt.max_frusta).astype(np.int64)
        key = (scene.cam_sensor if opt.camera_colour == "sensor"
               and len(scene.cam_sensor) else scene.cam_key)
        cc = _key_colour(key[idx])
        A, B = _frusta(scene, idx, step * opt.frustum_scale)
        _segments(buf, depth, A, B, np.tile(cc, (8, 1)),
                  eye, R, f, cx, cy, W, H, opt.near,
                  on_top=opt.cameras_on_top)
        x, y, z, ok = _project(scene.cam_centres, eye, R, f, cx, cy, opt.near)
        _plot(buf, depth, x, y, z, ok, _key_colour(key), 3, W, H,
              opt.cameras_on_top)
    return buf


def fit(scene: Scene, cam: Turntable) -> None:
    """Frame the whole walk."""
    ref = scene.trajectory if len(scene.trajectory) else scene.cam_centres
    if not len(ref):
        ref = scene.points
    if not len(ref):
        return
    cam.target = ref.mean(0).astype(np.float32)
    r = float(np.linalg.norm(ref - cam.target, axis=1).max())
    cam.distance = max(r, 1e-6) / math.sin(math.radians(cam.fov) / 2) * 1.05
