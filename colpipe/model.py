"""Read a COLMAP sparse model - cameras, images, points - for viewing and export.

Reads the binary form directly rather than shelling out to model_converter,
because the viewer wants the numbers in memory and a round trip through text
for a few million points is slower than parsing the binary.

COLMAP stores world-from-camera as a quaternion (qw, qx, qy, qz) and a
translation, both taking a world point into the camera frame. The camera centre
- the thing worth drawing - is therefore -R^T t, not t.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    image_id: int
    name: str
    camera_id: int
    qvec: np.ndarray             # w, x, y, z: world -> camera
    tvec: np.ndarray
    num_points: int = 0

    @property
    def rotation(self) -> np.ndarray:
        w, x, y, z = self.qvec
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    @property
    def centre(self) -> np.ndarray:
        return -self.rotation.T @ self.tvec


@dataclass
class Model:
    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, Image] = field(default_factory=dict)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    errors: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def centres(self) -> np.ndarray:
        return np.array([im.centre for im in self.images.values()]) \
            if self.images else np.zeros((0, 3))

    def summary(self) -> str:
        return (f"{len(self.images)} images, {len(self.cameras)} cameras, "
                f"{len(self.points)} points"
                + (f", mean error {self.errors.mean():.3f} px"
                   if len(self.errors) else ""))


def _read(f, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))


def read_model(path: Path) -> Model:
    """Read a model directory, binary if present and text otherwise."""
    path = Path(path)
    if (path / "images.bin").is_file():
        return _read_binary(path)
    if (path / "images.txt").is_file():
        return _read_text(path)
    raise FileNotFoundError(f"no COLMAP model in {path}")


def _read_binary(path: Path) -> Model:
    m = Model()
    with open(path / "cameras.bin", "rb") as f:
        (n,) = _read(f, "<Q")
        for _ in range(n):
            cid, model_id, w, h = _read(f, "<iiQQ")
            # parameter counts per model id, in COLMAP's order
            counts = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4,
                      9: 5, 10: 12, 11: 5}
            k = counts.get(model_id, 4)
            params = np.array(_read(f, f"<{k}d"))
            names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                     3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE"}
            m.cameras[cid] = Camera(cid, names.get(model_id, str(model_id)),
                                    w, h, params)
    with open(path / "images.bin", "rb") as f:
        (n,) = _read(f, "<Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "<idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (npts,) = _read(f, "<Q")
            f.seek(npts * 24, 1)          # x, y, point3D_id per observation
            m.images[iid] = Image(iid, name.decode("utf-8"), cid,
                                  np.array([qw, qx, qy, qz]),
                                  np.array([tx, ty, tz]), npts)
    xyz, rgb, err = [], [], []
    with open(path / "points3D.bin", "rb") as f:
        (n,) = _read(f, "<Q")
        for _ in range(n):
            _pid, x, y, z, r, g, b, e = _read(f, "<QdddBBBd")
            (track,) = _read(f, "<Q")
            f.seek(track * 8, 1)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
            err.append(e)
    m.points = np.array(xyz) if xyz else np.zeros((0, 3))
    m.colors = np.array(rgb, dtype=np.uint8) if rgb else np.zeros((0, 3), np.uint8)
    m.errors = np.array(err) if err else np.zeros(0)
    return m


def _read_text(path: Path) -> Model:
    m = Model()
    for line in (path / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        m.cameras[int(p[0])] = Camera(int(p[0]), p[1], int(p[2]), int(p[3]),
                                      np.array([float(x) for x in p[4:]]))
    lines = [l for l in (path / "images.txt").read_text(encoding="utf-8").splitlines()
             if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        m.images[int(p[0])] = Image(
            int(p[0]), p[9], int(p[8]),
            np.array([float(x) for x in p[1:5]]),
            np.array([float(x) for x in p[5:8]]),
            len(lines[i + 1].split()) // 3)
    xyz, rgb, err = [], [], []
    for line in (path / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        xyz.append([float(x) for x in p[1:4]])
        rgb.append([int(x) for x in p[4:7]])
        err.append(float(p[7]))
    m.points = np.array(xyz) if xyz else np.zeros((0, 3))
    m.colors = np.array(rgb, dtype=np.uint8) if rgb else np.zeros((0, 3), np.uint8)
    m.errors = np.array(err) if err else np.zeros(0)
    return m


def frames_from_names(model: Model, pattern: str = r"(?P<frame>\d+)\.\w+$") \
        -> dict[str, dict[int, np.ndarray]]:
    """Group camera centres by sensor folder and frame number.

    The layout names images ``camNN/<prefix>_<frame>.jpg``, so the folder is the
    sensor and the trailing number is the frame - which is what the viewer
    colours by and what a rig check compares across sensors.
    """
    import re
    rx = re.compile(pattern)
    out: dict[str, dict[int, np.ndarray]] = {}
    for im in model.images.values():
        folder, _, stem = im.name.replace("\\", "/").rpartition("/")
        mt = rx.search(stem)
        if not mt:
            continue
        out.setdefault(folder, {})[int(mt.group("frame"))] = im.centre
    return out
