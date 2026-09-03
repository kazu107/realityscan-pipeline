"""Turn a set of extraction directions into a COLMAP rig configuration.

The 360 extractor renders each view with ffmpeg's v360 filter from one
equirectangular frame, so every view of a frame shares the optical centre
exactly and differs only in orientation. That is a rig with zero translation
and known rotations - and it is worth telling COLMAP so, because without it
the views of a frame have no baseline between them at all. Measured on
1-mid-1: the cameras of one frame sit 0.54% of the frame-to-frame step apart,
which is nothing to triangulate from.

    directions -> rig_config.json for `colmap rig_configurator`

Rotation convention
-------------------
v360 yaw turns the view to the right about the world up axis, pitch tilts it
up. COLMAP cameras look down +Z with +X right and +Y down, so world up is -Y
in camera coordinates, and cam_from_rig for a view at (yaw, pitch) is

    R = Rx(-pitch) @ Ry(yaw)

applied to the reference camera's frame. The quaternion is written [w, x, y, z],
which is what COLMAP uses everywhere else (images.txt is QW QX QY QZ).

Both of those - the axis signs and the quaternion order - are conventions that
cannot be read off the documentation, so verify_against() checks a generated rig
against angles measured from a reconstruction that already exists.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Direction:
    """One extraction direction, in the extractor's own terms."""

    index: int
    yaw: float
    pitch: float

    @property
    def sensor_name(self) -> str:
        return f"cam{self.index:02d}"


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> list[list[float]]:
    """cam_from_rig for a view at (yaw, pitch), reference view at (0, 0)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(-p), math.sin(-p)
    # Ry(yaw) then Rx(-pitch)
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]]
    return [[sum(rx[i][k] * ry[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def matrix_to_quaternion(m: list[list[float]]) -> list[float]:
    """Rotation matrix -> [w, x, y, z], the order COLMAP writes elsewhere."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return [w / n, x / n, y / n, z / n]


def quaternion_angle(q: list[float]) -> float:
    """Rotation angle of a [w, x, y, z] quaternion, in degrees."""
    return math.degrees(2.0 * math.acos(min(1.0, abs(q[0]))))


def pinhole_params(fov_deg: float, width: int, height: int) -> list[float]:
    """PINHOLE fx, fy, cx, cy for a rectilinear view of this horizontal FOV.

    v360 renders a rectilinear projection, so fx = (w/2) / tan(hfov/2), and the
    vertical FOV follows from the aspect ratio rather than being independent -
    the extractor computes it the same way.
    """
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return [fx, fx, width / 2.0, height / 2.0]


def vertical_fov(fov_deg: float, width: int, height: int) -> float:
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return math.degrees(2.0 * math.atan((height / 2.0) / fx))


@dataclass
class RigSpec:
    directions: list[Direction]
    fov: float
    width: int
    height: int
    ref_index: int = 0
    camera_model: str = "PINHOLE"
    #: Bind the directions into one rig, so every frame is a single 6-DOF pose.
    #: Measured on 1-mid-1 and left off because of it - see to_config.
    coupled: bool = False

    def sensor_folder(self, d: Direction) -> str:
        return d.sensor_name

    def to_config(self, image_subdir: str = "") -> list[dict]:
        """The list COLMAP's rig_configurator expects.

        With ``coupled`` off this is one single-camera rig per direction: the
        directions still get their own camera model, and frames still group by
        file name, but nothing ties the eight poses of a frame together.

        Coupling them is the obvious thing to do and it is wrong here. On 150
        frames of 1-mid-1, with identical features and matches, the only
        difference being the rig:

            against RealityScan   coupled   independent   (RealityScan)
            residual median          8.81          2.55             0
            step p10                 0.12          0.62          0.61
            step p90                17.84          1.61          1.13
            breaks over 3x             21             5             0
            3D points             110,524       266,793             -
            reprojection           0.5494        0.4291        0.6210

        Coupled, runs of 40+ consecutive frames land on a single point: frames
        26-56 all sat within 0.2 of each other while RealityScan walked 20
        index-steps.

        Why is not established. The obvious suspect - a wrong rig - was
        checked and cleared: measured against RealityScan's own solve of these
        images the ring is 45.00 degrees to within 0.014, so the geometry being
        imposed is right (see _smoke/verify_rig_rotations.py). Focal length
        agrees to 0.01%, the principal point is at centre, there is no
        distortion, and the structure is not at infinity (median point depth is
        8 frame-steps). What is left is the coupling itself, and the numbers
        above are the reason for the default, not a theory about it.

        Letting it relax instead (ba_refine_sensor_from_rig 1) is not a way
        out: COLMAP 4.2.0 crashes, 0xC0000409, 62 frames into the same data.
        So a coupled rig here can be held rigid, which collapses the walk, or
        refined, which aborts.

        Independent, the eight views of a frame come out 4.9e-3 apart - about
        1% of a frame step - so the shared optical centre is recovered rather
        than imposed, which is all the constraint was worth.
        """
        params = pinhole_params(self.fov, self.width, self.height)
        cams = []
        for d in self.directions:
            prefix = f"{image_subdir}{d.sensor_name}/" if image_subdir \
                else f"{d.sensor_name}/"
            entry: dict = {
                "image_prefix": prefix,
                "camera_model_name": self.camera_model,
                "camera_params": params,
            }
            if d.index == self.ref_index:
                entry["ref_sensor"] = True
            else:
                ref = next(x for x in self.directions if x.index == self.ref_index)
                m = rotation_matrix(d.yaw - ref.yaw, d.pitch - ref.pitch)
                entry["cam_from_rig_rotation"] = matrix_to_quaternion(m)
                # every view of a frame shares the optical centre
                entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
            cams.append(entry)
        if self.coupled:
            return [{"cameras": cams}]
        # one rig each: drop the relative pose and make every sensor its own
        # reference, which is what "no rig" means to rig_configurator
        out = []
        for e in cams:
            e = dict(e)
            e.pop("cam_from_rig_rotation", None)
            e.pop("cam_from_rig_translation", None)
            e["ref_sensor"] = True
            out.append({"cameras": [e]})
        return out

    def write(self, path: Path, image_subdir: str = "") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_config(image_subdir), indent=2) + "\n",
                        encoding="utf-8")
        return path


def ring(count: int, pitch: float = 0.0, start_yaw: float = 0.0) -> list[Direction]:
    """Evenly spaced yaws at one pitch - what the extractor's ring button makes."""
    return [Direction(index=i, yaw=(start_yaw + i * 360.0 / count + 180.0) % 360.0 - 180.0,
                      pitch=pitch)
            for i in range(count)]


def from_extractor_settings(path: Path, set_name: str | None = None,
                            set_index: int | None = None) -> tuple[RigSpec, dict]:
    """Read directions, FOV and size out of the extractor GUI's settings file.

    The saved set names ("セット4") do not line up with their positions in the
    list - the set named "セット4" is entry 3 - so selecting by name is the only
    safe way. Returns the spec and the raw settings, because the caller has to
    look at reverse_direction_index_on_odd as well.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sets = data.get("direction_sets") or []
    chosen = None
    if set_name is not None:
        chosen = next((s for s in sets if s.get("name") == set_name), None)
        if chosen is None:
            raise KeyError(f"no direction set named {set_name!r}; "
                           f"available: {[s.get('name') for s in sets]}")
    elif set_index is not None:
        chosen = sets[set_index]
    else:
        chosen = {"directions": data.get("directions") or []}

    dirs = [Direction(index=i, yaw=float(d["yaw"]), pitch=float(d["pitch"]))
            for i, d in enumerate(chosen.get("directions") or [])]
    spec = RigSpec(directions=dirs,
                   fov=float(data.get("fov", 100)),
                   width=int(data.get("width", 2133)),
                   height=int(data.get("height", 2133)))
    return spec, data
