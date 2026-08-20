r"""RealityScan export-format registry.

Format GUIDs come from the format definitions shipped with the application:
    <install>\calibration.xml   (registration / camera parameters)
    <install>\structure.xml     (sparse point cloud)

The registration format is pinned from the CLI with

    -set "calexFileFormatId={GUID}"

which was verified to override both the file extension and whatever format the
last GUI export happened to leave behind.  Formats whose definition carries
``exportImages`` also dump undistorted copies of every image next to the
camera file; ``-set "calexExportImages=false"`` suppresses that.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegistrationFormat:
    key: str
    label: str
    guid: str
    filename: str
    #: format writes undistorted images unless calexExportImages is turned off
    writes_images: bool = False
    note: str = ""


REGISTRATION_FORMATS: dict[str, RegistrationFormat] = {
    f.key: f
    for f in (
        RegistrationFormat(
            key="intext_csv",
            label="Internal/External Camera Parameters (CSV)",
            guid="{0CA18733-1EBC-4254-9974-17197EB409BD}",
            filename="cameras.csv",
            writes_images=False,
            note="name,x,y,alt,yaw,pitch,roll,f_35mm,px,py,k1..k4,t1,t2 - keeps original file names",
        ),
        RegistrationFormat(
            key="opencv_csv",
            label="OpenCV-compliant Camera Parameters (CSV)",
            guid="{B5331837-609D-4B12-A931-2863653d19F7}",
            filename="cameras_opencv.csv",
            writes_images=True,
            note="name,t,R(row major),f_pix,px_pix,py_pix,k1,k2,t2,t1,k3,k4",
        ),
        RegistrationFormat(
            key="colmap",
            label="COLMAP (sparse/0/*.txt)",
            guid="{280B11A4-F9A3-47D1-AE58-C0DEA33487D8}",
            filename="colmap.txt",
            writes_images=True,
            note="writes sparse/0/{cameras,images,points3D}.txt (+ images/ unless disabled)",
        ),
        RegistrationFormat(
            key="rf_json",
            label="Radiance Fields transforms.json",
            guid="{314B5F22-C39F-4050-AE19-2236584B6932}",
            filename="transforms.json",
            writes_images=True,
            note="NeRF/3DGS style transforms.json",
        ),
        RegistrationFormat(
            key="bundler",
            label="Bundler v0.3 (.out)",
            guid="{ECC4131A-1665-466C-93BE-66DF2EBC9086}",
            filename="bundle.out",
            writes_images=False,
        ),
        RegistrationFormat(
            key="imagelist",
            label="Image List (.imagelist)",
            guid="{91058C36-2982-4FEB-9D5E-529D1CBD5EEC}",
            filename="registered.imagelist",
            writes_images=False,
            note="just the registered file names - cheap way to see what aligned",
        ),
    )
}

#: sparse point cloud: the extension alone is unambiguous in structure.xml
SPARSE_EXTENSIONS = [".ply", ".xyz", ".xyzrgb", ".obj"]

DEFAULT_REGISTRATION_FORMAT = "intext_csv"
