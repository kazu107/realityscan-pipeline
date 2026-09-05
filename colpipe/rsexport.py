"""Hand a COLMAP reconstruction back to RealityScan as a component.

Verified end to end on 258 cameras: exported, imported, exported again, and
the round trip is exact - scale 1.000000, residual median 3.3e-16, which is
1e-12 of a frame step. The poses survive.

The route is `-loadColmap`, not `-importComponent` and not `-loadBundler`:

*   `-importComponent` wants an .rsalign, RealityScan's own binary format.
*   `-loadBundler` gets as far as "Added 258 images" and then stops on "File
    not found", with the image list in every naming COLMAP writes, the Bundler
    convention, and the name RealityScan might derive. Not worth chasing when
    loadColmap works.
*   `-loadColmap` takes any of the three text files and builds a component.

Four things have to be right, each of which cost a run to find:

*   the image names must be the extraction's own. The workspace calls a view
    cam03/1-mid_0400.jpg, because that is what makes COLMAP give each
    direction its own camera; RealityScan knows the file as
    1-mid_0400_03.jpg, and a component whose images are not the project's
    images cannot be merged with it.

*   those names must be relative to the folder the text files are in.
    RealityScan prepends that folder unconditionally, so an absolute path
    comes back as "K:\\...\\txt\\K:\\data\\jpeg\\...".

*   the list must not have CRLF endings, or every path carries a trailing
    carriage return and none of them resolve.

*   `-exportRegistration` needs `-selectMaximalComponent` first. Without it,
    it reports "Exporting Registration completed" and writes nothing at all.

The untriangulated keypoints are dropped on the way out. A 258-image slice
wrote a 421 MB images.txt with them; the full 1-mid-1 model would be several
gigabytes of features that mean nothing to RealityScan.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .flatten import flatten


@dataclass
class RsExportResult:
    model: Path | None = None
    text_dir: Path | None = None
    project: Path | None = None
    images: int = 0
    seconds: float = 0.0
    messages: list[str] = field(default_factory=list)


def write_text_model(colmap_exe: str, model_dir: Path, out_dir: Path,
                     image_dir, *,
                     keep_all_keypoints: bool = False) -> RsExportResult:
    """Write ``model_dir`` as COLMAP text that RealityScan can open.

    ``image_dir`` is one folder, or a mapping from workspace folder prefix to
    folder when the model holds more than one capture - a combined 1-mid and
    1-low model has its originals in two different places, and the names
    written into images.txt have to point at the right one for each.
    """
    import time

    t0 = time.time()
    res = RsExportResult()
    out_dir = Path(out_dir)
    text_dir = out_dir / "colmap_txt"
    text_dir.mkdir(parents=True, exist_ok=True)
    # flatten writes its model under <out>/sparse/0; the names it puts inside
    # have to be relative to where the *text* ends up, which is text_dir
    if isinstance(image_dir, dict):
        rel = {k: os.path.relpath(Path(v).resolve(), text_dir).replace("\\", "/")
               for k, v in image_dir.items()}
        for k, v in sorted(rel.items()):
            res.messages.append(f"images under {k or '(no prefix)'!r} "
                                f"referenced as {v}/<name>")
    else:
        rel = os.path.relpath(Path(image_dir).resolve(),
                              text_dir).replace("\\", "/")
        res.messages.append(f"images referenced as {rel}/<name>")

    fr = flatten(Path(model_dir), out_dir / "_rs", Path(model_dir),
                 write_masks=False, write_images=False,
                 keep_all_keypoints=keep_all_keypoints,
                 name_mode="source", image_root_rel=rel)
    res.images = fr.images
    res.messages += [m for m in fr.messages if "images:" not in m]

    binary = out_dir / "_rs" / "sparse" / "0"
    p = subprocess.run(
        [colmap_exe, "model_converter", "--input_path", str(binary.resolve()),
         "--output_path", str(text_dir.resolve()), "--output_type", "TXT"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        res.messages.append(f"model_converter failed: {p.stdout[-400:]}")
        return res

    # COLMAP writes LF already, but say so rather than trust it: a CRLF here
    # puts a carriage return on the end of every path and nothing resolves
    img = text_dir / "images.txt"
    raw = img.read_bytes()
    if b"\r\n" in raw:
        img.write_bytes(raw.replace(b"\r\n", b"\n"))
        res.messages.append("rewrote images.txt with LF endings")

    res.model = binary
    res.text_dir = text_dir
    res.seconds = time.time() - t0
    sizes = ", ".join(f"{f.name} {f.stat().st_size/1048576:.1f} MB"
                      for f in sorted(text_dir.iterdir()) if f.is_file())
    res.messages.append(f"text model written: {sizes}")
    return res


def import_into_realityscan(rs_exe: str, text_dir: Path, project: Path,
                            *, timeout_s: int = 7200,
                            log_path: Path | None = None) -> RsExportResult:
    """Run RealityScan headless to turn the text model into a project."""
    import time

    t0 = time.time()
    res = RsExportResult(text_dir=Path(text_dir), project=Path(project))
    project = Path(project)
    project.parent.mkdir(parents=True, exist_ok=True)
    # a project left open by a killed run keeps a .lock and the next load
    # fails with "already open in another instance"
    lock = project.with_suffix("") / ".lock"
    if lock.is_file():
        try:
            lock.unlink()
            res.messages.append("removed a stale .lock from an earlier run")
        except OSError:
            pass
    args = [rs_exe, "-headless", "-stdConsole",
            "-silent", str(project.parent / "crash"),
            "-loadColmap", str((Path(text_dir) / "cameras.txt").resolve()),
            "-save", str(project.resolve()),
            "-quit"]
    log = Path(log_path) if log_path else project.with_suffix(".log")
    with log.open("w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.kill()
            res.messages.append(f"RealityScan did not finish in "
                                f"{timeout_s/60:.0f} min - see {log.name}; a "
                                f"suppressed dialog is the usual cause")
            return res
    text = log.read_text(encoding="utf-8", errors="replace")
    res.seconds = time.time() - t0
    for line in text.splitlines():
        if "Suppressed message box" in line or "failed" in line.lower():
            res.messages.append(line.strip()[:200])
    if project.is_file():
        res.messages.append(f"{project.name} written "
                            f"({project.stat().st_size/1024:.0f} KB) in "
                            f"{res.seconds:.0f}s")
    else:
        res.messages.append(f"no project written; exit {p.returncode}")
    return res


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="a COLMAP model directory")
    ap.add_argument("out", help="where to write the text model and project")
    ap.add_argument("--image-dir", required=True, action="append",
                    help="the flat extraction the images live in. Repeat as "
                         "prefix=path for a model holding two captures, e.g. "
                         "--image-dir =K:/data/jpeg/1-mid-1 "
                         "--image-dir low_=K:/data/jpeg/1-low")
    ap.add_argument("--colmap", default=r"D:\h3dgs-work\tools\colmap420\bin\colmap.exe")
    ap.add_argument("--realityscan",
                    default=r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe")
    ap.add_argument("--project", default="",
                    help="write an .rsproj here as well as the text model")
    ap.add_argument("--keep-all-keypoints", action="store_true")
    a = ap.parse_args(argv)

    dirs = a.image_dir
    if len(dirs) == 1 and "=" not in dirs[0]:
        where = Path(dirs[0])
    else:
        where = {d.split("=", 1)[0]: Path(d.split("=", 1)[1]) for d in dirs}
    r = write_text_model(a.colmap, Path(a.model), Path(a.out), where,
                         keep_all_keypoints=a.keep_all_keypoints)
    for m in r.messages:
        print(" ", m)
    if not r.text_dir:
        return 1
    if a.project:
        r2 = import_into_realityscan(a.realityscan, r.text_dir, Path(a.project))
        for m in r2.messages:
            print(" ", m)
        return 0 if Path(a.project).is_file() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
