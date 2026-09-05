"""Find the image pairs that cross between two captures.

Two walks of the same place share no image, so nothing in COLMAP can align
their reconstructions - model_merger needs common registered images, measured:
disjoint halves of one model gave "Merge failed" while halves overlapping by
ten frames merged. The only way to combine them is one database with pairs
that cross between the sets, and the frame index cannot express those: 1-low's
frame 400 and 1-mid's frame 400 are unrelated numbers.

A vocabulary tree answers it. It is an image retrieval index - descriptors
clustered into "visual words", each image a sparse histogram of them - and it
returns the images most likely to see the same thing, without matching
anything. What it proposes still has to be matched and verified.

Two things are done here that `vocab_tree_matcher` on its own would not:

*   only cross-set pairs are kept. Retrieval is dominated by an image's own
    temporal neighbours and its own frame's adjacent views - measured on
    1-mid-1, querying cam07/1-mid_0029 returns cam07/1-mid_0028 and
    cam06/1-mid_0029 next - and those are already matched. Spending the
    retrieval budget on them again would waste hours.

*   the pairs go through `matches_importer`, the same path the loop seam took,
    so the settings and the accounting are the ones already validated.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

QUERY = re.compile(r"Querying for image (?P<name>.+?) \[(?P<i>\d+)/(?P<n>\d+)\]")
HIT = re.compile(r"image_name=(?P<name>.+?), score=(?P<score>[-\d.eE+]+)")


@dataclass
class RetrieveResult:
    queries: int = 0
    hits: int = 0
    cross: int = 0
    pairs: int = 0
    seconds: float = 0.0
    messages: list[str] = field(default_factory=list)


def image_names(database: Path) -> list[str]:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        return [n for (n,) in db.execute("select name from images order by name")]


def split_by_prefix(names: list[str], prefix: str) -> tuple[list[str], list[str]]:
    """(names starting with prefix, the rest) - the folder prefix names the set."""
    a = [n for n in names if n.replace("\\", "/").startswith(prefix)]
    return a, [n for n in names if n not in set(a)]


def write_list(path: Path, names: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


def retrieve(exe: str, database: Path, tree: Path, query: list[str],
             pool: list[str], work: Path, *, num_neighbors: int = 40,
             max_num_features: int = 4000,
             log_path: Path | None = None) -> dict[str, list[tuple[str, float]]]:
    """For each query image, the pool images the tree thinks it resembles."""
    ql = write_list(work / "retrieve_query.txt", query)
    pl = write_list(work / "retrieve_pool.txt", pool)
    args = [
        exe, "vocab_tree_retriever",
        "--database_path", str(Path(database).resolve()),
        "--vocab_tree_path", str(Path(tree).resolve()),
        "--query_image_list_path", str(ql.resolve()),
        "--database_image_list_path", str(pl.resolve()),
        "--num_neighbors", str(num_neighbors),
        "--max_num_features", str(max_num_features),
    ]
    p = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    if log_path:
        # An hour of retrieval must not be lost because a log file could not be
        # written - which is exactly what an unsanitised label in the name did
        # once. Sanitise, and if it still fails, carry on: the results are in
        # `out` either way.
        try:
            safe = re.sub(r'[<>:"/\|?*]', "_", log_path.name)
            path = log_path.with_name(safe)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(out, encoding="utf-8")
        except OSError:
            pass
    if p.returncode != 0:
        raise RuntimeError(f"vocab_tree_retriever failed: {out[-600:]}")

    found: dict[str, list[tuple[str, float]]] = {}
    current: str | None = None
    for line in out.splitlines():
        if (mt := QUERY.search(line)):
            current = mt.group("name")
            found.setdefault(current, [])
            continue
        if current and (mt := HIT.search(line)):
            found[current].append((mt.group("name"), float(mt.group("score"))))
    return found


def cross_pairs(found: dict[str, list[tuple[str, float]]], *,
                is_other, min_score: float = 0.0,
                per_query: int = 0) -> list[tuple[str, str, float]]:
    """Keep only the hits that land in the other set."""
    out = []
    for q, hits in found.items():
        kept = 0
        for name, score in hits:
            if name == q or not is_other(q, name) or score < min_score:
                continue
            out.append((q, name, score))
            kept += 1
            if per_query and kept >= per_query:
                break
    return out


def build_cross_list(exe: str, database: Path, tree: Path, prefix: str,
                     work: Path, *, num_neighbors: int = 40,
                     max_num_features: int = 4000, min_score: float = 0.0,
                     per_query: int = 10) -> tuple[Path, RetrieveResult]:
    """Retrieve both ways across the split and write one pair list.

    Both directions, because retrieval is not symmetric: an image of the
    denser set can fail to appear in the other's top N while the reverse query
    finds it.
    """
    import time

    res = RetrieveResult()
    t0 = time.time()
    names = image_names(Path(database))
    a, b = split_by_prefix(names, prefix)
    res.messages.append(f"{len(a)} images under {prefix!r}, {len(b)} outside")
    if not a or not b:
        res.messages.append("one side is empty - nothing crosses")
        return work / "cross_pairs.txt", res

    seta, setb = set(a), set(b)

    def other(q: str, h: str) -> bool:
        return (q in seta) != (h in seta)

    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for label, q, pool in (("A->B", a, b), ("B->A", b, a)):
        found = retrieve(exe, database, tree, q, pool, work,
                         num_neighbors=num_neighbors,
                         max_num_features=max_num_features,
                         log_path=work / f"retrieve_{label.replace('->','_')}.log")
        res.queries += len(found)
        res.hits += sum(len(v) for v in found.values())
        cp = cross_pairs(found, is_other=other, min_score=min_score,
                         per_query=per_query)
        res.cross += len(cp)
        for qn, hn, _s in cp:
            key = (qn, hn) if qn < hn else (hn, qn)
            if key not in seen:
                seen.add(key)
                lines.append(f"{key[0]} {key[1]}")
        res.messages.append(f"{label}: {len(found)} queries, {len(cp)} "
                            f"crossing hits")

    path = write_list(work / "cross_pairs.txt", lines)
    res.pairs = len(lines)
    res.seconds = time.time() - t0
    res.messages.append(f"{res.pairs} distinct crossing pairs written to "
                        f"{path.name} in {res.seconds/60:.0f} min")
    return path, res
