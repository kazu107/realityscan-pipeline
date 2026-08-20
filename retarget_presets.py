"""Rewrite the absolute paths in presets for a machine with different drives.

Presets store absolute paths - the image folder, the output root, and the seed
components inside it. Moving the pipeline to another machine that has no K:
drive means every one of those has to move too, including the ones buried in
the multi-line seed spec. Missing one shows up much later as a headless
RealityScan sitting on a dialog nobody can see.

    python retarget_presets.py K:\=D:\ [--dir presets] [--apply]

Prints what would change and leaves the files alone unless --apply is given.
Matching ignores case and treats / and \ as the same separator, because that
is how Windows resolves them.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def rewrite(value: str, rules: list[tuple[str, str]]) -> str:
    for old, new in rules:
        pat = re.compile(re.escape(old).replace(r"\\", r"[\\/]"), re.IGNORECASE)
        value = pat.sub(lambda _m: new, value)
    return value


def walk(node, rules: list[tuple[str, str]], hits: list[tuple[str, str]]):
    if isinstance(node, dict):
        return {k: walk(v, rules, hits) for k, v in node.items()}
    if isinstance(node, list):
        return [walk(v, rules, hits) for v in node]
    if isinstance(node, str):
        out = rewrite(node, rules)
        if out != node:
            for a, b in zip(node.splitlines() or [node], out.splitlines() or [out]):
                if a != b:
                    hits.append((a, b))
        return out
    return node


def main() -> int:
    args = [a for a in sys.argv[1:]]
    apply = "--apply" in args
    if apply:
        args.remove("--apply")
    directory = Path("presets")
    if "--dir" in args:
        i = args.index("--dir")
        directory = Path(args[i + 1])
        del args[i:i + 2]
    rules = []
    for a in args:
        if "=" not in a:
            print(__doc__)
            return 2
        old, new = a.split("=", 1)
        rules.append((old, new))
    if not rules:
        print(__doc__)
        return 2

    files = sorted(directory.glob("*.json"))
    print(f"{len(files)} preset(s) in {directory}, rules: "
          + ", ".join(f"{o} -> {n}" for o, n in rules))
    total = 0
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        hits: list[tuple[str, str]] = []
        out = walk(data, rules, hits)
        if not hits:
            continue
        total += len(hits)
        print(f"\n{f.name}: {len(hits)} path(s)")
        for a, b in hits[:4]:
            print(f"    {a.strip()}\n -> {b.strip()}")
        if len(hits) > 4:
            print(f"    ... and {len(hits) - 4} more")
        if apply:
            f.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"\n{total} path(s) {'rewritten' if apply else 'would change'}")
    if not apply:
        print("re-run with --apply to write the files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
