"""Collect every example scene in the repository into standalone modules.

Sources:
* ``.. manim::`` directives in docstrings under ``manim/`` and in ``docs/source``
  (the same examples the documentation renders);
* ``Scene`` subclasses defined in ``example_scenes/*.py``.

Writes ``<out>/scenes/<id>.py`` (one scene each, with ``from manim import *``)
and ``<out>/scenes.json``::

    python tools/scenes/collect.py --out scenes
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTIVE = re.compile(r"^(?P<indent>[ \t]*)\.\. manim::[ \t]*(?P<name>\w+)[ \t]*$")
OPTION = re.compile(r"^[ \t]*:[\w-]+:.*$")


def directive_blocks(text: str):
    """Yield (class_name, options, code) for every ``.. manim::`` block."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = DIRECTIVE.match(lines[i])
        if not match:
            i += 1
            continue
        base = len(match["indent"].expandtabs())
        name = match["name"]
        options: list[str] = []
        body: list[str] = []
        i += 1
        while i < len(lines) and OPTION.match(lines[i]):
            options.append(lines[i].strip())
            i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line.expandtabs()) - len(line.expandtabs().lstrip()) <= base:
                break
            body.append(line)
            i += 1
        code = textwrap.dedent("\n".join(body)).strip("\n")
        if code.startswith(">>> "):
            # A doctest: the directive runs its >>> and ... lines.
            code = "\n".join(
                line[4:] for line in code.splitlines() if line.startswith((">>> ", "... "))
            )
        if code:
            yield name, options, code


def docstring_sources():
    for path in sorted((ROOT / "manim").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=True)
                if doc and ".. manim::" in doc:
                    yield path, doc
    for path in sorted((ROOT / "docs" / "source").rglob("*.rst")):
        text = path.read_text(encoding="utf-8")
        if ".. manim::" in text:
            yield path, text


def example_scene_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
            isinstance(base, ast.Name) and base.id.endswith("Scene") for base in node.bases
        ):
            names.append(node.name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("scenes"))
    args = parser.parse_args()
    scene_dir = args.out / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    seen: dict[str, int] = {}

    for path, text in docstring_sources():
        rel = path.relative_to(ROOT).as_posix()
        for name, options, code in directive_blocks(text):
            stem = re.sub(r"\W+", "_", rel.rsplit(".", 1)[0])
            scene_id = f"{stem}__{name}"
            seen[scene_id] = seen.get(scene_id, 0) + 1
            if seen[scene_id] > 1:
                scene_id += f"_{seen[scene_id]}"
            module = scene_dir / f"{scene_id}.py"
            module.write_text("from manim import *\n\n" + code + "\n", encoding="utf-8")
            entries.append(
                {
                    "id": scene_id,
                    "module": module.name,
                    "scene": name,
                    "source": rel,
                    "still": any(o.startswith(":save_last_frame:") for o in options),
                }
            )

    for path in sorted((ROOT / "example_scenes").glob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for name in example_scene_names(path):
            scene_id = f"example_scenes__{path.stem}__{name}"
            module = scene_dir / f"{scene_id}.py"
            module.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            entries.append(
                {"id": scene_id, "module": module.name, "scene": name, "source": rel, "still": False}
            )

    (args.out / "scenes.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    print(f"collected {len(entries)} scenes into {scene_dir}")


if __name__ == "__main__":
    main()
