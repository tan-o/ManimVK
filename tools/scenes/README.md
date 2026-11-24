# Example scenes

Renders every example scene in the repository and fails when one raises an
error that is not listed, with its reason, in `known_errors.json`.  CI runs it
on every push (`.github/workflows/scenes.yml`).

## Run

Run this from the repository root, with manim installed in editable mode and
the native modules built:

```bash
python tools/scenes/collect.py --out scenes             # writes scenes/scenes/ + scenes.json
python tools/scenes/render.py --scenes scenes --jobs 2   # renders, writes scenes/report.md
```

* `--only REGEX` renders a subset, for example after a fix: `--only "ThreeD|Surface"`.
* `--timeout` (default 600s) is per scene.
* Keep `--jobs` low enough that every worker gets a GPU device.

## Which scenes

`collect.py` gathers scenes from:

* every `.. manim::` directive in the docstrings under `manim/`;
* every `.. manim::` directive in the `.rst` files under `docs/source/`;
* every `Scene` subclass in `example_scenes/*.py`.

## What is kept

`render.py` runs each scene in its own process, with Python's and NumPy's
random generators seeded, and saves under `scenes/frames/<id>/`:

| File | What |
|---|---|
| `eval_first.png`, `eval_mid.png`, `eval_last.png` | the scene evaluated at the first, middle and last sampled time (0.5s steps) |
| `final.png` | last frame of a real render to PNG |
| `status.json` | ok/error, traceback, timings |
