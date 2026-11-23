# Golden comparison: Cairo vs Vulkan

Renders every example scene in the repository with both renderers and compares
the frames. Cairo is deleted only after the Vulkan run has no `FAIL` or
`ERROR_CANDIDATE` rows and every `REVIEW` row has been checked by eye.

## Run

Run this from the repository root, with manim installed in editable mode and
the native module built:

```bash
python tools/golden/collect.py --out golden                    # writes golden/scenes/ + scenes.json
python tools/golden/render.py --scenes golden --out golden --renderer cairo  --jobs 8
python tools/golden/render.py --scenes golden --out golden --renderer vulkan --jobs 4
python tools/golden/compare.py --reference golden/cairo --candidate golden/vulkan --out golden/report
```

Then open `golden/report/report.md` (table) and `golden/report/report.html`
(reference | candidate | difference images for every non-PASS frame).

* `--only REGEX` renders a subset, for example after a fix:
  `--only "ThreeD|Surface"`.
* `--timeout` (default 600s) is per scene.
* For Vulkan, keep `--jobs` low enough that every worker gets a GPU device.

## Which scenes

`collect.py` gathers scenes from:

* every `.. manim::` directive in the docstrings under `manim/`;
* every `.. manim::` directive in the `.rst` files under `docs/source/`;
* every `Scene` subclass in `example_scenes/*.py`.

## Which frames

`render.py` runs each scene in its own process and saves:

| File | What |
|---|---|
| `eval_first.png`, `eval_mid.png`, `eval_last.png` | the scene evaluated at the first, middle and last sampled time (0.5s steps) |
| `final.png` | last frame of a real render to PNG |
| `status.json` | ok/error, traceback, timings |

## Verdicts

| Verdict | Meaning |
|---|---|
| `FAIL` | different size, 8×8-downsampled RGB MAE > 3, or alpha MAE > 2: the drawing differs |
| `REVIEW` | drawing matches coarsely, but full-resolution MAE > 1 or > 10% of pixels differ by more than 2/255; usually anti-aliasing or glyph rasterization |
| `ERROR_CANDIDATE` | only Vulkan raised: a regression |
| `ERROR_REFERENCE` | only Cairo raised |
| `ERROR_BOTH` | both raised; usually a missing tool such as LaTeX or dvisvgm |
| `PASS` | otherwise |

You can tune the thresholds with `--coarse-fail`, `--alpha-fail`,
`--mae-review` and `--changed-review`.
