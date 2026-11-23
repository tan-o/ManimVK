"""Render the collected example scenes with one renderer.

For every scene this stores, under ``<out>/<renderer>/<id>/``:

* ``eval_first.png``, ``eval_mid.png``, ``eval_last.png``: frames drawn by
  :meth:`.Manager.evaluate` at the first, middle and last sampled times
  (animated scenes only);
* ``final.png``: the last frame written by a real render (``format=png``),
  which goes through the renderer's normal frame path;
* ``status.json``: success flag, error text and timing.

Each scene runs in its own subprocess with a timeout, so a crash or hang in one
example cannot stop the run::

    python tools/golden/render.py --scenes golden --renderer cairo --jobs 4
    python tools/golden/render.py --scenes golden --renderer vulkan --jobs 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_STEP = 0.5
MAX_SAMPLED_TIME = 240.0


def run_worker(entry: dict, scenes: Path, renderer: str, out: Path, width: int, height: int) -> None:
    """Render one scene in this process (called through ``--worker``)."""
    import importlib.util
    import random

    import numpy as np  # noqa: F401  (examples use np without importing it)

    from manim import tempconfig
    from manim.manager import Manager

    def seed() -> None:
        # Examples use both Python's and NumPy's global RNGs for layouts and
        # StreamLines phases.  Cairo and Vulkan run in separate processes, so
        # without an explicit seed they compare two different random scenes.
        random.seed(0)
        np.random.seed(0)

    # networkx layout functions use ``np.random.default_rng(seed)``; a fresh
    # generator with seed=None ignores ``np.random.seed``.  Force a deterministic
    # seed for the comparison only.
    import networkx.drawing.layout as nx_layout

    for _name in ("spring_layout", "random_layout"):
        _original = getattr(nx_layout, _name)

        def _seeded_layout(*args, __original=_original, **kwargs):
            kwargs.setdefault("seed", 0)
            return __original(*args, **kwargs)

        setattr(nx_layout, _name, _seeded_layout)

    target = out / renderer / entry["id"]
    target.mkdir(parents=True, exist_ok=True)
    status: dict = {"id": entry["id"], "ok": False, "error": None, "frames": []}
    started = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location(f"golden_{entry['id']}", scenes / "scenes" / entry["module"])
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        seed()
        spec.loader.exec_module(module)
        scene_class = getattr(module, entry["scene"])
        original_construct = scene_class.construct

        def seeded_construct(self) -> None:
            # Seed as late as possible: scene initialization also consumes
            # Python's global RNG in some examples.
            seed()
            original_construct(self)

        scene_class.construct = seeded_construct
        settings = {
            "renderer": renderer,
            "pixel_width": width,
            "pixel_height": height,
            "frame_rate": 15,
            "disable_caching": True,
            "progress_bar": "none",
            "verbosity": "WARNING",
            "media_dir": str(target / "media"),
        }
        # 1. Frames as evaluation draws them.
        with tempconfig({**settings, "format": "none"}):
            frames = []
            seed()
            times = np.arange(0.0, MAX_SAMPLED_TIME, SAMPLE_STEP).tolist()
            seed()
            Manager(scene_class()).evaluate(frame_times=times, on_frame=frames.append)
            if frames:
                picks = {"eval_first": frames[0], "eval_mid": frames[len(frames) // 2], "eval_last": frames[-1]}
                for name, frame in picks.items():
                    frame.image.save(target / f"{name}.png")
                    status["frames"].append({"name": name, "time": frame.time})
        # 2. The last frame of a real render.
        with tempconfig({**settings, "format": "png"}):
            seed()
            scene = scene_class()
            scene.render()
            images = sorted((target / "media").rglob("*.png"), key=lambda p: p.stat().st_mtime)
            if not images:
                raise RuntimeError("render produced no PNG")
            images[-1].replace(target / "final.png")
        status["ok"] = True
        shutil.rmtree(target / "media", ignore_errors=True)
    except BaseException as error:  # noqa: BLE001 - report every failure
        status["error"] = f"{type(error).__name__}: {error}"[:2000]
    status["seconds"] = round(time.perf_counter() - started, 3)
    (target / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")


def cwd_for(entry: dict) -> Path:
    if entry["source"].startswith("example_scenes/"):
        return ROOT / "example_scenes"
    return ROOT / "docs" / "source"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", type=Path, default=Path("golden"), help="output of collect.py")
    parser.add_argument("--renderer", choices=["cairo", "vulkan"], required=True)
    parser.add_argument("--out", type=Path, default=None, help="defaults to --scenes")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per scene")
    parser.add_argument("--only", default=None, help="regex on scene ids")
    parser.add_argument("--size", default="854x480")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = (args.out or args.scenes).resolve()
    scenes = args.scenes.resolve()
    width, height = (int(v) for v in args.size.split("x"))

    if args.worker is not None:
        run_worker(json.loads(args.worker), scenes, args.renderer, out, width, height)
        return

    entries = json.loads((scenes / "scenes.json").read_text(encoding="utf-8"))
    if args.only:
        entries = [e for e in entries if re.search(args.only, e["id"])]

    def run(entry: dict) -> tuple[str, bool, str]:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--scenes", str(scenes), "--out", str(out),
            "--renderer", args.renderer, "--size", args.size, "--worker", json.dumps(entry),
        ]
        target = out / args.renderer / entry["id"]
        try:
            result = subprocess.run(
                command, cwd=cwd_for(entry), capture_output=True, text=True,
                timeout=args.timeout, env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0"},
            )
            if not (target / "status.json").exists():
                target.mkdir(parents=True, exist_ok=True)
                (target / "status.json").write_text(json.dumps({
                    "id": entry["id"], "ok": False,
                    "error": f"worker exited {result.returncode}: {result.stderr[-2000:]}",
                }), encoding="utf-8")
        except subprocess.TimeoutExpired:
            target.mkdir(parents=True, exist_ok=True)
            (target / "status.json").write_text(json.dumps({
                "id": entry["id"], "ok": False, "error": f"timeout after {args.timeout}s",
            }), encoding="utf-8")
        status = json.loads((target / "status.json").read_text(encoding="utf-8"))
        return entry["id"], status["ok"], status.get("error") or ""

    started = time.perf_counter()
    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, (scene_id, ok, error) in enumerate(pool.map(run, entries), 1):
            failed += not ok
            mark = "ok " if ok else "ERR"
            print(f"[{index}/{len(entries)}] {mark} {scene_id} {error[:120]}", flush=True)
    print(f"{len(entries) - failed} ok, {failed} errors, {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
