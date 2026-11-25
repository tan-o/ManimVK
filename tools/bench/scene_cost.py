"""Time every frame of a scene, per animation, without writing a video.

Draws each frame of the scene through ``Manager.evaluate`` (no encoding, no
partial movie files) and prints, per ``play`` and ``wait``, the frames, the
wall time per frame, where the native display lists spent it, and how many
chains of depth pieces occlusion culling left out.  Compare runs of the same
build with and without ``--no-culling`` rather than builds run at different
times.

::

    python tools/bench/scene_cost.py example_scenes/hairy_ball/hairy_ball.py HairyEarth \\
        --size 1920x1080 --frame-rate 30 [--no-culling] [--assets DIR]

Stages (milliseconds per frame): ``input`` reading objects and meshes from
Python, ``depth`` the Vulkan depth passes, ``occlusion`` rasterising
occluders, ``paths`` building paths and culling, ``skia`` Skia recording the
paths (stroking included), ``meshes`` drawing meshes.  ``other`` is the rest
of the frame: the scene's updaters and Python drawing code, the GPU, and
reading the frame back.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from manim import tempconfig
from manim.manager import Manager

STAGES = ("input", "depth", "occlusion", "paths", "skia", "meshes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("scene")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--no-culling", action="store_true")
    parser.add_argument("--assets", type=Path, default=None)
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="replace the native module with no-ops, to time the Python side alone",
    )
    args = parser.parse_args()
    if args.python_only:
        from frame_cost import _NullDevice

        from manim.renderer.skia import renderer as skia_renderer

        skia_renderer.manim_vk.Device = _NullDevice
    width, height = (int(value) for value in args.size.split("x"))

    spec = importlib.util.spec_from_file_location(args.file.stem, args.file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    options = {
        "pixel_width": width,
        "pixel_height": height,
        "frame_rate": args.frame_rate,
        "disable_caching": True,
        "progress_bar": "none",
        "verbosity": "WARNING",
    }
    if args.assets is not None:
        options["assets_dir"] = str(args.assets)
    frames: list[tuple[float, float, dict[str, float]]] = []
    with tempconfig(options):
        scene = getattr(module, args.scene)()
        renderer = scene.renderer
        renderer.set_occlusion_culling(not args.no_culling)
        manager = Manager(scene)
        last = [time.perf_counter()]

        def record(frame) -> None:
            now = time.perf_counter()
            frames.append((frame.time, now - last[0], renderer.take_draw_stats()))
            last[0] = time.perf_counter()

        # Every frame start up to a day; times past the end are ignored.
        times = np.arange(0, 86400, 1 / args.frame_rate)
        started = time.perf_counter()
        manager.evaluate(capture_timeline=True, frame_times=times, on_frame=record)
        total = time.perf_counter() - started
        entries = [
            entry
            for entry in manager.timeline.to_dict()["entries"]
            if entry["kind"] in ("play", "wait")
        ]

    times = np.array([frame[0] for frame in frames])
    print(
        f"{args.scene}: {len(frames)} frames at {width}x{height}, {total:.1f} s, "
        f"culling {'off' if args.no_culling else 'on'}"
    )
    header = f"{'#':>3} {'kind':5} {'frames':>6} {'ms/frame':>9} " + " ".join(
        f"{stage:>9}" for stage in (*STAGES, "other")
    )
    print(header + f" {'culled':>14}")
    for index, entry in enumerate(entries):
        inside = np.flatnonzero((times >= entry["start"] - 1e-9) & (times < entry["end"] - 1e-9))
        if not len(inside):
            continue
        wall = np.mean([frames[i][1] for i in inside]) * 1000
        stages = {
            stage: np.mean([frames[i][2].get(stage, 0.0) for i in inside]) * 1000
            for stage in STAGES
        }
        other = wall - sum(stages.values())
        tested = sum(frames[i][2].get("tested", 0) for i in inside)
        culled = sum(frames[i][2].get("culled", 0) for i in inside)
        share = f"{100 * culled / tested:5.1f}% of {tested / len(inside):6.0f}" if tested else ""
        print(
            f"{index:>3} {entry['kind']:5} {len(inside):>6} {wall:>9.1f} "
            + " ".join(f"{stages[stage]:>9.1f}" for stage in STAGES)
            + f" {other:>9.1f} {share:>14}"
        )


if __name__ == "__main__":
    main()
