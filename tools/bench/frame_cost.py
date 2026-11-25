"""Per-frame cost of the renderer on two stress scenes.

* ``dense``: two depth-tested VMobjects of 50 000 short strokes each over a
  translucent sphere in a ThreeDScene camera (like HairyEarth's dense field).
* ``many``: 3000 separate VMobjects in a 2D scene.

Each frame is drawn and read back.  The time is split into ``cpu`` (Python,
Rust and Skia recording its GPU commands: ``render_mobjects``) and ``gpu``
(waiting for the GPU to finish and reading the frame back: ``get_frame``).
GPU memory is read after every frame: the peak, and how much it grew from
the first frame to the last (a steady frame should not grow).
With ``--python-only`` the native target is replaced by one that does
nothing, which leaves the Python side alone::

    python tools/bench/frame_cost.py
    python tools/bench/frame_cost.py --python-only --frames 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from manim import (
    BLUE,
    DEGREES,
    MeshSurface,
    Sphere,
    Square,
    VGroup,
    VMobject,
    tempconfig,
)
from manim.camera import Camera, ThreeDCamera
from manim.mobject.geometry.arc import Dot
from manim.renderer.skia import renderer as skia_renderer


def _strokes(count: int, inner: float, outer: float, seed: int) -> VMobject:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    start, end = inner * directions, outer * directions
    delta = end - start
    points = np.stack([start, start + delta / 3, start + 2 * delta / 3, end], axis=1)
    strokes = VMobject(stroke_width=1, stroke_opacity=0.5)
    strokes.set_points(points.reshape(-1, 3))
    return strokes.apply_depth_test()


def dense():
    camera = ThreeDCamera(phi=60 * DEGREES, theta=30 * DEGREES)
    sphere = MeshSurface(Sphere(radius=3, resolution=(48, 24)), color=BLUE, opacity=0.3)
    field = VGroup(_strokes(50_000, 3.02, 3.4, 0), _strokes(50_000, 3.4, 3.5, 0))

    def step(frame: int) -> None:
        camera.set_theta((30 + frame) * DEGREES)

    return camera, [sphere, field], step


def many():
    rng = np.random.default_rng(0)
    dots = VGroup(*[Dot(rng.uniform(-6, 6, 3) * [1, 0.5, 0]) for _ in range(2000)])
    squares = VGroup(
        *[
            Square(0.2).move_to(rng.uniform(-6, 6, 3) * [1, 0.5, 0]).set_fill(BLUE, 0.5)
            for _ in range(1000)
        ]
    )

    def step(frame: int) -> None:
        dots.shift(0.001 * np.array([1.0, 0.0, 0.0]))

    return Camera(), [dots, squares], step


class _NullTarget:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def restore_static_layer(self) -> bool:
        return False

    def commit_into(self, frame) -> None:
        pass

    def memory_usage(self) -> tuple[None, int, int]:
        return (None, 0, 0)

    def take_draw_stats(self) -> dict[str, float]:
        return {}


class _NullDevice:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def device_name(self) -> str:
        return "none (--python-only)"

    def create_skia_target(self, width: int, height: int) -> _NullTarget:
        return _NullTarget()


def _draw_report(stats: list[dict[str, float]]) -> str:
    """Median native drawing time per stage, and the chains culled."""
    if not stats or not stats[0]:
        return "native stages: not measured"
    median = {key: float(np.median([row[key] for row in stats])) for key in stats[0]}
    stages = ", ".join(
        f"{key} {median[key] * 1000:.1f}"
        for key in ("input", "depth", "occlusion", "paths", "skia", "meshes")
    )
    return (
        f"native ms: {stages}; culled {median['culled']:.0f} of "
        f"{median['tested']:.0f} chains"
    )


def _memory_report(memory) -> str:
    mb = 1 << 20
    skia = [usage.skia for usage in memory]
    report = (
        f"skia peak {max(skia) / mb:7.1f} MB, "
        f"growth {(skia[-1] - skia[0]) / mb:+.1f} MB"
    )
    device = [usage.device for usage in memory if usage.device is not None]
    if device:
        report = (
            f"device peak {max(device) / mb:7.1f} MB, "
            f"growth {(device[-1] - device[0]) / mb:+.1f} MB; " + report
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scenes", nargs="*", default=["dense", "many"])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument(
        "--no-culling", action="store_true", help="draw depth pieces behind meshes too"
    )
    args = parser.parse_args()
    if args.python_only:
        skia_renderer.manim_vk.Device = _NullDevice
    width, height = (int(value) for value in args.size.split("x"))
    builders = {"dense": dense, "many": many}
    with tempconfig(
        {
            "pixel_width": width,
            "pixel_height": height,
            "format": "none",
            "verbosity": "ERROR",
        }
    ):
        for name in args.scenes:
            camera, mobjects, step = builders[name]()
            renderer = skia_renderer.SkiaRenderer(camera=camera)
            if args.no_culling:
                renderer.set_occlusion_culling(False)
            try:
                renderer.render_mobjects(mobjects)
                renderer.get_frame()
                cpu, gpu, memory, culled = [], [], [], []
                for frame in range(args.frames):
                    step(frame)
                    started = time.perf_counter()
                    renderer.render_mobjects(mobjects)
                    recorded = time.perf_counter()
                    renderer.get_frame()
                    cpu.append(recorded - started)
                    gpu.append(time.perf_counter() - recorded)
                    memory.append(renderer.memory_usage())
                    culled.append(renderer.take_draw_stats())
            finally:
                renderer.close()
            cpu_ms, gpu_ms = 1000 * np.array(cpu), 1000 * np.array(gpu)
            total = cpu_ms + gpu_ms
            print(
                f"{name:6s} median {np.median(total):7.1f} ms/frame "
                f"(cpu {np.median(cpu_ms):6.1f}, gpu {np.median(gpu_ms):6.1f}), "
                f"min {total.min():7.1f}, max {total.max():7.1f} "
                f"({args.frames} frames at {width}x{height})"
            )
            print(f"{'':6s} {_memory_report(memory)}")
            print(f"{'':6s} {_draw_report(culled)}")


if __name__ == "__main__":
    main()
