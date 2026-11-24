"""Per-frame cost of the renderer on two stress scenes.

* ``dense``: two depth-tested VMobjects of 50 000 short strokes each over a
  translucent sphere in a ThreeDScene camera (like HairyEarth's dense field).
* ``many``: 3000 separate VMobjects in a 2D scene.

Each frame is drawn and read back, so the time includes the GPU.  With
``--python-only`` the native target is replaced by one that does nothing,
which leaves the Python side alone::

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


class _NullDevice:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def device_name(self) -> str:
        return "none (--python-only)"

    def create_skia_target(self, width: int, height: int) -> _NullTarget:
        return _NullTarget()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scenes", nargs="*", default=["dense", "many"])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--python-only", action="store_true")
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
            try:
                renderer.render_mobjects(mobjects)
                renderer.get_frame()
                times = []
                for frame in range(args.frames):
                    step(frame)
                    started = time.perf_counter()
                    renderer.render_mobjects(mobjects)
                    renderer.get_frame()
                    times.append(time.perf_counter() - started)
            finally:
                renderer.close()
            times_ms = 1000 * np.array(times)
            print(
                f"{name:6s} median {np.median(times_ms):7.1f} ms/frame, "
                f"min {times_ms.min():7.1f}, max {times_ms.max():7.1f} "
                f"({args.frames} frames at {width}x{height})"
            )


if __name__ == "__main__":
    main()
