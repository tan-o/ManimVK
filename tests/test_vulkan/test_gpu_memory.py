"""Frame after frame of the same animation, GPU memory stays level.

The scene is HairyEarth's heaviest kind of frame, scaled down: a translucent
sphere, a textured opaque Earth, wireframe lines and depth-sorted strokes
and tips, under a moving camera, read back pipelined as the video writer
does.  Every allocation that outlives its frame would pile up here.

Software Vulkan (lavapipe, as in CI) keeps device memory in the process, so
the resident set counts it; a GPU driver's own count
(``VK_EXT_memory_budget``) is checked as well where it is real.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("manim_vk")

from manim import (  # noqa: E402
    BLUE,
    DEGREES,
    GREY,
    MeshSurface,
    Sphere,
    TexturedSurface,
    VGroup,
    VMobject,
    tempconfig,
)
from manim.camera import ThreeDCamera  # noqa: E402
from manim.renderer.skia.renderer import SkiaRenderer  # noqa: E402

WARM_UP, MEASURED = 10, 40
MB = 1 << 20


def _rss() -> int:
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * 4096


def _strokes(count: int, seed: int) -> VGroup:
    rng = np.random.default_rng(seed)
    start = rng.normal(size=(count, 3))
    start *= 3.02 / np.linalg.norm(start, axis=1, keepdims=True)
    end = start * 1.1
    delta = end - start
    shafts = VMobject(stroke_color=BLUE, stroke_width=2, stroke_opacity=0.35)
    shafts.set_points(
        np.stack([start, start + delta / 3, start + 2 * delta / 3, end], axis=1).reshape(-1, 3)
    )
    side = 0.02 * np.cross(delta, [0.0, 0.0, 1.0])
    corners = [end - side, end + 0.3 * delta, end + side]
    edges = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        d = b - a
        edges.append(np.stack([a, a + d / 3, a + 2 * d / 3, b], axis=1))
    tips = VMobject(fill_color=BLUE, fill_opacity=0.35, stroke_width=0)
    tips.set_points(np.concatenate(edges, axis=1).reshape(-1, 3))
    return VGroup(shafts, tips).apply_depth_test()


def _wireframe() -> VMobject:
    lines = VMobject(stroke_color=GREY, stroke_width=1, stroke_opacity=0.15)
    t = np.linspace(0, 2 * np.pi, 49)
    for latitude in np.linspace(-1.2, 1.2, 9):
        ring = 3 * np.stack(
            [np.cos(latitude) * np.cos(t), np.cos(latitude) * np.sin(t), np.full_like(t, np.sin(latitude))],
            axis=1,
        )
        lines.start_new_path(ring[0])
        lines.add_points_as_corners(ring[1:])
    return lines.apply_depth_test()


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/statm")
def test_a_steady_animation_keeps_gpu_memory_level(tmp_path):
    rng = np.random.default_rng(1)
    day, night = tmp_path / "day.png", tmp_path / "night.png"
    for path in (day, night):
        Image.fromarray(rng.integers(0, 256, (256, 512, 3), dtype=np.uint8)).save(path)
    camera = ThreeDCamera(phi=60 * DEGREES, theta=30 * DEGREES)
    mobjects = [
        TexturedSurface(Sphere(radius=2.0, resolution=(48, 24)), str(day), str(night)),
        MeshSurface(Sphere(radius=3, resolution=(48, 24)), color=BLUE, opacity=0.3),
        _wireframe(),
        _strokes(5000, 0),
    ]
    usage = []
    with tempconfig({"pixel_width": 1280, "pixel_height": 720}):
        renderer = SkiaRenderer(camera=camera)
        try:
            in_flight = None
            for frame in range(WARM_UP + MEASURED):
                camera.set_theta((30 + 2 * frame) * DEGREES)
                renderer.render_mobjects(mobjects)
                if in_flight is not None:
                    in_flight()
                in_flight = renderer.request_yuv420_frame()
                memory = renderer.memory_usage()
                usage.append((_rss(), memory.device, memory.skia))
            in_flight()
        finally:
            renderer.close()
    rss = np.array([row[0] for row in usage[WARM_UP:]]) / MB
    skia = np.array([row[2] for row in usage[WARM_UP:]]) / MB
    print(
        "resident MB per frame:", np.round(rss, 1).tolist(),
        "\nskia MB per frame:", np.round(skia, 1).tolist(),
        "\ndevice MB per frame:", [row[1] and round(row[1] / MB, 1) for row in usage[WARM_UP:]],
    )
    # Skia keeps what a frame needed, up to its 256 MB budget.  Tessellated
    # paths need a few render targets; CPU masks of depth slices would need
    # a 1280x720 texture per slice and object, and fill the budget.
    assert skia.max() < 128
    # One leaked 1280x720 RGBA texture a frame would add 140 MB.
    assert rss[-1] - rss[0] < 48
    assert skia[-1] - skia[0] < 16
    device = [row[1] for row in usage[WARM_UP:]]
    if device[0] is not None:
        assert (device[-1] - device[0]) / MB < 48
