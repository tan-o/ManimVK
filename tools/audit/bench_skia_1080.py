"""1080p readback benchmark: old eager commit vs deferred direct write."""

from __future__ import annotations

import os
import time

import numpy as np

from manim import Circle, Rotate, Scene, VGroup, PI, tempconfig
from manim.renderer.skia.target import SkiaRenderTarget


class Perf1080(Scene):
    def construct(self) -> None:
        group = VGroup(
            *[
                Circle(radius=0.15, color="#61DDAA").shift(
                    np.array([-2.0 + 0.4 * index, 0.0, 0.0])
                )
                for index in range(10)
            ]
        )
        self.add(group)
        self.play(Rotate(group, PI / 2), run_time=2.0)


def old_read_pixels(self: SkiaRenderTarget) -> np.ndarray:
    self.commit()
    return self._pixels.copy()


if os.environ.get("PERF_OLD"):
    from manim.renderer.skia.rendering import SkiaDrawingContext

    _old_display = SkiaDrawingContext._display_non_background_colored_vmobjects

    def eager_display(self, vmobjects):
        _old_display(self, vmobjects)
        self.target.commit()

    SkiaDrawingContext._display_non_background_colored_vmobjects = eager_display
    SkiaRenderTarget.read_pixels = old_read_pixels  # type: ignore[method-assign]

started = time.perf_counter()
with tempconfig(
    {
        "pixel_width": 1920,
        "pixel_height": 1080,
        "frame_rate": 60,
        "renderer": "vulkan",
        "format": "mp4",
        "media_dir": os.path.join("media", "perf_bench_1080"),
        "disable_caching": True,
        "progress_bar": "none",
    }
):
    Perf1080().render()
print(f"PERF_OLD={bool(os.environ.get('PERF_OLD'))} elapsed={time.perf_counter() - started:.3f}s")
