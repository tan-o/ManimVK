"""Tiny local benchmark for the Skia/Vulkan readback path.

Set ``PERF_OLD=1`` to run the previous commit()-then-copy readback.
"""

from __future__ import annotations

import os
import time

import numpy as np

from manim import Circle, Rotate, Scene, tempconfig, PI
from manim.renderer.skia.target import SkiaRenderTarget


class PerfScene(Scene):
    def construct(self) -> None:
        group = self.build_group()
        self.add(group)
        self.play(Rotate(group, PI / 2), run_time=2.5)

    def build_group(self):
        from manim import VGroup

        colors = ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A"]
        return VGroup(
            *[
                Circle(radius=0.035, color=colors[index % len(colors)]).shift(
                    np.array(
                        [
                            -3.2 + 0.16 * (index % 41),
                            -2.0 + 0.16 * (index // 41),
                            0.0,
                        ]
                    )
                )
                for index in range(41 * 25)
            ]
        )


def old_read_pixels(self: SkiaRenderTarget) -> np.ndarray:
    self.commit()
    return self._pixels.copy()


if os.environ.get("PERF_OLD"):
    from manim.renderer.skia.rendering import SkiaDrawingContext

    _old_display = (
        SkiaDrawingContext._display_non_background_colored_vmobjects
    )

    def eager_display(self, vmobjects):
        _old_display(self, vmobjects)
        self.target.commit()

    SkiaDrawingContext._display_non_background_colored_vmobjects = eager_display
    SkiaRenderTarget.read_pixels = old_read_pixels  # type: ignore[method-assign]

started = time.perf_counter()
with tempconfig(
    {
        "quality": "low_quality",
        "renderer": "vulkan",
        "format": "mp4",
        "media_dir": os.path.join("media", "perf_bench"),
        "disable_caching": True,
        "progress_bar": "none",
    }
):
    PerfScene().render()
elapsed = time.perf_counter() - started
print(f"PERF_OLD={bool(os.environ.get('PERF_OLD'))} elapsed={elapsed:.3f}s")
