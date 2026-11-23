"""Text-heavy benchmark for the opaque-fill contour batch."""

from __future__ import annotations

import os
import time

from manim import Scene, Text, Rotate, PI, tempconfig
from manim.renderer.skia.rendering import SkiaDrawingContext


class TextPerf(Scene):
    def construct(self) -> None:
        text = Text("Vulkan on Windows " * 24, font_size=24)
        self.add(text)
        self.play(Rotate(text, PI / 8), run_time=2.0)


if os.environ.get("PERF_NO_BATCH"):
    SkiaDrawingContext._solid_fill_style = staticmethod(lambda vmobject: None)  # type: ignore[method-assign]

started = time.perf_counter()
with tempconfig(
    {
        "quality": "low_quality",
        "renderer": "vulkan",
        "format": "mp4",
        "media_dir": os.path.join("media", "perf_text_batch"),
        "disable_caching": True,
        "progress_bar": "none",
    }
):
    TextPerf().render()
print(
    f"PERF_NO_BATCH={bool(os.environ.get('PERF_NO_BATCH'))} "
    f"elapsed={time.perf_counter() - started:.3f}s"
)
