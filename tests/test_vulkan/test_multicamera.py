"""MultiCamera composition must stay on the GPU in the Vulkan renderer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("manim_vk")

from manim import Square, tempconfig  # noqa: E402
from manim.mobject.types.image_mobject import ImageMobjectFromCamera  # noqa: E402
from manim.camera import Camera, MultiCamera  # noqa: E402
from manim.renderer.skia.renderer import SkiaRenderer  # noqa: E402
from manim.renderer.skia.target import SkiaRenderTarget  # noqa: E402


def test_multicamera_subviews_are_composited_without_readback(monkeypatch):
    with tempconfig(
        {
            "pixel_width": 160,
            "pixel_height": 90,
            "format": "none",
            "dry_run": True,
            "verbosity": "ERROR",
        }
    ):
        reads = []
        read_pixels = SkiaRenderTarget.read_pixels
        monkeypatch.setattr(
            SkiaRenderTarget, "read_pixels", lambda target: reads.append(target) or read_pixels(target)
        )
        view = ImageMobjectFromCamera(Camera(background_color="#ffffff"))
        view.set(width=7.5)
        renderer = SkiaRenderer(camera=MultiCamera([view]))
        try:
            renderer.render_mobjects(
                [Square(fill_color="#ffffff", fill_opacity=1, stroke_width=0), view]
            )
            frame = renderer.get_frame()
            # Only the frame is read back: the sub-view is sampled on the GPU.
            assert reads == [renderer._target]
            assert frame.shape == (90, 160, 4)
            assert np.asarray(frame)[..., :3].max() > 0
        finally:
            renderer.close()
