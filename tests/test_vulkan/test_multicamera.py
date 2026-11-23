"""MultiCamera composition must stay on the GPU in the Vulkan renderer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("manim_vk")

from manim import Square, config, tempconfig  # noqa: E402
from manim.mobject.types.image_mobject import ImageMobjectFromCamera  # noqa: E402
from manim.renderer.cairo.camera import Camera, MultiCamera  # noqa: E402
from manim.renderer.skia.renderer import SkiaRenderer  # noqa: E402


def test_multicamera_subviews_are_composited_without_readback():
    with tempconfig(
        {
            "renderer": "vulkan",
            "pixel_width": 160,
            "pixel_height": 90,
            "format": "none",
            "dry_run": True,
            "verbosity": "ERROR",
        }
    ):
        view = ImageMobjectFromCamera(Camera(background_color="#ffffff"))
        view.set(width=7.5)
        renderer = SkiaRenderer(camera=MultiCamera([view]))
        try:
            renderer.render_mobjects(
                [Square(fill_color="#ffffff", fill_opacity=1, stroke_width=0), view]
            )
            sub_target = renderer._sub_targets[id(view)]
            # The sub-view surface never had to be copied into host pixels.
            assert sub_target._last_readback is None
            frame = renderer.get_frame()
            assert frame.shape == (90, 160, 4)
            assert np.asarray(frame)[..., :3].max() > 0
        finally:
            renderer.close()
