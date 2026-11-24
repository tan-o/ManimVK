"""Public Mobject image APIs render through the Skia/Vulkan renderer."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("manim_vk")

from PIL import Image  # noqa: E402
from manim import Circle, tempconfig  # noqa: E402
from manim.renderer.skia import SkiaRenderer  # noqa: E402


def test_mobject_get_image_renders_with_vulkan():
    with tempconfig(
        {
            "pixel_width": 64,
            "pixel_height": 36,
            "format": "png",
            "dry_run": True,
            "verbosity": "ERROR",
        }
    ):
        with mock.patch.object(
            SkiaRenderer, "render_mobjects", autospec=True, side_effect=SkiaRenderer.render_mobjects
        ) as render:
            image = Circle().get_image()
        render.assert_called_once()

    assert isinstance(image, Image.Image)
    assert image.size == (64, 36)
