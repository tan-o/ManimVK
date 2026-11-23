"""Public Mobject image APIs must stay on the Vulkan/Skia path by default."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("manim_vk")

from PIL import Image  # noqa: E402
from manim import Circle, tempconfig  # noqa: E402


def test_mobject_get_image_does_not_construct_cairo_renderer():
    with tempconfig(
        {
            "renderer": "vulkan",
            "pixel_width": 64,
            "pixel_height": 36,
            "format": "png",
            "dry_run": True,
            "verbosity": "ERROR",
        }
    ):
        with mock.patch(
            "manim.renderer.cairo.CairoRenderer",
            side_effect=AssertionError("CairoRenderer must not be used by default"),
        ) as cairo:
            image = Circle().get_image()
            cairo.assert_not_called()

    assert isinstance(image, Image.Image)
    assert image.size == (64, 36)
