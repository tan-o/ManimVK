"""SkSL entry point regression: valid effects draw, invalid source raises."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("manim_vk")

from manim import SkSLMobject, tempconfig  # noqa: E402
from manim.renderer.skia.renderer import SkiaRenderer  # noqa: E402

VALID_SKSL = """
uniform float2 u_resolution;
half4 main(float2 fragCoord) {
    float2 uv = fragCoord / u_resolution;
    return half4(uv.x, uv.y, 0.5, 1.0);
}
"""


def test_sksl_effect_renders_and_invalid_source_raises():
    with tempconfig(
        {
            "pixel_width": 64,
            "pixel_height": 36,
            "format": "none",
            "dry_run": True,
            "verbosity": "ERROR",
        }
    ):
        renderer = SkiaRenderer()
        try:
            renderer.render_mobjects([SkSLMobject(VALID_SKSL)])
            frame = renderer.get_frame()
            assert frame.shape == (36, 64, 4)
            assert np.asarray(frame)[..., :3].max() > 0

            with pytest.raises(RuntimeError, match="SkSL compile failed"):
                renderer.render_mobjects(
                    [SkSLMobject("half4 main(float2 p) { return half4(1.0);")]
                )
        finally:
            renderer.close()
