"""User-authored SkSL effects for the Vulkan/Skia renderer."""

from __future__ import annotations

__all__ = ["SkSLMobject"]

import numpy as np

from manim._config import config
from manim.mobject.mobject import Mobject


class SkSLMobject(Mobject):
    """A rectangular quad painted by a user-supplied SkSL runtime effect.

    Parameters
    ----------
    source
        SkSL source containing a ``half4 main(float2 fragCoord)`` entry point.
        Compile errors raise a readable Python exception.
    uniforms
        Optional float uniforms passed to the shader.  ``u_resolution`` is
        supplied automatically when the shader declares it.
    width, height
        Initial size in scene units.  Defaults to the current frame size.
    """

    def __init__(
        self,
        source: str,
        uniforms: dict[str, object] | None = None,
        *,
        width: float | None = None,
        height: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.uniforms = dict(uniforms or {})
        half_width = (width or config.frame_width) / 2
        half_height = (height or config.frame_height) / 2
        self.set_points(
            np.array(
                [
                    [-half_width, half_height, 0.0],
                    [half_width, half_height, 0.0],
                    [half_width, -half_height, 0.0],
                    [-half_width, -half_height, 0.0],
                ]
            )
        )
