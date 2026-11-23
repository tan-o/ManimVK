from __future__ import annotations

from manim import Mobject
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL


def test_metaclass_registry(config):
    """The compatibility metaclass no longer rebases classes at runtime."""

    class SomeTestMobject(Mobject, metaclass=ConvertToOpenGL):
        pass

    assert SomeTestMobject in ConvertToOpenGL._converted_classes

    config.renderer = "opengl"
    # The Vulkan/Skia backend uses the classic Mobject hierarchy for every
    # renderer value; the retired moderngl backend must not swap bases.
    assert SomeTestMobject.__bases__ == (Mobject,)

    config.renderer = "cairo"
    assert SomeTestMobject.__bases__ == (Mobject,)
