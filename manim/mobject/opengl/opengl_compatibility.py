from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING, Any

__all__ = ["ConvertToOpenGL"]

if TYPE_CHECKING:
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject
    from manim.mobject.opengl.opengl_point_cloud_mobject import OpenGLPMobject
    from manim.mobject.opengl.opengl_three_dimensions import OpenGLSurface
    from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject


class ConvertToOpenGL(ABCMeta):
    """Compatibility metaclass retained for the classic mobject hierarchy.

    The Vulkan/Skia backend renders the classic mobject family.  The former
    runtime swap to ``OpenGL*`` classes was part of the retired moderngl
    backend, so the metaclass no longer converts classes based on
    ``config.renderer``.
    """

    _converted_classes: list[type] = []

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        return super().__new__(mcls, name, bases, namespace)

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        super().__init__(name, bases, namespace)
        cls._converted_classes.append(cls)
