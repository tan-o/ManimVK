"""Public OpenGL-family entry point.

The rendering backend for this build is Skia's Ganesh Vulkan backend.  The
``OpenGL*`` names are kept as a public compatibility facade, but they now map
to the classic mobject classes that the Skia renderer understands.  This keeps
existing user code import-compatible without routing anything through the
legacy OpenGL bridge.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    from dearpygui import dearpygui as dpg


from manim.mobject.opengl.dot_cloud import *
from manim.mobject.opengl.opengl_geometry import *
from manim.mobject.opengl.opengl_image_mobject import *
from manim.mobject.opengl.opengl_mobject import *
from manim.mobject.opengl.opengl_point_cloud_mobject import *
from manim.mobject.opengl.opengl_surface import *
from manim.mobject.opengl.opengl_three_dimensions import *
from manim.mobject.opengl.opengl_vectorized_mobject import *

from ..renderer.opengl.shader import *
from ..utils.opengl import *


# The names below are deliberately rebound after the wildcard imports above.
# They are the Vulkan/Skia equivalents of the original OpenGL classes.  The
# implementations live in the classic mobject tree, so animations, caching,
# camera handling, file writing and the rest of the shared Manim machinery do
# not need a second code path.
from manim.mobject.geometry.arc import (
    AnnularSector,
    Annulus,
    Arc,
    ArcBetweenPoints,
    Circle,
    CubicBezier,
    CurvedArrow,
    CurvedDoubleArrow,
    Dot,
    Ellipse,
    Sector,
    TipableVMobject,
)
from manim.mobject.geometry.line import (
    Arrow,
    DashedLine,
    DoubleArrow,
    Elbow,
    Line,
    TangentLine,
    Vector,
)
from manim.mobject.geometry.polygram import (
    Polygon,
    Rectangle,
    RegularPolygon,
    RoundedRectangle,
    Square,
    Triangle,
)
from manim.mobject.geometry.tips import ArrowTip
from manim.mobject.mobject import Group, Mobject
from manim.mobject.three_d.three_dimensions import Surface
from manim.mobject.types.image_mobject import ImageMobject
from manim.mobject.text.tex_mobject import MathTex, Tex
from manim.mobject.text.text_mobject import Text
from manim.mobject.types.point_cloud_mobject import (
    PGroup,
    PMobject,
    Point,
    PointCloudDot,
)
from manim.mobject.types.vectorized_mobject import (
    CurvesAsSubmobjects,
    DashedVMobject,
    VGroup,
    VMobject,
    VectorizedPoint,
)

OpenGLMobject = Mobject
OpenGLGroup = Group
OpenGLPoint = Point
OpenGLVMobject = VMobject
OpenGLVGroup = VGroup
OpenGLVectorizedPoint = VectorizedPoint
OpenGLCurvesAsSubmobjects = CurvesAsSubmobjects
OpenGLDashedVMobject = DashedVMobject
OpenGLTipableVMobject = TipableVMobject
OpenGLArc = Arc
OpenGLArcBetweenPoints = ArcBetweenPoints
OpenGLCurvedArrow = CurvedArrow
OpenGLCurvedDoubleArrow = CurvedDoubleArrow
OpenGLCircle = Circle
OpenGLDot = Dot
OpenGLEllipse = Ellipse
OpenGLAnnularSector = AnnularSector
OpenGLSector = Sector
OpenGLAnnulus = Annulus
OpenGLLine = Line
OpenGLDashedLine = DashedLine
OpenGLTangentLine = TangentLine
OpenGLElbow = Elbow
OpenGLArrow = Arrow
OpenGLVector = Vector
OpenGLDoubleArrow = DoubleArrow
OpenGLCubicBezier = CubicBezier
OpenGLPolygon = Polygon
OpenGLRegularPolygon = RegularPolygon
OpenGLTriangle = Triangle
OpenGLArrowTip = ArrowTip
OpenGLRectangle = Rectangle
OpenGLSquare = Square
OpenGLRoundedRectangle = RoundedRectangle
OpenGLPMobject = PMobject
OpenGLPGroup = PGroup
OpenGLPMPoint = Point
OpenGLImageMobject = ImageMobject
OpenGLText = Text
OpenGLMathTex = MathTex
OpenGLTex = Tex
DotCloud = PointCloudDot
TrueDot = Dot

from .compat import OpenGLSurface, OpenGLSurfaceMesh, OpenGLTexturedSurface

Surface = OpenGLSurface
SurfaceMesh = OpenGLSurfaceMesh
TexturedSurface = OpenGLTexturedSurface

__all__ = [name for name in globals() if not name.startswith("_")]
__all__ += ["OpenGLCamera", "Camera", "ThreeDCamera"]


def __getattr__(name: str):
    # Import these lazily: importing the camera modules while manim.opengl is
    # first initialized would create a circular import through manim.renderer.
    if name in {"Camera", "OpenGLCamera"}:
        from manim.renderer.cairo.camera import Camera

        return Camera
    if name == "ThreeDCamera":
        from manim.renderer.cairo.camera import ThreeDCamera

        return ThreeDCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
