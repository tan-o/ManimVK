"""Classic-mobject adapters for the small set of OpenGL surface helpers.

These are deliberately not GLSL surfaces.  They preserve the public
``OpenGLTexturedSurface`` / ``OpenGLSurfaceMesh`` construction API while
building ordinary ``Surface`` / ``VGroup`` objects, so the Skia/Vulkan renderer,
animations and camera code can handle them unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from manim.mobject.mobject import Mobject
from manim.mobject.three_d.three_dimensions import Surface
from manim.mobject.types.vectorized_mobject import VGroup, VMobject


def _resolution_pair(resolution: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(resolution, int):
        return (resolution, resolution)
    values = tuple(int(value) for value in resolution)
    if len(values) != 2:
        raise ValueError("resolution must be an int or a 2-tuple")
    return values


def _image_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


class OpenGLSurface(Surface):
    """Classic ``Surface`` with the OpenGL constructor spelling.

    ``OpenGLSurface`` used ``uv_func`` and separate ``color`` / ``opacity``
    keywords.  Map those to the classic ``Surface`` arguments so old scene code
    constructs a normal, animatable surface.
    """

    def __init__(
        self,
        uv_func: Any = None,
        u_range: tuple[float, float] = (0, 1),
        v_range: tuple[float, float] = (0, 1),
        resolution: int | Iterable[int] = 32,
        axes: Any = None,
        color: Any = None,
        colorscale: Any = None,
        colorscale_axis: int = 2,
        opacity: float = 1.0,
        gloss: float = 0.3,
        shadow: float = 0.4,
        prefered_creation_axis: int = 1,
        epsilon: float = 1e-5,
        render_primitive: Any = None,
        depth_test: bool = True,
        shader_folder: Any = None,
        **kwargs: Any,
    ):
        self.axes = axes
        self.colorscale = colorscale
        self.colorscale_axis = colorscale_axis
        self.prefered_creation_axis = prefered_creation_axis
        self.epsilon = epsilon
        super().__init__(
            func=uv_func if uv_func is not None else (lambda u, v: np.array([u, v, 0.0])),
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            fill_color=color if color is not None else kwargs.pop("fill_color", None) or "#888888",
            fill_opacity=opacity,
            **kwargs,
        )

    def get_surface_points_and_nudged_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = self.get_all_points()
        return points, points, points

    def get_unit_normals(self) -> np.ndarray:
        return np.zeros_like(self.get_all_points())

    def get_triangle_indices(self) -> np.ndarray:
        return np.zeros(0, dtype=int)


class OpenGLTexturedSurface(Surface):
    """A classic ``Surface`` with per-face colours sampled from an image.

    The original OpenGL object sampled a texture in a fragment shader.  This
    adapter keeps the same constructor arguments and produces a real,
    animatable classic surface; each face gets the image colour at its UV
    position, which is enough for the standard ``SurfaceExample`` workflow on
    the Vulkan path without reintroducing a GLSL pipeline.
    """

    def __init__(
        self,
        uv_surface: Surface,
        image_file: str | Path,
        dark_image_file: str | Path | None = None,
        image_mode: str | Iterable[str] = "RGBA",
        shader_folder: str | Path | None = None,
        **kwargs: Any,
    ):
        if not isinstance(uv_surface, Surface):
            raise TypeError("uv_surface must be a classic Surface")
        self.uv_surface = uv_surface
        self.texture_image = _image_rgb(image_file)
        self.dark_texture_image = (
            _image_rgb(dark_image_file) if dark_image_file is not None else None
        )
        resolution = _resolution_pair(uv_surface.resolution)
        super().__init__(
            func=uv_surface.func,
            u_range=uv_surface.u_range,
            v_range=uv_surface.v_range,
            resolution=resolution,
            checkerboard_colors=False,
            fill_color=uv_surface.fill_color,
            fill_opacity=uv_surface.fill_opacity,
            stroke_color=uv_surface.stroke_color,
            stroke_width=uv_surface.stroke_width,
            **kwargs,
        )
        self._apply_texture_colors()

    def _apply_texture_colors(self) -> None:
        u_res, v_res = _resolution_pair(self.uv_surface.resolution)
        image = self.texture_image
        height, width = image.shape[:2]
        faces = self.list_of_faces
        for i in range(u_res):
            u = (i + 0.5) / u_res
            for j in range(v_res):
                v = (j + 0.5) / v_res
                face = faces[i * v_res + j] if i * v_res + j < len(faces) else None
                if face is None:
                    continue
                x = min(width - 1, int(u * width))
                y = min(height - 1, int((1.0 - v) * height))
                color = image[y, x]
                rgb = tuple(int(round(float(channel) * 255)) for channel in color)
                face.set_fill("#{:02x}{:02x}{:02x}".format(*rgb), opacity=1.0)
                face.set_stroke(width=0)

    def get_surface_points_and_nudged_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = self.get_all_points()
        return points, points, points

    def get_unit_normals(self) -> np.ndarray:
        return np.zeros_like(self.get_all_points())

    def get_triangle_indices(self) -> np.ndarray:
        return np.zeros(0, dtype=int)


class OpenGLSurfaceMesh(VGroup):
    """Line-grid overlay for a classic surface, replacing the OpenGL mesh helper."""

    def __init__(
        self,
        uv_surface: Surface,
        resolution: tuple[int, int] | None = None,
        stroke_width: float = 1,
        normal_nudge: float = 1e-2,
        depth_test: bool = True,
        flat_stroke: bool = False,
        **kwargs: Any,
    ):
        if not isinstance(uv_surface, Surface):
            raise TypeError("uv_surface must be a classic Surface")
        super().__init__()
        self.uv_surface = uv_surface
        self.resolution = resolution if resolution is not None else (21, 21)
        self.normal_nudge = normal_nudge
        self._build_lines(stroke_width=stroke_width, **kwargs)

    def _build_lines(self, *, stroke_width: float, **kwargs: Any) -> None:
        part_u, part_v = _resolution_pair(self.resolution)
        u_values = np.linspace(*self.uv_surface.u_range, part_u)
        v_values = np.linspace(*self.uv_surface.v_range, part_v)
        for u in u_values:
            points = np.array([self.uv_surface.func(u, v) for v in v_values])
            path = VMobject()
            path.set_points_smoothly(points)
            path.set_stroke(width=stroke_width, **kwargs)
            self.add(path)
        for v in v_values:
            points = np.array([self.uv_surface.func(u, v) for u in u_values])
            path = VMobject()
            path.set_points_smoothly(points)
            path.set_stroke(width=stroke_width, **kwargs)
            self.add(path)
