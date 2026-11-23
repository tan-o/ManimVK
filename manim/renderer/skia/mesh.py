"""Minimal Skia/Vulkan adapter for the public ``Mesh``/``Object3D`` API.

The original OpenGL renderer evaluated user GLSL.  The Vulkan build deliberately
does not keep a second GL pipeline; instead it consumes the mesh's vertex
attributes and draws their triangles through the same Skia/Ganesh Vulkan target
used by classic mobjects.  This covers the common ``Mesh`` use cases (coloured
geometry, full-screen quads with uniform colours, custom attribute meshes) while
leaving arbitrary fragment-shader authoring as a separate feature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manim.renderer.cairo.camera import Camera
    from manim.renderer.opengl.shader import Mesh, Object3D
    from manim.renderer.skia.target import SkiaRenderTarget


_TRIANGLES = 4
_TRIANGLE_STRIP = 5
_TRIANGLE_FAN = 6


def _attribute_array(mesh: Mesh, names: tuple[str, ...]) -> np.ndarray | None:
    attributes = getattr(mesh, "attributes", None)
    if attributes is None or getattr(attributes, "dtype", None) is None:
        return None
    dtype_names = attributes.dtype.names or ()
    for name in names:
        if name in dtype_names:
            return attributes[name]
    return None


def _color_array(mesh: Mesh, count: int) -> np.ndarray:
    colors = _attribute_array(mesh, ("in_color", "color", "rgba", "fill_color"))
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.ndim == 1:
            colors = np.repeat(colors.reshape(1, -1), count, axis=0)
        if colors.shape[1] == 3:
            colors = np.concatenate(
                [colors, np.ones((len(colors), 1), dtype=np.float32)],
                axis=1,
            )
        return colors[:, :4]

    # ``Mesh`` objects without a vertex colour attribute are usually driven by
    # a small uniform-only shader.  The shader object stores its uniforms even
    # though the Skia adapter does not compile the GLSL; use them as a flat
    # fallback colour.
    shader = getattr(mesh, "shader", None)
    program = getattr(shader, "shader_program", None)
    members = getattr(program, "_members", {}) if program is not None else {}
    uniform_color = members.get("u_color")
    if uniform_color is not None:
        value = np.asarray(uniform_color, dtype=np.float32).reshape(-1)
        if value.size == 3:
            value = np.concatenate([value, [1.0]])
        if value.size >= 4:
            return np.repeat(value[:4].reshape(1, 4), count, axis=0)
    channels = [members.get(f"v_{name}") for name in ("red", "green", "blue")]
    if all(channel is not None for channel in channels):
        try:
            value = np.array(
                [float(channels[0]), float(channels[1]), float(channels[2]), 1.0],
                dtype=np.float32,
            )
            return np.repeat(value.reshape(1, 4), count, axis=0)
        except (TypeError, ValueError):
            pass
    return np.ones((count, 4), dtype=np.float32)



def _triangle_indices(mesh: Mesh, count: int) -> list[tuple[int, int, int]]:
    indices = getattr(mesh, "indices", None)
    if indices is not None:
        indices_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(indices_array) >= 3:
            return [
                (int(indices_array[i]), int(indices_array[i + 1]), int(indices_array[i + 2]))
                for i in range(0, len(indices_array) - 2, 3)
            ]
    primitive = int(getattr(mesh, "primitive", _TRIANGLES))
    if primitive == _TRIANGLES:
        return [
            (i, i + 1, i + 2)
            for i in range(0, count - 2, 3)
        ]
    if primitive == _TRIANGLE_STRIP:
        return [
            (i, i + 1, i + 2) if i % 2 == 0 else (i + 1, i, i + 2)
            for i in range(count - 2)
        ]
    if primitive == _TRIANGLE_FAN:
        return [(0, i + 1, i + 2) for i in range(count - 2)]
    return []


def _mesh_vertices(mesh: Mesh) -> np.ndarray | None:
    vertices = _attribute_array(mesh, ("in_vert", "point", "position", "vertex"))
    if vertices is None:
        return None
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim == 1:
        vertices = vertices.reshape(1, -1)
    if vertices.shape[1] < 3:
        return None
    return vertices[:, :3]


def _interpolated_triangles(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    steps: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Approximate Gouraud shading with small flat-colour triangles.

    Skia's path filler is flat per path, so a directly drawn triangle loses
    the per-vertex colour gradient of the original OpenGL pipeline.  Splitting
    a triangle into a barycentric grid keeps the gradient visible while using
    the same vector path renderer as classic mobjects.
    """
    if colors.shape[0] != 3:
        return []
    spread = float(np.max(np.ptp(colors[:, :3], axis=0))) if len(colors) else 0.0
    steps = 1 if spread < 0.02 else max(1, int(steps))

    def barycentric(values: np.ndarray, i: int, j: int) -> np.ndarray:
        return (
            values[0] * (steps - i - j) + values[1] * i + values[2] * j
        ) / steps

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(steps):
        for j in range(steps - i):
            a = barycentric(points, i, j)
            b = barycentric(points, i + 1, j)
            c = barycentric(points, i, j + 1)
            ca = barycentric(colors, i, j)
            cb = barycentric(colors, i + 1, j)
            cc = barycentric(colors, i, j + 1)
            result.append(
                (
                    np.array([a, b, c]),
                    np.mean(np.array([ca, cb, cc]), axis=0),
                )
            )
            if i + j <= steps - 2:
                d = barycentric(points, i + 1, j + 1)
                cd = barycentric(colors, i + 1, j + 1)
                result.append(
                    (
                        np.array([b, d, c]),
                        np.mean(np.array([cb, cd, cc]), axis=0),
                    )
                )
    return result


def _merge_uniform_triangle_pair(
    triangles: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Merge the two triangles of a same-colour quad into one path.

    This avoids the anti-aliased diagonal seam produced when a full-screen
    quad is filled as two independent triangles.
    """
    if len(triangles) != 2:
        return None
    points_a, colors_a = triangles[0][1], triangles[0][2]
    points_b, colors_b = triangles[1][1], triangles[1][2]
    if not np.allclose(colors_a, colors_b, atol=1e-3):
        return None
    all_points = np.vstack([points_a, points_b])
    rounded = np.round(all_points, 6)
    unique_rounded, inverse = np.unique(rounded, axis=0, return_inverse=True)
    if len(unique_rounded) != 4:
        return None
    unique = np.zeros((4, 2), dtype=np.float64)
    for index, point in zip(inverse, all_points):
        unique[index] = point
    center = unique.mean(axis=0)
    order = np.argsort(
        np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0])
    )
    return unique[order], colors_a.mean(axis=0)


def _fullscreen_shader(mesh: Mesh) -> tuple[str, float] | None:
    """Recognise the full-screen shaders that have native SkSL counterparts."""
    shader = getattr(mesh, "shader", None)
    program = getattr(shader, "shader_program", None)
    members = getattr(program, "_members", {}) if program is not None else {}
    try:
        shader_time = float(members.get("u_time", 0.0))
    except (TypeError, ValueError):
        shader_time = 0.0

    if getattr(shader, "name", None) == "design_3":
        return ("design_3", shader_time)

    source = getattr(shader, "source", None)
    if not isinstance(source, dict):
        return None
    fragment = source.get("fragment_shader", "")
    if "hsb2rgb" in fragment and "gl_FragCoord" in fragment:
        return ("hsb", shader_time)
    if "palette(" in fragment and "rm(" in fragment and "gl_FragCoord" in fragment:
        return ("design_3", shader_time)
    return None


def _fullscreen_hsb_time(mesh: Mesh) -> float | None:
    """Backwards-compatible HSB-only probe used by older callers."""
    result = _fullscreen_shader(mesh)
    if result is not None and result[0] == "hsb":
        return result[1]
    return None


def draw_meshes(
    meshes: list[Object3D],
    *,
    camera: Camera,
    target: SkiaRenderTarget,
) -> None:
    """Draw the triangle meshes in ``meshes`` into ``target``.

    All meshes are flattened, projected with the classic camera and painted
    back-to-front so that a depth buffer is unnecessary for the common case.
    """
    special_meshes = [
        (mesh, _fullscreen_shader(mesh))
        for root in meshes
        for mesh in root.get_meshes()
        if _fullscreen_shader(mesh) is not None
    ]
    if special_meshes:
        for mesh, shader in special_meshes:
            kind, shader_time = shader or ("hsb", 0.0)
            if kind == "design_3":
                target.draw_design3_quad(shader_time)
            else:
                target.draw_hsb_quad(shader_time)
        return

    triangles: list[tuple[float, np.ndarray, np.ndarray]] = []
    for root in meshes:
        for mesh in root.get_meshes():
            if getattr(mesh, "skip_render", False):
                continue
            vertices = _mesh_vertices(mesh)
            if vertices is None or len(vertices) < 3:
                continue
            model = np.asarray(mesh.hierarchical_model_matrix(), dtype=np.float64)
            homogeneous = np.concatenate(
                [vertices, np.ones((len(vertices), 1))],
                axis=1,
            )
            world = homogeneous @ model.T
            is_three_d_camera = hasattr(camera, "phi_tracker")
            has_stored_euler = getattr(camera, "_euler_angles", None) is not None
            if is_three_d_camera or has_stored_euler:
                projected = camera.project_points(world[:, :3])
            else:
                # A classic 2D Camera applies its frame transform in the
                # render target matrix, so scene coordinates pass through.
                projected = world[:, :3]
            colors = _color_array(mesh, len(vertices))
            for i0, i1, i2 in _triangle_indices(mesh, len(vertices)):
                if max(i0, i1, i2) >= len(projected):
                    continue
                points = projected[[i0, i1, i2]]
                depth = float(np.mean(world[[i0, i1, i2], 2]))
                # Painter's algorithm: farther (smaller z) first.
                triangles.append((depth, points[:, :2], colors[[i0, i1, i2]]))

    if not triangles:
        return

    # Split strongly varying vertex-colour triangles into a small barycentric
    # grid, then draw the flat-colour pieces through the normal Skia path.
    context = target.get_context(camera)
    context.set_anti_alias(False)
    merged = _merge_uniform_triangle_pair(triangles)
    if merged is not None:
        quad, color = merged
        alpha = float(color[3]) if len(color) > 3 else 1.0
        if alpha > 0:
            context.set_source_rgba(float(color[0]), float(color[1]), float(color[2]), alpha)
            context.new_path()
            context.move_to(float(quad[0, 0]), float(quad[0, 1]))
            for point in quad[1:]:
                context.line_to(float(point[0]), float(point[1]))
            context.close_path()
            context.fill_preserve()
        context.set_anti_alias(True)
        return

    for _, points, colors in sorted(triangles, key=lambda item: item[0]):
        if not np.all(np.isfinite(points)):
            continue
        if np.all(colors[:, 3] <= 0):
            continue
        for triangle_points, color in _interpolated_triangles(points, colors):
            alpha = float(color[3]) if len(color) > 3 else 1.0
            if alpha <= 0:
                continue
            context.set_source_rgba(
                float(color[0]),
                float(color[1]),
                float(color[2]),
                alpha,
            )
            context.new_path()
            context.move_to(float(triangle_points[0, 0]), float(triangle_points[0, 1]))
            context.line_to(float(triangle_points[1, 0]), float(triangle_points[1, 1]))
            context.line_to(float(triangle_points[2, 0]), float(triangle_points[2, 1]))
            context.close_path()
            context.fill_preserve()
    context.set_anti_alias(True)
