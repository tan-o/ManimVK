"""Mesh surfaces hide each other per pixel, as a depth buffer would.

The frames are checked against ray casting: for every pixel, the ray from
the camera through it is intersected with the exact spheres, and the nearest
one says which colour the pixel must show.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("manim_vk")

from manim import (  # noqa: E402
    BLUE,
    DEGREES,
    OUT,
    RED,
    WHITE,
    MeshSurface,
    Sphere,
    VMobject,
    tempconfig,
)
from manim.camera import ThreeDCamera  # noqa: E402
from manim.renderer.skia.renderer import SkiaRenderer  # noqa: E402

WIDTH, HEIGHT = 480, 270


def _render(camera, mobjects, culling=True, stats=None):
    with tempconfig({"pixel_width": WIDTH, "pixel_height": HEIGHT}):
        renderer = SkiaRenderer(camera=camera)
        try:
            renderer.set_occlusion_culling(culling)
            renderer.render_mobjects(mobjects)
            if stats is not None:
                stats.append(renderer.culling_stats())
            return renderer.get_frame().astype(np.float64) / 255
        finally:
            renderer.close()


def _flat_sphere(center, radius, color, opacity=1.0, rotation=0.0):
    sphere = MeshSurface(
        Sphere(radius=radius, resolution=(96, 48)),
        color=color,
        opacity=opacity,
        shading=(0, 0, 0),
    )
    sphere.rotate(rotation, OUT, about_point=np.zeros(3))
    return sphere.shift(center)


def _cast(camera, spheres):
    """Per pixel: the index of the nearest sphere hit (-1 for none) and the
    pixels to leave out, where polygons and spheres may disagree.
    """
    with tempconfig({"pixel_width": WIDTH, "pixel_height": HEIGHT}):
        frame_width, frame_height = camera.frame_width, camera.frame_height
    rotation = camera.get_rotation_matrix()
    focal, zoom = camera.get_focal_distance(), camera.get_zoom()
    center = np.asarray(camera.frame_center, dtype=float)
    xs = ((np.arange(WIDTH) + 0.5) / WIDTH - 0.5) * frame_width
    ys = (0.5 - (np.arange(HEIGHT) + 0.5) / HEIGHT) * frame_height
    screen_x, screen_y = np.meshgrid(xs, ys)
    # In the camera's rotated frame the eye is at (0, 0, focal) and the ray
    # through screen point (X, Y) meets z = 0 at (X, Y) / zoom.
    directions = np.stack(
        [screen_x / zoom, screen_y / zoom, -focal * np.ones_like(screen_x)], axis=-1
    )
    eye = np.array([0.0, 0.0, focal])
    hits = []
    for sphere_center, radius in spheres:
        c = rotation @ (np.asarray(sphere_center, dtype=float) - center)
        oc = eye - c
        a = np.sum(directions**2, axis=-1)
        b = 2 * np.sum(directions * oc, axis=-1)
        disc = b**2 - 4 * a * (oc @ oc - radius**2)
        t = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
        # Close to the silhouette the polygon and the sphere differ: leave
        # out rays passing within 3% of the radius of it.
        miss = np.linalg.norm(np.cross(oc, directions), axis=-1) / np.sqrt(a)
        edge = np.abs(miss - radius) < 0.03 * radius
        hits.append((t, edge))
    ts = np.stack([t for t, _ in hits])
    order = np.sort(ts, axis=0)
    nearest = np.where(np.isfinite(order[0]), np.argmin(ts, axis=0), -1)
    # The distance between the nearest two surfaces along the ray, in scene
    # units: where it is tiny, the polygons may disagree with the spheres.
    length = np.linalg.norm(directions, axis=-1)
    gap = (
        (order[1] - order[0]) * length
        if len(ts) > 1
        else np.full(nearest.shape, np.inf)
    )
    ambiguous = np.any(np.stack([edge for _, edge in hits]), axis=0) | (gap < 0.02)
    return nearest, ambiguous


def _camera():
    return ThreeDCamera(phi=30 * DEGREES, theta=-80 * DEGREES)


def test_intersecting_opaque_spheres_hide_each_other_per_pixel():
    camera = _camera()
    spheres = [((-1.0, 0, 0), 2.2, RED), ((1.2, 0.3, 0), 2.0, BLUE)]
    frame = _render(camera, [_flat_sphere(c, r, color) for c, r, color in spheres])
    nearest, ambiguous = _cast(camera, [(c, r) for c, r, _ in spheres])
    red = (frame[..., 0] > 0.5) & (frame[..., 2] < 0.5)
    blue = (frame[..., 2] > 0.5) & (frame[..., 0] < 0.5)
    shown = np.where(red, 0, np.where(blue, 1, -1))
    wrong = (shown != nearest) & ~ambiguous
    # Away from silhouettes and from where the spheres are nearly as far, the
    # 96 x 48 polygons and the spheres agree; ordering triangles instead of
    # testing depth gets dozens of pixels along the intersection wrong.
    assert wrong.sum() <= 10, f"{wrong.sum()} pixels show the wrong sphere"
    assert (~ambiguous & (nearest == 0)).sum() > 3000
    assert (~ambiguous & (nearest == 1)).sum() > 3000


def test_translucent_surface_inside_an_opaque_one_is_hidden():
    # The Earth over a translucent sphere: 0.1% apart, triangles not aligned.
    camera = _camera()
    outer = _flat_sphere((0, 0, 0), 3.003, RED, rotation=-90 * DEGREES)
    inner = _flat_sphere((0, 0, 0), 3.0, BLUE, opacity=0.3)
    alone = _render(
        camera, [_flat_sphere((0, 0, 0), 3.003, RED, rotation=-90 * DEGREES)]
    )
    both = _render(camera, [inner, outer])
    np.testing.assert_array_equal(both, alone)


def test_opaque_surface_inside_a_translucent_one_is_drawn_through_it():
    camera = _camera()
    outer = _flat_sphere((0, 0, 0), 2.0, BLUE, opacity=0.5)
    inner = _flat_sphere((0, 0, 0), 1.0, RED)
    frame = _render(camera, [inner, outer])
    nearest, ambiguous = _cast(camera, [((0, 0, 0), 1.0)])
    inside = (nearest == 0) & ~ambiguous
    # Red under half-transparent blue, never the background.
    assert np.all(frame[inside, 0] > 0.3)
    assert np.all(frame[inside, 2] > 0.3)


def _pixels(camera, points):
    """Device pixel (column, row) of scene points, and whether each lies on
    the side of a sphere about the origin that faces the camera, with the
    cosine between the outward normal and the direction to the eye."""
    with tempconfig({"pixel_width": WIDTH, "pixel_height": HEIGHT}):
        frame_width, frame_height = camera.frame_width, camera.frame_height
    projected = camera.project_points(points)
    column = ((projected[:, 0] / frame_width + 0.5) * WIDTH).astype(int)
    row = ((0.5 - projected[:, 1] / frame_height) * HEIGHT).astype(int)
    rotation = camera.get_rotation_matrix()
    center = np.asarray(camera.frame_center, dtype=float)
    eye = center + rotation.T @ np.array([0.0, 0.0, camera.get_focal_distance()])
    normal = points / np.linalg.norm(points, axis=1, keepdims=True)
    to_eye = eye - points
    facing = np.sum(normal * to_eye, axis=1) / np.linalg.norm(to_eye, axis=1)
    return column, row, facing


def test_a_line_on_a_translucent_sphere_shows_on_its_near_side_only():
    # ManimGL's SurfaceMesh lines: on the sphere, nudged out by 0.01.  One
    # long line, the equator, passes in front of the sphere and behind it.
    camera = ThreeDCamera(phi=60 * DEGREES, theta=-80 * DEGREES)
    t = np.linspace(0, 2 * np.pi, 129)
    equator = 2.01 * np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    line = VMobject(stroke_color=WHITE, stroke_width=6, fill_opacity=0)
    line.set_points_as_corners(equator).apply_depth_test()
    sphere = _flat_sphere((0, 0, 0), 2.0, BLUE, opacity=0.5)
    alone = _render(camera, [sphere])
    both = _render(camera, [sphere, line])
    column, row, facing = _pixels(camera, equator)
    inside = (column >= 1) & (column < WIDTH - 1) & (row >= 1) & (row < HEIGHT - 1)
    difference = np.abs(both - alone).max(axis=-1)
    # The stroke is a few pixels wide and drawn at the depth of its centre
    # line, as ManimGL draws it: where the surface rises towards the camera
    # beside the line, the stroke's edge on that side is hidden.  Look at
    # the 3 x 3 pixels around each sample.
    around = np.max(
        [
            difference[row[inside] + dy, column[inside] + dx]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ],
        axis=0,
    )
    changed = around > 0.1
    # Close to the silhouette the near and far halves are a few pixels
    # apart on screen; leave out samples facing within 0.3 of sideways.
    near, far = facing[inside] > 0.3, facing[inside] < -0.3
    assert near.sum() > 20 and far.sum() > 20
    # Over the whole near side the line is drawn; behind the sphere, nowhere.
    assert changed[near].all(), f"{(~changed[near]).sum()} near samples missing"
    assert not changed[far].any(), f"{changed[far].sum()} far samples drawn"


def test_a_translucent_sphere_blends_its_near_side_once():
    camera = _camera()
    sphere = _flat_sphere((0, 0, 0), 2.0, BLUE, opacity=0.5)
    frame = _render(camera, [sphere])
    nearest, ambiguous = _cast(camera, [((0, 0, 0), 2.0)])
    inside = (nearest == 0) & ~ambiguous
    # Half of BLUE over the black background, not a quarter more for the
    # far side seen through the near one.
    blue = np.array(BLUE.to_rgb())
    error = np.abs(frame[inside, :3] - 0.5 * blue).max()
    assert error <= 2 / 255, f"{error * 255:.1f} levels off half of BLUE"


def test_a_fully_transparent_surface_hides_nothing():
    # HairyEarth adds the Earth at opacity 0 and fades it in.
    camera = ThreeDCamera(phi=60 * DEGREES, theta=-80 * DEGREES)
    t = np.linspace(0, 2 * np.pi, 129)
    line = VMobject(stroke_color=WHITE, stroke_width=6, fill_opacity=0)
    line.set_points_as_corners(
        1.5 * np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    ).apply_depth_test()
    hidden = _flat_sphere((0, 0, 0), 2.0, BLUE, opacity=0.0)
    np.testing.assert_array_equal(_render(camera, [hidden, line]), _render(camera, [line]))


def test_occlusion_culling_leaves_the_frame_unchanged():
    # Hairs all over a translucent sphere and an opaque one inside it, as in
    # HairyEarth: those behind are culled before drawing, not clipped after.
    camera = ThreeDCamera(phi=60 * DEGREES, theta=-80 * DEGREES)
    rng = np.random.default_rng(3)
    start = rng.normal(size=(4000, 3))
    start *= 2.02 / np.linalg.norm(start, axis=1, keepdims=True)
    end = start * 1.08
    hairs = VMobject(stroke_color=WHITE, stroke_width=2, fill_opacity=0)
    hairs.set_points(
        np.stack([start, (2 * start + end) / 3, (start + 2 * end) / 3, end], axis=1).reshape(-1, 3)
    )
    hairs.apply_depth_test()
    mobjects = [
        _flat_sphere((0, 0, 0), 2.0, BLUE, opacity=0.4),
        _flat_sphere((0.5, 0, 0), 1.2, RED),
        hairs,
    ]
    stats = []
    culled = _render(camera, mobjects, culling=True, stats=stats)
    np.testing.assert_array_equal(culled, _render(camera, mobjects, culling=False))
    tested, hidden = stats[0]
    # About a third of the hairs lie wholly behind the near side; the rest
    # are in front, on the rim, or reach past it.
    if tested:  # the native module (the CPU stand-in does not cull)
        assert hidden > tested / 8, f"only {hidden} of {tested} chains culled"
