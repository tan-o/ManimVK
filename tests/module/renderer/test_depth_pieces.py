"""Depth-tested mobjects and mesh surfaces in a ThreeDScene.

The camera orders depth pieces (subpaths of depth-tested VMobjects, triangles
of mesh surfaces) back to front in depth slices; the Skia drawing context sends
all VMobject runs between two mesh runs in one native call.  The native module
is a recorder here, so these check calls, not pixels.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from PIL import Image

from manim import (
    DEGREES,
    MeshSurface,
    Sphere,
    Square,
    TexturedSurface,
    ThreeDScene,
    VMobject,
    tempconfig,
)
from manim.camera import ThreeDCamera
from manim.renderer.skia import renderer as skia_renderer


def _hairs(count: int, radius: float = 1.0) -> VMobject:
    """``count`` short radial lines on a sphere, one subpath each."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    start, end = 1.02 * radius * directions, 1.2 * radius * directions
    delta = end - start
    points = np.stack([start, start + delta / 3, start + 2 * delta / 3, end], axis=1)
    hairs = VMobject()
    hairs.set_points(points.reshape(-1, 3))
    return hairs.apply_depth_test()


def test_depth_pieces_are_subpath_centers():
    hairs = _hairs(10)
    centers = hairs.get_depth_pieces()
    assert centers.shape == (10, 3)
    np.testing.assert_allclose(centers, hairs.points.reshape(10, 4, 3).mean(axis=1))
    assert VMobject().get_depth_pieces() is None
    assert _hairs(10).deactivate_depth_test().get_depth_pieces() is None


def test_draw_order_runs_far_to_near_and_flat_last():
    camera = ThreeDCamera(phi=0, theta=-90 * DEGREES)  # looking down the z axis
    hairs = _hairs(200)
    flat = Square()
    order = camera.get_draw_order([flat, hairs], {id(flat): 0, id(hairs): 1})
    runs = list(order.runs())
    assert runs[-1][0] is flat
    assert runs[-1][1] is None
    pieces = np.concatenate([p for m, p in runs if m is hairs])
    assert sorted(pieces.tolist()) == list(range(200))
    # Runs come far to near: the z of their pieces increases.
    run_depth = [
        hairs.get_depth_pieces()[p][:, 2].mean() for m, p in runs if m is hairs
    ]
    assert run_depth == sorted(run_depth)
    assert len(run_depth) <= camera.depth_slices


def test_without_pieces_draw_order_keeps_display_order():
    camera = ThreeDCamera()
    mobjects = [Square(), Square().shift(2 * np.array([1.0, 0, 0]))]
    assert list(camera.get_draw_order(mobjects).runs()) == [
        (mobjects[0], None),
        (mobjects[1], None),
    ]


def test_mesh_surface_follows_the_transformed_surface():
    sphere = Sphere(radius=2, resolution=(8, 4)).shift(np.array([1.0, 0, 0]))
    mesh = MeshSurface(sphere)
    assert mesh.resolution == (8, 4)
    assert len(mesh.points) == 9 * 5
    assert mesh.triangles.shape == (2 * 8 * 4, 3)
    np.testing.assert_allclose(
        np.linalg.norm(mesh.points - [1, 0, 0], axis=1), 2, atol=1e-9
    )
    assert mesh.get_depth_pieces().shape == (64, 3, 3)


def test_mesh_layers_blend_base_highlight_and_shadow():
    mesh = MeshSurface(Sphere(resolution=(6, 3)), shading=(0.3, 0.2, 0.4))
    positions, uvs, colors = mesh.get_shaded_mesh(ThreeDCamera(), np.array([0, 5]))
    # Two triangles, three layers each, three vertices per layer.
    assert (
        positions.shape == (18, 3)
    )
    assert (
        uvs.shape == (18, 2)
    )
    assert (
        colors.shape == (18, 4)
    )
    assert np.all(colors[:3, 3] == 1)  # the base colour is opaque
    np.testing.assert_allclose(colors[:3, :3], np.tile(mesh.color.to_rgb(), (3, 1)))
    unshaded = MeshSurface(Sphere(resolution=(6, 3)), shading=(0, 0, 0))
    assert unshaded.get_shaded_mesh(ThreeDCamera())[0].shape == (2 * 6 * 3 * 3, 3)


def test_textured_surface_atlas_holds_day_night_and_white(tmp_path):
    day = tmp_path / "day.png"
    night = tmp_path / "night.png"
    Image.new("RGB", (8, 4), (10, 200, 30)).save(day)
    Image.new("RGB", (16, 8), (5, 5, 50)).save(night)
    earth = TexturedSurface(Sphere(resolution=(6, 3)), day, night)
    atlas, key = earth.get_texture_atlas()
    assert key == earth.get_texture_atlas()[1]
    assert not atlas.flags.writeable
    assert atlas.shape[1] == 8
    assert tuple(atlas[0, 0]) == (10, 200, 30, 255)
    assert tuple(atlas[-1, 0]) == (255, 255, 255, 255)
    _, uvs, colors = earth.get_shaded_mesh(ThreeDCamera())
    # Night, day, highlight and shadow layers.
    assert len(uvs) == 2 * 6 * 3 * 3 * 4
    earth.fade(1)
    assert earth.get_opacity() == 0


class _Target:
    def __init__(self, calls: Counter, width: int, height: int) -> None:
        self.calls, self.width, self.height = calls, width, height
        self.paths: list[int] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls[name] += 1

        return record

    def draw_vmobjects(self, objects, runs, parts):
        self.calls["draw_vmobjects"] += 1
        assert runs.dtype == np.int64
        assert runs.shape[1] == 3
        assert parts.dtype == np.int64
        for index, start, end in runs.tolist():
            self.calls["subpaths"] += (
                end - start if end > start else len(objects[index][1])
            )

    def draw_textured_mesh(
        self, key, texture, width, height, positions, uvs, colors, alpha, occlude
    ):
        self.calls["draw_textured_mesh"] += 1
        self.calls["mesh_vertices"] += len(positions)
        assert positions.dtype == np.float32
        assert positions.shape[1] == 2
        assert len(positions) == len(uvs) == len(colors)
        if occlude is not None:
            self.calls["occlude"] += 1

    def restore_static_layer(self):
        return False

    def commit_into(self, frame):
        frame[...] = 0


class _Device:
    calls: Counter = Counter()

    def __init__(self, *args, **kwargs) -> None:
        pass

    def device_name(self) -> str:
        return "recorder"

    def create_skia_target(self, width, height):
        return _Target(self.calls, width, height)


@pytest.fixture
def calls(monkeypatch):
    _Device.calls = Counter()
    monkeypatch.setattr(skia_renderer.manim_vk, "Device", _Device)
    return _Device.calls


class _HairyBall(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)
        self.add(MeshSurface(Sphere(resolution=(12, 6)), opacity=0.5), _hairs(3000))


def test_hairs_and_mesh_draw_in_few_runs(calls, tmp_path):
    with tempconfig(
        {
            "pixel_width": 160,
            "pixel_height": 90,
            "media_dir": str(tmp_path),
            "save_last_frame": True,
            "write_to_movie": False,
            "disable_caching": True,
        }
    ):
        _HairyBall().render()
    # Every hair and every triangle is drawn once ...
    assert calls["subpaths"] == 3000
    assert calls["mesh_vertices"] == 2 * 12 * 6 * 3 * 3
    # ... in a few dozen calls, not one per piece.
    assert calls["draw_vmobjects"] <= ThreeDCamera.depth_slices
    assert calls["draw_textured_mesh"] <= ThreeDCamera.depth_slices
    # The translucent sphere hides what is behind it.
    assert calls["occlude"] == calls["draw_textured_mesh"]
