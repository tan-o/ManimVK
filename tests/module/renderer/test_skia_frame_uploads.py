"""Skia frames start from GPU-resident layers instead of uploading pixels.

Runs without a GPU: the native module is replaced by a recorder, so these tests
check the renderer's calls (one display-list call per frame, one static capture per
play, no per-frame pixel uploads), not pixels.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from manim import (
    Circle,
    Scene,
    Square,
    ValueTracker,
    Vector,
    VGroup,
    tempconfig,
)
from manim.renderer.skia import renderer as skia_renderer
from manim.renderer.skia.rendering import SkiaDrawingContext


class _RecordingTarget:
    def __init__(self, calls: Counter, width: int, height: int) -> None:
        self.calls, self.width, self.height = calls, width, height
        self.static = False

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls[name] += 1

        return record

    def capture_static_layer(self):
        self.calls["capture_static_layer"] += 1
        self.static = True

    def clear_static_layer(self):
        self.static = False

    def restore_static_layer(self):
        self.calls["restore_static_layer"] += 1
        return self.static

    def commit_into(self, frame):
        self.calls["commit_into"] += 1
        frame[...] = 0

    def request_yuv420(self):
        self.calls["request_yuv420"] += 1
        return self.calls["request_yuv420"]

    def collect_yuv420_into(self, ticket, frame):
        frame[...] = 0


class _RecordingDevice:
    calls: Counter = Counter()

    def __init__(self, *args, **kwargs) -> None:
        pass

    def device_name(self) -> str:
        return "recorder"

    def create_skia_target(self, width, height):
        return _RecordingTarget(self.calls, width, height)


@pytest.fixture
def calls(monkeypatch):
    _RecordingDevice.calls = Counter()
    monkeypatch.setattr(skia_renderer.manim_vk, "Device", _RecordingDevice)
    return _RecordingDevice.calls


class _Moving(Scene):
    def construct(self):
        self.add(
            VGroup(*[Square().shift(i * 0.5 * np.array([1.0, 0, 0])) for i in range(4)])
        )
        tracker = ValueTracker(0)
        vectors = [Vector([1, 0, 0]) for _ in range(5)]
        for index, vector in enumerate(vectors):
            vector.add_updater(
                lambda v, index=index: v.set_angle(tracker.get_value() + index)
            )
        self.add(*vectors)
        self.play(tracker.animate.set_value(1), run_time=1)
        self.play(tracker.animate.set_value(2), run_time=1)


def test_static_layer_is_captured_once_per_play_and_never_uploaded(calls, tmp_path):
    with tempconfig(
        {
            "frame_rate": 10,
            "pixel_width": 160,
            "pixel_height": 90,
            "media_dir": str(tmp_path),
            "disable_caching": True,
        }
    ):
        _Moving().render()
    # Movie frames are read as YUV on the GPU, not as RGBA.
    frames = calls["request_yuv420"]
    assert frames >= 20
    assert calls["commit_into"] == 0
    assert calls["capture_static_layer"] == 2
    assert calls["load_pixels"] == 0
    assert calls["restore_static_layer"] >= frames
    # All VMobjects of a frame go in one native call, never per point.
    assert calls["draw_display_list"] == calls["restore_static_layer"]
    assert calls["move_to"] == calls["curve_to"] == calls["set_cubic_path"] == 0


class _OpRecorder:
    def __init__(self) -> None:
        self.ops: list[tuple] = []

    def new_path(self):
        self.ops.clear()

    def new_sub_path(self):
        pass

    def move_to(self, x, y):
        self.ops.append(("M", x, y))

    def curve_to(self, *coords):
        self.ops.append(("C", *coords))

    def close_path(self):
        self.ops.append(("Z",))

    def set_cubic_path(self, points, subpaths):
        self.ops.clear()
        for start, end, closed in subpaths:
            self.ops.append(("M", points[start, 0], points[start, 1]))
            for index in range(start, end - 3, 4):
                handles = points[index + 1 : index + 4, :2].ravel()
                self.ops.append(("C", *handles))
            if closed:
                self.ops.append(("Z",))


def _reference_path_ops(recorder: _OpRecorder, camera, vmobject) -> None:
    """The path a VMobject describes, segment by segment: the loop the Cairo
    renderer drew with, kept as the reference for manim_core.cubic_subpaths.
    """
    recorder.new_path()
    points = camera.transform_points_pre_display(vmobject, vmobject.points)
    if len(points) == 0:
        return
    nppcc = vmobject.n_points_per_cubic_curve
    for start, end in vmobject.get_subpath_split_indices_from_points(points, n_dims=2):
        start, end = int(start), int(end)
        if end - start < nppcc:
            continue
        recorder.new_sub_path()
        recorder.move_to(*points[start, :2])
        for index in range(start, end - nppcc + 1, nppcc):
            recorder.curve_to(*points[index + 1 : index + 4, :2].ravel())
        if vmobject.consider_points_equals_2d(points[start], points[end - 1]):
            recorder.close_path()


@pytest.mark.parametrize(
    "mobject",
    [
        Circle(),
        Square(),
        Vector([1, 2, 0]),
        Circle().pointwise_become_partial(Circle(), 0.1, 0.6),
    ],
)
def test_one_path_call_describes_the_reference_path(mobject):
    from manim.camera import Camera

    camera = Camera()
    for vmobject in mobject.family_members_with_points():
        reference_ops, skia_ops = _OpRecorder(), _OpRecorder()
        skia_context = object.__new__(SkiaDrawingContext)
        skia_context.camera = camera
        _reference_path_ops(reference_ops, camera, vmobject)
        SkiaDrawingContext._set_path(skia_context, skia_ops, vmobject)
        assert len(reference_ops.ops) == len(skia_ops.ops)
        for a, b in zip(reference_ops.ops, skia_ops.ops, strict=True):
            assert a[0] == b[0]
            np.testing.assert_allclose(a[1:], b[1:])


class _ImageCalls(_RecordingTarget):
    draws: list[tuple] = []

    def draw_image_quad(
        self, key, pixels, width, height, quad, alpha, sampling, mix=None
    ):
        self.draws.append(
            (key, id(pixels), alpha, None if mix is None else (mix[0], mix[2]))
        )


class _ImageDevice(_RecordingDevice):
    def create_skia_target(self, width, height):
        return _ImageCalls(self.calls, width, height)


class _ImageFades(Scene):
    def construct(self):
        from manim import FadeIn, ImageMobject, Transform

        rng = np.random.default_rng(0)
        first = ImageMobject(rng.integers(0, 255, (8, 8, 4), dtype=np.uint8))
        second = ImageMobject(rng.integers(0, 255, (8, 8, 4), dtype=np.uint8))
        self.play(FadeIn(first), run_time=1)
        self.play(Transform(first, second), run_time=1)
        self.keys = first.texture_key, second.texture_key


def test_image_animations_reuse_their_textures(monkeypatch, tmp_path):
    _ImageCalls.draws = []
    monkeypatch.setattr(skia_renderer.manim_vk, "Device", _ImageDevice)
    with tempconfig(
        {
            "frame_rate": 10,
            "pixel_width": 160,
            "pixel_height": 90,
            "media_dir": str(tmp_path),
            "disable_caching": True,
        }
    ):
        scene = _ImageFades()
        scene.render()
    draws = _ImageCalls.draws
    # One source, under one key, is drawn for the whole fade: only the
    # opacity changes.  The transform draws the two sources with a weight.
    fade = [d for d in draws if d[3] is None and d[2] < 1]
    assert len(fade) >= 5
    assert len({key for key, _, _, _ in fade}) == 1
    assert [d[2] for d in fade] == sorted(d[2] for d in fade)
    mixes = [d for d in draws if d[3] is not None]
    assert len(mixes) >= 5
    first_key, second_key = mixes[0][0], mixes[0][3][0]
    assert first_key != second_key
    assert {d[0] for d in mixes} == {first_key}
    assert {d[3][0] for d in mixes} == {second_key}
    weights = [d[3][1] for d in mixes]
    assert weights == sorted(weights)
    assert weights[0] > 0
    assert weights[-1] < 1
