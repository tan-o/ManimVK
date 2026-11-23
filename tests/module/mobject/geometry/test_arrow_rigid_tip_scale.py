"""Arrow.scale's rigid-tip fast path must match the general pop/add-tip path."""

from __future__ import annotations

import numpy as np
import pytest

from manim import Arrow, Vector
from manim.mobject.geometry.line import Arrow as ArrowClass


def _general_scale(arrow: Arrow, factor: float) -> None:
    original = ArrowClass._scale_with_rigid_tip
    ArrowClass._scale_with_rigid_tip = lambda *args: False
    try:
        ArrowClass.scale(arrow, factor)
    finally:
        ArrowClass._scale_with_rigid_tip = original


@pytest.mark.parametrize("seed", range(5))
def test_rigid_tip_scale_matches_general_path_over_many_frames(seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        kwargs = {
            "buff": 0,
            "stroke_width": float(rng.uniform(1, 8)),
            "tip_length": float(rng.uniform(0.05, 0.4)),
            "max_tip_length_to_length_ratio": float(rng.uniform(0.1, 0.4)),
        }
        start = np.r_[rng.normal(size=2), 0.0]
        end = np.r_[rng.normal(size=2) * 2, 0.0]
        fast, general = Arrow(start, end, **kwargs), Arrow(start, end, **kwargs)
        for _ in range(15):
            length = float(np.exp(rng.normal()) * 0.5)
            angle = float(rng.uniform(-np.pi, np.pi))
            shift = np.r_[rng.normal(size=2), 0.0]
            fast.set_length(length)
            _general_scale(general, length / general.get_length())
            for arrow in (fast, general):
                arrow.set_angle(angle)
                arrow.shift(shift)
        np.testing.assert_allclose(fast.points, general.points, atol=1e-9)
        np.testing.assert_allclose(fast.tip.points, general.tip.points, atol=1e-9)
        assert fast.stroke_width == pytest.approx(general.stroke_width, abs=1e-9)


def test_rigid_tip_scale_falls_back_for_negative_factor_and_about_point():
    for call in (
        lambda arrow: arrow.scale(-0.5),
        lambda arrow: arrow.scale(0.5, about_point=np.array([1.0, 1.0, 0.0])),
    ):
        fast, general = Vector([2, 1, 0]), Vector([2, 1, 0])
        call(fast)
        original = ArrowClass._scale_with_rigid_tip
        ArrowClass._scale_with_rigid_tip = lambda *args: False
        try:
            call(general)
        finally:
            ArrowClass._scale_with_rigid_tip = original
        np.testing.assert_allclose(fast.points, general.points)
        np.testing.assert_allclose(fast.tip.points, general.tip.points)
