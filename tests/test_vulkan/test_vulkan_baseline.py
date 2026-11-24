"""Vulkan/Skia regression baselines.

These tests render a small set of representative scenes through the Vulkan
backend and compare the resulting PNG against a Vulkan-specific baseline.
They intentionally skip when ``manim_vk`` or a Vulkan device is unavailable,
so the Cairo suite still runs on machines without a GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("manim_vk")

from manim import (  # noqa: E402
    BLUE,
    GREEN,
    RED,
    Circle,
    ImageMobject,
    PointCloudDot,
    Square,
    Surface,
    Text,
    ThreeDScene,
    Scene,
    tempconfig,
)

BASELINE_DIR = Path(__file__).parent / "baselines"
WRITE_BASELINES = os.environ.get("MANIM_VULKAN_BASELINE") == "1"


def _vulkan_config(media_dir: Path) -> dict:
    return {
        "pixel_width": 320,
        "pixel_height": 180,
        "frame_rate": 10,
        "format": "png",
        "media_dir": str(media_dir),
        "disable_caching": True,
        "dry_run": False,
        "verbosity": "ERROR",
    }


def _render_once(scene_cls: type[Scene], media_dir: Path) -> np.ndarray:
    with tempconfig(_vulkan_config(media_dir)):
        scene = scene_cls()
        scene.render()
    images = sorted(media_dir.glob("images/**/*.png"))
    assert images, f"Vulkan renderer produced no PNG under {media_dir}"
    return np.asarray(Image.open(images[-1]).convert("RGBA"))


def _assert_baseline(name: str, frame: np.ndarray) -> None:
    expected_path = BASELINE_DIR / f"{name}.png"
    if WRITE_BASELINES:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame, "RGBA").save(expected_path)
        return
    assert expected_path.exists(), (
        f"missing Vulkan baseline {expected_path}; regenerate with "
        "MANIM_VULKAN_BASELINE=1"
    )
    expected = np.asarray(Image.open(expected_path).convert("RGBA"))
    assert frame.shape == expected.shape
    diff = np.abs(frame.astype(np.int16) - expected.astype(np.int16))
    mae = float(diff.mean())
    changed = float((diff.max(axis=2) > 2).mean())
    # Skia/Ganesh is deterministic on one machine; allow small driver-level
    # differences but reject real drawing regressions.
    assert mae <= 2.0, f"{name}: MAE={mae:.3f} exceeds 2.0"
    assert changed <= 0.02, f"{name}: {changed:.1%} pixels changed"


class _VectorScene(Scene):
    def construct(self):
        circle = Circle(radius=1.1, color=BLUE, fill_opacity=0.6)
        square = Square(side_length=1.4, color=RED, fill_opacity=0.4)
        square.shift(2.4 * np.array([1, 0, 0]))
        label = Text("Vulkan", font_size=34, color=GREEN)
        label.to_edge(np.array([0, -1, 0]), buff=0.35)
        self.add(circle, square, label)


class _TextScene(Scene):
    def construct(self):
        title = Text("Skia + Vulkan", font_size=40)
        subtitle = Text("one two three", font_size=26)
        subtitle.next_to(title, np.array([0, -1, 0]), buff=0.3)
        self.add(title, subtitle)


class _SurfaceScene(ThreeDScene):
    def construct(self):
        surface = Surface(
            lambda u, v: np.array([u, v, 0.35 * np.sin(3 * u) * np.cos(3 * v)]),
            u_range=(-2, 2),
            v_range=(-2, 2),
            resolution=(18, 18),
            fill_opacity=0.9,
        )
        surface.set_color_by_gradient(BLUE, GREEN, RED)
        self.add(surface)
        self.set_camera_orientation(phi=65 * np.pi / 180, theta=35 * np.pi / 180)


class _ImageScene(Scene):
    def construct(self):
        yy, xx = np.mgrid[0:64, 0:64]
        pixels = np.zeros((64, 64, 4), dtype=np.uint8)
        pixels[..., 0] = (xx * 4).astype(np.uint8)
        pixels[..., 1] = (yy * 4).astype(np.uint8)
        pixels[..., 2] = 180
        pixels[..., 3] = 255
        image = ImageMobject(pixels, scale_to_resolution=180)
        image.scale_to_fit_height(2.2)
        self.add(image)


class _PointCloudScene(Scene):
    def construct(self):
        points = PointCloudDot(radius=1.3, density=18, color=GREEN, stroke_width=3)
        self.add(points)


@pytest.mark.parametrize(
    "name, scene_cls",
    [
        ("vector", _VectorScene),
        ("text", _TextScene),
        ("surface", _SurfaceScene),
        ("image", _ImageScene),
        ("point_cloud", _PointCloudScene),
    ],
)
def test_vulkan_baseline(name, scene_cls, tmp_path):
    try:
        frame = _render_once(scene_cls, tmp_path)
    except RuntimeError as error:
        pytest.skip(f"Vulkan device unavailable: {error}")
    _assert_baseline(name, frame)
