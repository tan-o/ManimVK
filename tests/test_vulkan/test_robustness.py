"""Robustness checks for the Vulkan backend's failure paths."""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

pytest.importorskip("manim_vk")

import manim_vk  # noqa: E402

from manim import tempconfig  # noqa: E402
from manim.renderer.skia.renderer import SkiaRenderer  # noqa: E402


def test_missing_vulkan_device_has_actionable_error(monkeypatch):
    monkeypatch.setattr(
        manim_vk,
        "Device",
        Mock(side_effect=RuntimeError("no Vulkan physical device found")),
    )
    with tempconfig(
        {
            "renderer": "vulkan",
            "pixel_width": 32,
            "pixel_height": 18,
            "format": "none",
            "dry_run": True,
        }
    ):
        renderer = SkiaRenderer()
        try:
            with pytest.raises(RuntimeError, match="Install a working Vulkan driver"):
                renderer._ensure_device()
        finally:
            renderer.close()


def test_skia_target_rejects_cross_thread_access():
    with tempconfig(
        {
            "renderer": "vulkan",
            "pixel_width": 32,
            "pixel_height": 18,
            "format": "none",
            "dry_run": True,
        }
    ):
        renderer = SkiaRenderer()
        try:
            target = renderer._get_target()
            error: list[BaseException] = []

            def use_target():
                try:
                    target._ensure_open()
                except BaseException as caught:  # pragma: no cover - asserted below
                    error.append(caught)

            thread = threading.Thread(target=use_target)
            thread.start()
            thread.join()
            assert len(error) == 1
            assert isinstance(error[0], RuntimeError)
            assert "bound to the thread" in str(error[0])
        finally:
            renderer.close()
