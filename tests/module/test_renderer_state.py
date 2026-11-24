"""Every renderer reports where it is in its life through a read-only state."""

from unittest.mock import Mock

import pytest

from manim import Scene, Square, tempconfig
from manim.renderer.protocol import Renderer, RendererState
from manim.renderer.skia import SkiaRenderer


def test_the_renderer_implements_the_protocol():
    renderer = SkiaRenderer()
    try:
        assert isinstance(renderer, Renderer)
        assert renderer.state is RendererState.UNOPENED
        with pytest.raises(AttributeError):
            renderer.state = RendererState.OPEN
    finally:
        renderer.close()
    assert renderer.state is RendererState.CLOSED


SETTINGS = {"pixel_width": 16, "pixel_height": 8, "format": "none"}


def test_a_renderer_goes_from_unopened_to_open_to_closed():
    with tempconfig(SETTINGS):
        renderer = Scene().renderer
        assert renderer.state is RendererState.UNOPENED
        renderer.open()
        assert renderer.state is RendererState.OPEN
        renderer.close()
        assert renderer.state is RendererState.CLOSED
        assert not renderer.state.accepts_work
        with pytest.raises(RuntimeError, match="closed"):
            renderer.open()


def test_a_failed_close_can_be_retried():
    with tempconfig(SETTINGS):
        renderer = Scene().renderer
        renderer.open()
        renderer.render_mobjects([Square()])
        assert renderer._target is not None
        # Releasing fails and leaves the Skia target alive: close() keeps it
        # around and can be called again.
        target = renderer._target
        target_close = target.close
        target.close = Mock(side_effect=[OSError("release failed"), None])
        with pytest.raises(OSError, match="release failed"):
            renderer.close()
        assert renderer.state is RendererState.CLOSE_FAILED
        with pytest.raises(RuntimeError, match="close failed"):
            renderer.open()
        target.close = target_close
        renderer.close()
        assert renderer.state is RendererState.CLOSED
        assert target._closed


def test_an_interrupted_close_can_be_retried():
    with tempconfig(SETTINGS):
        renderer = SkiaRenderer()
        renderer.render_mobjects([Square()])
        # The first close of the frame target is interrupted.
        target = renderer._target
        target_close = target.close
        target.close = Mock(side_effect=[KeyboardInterrupt, None])
        with pytest.raises(KeyboardInterrupt):
            renderer.close()
        assert renderer.state is RendererState.CLOSE_FAILED
        with pytest.raises(RuntimeError, match="close failed"):
            renderer.open()
        assert not target._closed
        target.close = target_close
        renderer.close()
        assert renderer.state is RendererState.CLOSED
        assert target._closed


def test_a_close_raises_the_first_failure_and_logs_the_others(manim_caplog):
    with tempconfig(SETTINGS):
        renderer = SkiaRenderer()
        renderer.render_mobjects([Square()])
        renderer._target.close = Mock(side_effect=OSError("frame target"))
        renderer._sub_targets["view"] = Mock(close=Mock(side_effect=OSError("view")))
        with pytest.raises(OSError, match="frame target"):
            renderer.close()
    assert renderer.state is RendererState.CLOSE_FAILED
    assert "Additional renderer cleanup failure" in manim_caplog.text
    assert "view" in renderer._sub_targets


def test_renderer_cairo_in_a_config_draws_with_vulkan():
    # Manim Community scripts and configs name the Cairo renderer.
    with tempconfig({**SETTINGS, "renderer": "cairo"}):
        assert isinstance(Scene().renderer, SkiaRenderer)
