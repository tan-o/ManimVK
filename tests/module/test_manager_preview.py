"""Interactive redraws display frames without advancing animation time."""

from unittest.mock import Mock

import manim_vk
import pytest

from manim import Scene, tempconfig


@pytest.fixture
def scene(monkeypatch):
    # winit's event loop can only be created once per process; the unit tests
    # exercise the manager/embed contract, so use a fake native window.
    fake_window = Mock()
    fake_window.is_closing.return_value = False
    fake_window.error.return_value = None
    fake_window.adapter_name.return_value = ""
    monkeypatch.setattr(manim_vk, "NativeWindow", Mock(return_value=fake_window))
    with tempconfig({"renderer": "opengl", "format": "none", "live_preview": True}):
        scene = Scene()
        with scene._get_manager():
            yield scene


def test_preview_presents_without_advancing_time(scene, monkeypatch):
    manager = scene.manager
    manager.time = 2
    draw = Mock()
    monkeypatch.setattr(scene.renderer, "render", draw)
    window = Mock()
    monkeypatch.setattr(scene.renderer, "window", window)
    manager._render_preview_frame(-1)
    draw.assert_called_once_with(scene, -1, scene.moving_mobjects)
    window.swap_buffers.assert_called_once_with()
    assert scene.time == 2


def test_embed_redraws_through_manager_before_and_after_commands(scene, monkeypatch):
    redraw = Mock()
    monkeypatch.setattr(scene.manager, "_render_preview_frame", redraw)
    shell = Mock()
    monkeypatch.setattr("IPython.terminal.embed.InteractiveShellEmbed", lambda: shell)
    with pytest.raises(Exception, match="Exiting scene"):
        scene.embed()
    redraw.assert_called_once_with(-1)
    event, callback = shell.events.register.call_args.args
    assert event == "post_run_cell"
    callback()
    assert redraw.call_count == 2
    assert scene.time == 0


def test_interactive_idle_redraw_uses_manager(scene, monkeypatch):
    monkeypatch.setattr("manim.scene.scene.Observer", Mock())

    def redraw(_):
        scene.quit_interaction = True

    redraw = Mock(side_effect=redraw)
    monkeypatch.setattr(scene.manager, "_render_preview_frame", redraw)
    scene.interact(Mock(pt_app=None), Mock())
    redraw.assert_called_once()
    assert scene.time == 0
