"""The preview session's controls, driven by a scripted window."""

from __future__ import annotations


import pytest

from manim import Circle, Create, FadeOut, Scene, Square, Transform, tempconfig
from manim.preview import PreviewSession, capture_storyboard
from manim.renderer.skia import renderer as skia_renderer


class _Target:
    def __init__(self, width, height):
        self.width, self.height = width, height

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def restore_static_layer(self):
        return False

    def commit_into(self, frame):
        frame[...] = 0


class _Device:
    def __init__(self, *args, **kwargs):
        pass

    def device_name(self):
        return "recorder"

    def create_skia_target(self, width, height):
        return _Target(width, height)


@pytest.fixture(autouse=True)
def native(monkeypatch):
    monkeypatch.setattr(skia_renderer.manim_vk, "Device", _Device)
    with tempconfig({"frame_rate": 10, "pixel_width": 64, "pixel_height": 36}):
        yield


class Shots(Scene):
    def construct(self):
        self.next_section("one")
        circle = Circle()
        self.play(Create(circle), run_time=1)
        self.play(Transform(circle, Square()), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(circle), run_time=0.5)


def test_storyboard_lists_one_shot_per_call():
    board = capture_storyboard(Shots)
    assert [shot.kind for shot in board.shots] == ["play", "play", "wait", "play"]
    assert board.shots[1].start == pytest.approx(1.0)
    assert board.shots[0].section == "one"
    assert board.shots[0].label == "0 Create"
    assert board.shot_at(1.2).index == 1
    assert board.frame_times(2.5) == pytest.approx([2.5, 2.6, 2.7, 2.8, 2.9])


class ScriptedWindow:
    """Feeds ``script[n]`` (a list of events) after the n-th presented frame."""

    def __init__(self, script):
        self.script = script
        self.frames = []
        self.presented = 0
        self.closed = False

    def present(self, rgba):
        assert rgba.shape == (36, 64, 4)
        self.frames.append(rgba)
        self.presented += 1

    def poll_events(self):
        return self.script.pop(self.presented, [])

    def is_closing(self):
        return False

    def close(self):
        self.closed = True


def _run(script, **kwargs):
    window = ScriptedWindow(script)
    session = PreviewSession(
        "shots.py",
        window_factory=lambda w, h, title: window,
        load_scene=lambda file, name: Shots,
        **kwargs,
    )
    session.speed = 8.0
    times = []
    original = session._on_frame

    def record(frame):
        times.append(round(frame.time, 3))
        original(frame)

    session._on_frame = record
    session.run()
    return session, window, times


def key(name):
    return [("key", name, True, False)]


def test_next_shot_jumps_and_quit_closes_the_window():
    session, window, times = _run({3: key("arrowright"), 6: key("q")})
    assert times[:3] == [0.0, 0.1, 0.2]
    assert times[3] == pytest.approx(1.0)  # the next shot starts at 1 s
    assert window.closed


def test_loop_repeats_the_current_shot():
    session, window, times = _run({2: key("l"), 25: key("q")})
    assert session.loop_shot == 0
    assert max(times) < 1.0 and times.count(0.0) >= 2


def test_click_on_bar_seeks():
    session, window, times = _run(
        {2: [("cursor", 48.0, 30.0), ("mouse", "left", True)], 4: key("q")}
    )
    assert times[2] == pytest.approx(2.2)  # 3/4 of the 3 s scene, on a frame


def test_pause_step_and_back():
    script = {2: key("space") + key("."), 3: key(","), 4: key("q")}
    session, window, times = _run(script)
    assert session.paused
    assert times == [0.0, 0.1, 0.2]
    assert session.time == pytest.approx(0.1)


def test_load_error_is_shown_and_the_window_stays():
    window = ScriptedWindow({3: key("q")})

    def broken(file, name):
        raise SyntaxError("bad line")

    session = PreviewSession("x.py", window_factory=lambda w, h, t: window, load_scene=broken)
    session.run()
    assert "SyntaxError" in session.error
    assert window.presented >= 3


def test_grid_and_editor(monkeypatch):
    opened = []
    monkeypatch.setattr("subprocess.Popen", lambda args: opened.append(args))
    monkeypatch.setenv("MANIM_EDITOR", "myeditor {file}:{line}")
    session, window, times = _run({2: key("g") + key("e"), 3: key("q")})
    assert session.grid
    line = session.storyboard.shots[0].line
    assert opened == [["myeditor", f"{session.file.resolve()}:{line}"]]
