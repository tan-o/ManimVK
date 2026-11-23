"""An interactive preview of a scene file: play, pause, seek by shot, reload on save.

How it works
------------
A Manim scene is code that runs forward: the state at second 12 is whatever
the plays and updaters before it left behind.  The preview therefore keeps
Manim's own model.  To show the scene from time ``t`` it evaluates a new scene
object and only draws the frames from ``t`` on.  Earlier plays still run, so
updaters see the same history as in a render, but nothing is drawn for them.
Seeking, jumping to a shot and reloading all restart the evaluation this way.

Frames already shown are kept in a short in-memory history, so stepping
backwards while paused needs no evaluation.

Controls
--------
========================  ===================================================
Space                     pause / play
Left / Right              previous / next shot
``,`` / ``.``             one frame back (from history) / forward (paused)
Home                      back to the start
``l``                     loop the current shot
``r``                     reload the file now (also happens on save)
``s``                     save the current frame as a PNG
``[`` / ``]``             half / double the playback speed
``g``                     coordinate grid; the bar shows the cursor's scene
                          coordinates
``e``                     open the current shot's line in an editor
                          (``MANIM_EDITOR``, default ``code -g {file}:{line}``)
click on the bar          seek
``q`` / Escape            quit
========================  ===================================================
"""

from __future__ import annotations

__all__ = ["PreviewSession"]

import collections
import os
import shlex
import subprocess
import time as clock
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

from .storyboard import Storyboard, capture_storyboard

if TYPE_CHECKING:
    from manim.scene.scene import Scene

#: Height in pixels of the timeline bar drawn over the bottom of the frame.
BAR_HEIGHT = 22


class PreviewWindow(Protocol):
    """What the session needs from a window (see :class:`NativePreviewWindow`)."""

    def present(self, rgba: np.ndarray) -> None: ...
    def poll_events(self) -> list[tuple[Any, ...]]: ...
    def is_closing(self) -> bool: ...
    def close(self) -> None: ...


class NativePreviewWindow:
    """The ``manim_vk`` preview window: presents RGBA frames and reports input."""

    def __init__(self, width: int, height: int, title: str) -> None:
        import manim_vk

        self._native = manim_vk.NativeWindow(width, height, title)

    def present(self, rgba: np.ndarray) -> None:
        self._native.present_rgba(np.ascontiguousarray(rgba, dtype=np.uint8).tobytes())

    def poll_events(self) -> list[tuple[Any, ...]]:
        return self._native.poll_events()

    def is_closing(self) -> bool:
        return bool(self._native.is_closing())

    def close(self) -> None:
        self._native.close()


class _Restart(Exception):
    """Abort the running evaluation and show the scene from ``time``."""

    def __init__(self, time: float, reload: bool = False) -> None:
        super().__init__(time)
        self.time = time
        self.reload = reload


class _Quit(Exception):
    """Leave the preview."""


class PreviewSession:
    """Preview one scene of a file in a window.

    Parameters
    ----------
    file
        The scene file; it is re-imported whenever it changes on disk.
    scene_name
        The scene class to show; the only scene of the file if omitted.
    window_factory
        ``(width, height, title) -> window``; the native window by default.
    history
        How many shown frames to keep for stepping backwards.
    load_scene
        ``(file, scene_name) -> scene class``; reads the file by default.
    """

    def __init__(
        self,
        file: str | Path,
        scene_name: str | None = None,
        *,
        window_factory: Callable[[int, int, str], PreviewWindow] | None = None,
        history: int = 120,
        load_scene: Callable[[Path, str | None], type[Scene]] | None = None,
    ) -> None:
        self.file = Path(file)
        self.scene_name = scene_name
        self._window_factory = window_factory or NativePreviewWindow
        self._load_scene = load_scene or _load_scene_class
        self.window: PreviewWindow | None = None
        self.storyboard: Storyboard | None = None
        self.scene_class: type[Scene] | None = None
        self.history: collections.deque[tuple[float, np.ndarray]] = collections.deque(
            maxlen=history
        )
        self.paused = False
        self.loop_shot: int | None = None
        self.speed = 1.0
        self.time = 0.0
        self.error: str | None = None
        self._mtime = 0.0
        self._last_watch = 0.0
        self._cursor = (0.0, 0.0)
        self._wall_start = 0.0
        self._scene_start = 0.0
        self._step = False
        self._back = 0
        self.grid = False

    # -- the main loop ---------------------------------------------------
    def run(self) -> None:
        """Show the scene until the window is closed or ``q`` is pressed."""
        target, reload = 0.0, True
        try:
            while True:
                if reload:
                    self._reload()
                try:
                    if self.scene_class is None or self.storyboard is None:
                        self._idle()  # a load error: wait for the next save
                    else:
                        self._play_from(target)
                        self._at_end()
                except _Restart as restart:
                    target, reload = restart.time, restart.reload
                else:
                    reload = False
        except _Quit:
            pass
        finally:
            if self.window is not None:
                self.window.close()

    def _reload(self) -> None:
        self._mtime = self._file_mtime()
        try:
            self.scene_class = self._load_scene(self.file, self.scene_name)
            self.storyboard = capture_storyboard(self.scene_class)
            self.error = None
        except Exception as error:  # show it, keep the window, wait for a fix
            self.scene_class = None
            self.storyboard = None
            self.error = f"{type(error).__name__}: {error}"
        self.history.clear()

    def _play_from(self, start: float) -> None:
        from manim.manager import Manager

        assert self.scene_class is not None and self.storyboard is not None
        start = min(max(start, 0.0), self.storyboard.duration)
        times = self.storyboard.frame_times(start)
        self._wall_start = clock.perf_counter()
        self._scene_start = start
        if not times:
            return
        Manager(self.scene_class()).evaluate(frame_times=times, on_frame=self._on_frame)

    def _at_end(self) -> None:
        """The last frame was shown: loop the shot, or wait for input."""
        assert self.storyboard is not None
        if self.loop_shot is not None:
            raise _Restart(self.storyboard.shots[self.loop_shot].start)
        self.paused = True
        while True:
            self._handle_input()
            self._show(None)
            clock.sleep(1 / 60)

    def _idle(self) -> None:
        while True:
            self._handle_input()
            self._show(None)
            clock.sleep(1 / 30)

    # -- frames ------------------------------------------------------------
    def _on_frame(self, frame: Any) -> None:
        self.time = frame.time
        rgba = np.asarray(frame.image)
        self.history.append((frame.time, rgba))
        self._back = 0
        self._show(rgba)
        self._handle_input()
        storyboard = self.storyboard
        assert storyboard is not None
        if self.loop_shot is not None:
            shot = storyboard.shots[self.loop_shot]
            if frame.time + 1 / storyboard.frame_rate >= shot.end - 1e-9:
                raise _Restart(shot.start)
        # Pace to wall-clock time; a frame that took longer is shown late
        # instead of skipping frames, so nothing is missed.
        due = self._wall_start + (frame.time - self._scene_start) / self.speed
        delay = due - clock.perf_counter()
        if delay > 0:
            clock.sleep(delay)
        while self.paused and not self._step:
            self._handle_input()
            self._show(None)
            clock.sleep(1 / 60)
        if self._step:
            self._step = False
        # Restart the clock after a pause or a slow frame.
        if self.paused or delay < -0.25:
            self._wall_start = clock.perf_counter()
            self._scene_start = frame.time

    def _window(self, width: int, height: int) -> PreviewWindow:
        if self.window is None:
            self.window = self._window_factory(width, height, f"Manim preview: {self.file.name}")
        return self.window

    def _current_rgba(self) -> np.ndarray | None:
        if not self.history:
            return None
        index = len(self.history) - 1 - self._back
        return self.history[index][1]

    def _show(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            rgba = self._current_rgba()
        if rgba is None:
            from manim import config

            rgba = np.zeros((config.pixel_height, config.pixel_width, 4), dtype=np.uint8)
            rgba[..., 3] = 255
        window = self._window(rgba.shape[1], rgba.shape[0])
        window.present(self._with_overlay(rgba))

    def _with_overlay(self, rgba: np.ndarray) -> np.ndarray:
        """The frame with the timeline bar: shots, playhead and status."""
        image = Image.fromarray(rgba).convert("RGBA")
        width, height = image.size
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if self.grid:
            self._draw_grid(draw, width, height)
        top = height - BAR_HEIGHT
        draw.rectangle((0, top, width, height), fill=(0, 0, 0, 170))
        storyboard = self.storyboard
        if storyboard is not None and storyboard.duration > 0:
            scale = width / storyboard.duration
            current = storyboard.shot_at(self.time)
            for shot in storyboard.shots:
                x0, x1 = shot.start * scale, max(shot.end * scale - 1, shot.start * scale)
                color = (
                    (90, 160, 255, 200)
                    if current is not None and shot.index == current.index
                    else (110, 110, 110, 160)
                )
                if shot.index == self.loop_shot:
                    color = (255, 190, 60, 220)
                draw.rectangle((x0, top + 4, x1, top + 9), fill=color)
            playhead = self.time * scale
            draw.line((playhead, top + 1, playhead, top + 12), fill=(255, 255, 255, 255), width=2)
            status = "paused" if self.paused else f"{self.speed:g}x"
            if self.grid:
                x, y = self._scene_coordinates(*self._cursor, width, height)
                status += f"  ({x:.2f}, {y:.2f})"
            label = current.label if current is not None else ""
            line = f"  line {current.line}" if current is not None and current.line else ""
            draw.text(
                (6, top + 10),
                f"{self.time:6.2f}/{storyboard.duration:.2f}s  {status}  shot {label}{line}",
                fill=(235, 235, 235, 255),
            )
        if self.error:
            draw.rectangle((0, 0, width, 18), fill=(150, 20, 20, 210))
            draw.text((6, 3), self.error[:200], fill=(255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(image, overlay))

    @staticmethod
    def _scene_coordinates(x: float, y: float, width: int, height: int) -> tuple[float, float]:
        """Frame pixels to scene units of the default camera frame."""
        from manim import config

        return (
            (x / width - 0.5) * config.frame_width,
            (0.5 - y / height) * config.frame_height,
        )

    def _draw_grid(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        from manim import config

        per_unit = width / config.frame_width
        cx, cy = width / 2, height / 2
        for k in range(-int(config.frame_width), int(config.frame_width) + 1):
            x = cx + k * per_unit
            draw.line((x, 0, x, height), fill=(255, 255, 255, 110 if k == 0 else 40))
            if k and 0 <= x < width:
                draw.text((x + 2, cy + 2), str(k), fill=(255, 255, 255, 140))
        for k in range(-int(config.frame_height), int(config.frame_height) + 1):
            y = cy - k * per_unit
            draw.line((0, y, width, y), fill=(255, 255, 255, 110 if k == 0 else 40))
            if k and 0 <= y < height:
                draw.text((cx + 2, y + 2), str(k), fill=(255, 255, 255, 140))

    def _open_editor(self, line: int | None) -> None:
        command = os.environ.get("MANIM_EDITOR", "code -g {file}:{line}")
        args = shlex.split(command.format(file=self.file.resolve(), line=line or 1))
        try:
            subprocess.Popen(args)
        except OSError as error:
            self.error = f"cannot open editor ({command}): {error}"

    # -- input -------------------------------------------------------------
    def _handle_input(self) -> None:
        window = self.window
        if window is None:
            return
        if window.is_closing():
            raise _Quit
        self._watch_file()
        for event in window.poll_events():
            kind = event[0]
            if kind == "key" and event[2]:
                self._on_key(event[1])
            elif kind == "cursor":
                self._cursor = (event[1], event[2])
            elif kind == "mouse" and event[1] == "left" and event[2]:
                self._on_click()

    def _on_key(self, name: str) -> None:
        storyboard = self.storyboard
        shot = storyboard.shot_at(self.time) if storyboard is not None else None
        if name in ("q", "escape"):
            raise _Quit
        if name == "space":
            self.paused = not self.paused
        elif name == "r":
            raise _Restart(shot.start if shot else 0.0, reload=True)
        elif name == "s":
            self._save_still()
        elif name in ("[", "]"):
            self.speed = min(max(self.speed * (0.5 if name == "[" else 2.0), 0.125), 8.0)
        elif name == "g":
            self.grid = not self.grid
        elif name == "e":
            self._open_editor(shot.line if shot is not None else None)
        elif storyboard is None or shot is None:
            return
        elif name == "arrowright":
            if shot.index + 1 < len(storyboard.shots):
                raise _Restart(storyboard.shots[shot.index + 1].start)
        elif name == "arrowleft":
            # Like a media player: back to the start of this shot, or to the
            # previous shot when already at its start.
            near_start = self.time - shot.start < 0.25
            index = shot.index - 1 if near_start and shot.index > 0 else shot.index
            raise _Restart(storyboard.shots[index].start)
        elif name == "home":
            raise _Restart(0.0)
        elif name == "l":
            self.loop_shot = None if self.loop_shot == shot.index else shot.index
        elif name == ".":
            self.paused = True
            if self._back > 0:
                self._back -= 1
            else:
                self._step = True
        elif name == ",":
            self.paused = True
            self._back = min(self._back + 1, len(self.history) - 1)
            if self.history:
                self.time = self.history[len(self.history) - 1 - self._back][0]

    def _on_click(self) -> None:
        storyboard = self.storyboard
        rgba = self._current_rgba()
        if storyboard is None or rgba is None:
            return
        x, y = self._cursor
        height, width = rgba.shape[:2]
        if y >= height - BAR_HEIGHT and width > 0:
            raise _Restart(min(max(x / width, 0.0), 1.0) * storyboard.duration)

    def _save_still(self) -> None:
        rgba = self._current_rgba()
        if rgba is None:
            return
        path = self.file.with_name(f"{self.file.stem}_{self.time:.2f}s.png")
        Image.fromarray(rgba).save(path)

    def _file_mtime(self) -> float:
        try:
            return self.file.stat().st_mtime
        except OSError:
            return 0.0

    def _watch_file(self) -> None:
        now = clock.perf_counter()
        if now - self._last_watch < 0.25:
            return
        self._last_watch = now
        if self._file_mtime() != self._mtime:
            storyboard = self.storyboard
            shot = storyboard.shot_at(self.time) if storyboard is not None else None
            raise _Restart(shot.start if shot else 0.0, reload=True)


def _load_scene_class(file: Path, scene_name: str | None) -> type[Scene]:
    """Import ``file`` afresh and return its scene class ``scene_name``."""
    from manim.utils.module_ops import get_module, get_scene_classes_from_module

    module = get_module(file)
    classes = get_scene_classes_from_module(module)
    if scene_name is not None:
        for scene_class in classes:
            if scene_class.__name__ == scene_name:
                return scene_class
        raise ValueError(f"{file.name} has no scene named {scene_name}")
    if len(classes) != 1:
        names = ", ".join(c.__name__ for c in classes) or "none"
        raise ValueError(f"name the scene to preview; {file.name} has: {names}")
    return classes[0]
