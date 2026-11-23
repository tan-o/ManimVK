"""A scene's storyboard: its shots, one per ``play()`` or ``wait()`` call.

The storyboard is read from the scene's :doc:`timeline </guides/timeline>`,
captured by evaluating the scene without drawing it.  Each shot keeps where
it starts and ends, what it plays, the section it belongs to and the line of
code that made it.  :func:`storyboard_sheet` draws the storyboard as a contact
sheet, one thumbnail per shot.
"""

from __future__ import annotations

__all__ = ["Shot", "Storyboard", "capture_storyboard", "storyboard_sheet"]

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from manim.scene.scene import Scene
    from manim.timeline import Timeline


@dataclass(frozen=True, slots=True)
class Shot:
    """One ``play()`` or ``wait()`` call of a scene."""

    index: int
    kind: str
    start: float
    end: float
    animations: tuple[str, ...]
    section: str | None
    line: int | None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def label(self) -> str:
        """A short description, such as ``"3 Create, FadeIn"`` or ``"4 wait"``."""
        if self.kind == "wait":
            return f"{self.index} wait"
        names = [
            "animate" if name.endswith("_MethodAnimation") else name.rsplit(".", 1)[-1]
            for name in self.animations
        ]
        shown = ", ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else "")
        return f"{self.index} {shown}"


@dataclass(frozen=True, slots=True)
class Storyboard:
    """The shots of a scene, in order."""

    scene: str
    frame_rate: float
    duration: float
    shots: tuple[Shot, ...]

    @classmethod
    def from_timeline(cls, timeline: Timeline) -> Storyboard:
        data = timeline.to_dict()
        shots: list[Shot] = []
        section: str | None = None
        for entry in data["entries"]:
            kind = entry["kind"]
            if kind == "section":
                section = entry.get("name")
                continue
            if kind not in ("play", "wait") or entry.get("end") is None:
                continue
            source = entry.get("source") or {}
            shots.append(
                Shot(
                    index=len(shots),
                    kind=kind,
                    start=float(entry["start"]),
                    end=float(entry["end"]),
                    animations=tuple(a["type"] for a in entry.get("animations", ())),
                    section=section,
                    line=source.get("line"),
                )
            )
        return cls(
            scene=data["scene"],
            frame_rate=float(data["frame_rate"]),
            duration=float(data["duration"]),
            shots=tuple(shots),
        )

    def shot_at(self, time: float) -> Shot | None:
        """The shot playing at ``time``; the last one after the end."""
        if not self.shots:
            return None
        starts = [shot.start for shot in self.shots]
        index = max(bisect.bisect_right(starts, time) - 1, 0)
        return self.shots[index]

    def frame_times(self, start: float = 0.0) -> list[float]:
        """The start times of the frames from ``start`` to the end."""
        rate = self.frame_rate
        first = int(np.floor(start * rate + 1e-9))
        last = int(np.ceil(self.duration * rate - 1e-9))
        return [index / rate for index in range(first, last)]


def capture_storyboard(scene_class: type[Scene]) -> Storyboard:
    """Evaluate a new ``scene_class()`` without drawing it and return its storyboard."""
    from manim.manager import Manager

    manager = Manager(scene_class())
    manager.evaluate(capture_timeline=True)
    return Storyboard.from_timeline(manager.timeline)


def shot_thumbnails(scene_class: type[Scene], storyboard: Storyboard) -> list[Image.Image]:
    """The last frame of each shot, drawn in one evaluation of a new scene."""
    from manim.manager import Manager

    rate = storyboard.frame_rate
    times = [max(shot.start, shot.end - 1 / rate) for shot in storyboard.shots]
    images: dict[float, Image.Image] = {}

    def keep(frame: Any) -> None:
        images[frame.requested_time] = frame.image

    Manager(scene_class()).evaluate(frame_times=times, on_frame=keep)
    blank = Image.new("RGBA", (16, 9))
    return [images.get(time, blank) for time in times]


def storyboard_sheet(
    storyboard: Storyboard,
    thumbnails: Sequence[Image.Image],
    columns: int = 4,
    thumbnail_width: int = 320,
) -> Image.Image:
    """A contact sheet: each shot's last frame with its number, animations,
    time range and line of code."""
    if not thumbnails:
        return Image.new("RGB", (thumbnail_width, 40), "black")
    aspect = thumbnails[0].height / thumbnails[0].width
    width, height = thumbnail_width, int(thumbnail_width * aspect)
    caption = 34
    rows = -(-len(thumbnails) // columns)
    sheet = Image.new("RGB", (columns * (width + 8) + 8, rows * (height + caption + 8) + 8), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for shot, image in zip(storyboard.shots, thumbnails, strict=True):
        x = 8 + (shot.index % columns) * (width + 8)
        y = 8 + (shot.index // columns) * (height + caption + 8)
        sheet.paste(image.convert("RGB").resize((width, height)), (x, y))
        where = f"line {shot.line}" if shot.line else ""
        section = f"[{shot.section}] " if shot.section else ""
        draw.text((x + 4, y + height + 2), f"{section}{shot.label}", fill=(240, 240, 240))
        draw.text(
            (x + 4, y + height + 17),
            f"{shot.start:.2f}-{shot.end:.2f}s  {where}",
            fill=(160, 160, 160),
        )
    return sheet
