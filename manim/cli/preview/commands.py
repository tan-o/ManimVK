"""``manim preview`` and ``manim storyboard``: look at a scene while writing it."""

from __future__ import annotations

from pathlib import Path

import cloup

from manim._config import config, console
from manim.constants import EPILOG, QUALITIES

__all__ = ["preview", "storyboard"]

_QUALITY = {q["flag"]: name for name, q in QUALITIES.items() if q["flag"]}


def _configure(quality: str, frame_rate: float | None) -> None:
    config.renderer = "vulkan"
    config.quality = _QUALITY[quality]
    if frame_rate is not None:
        config.frame_rate = frame_rate
    config.progress_bar = "none"


_file_argument = cloup.argument(
    "file", type=cloup.Path(path_type=Path, exists=True, dir_okay=False), required=True
)
_scene_argument = cloup.argument("scene", required=False)
_quality_option = cloup.option(
    "-q", "--quality", type=cloup.Choice(sorted(_QUALITY)), default="l", show_default=True,
    help="Render quality of the preview (l: 480p15, m: 720p30, ...).",
)
_fps_option = cloup.option("--fps", "frame_rate", type=float, default=None, help="Frame rate override.")


@cloup.command(context_settings=None, no_args_is_help=True, epilog=EPILOG)
@_file_argument
@_scene_argument
@_quality_option
@_fps_option
def preview(file: Path, scene: str | None, quality: str, frame_rate: float | None) -> None:
    """Play a scene in a window; reload it whenever FILE is saved.

    Space pauses, Left/Right jump between shots (play/wait calls), ',' and '.'
    step frames, 'l' loops the shot, 'r' reloads, 's' saves a still, '[' and
    ']' change the speed, clicking the bar seeks, 'q' quits.
    """
    from manim.preview import PreviewSession

    _configure(quality, frame_rate)
    PreviewSession(file, scene).run()


@cloup.command(context_settings=None, no_args_is_help=True, epilog=EPILOG)
@_file_argument
@_scene_argument
@_quality_option
@_fps_option
@cloup.option(
    "-o", "--output", type=cloup.Path(path_type=Path, dir_okay=False), default=None,
    help="Image to write; <file>_<scene>_storyboard.png by default.",
)
@cloup.option("--columns", type=int, default=4, show_default=True)
def storyboard(
    file: Path,
    scene: str | None,
    quality: str,
    frame_rate: float | None,
    output: Path | None,
    columns: int,
) -> None:
    """Draw a contact sheet with the last frame of every shot of a scene."""
    from manim.preview.session import _load_scene_class
    from manim.preview.storyboard import capture_storyboard, shot_thumbnails, storyboard_sheet

    _configure(quality, frame_rate)
    scene_class = _load_scene_class(file, scene)
    board = capture_storyboard(scene_class)
    sheet = storyboard_sheet(board, shot_thumbnails(scene_class, board), columns=columns)
    output = output or file.with_name(f"{file.stem}_{scene_class.__name__}_storyboard.png")
    sheet.save(output)
    console.print(f"{len(board.shots)} shots, {board.duration:.2f}s -> {output}")
