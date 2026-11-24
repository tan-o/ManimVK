from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from pathlib import Path

import numpy as np

from manim.typing import PixelArray

from ._show_diff import show_diff_helper

# Frames are drawn on the GPU, whose anti-aliasing differs from the rasterizer
# that made the control data and between GPUs.  They are compared as
# tools/golden compares renders: RGB after 8x8 area averaging, where edge
# differences average out but a missing, moved or recoloured object does not,
# and alpha.  Both are mean absolute differences on the 0-255 scale.
COARSE_FACTOR = 8
COARSE_RGB_TOLERANCE = 3.0
ALPHA_TOLERANCE = 2.0


def _coarse(frame: np.ndarray) -> np.ndarray:
    h = frame.shape[0] // COARSE_FACTOR * COARSE_FACTOR
    w = frame.shape[1] // COARSE_FACTOR * COARSE_FACTOR
    frame = frame[:h, :w]
    return frame.reshape(h // COARSE_FACTOR, COARSE_FACTOR, w // COARSE_FACTOR, COARSE_FACTOR, -1).mean(axis=(1, 3))


def frame_difference(frame: PixelArray, control: PixelArray) -> tuple[float, float]:
    """``(coarse RGB MAE, alpha MAE)`` between two RGBA frames of one size."""
    frame = np.asarray(frame, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    coarse = float(np.abs(_coarse(frame)[..., :3] - _coarse(control)[..., :3]).mean())
    alpha = float(np.abs(frame[..., 3] - control[..., 3]).mean())
    return coarse, alpha


logger = logging.getLogger("manim")


class _FramesTester:
    def __init__(self, file_path: Path, show_diff: bool = False) -> None:
        self._file_path = file_path
        self._show_diff = show_diff
        self._frames: np.ndarray
        self._number_frames: int = 0
        self._frames_compared = 0

    @contextlib.contextmanager
    def testing(self) -> Generator[None, None, None]:
        with np.load(self._file_path) as data:
            self._frames = data["frame_data"]
            # For backward compatibility, when the control data contains only one frame (<= v0.8.0)
            if len(self._frames.shape) != 4:
                self._frames = np.expand_dims(self._frames, axis=0)
            logger.debug(self._frames.shape)
            self._number_frames = np.ma.size(self._frames, axis=0)
            yield
            assert self._frames_compared == self._number_frames, (
                f"The scene tested contained {self._frames_compared} frames, "
                f"when there are {self._number_frames} control frames for this test."
            )

    def check_frame(self, frame_number: int, frame: PixelArray) -> None:
        assert frame_number < self._number_frames, (
            f"The tested scene is at frame number {frame_number} "
            f"when there are {self._number_frames} control frames."
        )
        control = self._frames[frame_number]
        assert frame.shape == control.shape, (
            f"Frame no {frame_number} has shape {frame.shape}, the control frame {control.shape}."
        )
        coarse, alpha = frame_difference(frame, control)
        if coarse <= COARSE_RGB_TOLERANCE and alpha <= ALPHA_TOLERANCE:
            self._frames_compared += 1
            return
        if self._show_diff:
            show_diff_helper(frame_number, frame, control, self._file_path.name)
        raise AssertionError(
            f"Frame no {frame_number} differs from {self._file_path.name}: coarse RGB MAE "
            f"{coarse:.2f} (tolerance {COARSE_RGB_TOLERANCE}), alpha MAE {alpha:.2f} "
            f"(tolerance {ALPHA_TOLERANCE}). You can use --show_diff to visually show the difference."
        )


class _ControlDataWriter(_FramesTester):
    def __init__(self, file_path: Path, size_frame: tuple) -> None:
        self.file_path = file_path
        self.frames = np.empty((0, *size_frame, 4))
        self._number_frames_written: int = 0

    # Actually write a frame.
    def check_frame(self, index: int, frame: PixelArray) -> None:
        frame = frame[np.newaxis, ...]
        self.frames = np.concatenate((self.frames, frame))
        self._number_frames_written += 1

    @contextlib.contextmanager
    def testing(self) -> Generator[None, None, None]:
        yield
        self.save_contol_data()

    def save_contol_data(self) -> None:
        self.frames = self.frames.astype("uint8")
        np.savez_compressed(self.file_path, frame_data=self.frames)
        logger.info(
            f"{self._number_frames_written} control frames saved in {self.file_path}",
        )
