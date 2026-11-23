"""Encoding for one silent cached video segment."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.reformatter import Colorspace

from manim._config.video_encoder import VideoEncoderSpec
from manim.typing import RGBAPixelArray

__all__ = ["VideoSegmentEncoder", "YUV420Frame"]

# H.264/H.265 VUI codes: BT.709 matrix and primaries, sRGB transfer, limited
# range.  Manim's colours are sRGB; tagging them keeps players from guessing.
_BT709, _SRGB_TRANSFER, _LIMITED_RANGE = 1, 13, 1


@dataclass(frozen=True)
class YUV420Frame:
    """An opaque frame already converted to video: I420, BT.709 limited range.

    ``data`` is a C-contiguous ``uint8`` array of shape
    ``(height * 3 // 2, width)``: the Y plane, then the U and V planes at half
    width and height, packed.  A GPU renderer reads frames this way to skip
    the encoder's own conversion.
    """

    data: np.ndarray


class VideoSegmentEncoder:
    """Encode top-left-origin frames into one video segment.

    Frames are C-contiguous RGBA arrays, or :class:`YUV420Frame` when the
    stream is ``yuv420p``.  ``yuv420p`` streams are BT.709 and tagged so.
    """

    def __init__(self, *, target: Path, spec: VideoEncoderSpec) -> None:
        self.target = target
        self.spec = spec
        self._next_pts = 0
        self._closed = False
        self.target.parent.mkdir(parents=True, exist_ok=True)

        container = None
        try:
            container = av.open(
                self.target,
                mode="w",
                format=self.spec.container_format,
            )
            stream = container.add_stream(
                self.spec.codec,
                rate=self.spec.frame_rate,
                options=dict(self.spec.options),
            )
            stream.pix_fmt = self.spec.pixel_format
            stream.width = self.spec.width
            stream.height = self.spec.height
            if self.spec.pixel_format == "yuv420p":
                codec_context = stream.codec_context
                codec_context.colorspace = _BT709
                codec_context.color_primaries = _BT709
                codec_context.color_trc = _SRGB_TRANSFER
                codec_context.color_range = _LIMITED_RANGE
        except BaseException as error:
            if container is not None:
                with suppress(BaseException):
                    container.close()
            with suppress(BaseException):
                self.target.unlink(missing_ok=True)
            if not isinstance(error, Exception):
                raise
            raise self._operation_error("open", error) from error

        self._container = container
        self._stream = stream

    def _profile_description(self) -> str:
        return (
            f"{self.spec.container_format}/{self.spec.codec}/"
            f"{self.spec.pixel_format}, {self.spec.width}x{self.spec.height} "
            f"at {self.spec.frame_rate} fps"
        )

    def _operation_error(self, operation: str, error: BaseException) -> RuntimeError:
        return RuntimeError(
            f"Failed to {operation} video segment {self.target} "
            f"({self._profile_description()}): {error}",
        )

    def _validate_frame(
        self, pixels: RGBAPixelArray | YUV420Frame, repeat: int
    ) -> None:
        if self._closed:
            raise RuntimeError(f"Video segment encoder for {self.target} is closed.")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
            raise ValueError("Frame repeat must be a positive integer.")
        expected_shape = (self.spec.height, self.spec.width, 4)
        if isinstance(pixels, YUV420Frame):
            if self.spec.pixel_format != "yuv420p":
                raise ValueError(
                    f"YUV 4:2:0 frames need a yuv420p stream; got {self.spec.pixel_format}."
                )
            pixels = pixels.data
            expected_shape = (self.spec.height * 3 // 2, self.spec.width)
        if not isinstance(pixels, np.ndarray):
            raise TypeError("Video segment frames must be NumPy arrays.")
        if pixels.shape != expected_shape:
            raise ValueError(
                f"Video segment frames must have shape {expected_shape}; "
                f"got {pixels.shape}.",
            )
        if pixels.dtype != np.uint8:
            raise TypeError(
                f"Video segment frames must use uint8; got {pixels.dtype}.",
            )
        if not pixels.flags.c_contiguous:
            raise ValueError("Video segment frames must be C-contiguous.")

    def _video_frame(self, pixels: RGBAPixelArray | YUV420Frame) -> av.VideoFrame:
        if isinstance(pixels, YUV420Frame):
            return av.VideoFrame.from_ndarray(pixels.data, format="yuv420p")
        frame = av.VideoFrame.from_ndarray(pixels, format="rgba")
        if self.spec.pixel_format == "yuv420p":
            # The conversion the stream is tagged with; swscale's own
            # default is BT.601.
            frame = frame.reformat(format="yuv420p", dst_colorspace=Colorspace.ITU709)
        return frame

    def write_frame(
        self, pixels: RGBAPixelArray | YUV420Frame, *, repeat: int = 1
    ) -> None:
        """Encode ``pixels`` at consecutive segment-local PTS values."""
        self._validate_frame(pixels, repeat)
        time_base = Fraction(
            self.spec.frame_rate.denominator,
            self.spec.frame_rate.numerator,
        )
        try:
            # libavcodec takes its own reference to each frame it is sent, so
            # one converted frame serves every repeat.
            frame = self._video_frame(pixels)
            for _ in range(repeat):
                frame.pts = self._next_pts
                frame.time_base = time_base
                self._next_pts += 1
                for packet in self._stream.encode(frame):
                    self._container.mux(packet)
        except Exception as error:
            raise self._operation_error("encode", error) from error

    def finish(self) -> None:
        """Flush encoded packets and close the segment."""
        if self._closed:
            return
        self._closed = True
        # Close even after an interruption, preserving a flush failure if close fails too.
        first_error: BaseException | None = None
        try:
            try:
                for packet in self._stream.encode():
                    self._container.mux(packet)
            except BaseException as error:
                first_error = error
        finally:
            try:
                self._container.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            if not isinstance(first_error, Exception):
                raise first_error
            raise self._operation_error("finish", first_error) from first_error

    def abort(self) -> None:
        """Close resources and remove the incomplete target."""
        # An interrupted close must still remove the target and remain the primary error.
        first_error: BaseException | None = None
        try:
            if not self._closed:
                self._closed = True
                try:
                    self._container.close()
                except BaseException as error:
                    first_error = error
        finally:
            try:
                self.target.unlink(missing_ok=True)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            if not isinstance(first_error, Exception):
                raise first_error
            raise self._operation_error("abort", first_error) from first_error
