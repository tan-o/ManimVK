"""P1 smoke test: native Vulkan offscreen target -> PNG + MP4.

This does not touch Manim's scene graph yet.  It verifies the packaged Rust
extension can own a Vulkan device, clear an RGBA8 image, read the pixels back,
and feed those pixels into the same ffmpeg path Manim uses for video output.

Run under lavapipe:

    DISPLAY= VK_ICD_FILENAMES=/home/user/vkrt/lvp_icd.x86_64.json \
    LD_LIBRARY_PATH=/home/user/vkrt python3 scripts/render_p1_native.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import manim_vk
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("media/native_vk/p1_clear.mp4"),
    )
    args = parser.parse_args()

    frames = max(1, int(round(args.fps * args.seconds)))
    device = manim_vk.Device("ManimP1")
    print(f"device: {device.device_name()}  api: {device.api_version()}")
    target = device.create_target(args.width, args.height)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.out.with_suffix(".png")

    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(args.out),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None

    for index in range(frames):
        # A slow sweep makes it obvious that every frame really went through
        # Vulkan instead of a single still image being looped.
        t = index / max(1, frames - 1)
        pixels = bytes(
            target.clear(
                0.04 + 0.02 * t,
                0.09 + 0.03 * t,
                0.20 + 0.05 * t,
                1.0,
            )
        )
        encoder.stdin.write(pixels)
        if index == 0:
            Image.frombytes("RGBA", (args.width, args.height), pixels).save(png_path)

    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed")

    print(f"wrote {args.out} ({frames} frames) and {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
