"""P2 smoke test: Skia/Ganesh Vulkan draws into manim_vk's VkImage.

    DISPLAY= VK_ICD_FILENAMES=/home/user/vkrt/lvp_icd.x86_64.json \
    LD_LIBRARY_PATH=/home/user/vkrt python3 scripts/render_skia_vk_test.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import manim_vk
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("media/native_vk/skia_vk_test.png"),
    )
    args = parser.parse_args()

    device = manim_vk.Device("ManimSkiaP2")
    target = device.create_target(args.width, args.height)
    pixels = bytes(target.skia_draw_test())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGBA", (args.width, args.height), pixels).save(args.out)
    print(f"wrote {args.out} ({args.width}x{args.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
