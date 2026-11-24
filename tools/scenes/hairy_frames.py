"""Draw key frames of the hairy ball scenes as PNG files.

The scenes in ``example_scenes/hairy_ball`` are too heavy to render whole
on CI's software Vulkan, so ``tools/scenes`` does not collect them; this
draws a few frames of each through ``Manager.evaluate``, which runs every
animation step but draws only the requested frames.  HairyEarth's Earth
images are not in the repository: unless ``--assets`` holds them, plain
stand-ins are made (a latitude and longitude grid by day, dots by night).

::

    python tools/scenes/hairy_frames.py --out scenes/hairy --size 854x480
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from manim import tempconfig
from manim.manager import Manager

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "example_scenes" / "hairy_ball" / "hairy_ball.py"

#: Seconds into each scene to draw: one or two frames of every stage.
FRAMES = {
    "SingleNullPointHairyBall": [0.5, 3.0, 6.0],
    "PairsOfNullPoints": [2.0, 6.0, 12.0, 20.0],
    "HairyEarth": [1.5, 6.0, 16.0, 20.0, 22.5, 25.0, 29.0, 35.0, 44.0, 60.0],
}


def _stand_in_earth(folder: Path) -> None:
    """Equirectangular stand-ins for EarthTextureMap and NightEarthTextureMap."""
    height, width = 256, 512
    v, u = np.mgrid[0:height, 0:width]
    day = np.zeros((height, width, 3), np.uint8)
    day[...] = (30, 90, 160)
    land = (np.sin(u / width * 6 * np.pi) * np.cos(v / height * 4 * np.pi)) > 0.3
    day[land] = (60, 140, 60)
    day[(u % 32 == 0) | (v % 32 == 0)] = (230, 230, 230)
    night = np.zeros((height, width, 3), np.uint8)
    rng = np.random.default_rng(0)
    lights = rng.random((height, width)) > 0.995
    night[lights & land] = (255, 220, 120)
    Image.fromarray(day).save(folder / "EarthTextureMap.png")
    Image.fromarray(night).save(folder / "NightEarthTextureMap.png")


def _jpeg(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("scenes/hairy"))
    parser.add_argument("--size", default="854x480")
    parser.add_argument("--frame-rate", type=int, default=15)
    parser.add_argument("--assets", type=Path, default=None)
    parser.add_argument(
        "--print-jpeg",
        action="store_true",
        help="also print each frame as a base64 JPEG line, for reading it from a CI log",
    )
    parser.add_argument("scenes", nargs="*", default=list(FRAMES))
    args = parser.parse_args()
    width, height = (int(value) for value in args.size.split("x"))
    args.out.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("hairy_ball", SCENES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as temporary:
        assets = args.assets
        if assets is None:
            assets = Path(temporary)
            _stand_in_earth(assets)
        for name in args.scenes:
            with tempconfig(
                {
                    "pixel_width": width,
                    "pixel_height": height,
                    "frame_rate": args.frame_rate,
                    "assets_dir": str(assets),
                    "disable_caching": True,
                    "progress_bar": "none",
                    "verbosity": "WARNING",
                }
            ):
                scene = getattr(module, name)()

                def save(frame, name=name) -> None:
                    path = args.out / f"{name}_{frame.requested_time:05.1f}s.png"
                    frame.image.save(path)
                    print(path, flush=True)
                    if args.print_jpeg:
                        print(f"JPEG {path.name} {_jpeg(frame.image)}", flush=True)

                Manager(scene).evaluate(frame_times=FRAMES[name], on_frame=save)


if __name__ == "__main__":
    main()
