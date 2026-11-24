# Hairy ball

The hairy ball scenes from 3Blue1Brown's "The Hairy Ball Theorem" (2026),
ported from `3b1b/videos` (`_2026/hairy_ball/spheres.py`, ManimGL) to the
Manim Community API.  They are a stress test of the Vulkan renderer: up to
50 000 depth-sorted vectors over a translucent sphere.

```
manim -pqh hairy_ball.py SingleNullPointHairyBall
manim -pqh hairy_ball.py PairsOfNullPoints
manim -pqh hairy_ball.py HairyEarth
```

`HairyEarth` needs two equirectangular Earth images in this folder (or the
assets directory), named `EarthTextureMap` and `NightEarthTextureMap`
(`.jpg` or `.png`), for instance NASA's Blue Marble and Black Marble.

The scenes are too heavy for CI's software Vulkan, so `tools/scenes` does not
collect this folder (it collects `example_scenes/*.py` only).
