# Rendering audit, November 2026

After three bugs in a row in the 3D and batching code (GPU memory, translucent
meshes, depth of long strokes), every drawing path of the Skia/Vulkan renderer
was checked against what it replaces: ManimCE's Cairo camera (v0.19.0) for
the Community API, ManimGL (v1.7.2) for the mesh surfaces and depth test.
Each row says how it is verified, and by which test.

## The three bugs

| Bug | Cause | Fix | Test |
|---|---|---|---|
| GPU memory ran out in HairyEarth's dense field | A depth slice was drawn as one path of thousands of strokes spanning the frame; with coverage anti-aliasing Ganesh has no GPU renderer for it and rasterised each on the CPU into a cached frame-sized mask (2 MB per slice and object at 1080p, hundreds a frame). On CI the resident set grew 58 MB a frame and Skia's cache sat at its 256 MB limit. | Depth runs are drawn one path per 128-pixel tile, small enough for Ganesh's GPU path atlas. 2D drawing is unchanged. | `test_gpu_memory.py`: 50 frames of a scaled-down HairyEarth; Skia's cache under 128 MB and no growth. |
| Translucent surfaces showed through / hid the wrong things | Depth was emulated with the painter's algorithm and a clear to the background. | A Vulkan depth pass renders the nearest opaque mesh and the nearest mesh of any opacity per pixel; meshes test against them. | `test_mesh_depth.py`: intersecting spheres, nested surfaces, near side blended once, transparent surface hides nothing. |
| HairyEarth's mesh lines vanished across the sphere | A whole line was one depth piece at its mean depth (the sphere's centre). | A stroke is ordered per cubic curve; each run is clipped per pixel against mesh depth with its nearest piece's key. | `test_a_line_on_a_translucent_sphere_shows_on_its_near_side_only` (ray-cast reference). |

Found while auditing and fixed with them: a fully transparent mesh wrote
depth (HairyEarth's Earth before it fades in would have hidden the stream
lines); a single depth-tested mobject made one run across every slice, so its
depth key was meaningless; HairyEarth's mesh lines lacked ManimGL's
`SurfaceMesh` normal nudge (0.01).

## Community API against Cairo

The graphical unit tests compare frames with ManimCE's Cairo control frames
(239 files from ManimCE, 2 replaced, below).  The comparison averages 8x8
blocks, which hides a thin missing line, so CI now also logs the largest block
difference of every frame (`MANIM_FRAME_REPORT`).

| Path | Verified by | Result |
|---|---|---|
| VMobject path: subpaths, closing, curves | Graphical units (231 tests, 799 frames) | Largest 8x8 block difference 35/255; median of the per-test maxima 25 |
| Fill, stroke, background stroke, order | Same; code read against `display_vectorized` | Same order (background stroke, fill, stroke) |
| Gradients (sheen, stops) | Same; code read against `set_cairo_context_color` | Same start/end points, evenly spaced stops |
| Line width, joins, caps, miter limit | Same; `test_sharp_miter_joins_use_cairos_miter_limit` | AUTO is Cairo's default (miter, butt), not Cairo's leftover state from the previous mobject |
| Text, Tex, SVG | Graphical units | Within the block bound |
| Images | Graphical units; code read against `display_image_mobject` | Two control frames replaced: Cairo rounds rotation to whole degrees and positions to whole pixels; the quad is drawn exactly here, with Catmull-Rom (PIL's bicubic when enlarging) |
| 3D camera: projection, z order, shading | Graphical units (`threed`, 16 tests); `ThreeDCamera` methods diffed against ManimCE | Same logic; projection moved to Rust with the same formulas |
| Static layer, multicamera, SkSL, video | `test_camera.py`, `test_multicamera.py`, `test_sksl.py`, CI movie | Pass |

The 32 tests whose largest block differs by more than 24 levels are all axes
and plots: thin (about 1.2 pixel) strokes are a little brighter in places on
the GPU than in Cairo's exact coverage.  Open; a fidelity difference, not a
missing or moved object.

## ManimGL features

| Feature | Verified by |
|---|---|
| Surface shading (`finalize_color`, `surface`, `textured_surface`) | SkSL ported line by line; HairyEarth frames on CI (`tools/scenes/hairy_frames.py`) |
| Depth test between meshes | Ray-cast references in `test_mesh_depth.py` |
| Depth test of strokes against meshes | Ray-cast reference for a line on a sphere |

## Known limits

- A VMobject run is tested against mesh depth with its nearest piece's key:
  a piece up to one depth slice (1/128 of the depth range) behind a surface
  may show.
- A stroke is tested at the depth of its centre line, as in ManimGL: where a
  surface rises towards the camera beside a line lying on it, the stroke's
  edge on that side is hidden (hence ManimGL's normal nudge).
- With `exponential_projection`, mesh depth is interpolated linearly in
  `1/(d - z)`, which is exact only for the standard projection.
- Images, point clouds and SkSL mobjects are ordered in 3D but not tested
  against mesh depth, as in Cairo.
