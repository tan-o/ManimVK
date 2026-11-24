# manim_core native extension

CPU kernels for Mobject geometry.  This crate has no GPU or Skia dependency,
so it builds on any machine with a Rust toolchain.  It handles work that runs
every frame on point arrays:

| Function | Used by |
|---|---|
| `transform_points(arrays, matrix, about, shift)` | `Mobject.shift`, `scale`, `rotate`, `stretch`, `apply_matrix` |
| `bounds(arrays, curve_size)` | `get_critical_point`, `get_center`, `move_to`, `width`/`height` |
| `lerp(start, end, alpha)` | `Mobject.interpolate` along straight paths (`Transform`) |
| `subpath_split_indices(points, curve_size, n_dims, atol)` | `VMobject.get_subpath_split_indices_from_points` |
| `cubic_subpaths(points, curve_size, n_dims, atol, limit)` | the Skia renderer's path building |
| `tame_huge_curves(points, curve_size, limit)` | the Skia renderer's depth-sorted paths |
| `project_points(points, center, rotation, focal_distance, zoom, exponential)` | `ThreeDCamera.project_points` |
| `range_means(points, ranges)` | `VMobject.get_depth_pieces` (subpath centers; parallel) |
| `depth_order(depth, group, depth_slices)` | `ThreeDCamera.get_draw_order` (parallel sort) |
| `line_points`, `circle_points`, `arrow_points` | `LineBatch`, `CircleBatch`/`DotBatch`, `ArrowBatch` |

Every function takes NumPy arrays and returns new arrays.  Manim assigns new
point arrays instead of mutating shared ones, so the kernels never write into
their inputs.

The pure Rust parts (`geometry`, `shapes`) do not depend on PyO3.  The
renderer crate can call them directly.  `range_means` and `depth_order` run
on all cores (rayon) and release the GIL; the other kernels are single
passes, cheap enough that threads would cost more than they save.

`API_VERSION` in `src/lib.rs` must match `manim/_native.py`; bump both when a
binding changes.

## Build

```powershell
cd native\manim_core
maturin build --release
uv pip install --python ..\..\.venv\Scripts\python.exe --no-deps --reinstall `
  target\wheels\manim_core-0.1.0-cp311-abi3-win_amd64.whl
```

or `maturin develop --release` inside the virtual environment.

`cargo test --lib` runs the Rust unit tests; `tests/module/mobject/test_batch_and_core.py`
checks the bindings against the NumPy reference behaviour.
