# manim_vk native extension

Python bindings for the offscreen Vulkan target, Skia/Ganesh drawing, and the
read-only preview window used by the ManimVK renderer.

## Build (Windows x64)

Requirements:

- Rust stable toolchain with the MSVC target
- Vulkan SDK or loader with a working ICD (NVIDIA/AMD/Intel, or lavapipe)
- Skia binaries available locally, referenced by `SKIA_BINARIES_URL`
- `maturin`

From this directory:

```powershell
$env:SKIA_BINARIES_URL = 'file://C:/skia/skia-binaries.tar.gz'
$env:LIB = '%USERPROFILE%\miniconda3\libs;' + $env:LIB
maturin build --release
uv pip install --python ..\..\.venv\Scripts\python.exe --no-deps --reinstall `
  target\wheels\manim_vk-0.1.0-cp311-abi3-win_amd64.whl
```

The extension is `abi3` for Python 3.11+.  A clean build downloads the Skia
binary archive into the Cargo target directory.

## Type-check without Skia binaries

`skia-bindings` skips its native build and uses its pre-generated bindings
when `DOCS_RS` is set, so the crate can be checked (not linked) on a machine
without Skia's binaries, such as CI:

```sh
DOCS_RS=1 cargo clippy
```

## Platform status

- Windows x64: built and exercised locally.
- Linux: built and tested in CI (`.github/workflows/scenes.yml`) on Mesa's
  lavapipe, the real Skia Ganesh Vulkan path in software.  No prebuilt wheel
  is published.
- macOS: Vulkan requires MoltenVK; no prebuilt wheel or CI job is provided yet.

## Behaviour notes

- Interface version: `API_VERSION` in `src/lib.rs` must match
  `manim/_native.py`; bump both when a binding changes.  `import manim`
  refuses an extension built from older sources.
- Display list (`display_list.rs`): `draw_display_list(objects, meshes, runs,
  parts, keys)` draws a frame's VMobjects and surface meshes in one call.
  Each VMobject comes once (projected points, `(start, end, flags)` rows of
  subpaths or depth pieces, prebuilt paints), each mesh once (vertices,
  triangles, images, lighting), then `(kind, object, start, end)` runs in
  drawing order, each drawing a mesh, a whole VMobject or its depth pieces
  `parts[start..end]`, with one float32 depth key per run.  The pieces of a
  run are drawn one path per 128-pixel screen tile, small enough for Ganesh's
  GPU path atlas (a larger anti-aliased path is rasterised on the CPU into a
  cached mask the size of its bounds).
- Meshes (`meshes.rs`) are Skia custom meshes (`SkMesh`) whose SkSL is
  ManimGL's surface shading (`finalize_color.glsl`, `surface`,
  `textured_surface`).  Their depth (`depth.rs`) comes from a Vulkan render
  pass of our own with a real depth attachment, on the same device and queue
  as Skia, run twice a frame: the nearest opaque mesh per pixel, and the
  nearest mesh of any opacity.  An opaque mesh is tested against the first,
  a translucent one against the second; each VMobject run with a depth key is
  clipped (`clipShader`) to where the second holds nothing nearer.  The pass's
  shaders are GLSL, compiled to SPIR-V with naga.
- Occlusion culling (`occlusion.rs`): Intel's Masked Occlusion Culling
  (`third_party/masked_occlusion_culling`, Apache-2.0, built by `build.rs` on
  x86 with SSE4.1 and AVX2 variants picked at run time) rasterises the
  frame's meshes on the CPU; chains of depth pieces outside the frame or
  wholly behind them at their run's depth key are skipped.
  `set_occlusion_culling(enabled)`; `take_draw_stats()` returns the time
  display lists spent per stage and the chains tested and culled.
- `memory_usage()` returns the driver's device-local heap usage
  (`VK_EXT_memory_budget`, None without it) and Skia's budgeted cache.

- Keyed textures (`textures.rs`): Python names every read-only pixel array
  (image files, `ImageMobject` sources, mesh atlases) with a process-unique
  key from `manim.utils.images.new_texture_key`.  The first draw uploads a
  GPU texture; later draws reuse it.  A texture not drawn for two frames is
  dropped.
- Video frames are read with `request_yuv420` / `collect_yuv420_into`: Skia
  converts to BT.709 I420 on the GPU and reads 1.5 bytes per pixel
  asynchronously, so frame N is collected after frame N+1 is drawn.

- One `Device` owns one Vulkan instance/device and one shared `DirectContext`.
  Targets created from it clone the context reference rather than creating a
  second Ganesh context.
- All target and preview objects are single-thread bound.  Cross-thread use
  raises a readable `RuntimeError` instead of panicking.
- Preview adapter selection uses `WGPU_ADAPTER_NAME` internally, populated from
  the offscreen device name so both paths normally use the same GPU.
