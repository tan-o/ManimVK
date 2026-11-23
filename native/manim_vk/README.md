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

- Windows x64: built and exercised in this repository.
- Linux: the source is portable; build with the system Vulkan loader and a
  local Skia binary cache.  No prebuilt wheel is checked in.
- macOS: Vulkan requires MoltenVK; no prebuilt wheel or CI job is provided yet.

## Behaviour notes

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
