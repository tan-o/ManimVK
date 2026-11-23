Vulkan / Skia backend
====================

ManimVK renders the classic Manim object model through Skia's Ganesh backend on
top of Vulkan.  It is the default renderer for this fork; Cairo remains available
as an explicit reference rasterizer.

Default behaviour
-----------------

``config.renderer`` defaults to ``"vulkan"``.  Both ``--renderer vulkan`` and the
historical ``--renderer opengl`` spelling select the Skia/Vulkan backend; the old
moderngl pipeline is retired.  ``--renderer cairo`` is the only way to request
Cairo.

The renderer keeps one Vulkan device per process and one Skia Ganesh context per
device.  Vector paths, text, meshes, images, point clouds, background-filled
shapes and ``MultiCamera`` sub-views all draw and composite on the GPU.  A frame
is copied to host memory only when a video writer, still image, or preview
needs pixels.

Selecting a GPU
---------------

Set ``MANIM_VK_DEVICE`` to a case-insensitive substring of the Vulkan adapter
name, for example::

    set MANIM_VK_DEVICE=RTX 3050
    manim -qm scene.py Scene

Without the variable, a discrete GPU is preferred, then an integrated GPU, then a
CPU device such as lavapipe.  If no Vulkan physical device exists the renderer
raises ``RuntimeError`` with installation guidance.

Custom shaders
--------------

Legacy GLSL ``Shader`` objects are not compiled.  Use
:class:`~manim.mobject.sksl.SkSLMobject` with a SkSL source string instead::

    from manim import Scene, SkSLMobject

    class ShaderScene(Scene):
        def construct(self):
            self.add(SkSLMobject("""
                uniform float2 u_resolution;
                half4 main(float2 fragCoord) {
                    float2 uv = fragCoord / u_resolution;
                    return half4(uv.x, uv.y, 0.5, 1.0);
                }
            """))

The backend injects ``u_resolution`` when it is not supplied.  Invalid SkSL
raises ``RuntimeError`` with the compiler diagnostic.

Known limitations
-----------------

- The preview window is read-only.  ``interactive_embed()`` warns and returns;
  mouse and keyboard events are not forwarded.
- ``fullscreen``, ``window_position`` and ``window_monitor`` are applied by the
  native preview window, but HiDPI scaling may differ from Cairo's preview.
- An explicit Cairo request requires the optional ``legacy`` dependencies
  (``pip install manim[legacy]`` or ``uv sync --extra legacy``).
- Prebuilt native wheels currently target Windows x64.  Building on Linux
  requires a Vulkan loader and the Skia binary cache; macOS additionally needs
  MoltenVK.  See ``native/manim_vk/README.md`` for build commands.
