Vulkan / Skia backend
====================

ManimVK renders the classic Manim object model through Skia's Ganesh backend on
top of Vulkan, its only renderer.  Scene scripts written for Manim Community run
unchanged.

Default behaviour
-----------------

The OpenGL renderer, the ``OpenGL*`` mobject family and the GLSL
``Shader``/``Mesh`` API are removed; custom effects use :class:`.SkSLMobject`.
``config.renderer`` and ``--renderer`` still accept ``"cairo"`` and
``"vulkan"``, as Manim Community scripts and configs use them; both draw with
Skia/Vulkan.

The renderer keeps one Vulkan device per process and one Skia Ganesh context per
device.  Vector paths, text, images, point clouds, background-filled
shapes and ``MultiCamera`` sub-views all draw and composite on the GPU.  A frame
is copied to host memory only when a video writer, still image, or preview
needs pixels.

``SkiaRenderer.memory_usage()`` reports the GPU memory in use, and
``tools/bench/frame_cost.py`` prints it per scene.

Selecting a GPU
---------------

Set ``MANIM_VK_DEVICE`` to a case-insensitive substring of the Vulkan adapter
name, for example::

    set MANIM_VK_DEVICE=RTX 3050
    manim -qm scene.py Scene

Without the variable, a discrete GPU is preferred, then an integrated GPU, then a
CPU device such as lavapipe.  If no Vulkan physical device exists the renderer
raises ``RuntimeError`` with installation guidance.

Depth in 3D scenes
------------------

In a :class:`.ThreeDScene`, mobjects with ``shade_in_3d`` and mobjects marked
with :meth:`.Mobject.apply_depth_test` are drawn back to front.  A depth-tested
:class:`.VMobject` is ordered in parts: every cubic curve of a stroke, or
every subpath when it has a fill (see
:meth:`.VMobject.get_depth_piece_ranges`).  One path holding thousands of short
strokes, such as the hairs of a vector field on a sphere, or one line running
around a sphere, is partly hidden behind the sphere and partly in front of it.

The camera sorts the pieces into :attr:`.ThreeDCamera.depth_slices` slices of
the frame's depth range (far to near; inside a slice by top-level mobject,
then by mobject) and returns the order as arrays.  The pieces of one mobject
in one slice form a run.  The renderer sends a frame's VMobjects and mesh
surfaces, each once, and the runs that draw them in that order to the native
module in one call, so a field of 50 000 arrows costs one Python-to-native
call per frame, not one per arrow.  A run is drawn as one path per 128-pixel
screen tile, small enough for Skia's GPU path atlas.

:class:`.MeshSurface` draws a :class:`.Surface` as one smoothly lit triangle
mesh and :class:`.TexturedSurface` stretches an image over it, optionally with
a second image on the side facing away from the light.  Their shading is
ManimGL's: an untextured surface is lit per vertex and a textured one per
pixel, with the day and night images mixed across the terminator.

Mesh surfaces get exact depth from a Vulkan depth buffer.  Each frame, a small
Vulkan render pass draws the mesh surfaces with a depth test into two depth
images: the nearest opaque surface at every pixel, and the nearest surface of
any opacity.  Meshes draw first, opaque before translucent.  An opaque
surface is discarded per pixel behind a nearer opaque one, however close (an
Earth 0.1% outside a translucent sphere) or intersecting they are.  A
translucent surface is discarded behind any nearer surface, so it shows only
its near side, blended once, and an opaque surface behind it shows through.
Every 3D run of VMobjects is clipped per pixel to where no mesh surface is
nearer than its nearest piece, as a depth buffer would hide it: a
translucent surface hides the depth-tested mobjects behind it and shows the
camera background through itself.  Unlike ManimGL, this does not depend on
the order mobjects were added.

Pieces that cannot be seen are not drawn at all: as in Unity's Burst
Occlusion Culling, the mesh surfaces are rasterised each frame on the CPU
with Intel's Masked Occlusion Culling into a conservative hierarchical depth
buffer, and a chain of pieces whose screen bounds lie outside the frame, or
wholly behind the surfaces at its run's depth, is skipped before its path is
built.  The test only culls what the per-pixel test would hide entirely, so
the frame is the same with culling off
(``SkiaRenderer.set_occlusion_culling(False)``); ``take_draw_stats()`` counts
the chains tested and culled.  The library is built for x86; elsewhere
nothing is culled.

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
  native preview window; HiDPI scaling may differ from Manim Community's.
- ``ImageMobject.pixel_array`` can be written in place, as in Manim Community:
  writes by index or with in-place operators upload the new pixels.
  ``np.copyto`` and ``out=`` arguments are not noticed; assign
  ``pixel_array`` after using them.
- The native modules ``manim_core`` and ``manim_vk`` are built from
  ``native/``; no prebuilt wheels are published.  Each reports an interface
  version, and ``import manim`` refuses a module built from older sources:
  rebuild both after pulling changes to ``native/``.  Windows x64 is built
  locally and Linux in CI (on lavapipe); macOS additionally needs MoltenVK.
  See ``native/manim_vk/README.md`` for build commands.
