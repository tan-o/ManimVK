"""Skia/Vulkan renderer for the classic (Cairo-family) Manim mobjects.

The public protocol is inherited from :class:`CairoRenderer`; only the raster
target and the drawing context are replaced.  This keeps Scene, Camera,
Animation, SceneFileWriter and every classic Mobject unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import manim_vk
from PIL import Image

from manim import logger
from manim.mobject.mobject import Mobject
from manim.mobject.types.image_mobject import ImageMobjectFromCamera
from manim.camera import Camera, MultiCamera
from manim.renderer.cairo.renderer import CairoRenderer
from manim.renderer.cairo.target import _ImageFiles
from manim.renderer.protocol import RendererCapabilities, RendererState
from manim.utils.iterables import list_update

from .rendering import SkiaDrawingContext
from .target import SkiaRenderTarget

if TYPE_CHECKING:
    from manim.scene.scene import Scene
    from manim.typing import RGBAPixelArray

__all__ = ["SkiaRenderer"]


class SkiaRenderer(CairoRenderer):
    """Render classic Manim mobjects through Skia's Ganesh Vulkan backend."""

    capabilities = RendererCapabilities(live_preview=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vk_device = None
        self._skia_targets: list[SkiaRenderTarget] = []
        self._snapshot_renderer: SkiaRenderer | None = None
        self.window = None

    def init_scene(self, scene: Scene, session_spec) -> None:
        super().init_scene(scene, session_spec)
        if session_spec.presentation.live_preview:
            from manim import config
            from manim.renderer.vulkan.window import VulkanWindow
            from manim.renderer.vulkan.window_settings import _WindowSettings

            # Create the offscreen device first so the preview can ask wgpu
            # for the same adapter instead of silently opening a second GPU.
            self._ensure_device()
            self.window = VulkanWindow(
                self,
                _settings=_WindowSettings.from_config(config),
            )

    def _ensure_device(self):
        if self._vk_device is None:
            try:
                self._vk_device = manim_vk.Device("ManimCE")
            except Exception as error:
                raise RuntimeError(
                    "Unable to create a Vulkan device. Install a working Vulkan "
                    "driver (NVIDIA/AMD/Intel) or a software ICD such as lavapipe/"
                    "SwiftShader, and optionally select an adapter with "
                    "MANIM_VK_DEVICE."
                ) from error
            logger.info("Vulkan render device: %s", self._vk_device.device_name())
        return self._vk_device

    def _target_for(
        self,
        settings,
        *,
        image_files: _ImageFiles,
    ) -> SkiaRenderTarget:
        device = self._ensure_device()
        target = SkiaRenderTarget(
            settings,
            image_files,
            device=device,
        )
        self._skia_targets.append(target)
        return target

    def _get_target(self) -> SkiaRenderTarget:
        self._ensure_open()
        self._state = RendererState.OPEN
        if self._target is None:
            self._target = self._target_for(
                self._raster_settings,
                image_files=self._image_files,
            )
        return self._target

    def _sub_target_for(
        self,
        image_mobject: ImageMobjectFromCamera,
        *,
        parent_camera: Camera,
        parent_target: SkiaRenderTarget,
    ) -> SkiaRenderTarget:
        parent_settings = parent_target.settings
        pixel_height = max(
            1,
            int(parent_settings.pixel_height * image_mobject.height / parent_camera.frame_height),
        )
        pixel_width = max(
            1,
            int(parent_settings.pixel_width * image_mobject.width / parent_camera.frame_width),
        )
        key = id(image_mobject)
        target = self._sub_targets.get(key)
        if target is not None and (
            target.settings.pixel_width != pixel_width
            or target.settings.pixel_height != pixel_height
        ):
            target.close()
            target = None
        if target is None:
            target = self._target_for(
                parent_settings.resized(
                    pixel_width=pixel_width,
                    pixel_height=pixel_height,
                ),
                image_files=self._image_files,
            )
            self._sub_targets[key] = target
        return target

    def _render_camera(
        self,
        *,
        camera: Camera,
        target: SkiaRenderTarget,
        mobjects: Iterable[Mobject],
        include_submobjects: bool,
        excluded_mobjects: list[Mobject] | None,
        camera_stack: tuple[int, ...],
    ) -> None:
        camera_id = id(camera)
        if camera_id in camera_stack:
            raise RuntimeError("Skia camera views cannot contain a composition cycle.")
        next_stack = (*camera_stack, camera_id)
        mobject_list = list(mobjects)
        camera_view_targets: dict[int, SkiaRenderTarget] = {}

        if isinstance(camera, MultiCamera):
            for image_mobject in camera.image_mobjects_from_cameras:
                sub_target = self._sub_target_for(
                    image_mobject,
                    parent_camera=camera,
                    parent_target=target,
                )
                sub_target.reset(image_mobject.camera)
                sub_excluded_mobjects = list_update(
                    list(excluded_mobjects or []),
                    [image_mobject],
                )
                self._render_camera(
                    camera=image_mobject.camera,
                    target=sub_target,
                    mobjects=mobject_list,
                    include_submobjects=include_submobjects,
                    excluded_mobjects=sub_excluded_mobjects,
                    camera_stack=next_stack,
                )
                # Keep the finished sub-camera surface on the GPU; the parent
                # target samples it as a texture during composition.
                camera_view_targets[id(image_mobject)] = sub_target
                camera_view_targets[id(image_mobject.camera)] = sub_target
                # ``_draw_frame`` prunes sub-targets that are absent from this
                # bookkeeping dict.  Register the view without materialising
                # pixels; the resolver below returns the GPU target itself.
                self._camera_view_pixels[id(image_mobject)] = None
                self._camera_view_pixels[id(image_mobject.camera)] = None

        def resolve_image(
            image_mobject: ImageMobjectFromCamera,
        ) -> SkiaRenderTarget | RGBAPixelArray | None:
            target = camera_view_targets.get(id(image_mobject))
            if target is None:
                target = camera_view_targets.get(id(image_mobject.camera))
            if target is not None:
                return target
            return self._camera_view_pixels.get(id(image_mobject)) or self._camera_view_pixels.get(
                id(image_mobject.camera)
            )

        SkiaDrawingContext(
            camera=camera,
            target=target,
            image_resolver=resolve_image,
        ).draw(
            mobject_list,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )
        # Leave the finished frame in the GPU surface.  commit() is deferred
        # until a caller actually needs pixels (video/still/preview).  This
        # removes one GPU-to-CPU readback from evaluation and preview paths and
        # avoids doing it twice when a manager asks for get_frame().

    def update_frame(
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        *,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> None:
        """Draw the scene's mobjects.

        A frame starts from the play's static layer, which stays on the GPU,
        instead of uploading the static image as the Cairo renderer copies it.
        """
        self._ensure_open()
        if self.skip_animations and not ignore_skipping:
            return
        self._final_scene = None
        if not mobjects:
            mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        target = self._get_target()
        if not target.restore_static_layer():
            target.reset(self.camera)
        self._draw_frame(
            camera=self.camera,
            mobjects=mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )

    def save_static_frame_data(
        self,
        scene: Scene,
        static_mobjects: Iterable[Mobject],
    ) -> None:
        """Draw the play's static mobjects once and keep them on the GPU."""
        self._ensure_open()
        self.discard_static_frame()
        # Same rule as the Cairo renderer: nested camera views can show any
        # moving mobject, so such a play draws everything every frame.
        self._render_all_mobjects = (
            isinstance(self.camera, MultiCamera)
            and bool(self.camera.image_mobjects_from_cameras)
            and bool(scene.moving_mobjects)
        )
        if self._render_all_mobjects or not static_mobjects:
            return
        self.update_frame(scene, mobjects=static_mobjects)
        self._get_target().capture_static_layer()

    def discard_static_frame(self) -> None:
        if self._target is not None and not self._target._closed:
            self._target.clear_static_layer()

    def _animation_cache_identity(self, scene: Scene) -> tuple[str, Any]:
        # Its own backend name, so Cairo and Vulkan segments never share a cache key.
        return "vulkan", ()

    def _present_frame(self, scene: Scene, frame_offset: float) -> None:
        if self.window is not None:
            self.window.swap_buffers()

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None:
        if self.window is not None:
            self.window.swap_buffers()

    def _new_snapshot_renderer(self) -> SkiaRenderer:
        """A renderer for snapshots that shares this renderer's Vulkan device."""
        renderer = SkiaRenderer(camera=self.camera, _raster_settings=self._raster_settings)
        renderer._vk_device = self._ensure_device()
        return renderer

    @contextmanager
    def _scene_images(self) -> Iterator[None]:
        """Keep one snapshot renderer open while an evaluation requests images."""
        renderer = self._new_snapshot_renderer()
        self._snapshot_renderer = renderer
        try:
            yield
        finally:
            self._snapshot_renderer = None
            renderer.close()

    def _get_scene_image(self, scene: Scene) -> Image.Image:
        mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        if self._snapshot_renderer is not None:
            self._snapshot_renderer.render_mobjects(mobjects)
            return self._snapshot_renderer.get_image()
        with self._new_snapshot_renderer() as renderer:
            renderer.render_mobjects(mobjects)
            return renderer.get_image()

    def scene_finished(self, scene: Scene) -> None:
        """Finish video output and honour the interactive-static contract.

        An interactive scene with no animations should leave only the live
        preview visible; the classic renderer's automatic still-frame fallback
        must not write a PNG into ``media/images``.
        """
        self._ensure_open()
        output = self.file_writer.output_spec
        if self.num_plays and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        self.discard_static_frame()
        self._final_scene = scene
        if output.is_still or (
            not self.num_plays
            and output.fallback_to_still
            and not getattr(scene, "interactive_mode", False)
        ):
            self.file_writer.save_image(self.get_frame())

    def close(self) -> None:
        if getattr(self, "window", None) is not None:
            self.window.close()
            self.window = None
        super().close()
        self._skia_targets.clear()
