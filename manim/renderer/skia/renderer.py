"""Skia/Vulkan renderer for the classic (Cairo-family) Manim mobjects.

The public protocol is inherited from :class:`CairoRenderer`; only the raster
target and the drawing context are replaced.  This keeps Scene, Camera,
Animation, SceneFileWriter and every classic Mobject unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import manim_vk
from PIL import Image

from manim import logger
from manim.mobject.mobject import Mobject
from manim.mobject.types.image_mobject import ImageMobjectFromCamera
from manim.renderer.cairo.camera import Camera, MultiCamera
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
        self.window = None
        # ``Shader`` objects in user scenes are accepted with this value; the
        # Skia mesh adapter reads raw vertex attributes and never compiles GLSL.
        self.context = None

    def init_scene(self, scene: Scene, session_spec) -> None:
        super().init_scene(scene, session_spec)
        if session_spec.presentation.live_preview:
            from manim import config
            from manim.renderer.vulkan.window import VulkanWindow
            from manim.renderer.vulkan.window_settings import _WindowSettings

            self.window = VulkanWindow(
                self,
                _settings=_WindowSettings.from_config(config),
            )

    def _target_for(
        self,
        settings,
        *,
        image_files: _ImageFiles,
    ) -> SkiaRenderTarget:
        if self._vk_device is None:
            self._vk_device = manim_vk.Device("ManimCE")
            logger.info("Vulkan render device: %s", self._vk_device.device_name())
        target = SkiaRenderTarget(
            settings,
            image_files,
            device=self._vk_device,
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
        camera_view_pixels: dict[int, RGBAPixelArray] = {}

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
                # A composition needs the sub-camera image as CPU pixels;
                # normal single-camera frames stay on the GPU until get_frame().
                sub_target.commit()
                camera_view_pixels[id(image_mobject)] = sub_target.pixels
                camera_view_pixels[id(image_mobject.camera)] = sub_target.pixels

        def resolve_image(
            image_mobject: ImageMobjectFromCamera,
        ) -> RGBAPixelArray | None:
            pixels = self._camera_view_pixels.get(id(image_mobject))
            if pixels is None:
                pixels = camera_view_pixels.get(id(image_mobject.camera))
            return pixels

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
        """Draw classic mobjects and then any public ``Mesh`` objects."""
        from manim.renderer.opengl.shader import Object3D

        if mobjects is None:
            mobjects = list_update(scene.mobjects, scene.foreground_mobjects)
        classic_mobjects: list[Mobject] = []
        mesh_roots: list[Object3D] = []
        for mob in mobjects:
            if isinstance(mob, Object3D):
                mesh_roots.append(mob)
            else:
                classic_mobjects.append(mob)
        for mesh in getattr(scene, "meshes", []):
            if mesh not in mesh_roots:
                mesh_roots.append(mesh)

        super().update_frame(
            scene,
            classic_mobjects,
            include_submobjects,
            ignore_skipping,
            excluded_mobjects=excluded_mobjects,
        )
        if self.skip_animations and not ignore_skipping:
            return
        if mesh_roots:
            from .mesh import draw_meshes

            draw_meshes(
                mesh_roots,
                camera=self.camera,
                target=self._get_target(),
            )

    def _present_frame(self, scene: Scene, frame_offset: float) -> None:
        if self.window is not None:
            self.window.swap_buffers()

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None:
        if self.window is not None:
            self.window.swap_buffers()

    def _get_scene_image(self, scene: Scene) -> Image.Image:
        with SkiaRenderer(
            camera=self.camera,
            _raster_settings=self._raster_settings,
        ) as renderer:
            renderer.render_mobjects(
                list_update(scene.mobjects, scene.foreground_mobjects)
            )
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
        self.static_image = None
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
