"""The renderer: Manim's mobjects drawn through Skia's Ganesh Vulkan backend."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

import manim_vk
from PIL import Image

from manim import config, logger
from manim._run.frames import frame_count
from manim.camera import Camera, MultiCamera, ThreeDCamera
from manim.mobject.mobject import Mobject, _AnimationBuilder
from manim.mobject.types.image_mobject import ImageMobjectFromCamera
from manim.renderer._execution import _RendererExecutionView
from manim.renderer.protocol import RendererCapabilities, RendererState
from manim.scene.scene_file_writer import SceneFileWriter
from manim.utils.iterables import list_update

from .rendering import SkiaDrawingContext
from .target import GpuMemory, SkiaRenderTarget, _ImageFiles, _RasterSettings

if TYPE_CHECKING:
    from manim._config.render_session import RenderSessionSpec
    from manim.animation.animation import Animation
    from manim.scene.scene import Scene
    from manim.scene.video_segment_encoder import YUV420Frame
    from manim.typing import RGBAPixelArray

__all__ = ["SkiaRenderer"]


class SkiaRenderer(_RendererExecutionView):
    """Draw a scene's mobjects on the GPU and read frames back on request.

    The camera's view, and the views of a :class:`.MultiCamera`, are drawn
    into Skia surfaces on one Vulkan device.  Each play's static mobjects are
    drawn once and kept on the GPU as the background of its frames.  The
    device is created on the first draw.
    """

    capabilities = RendererCapabilities(live_preview=True)

    def __init__(
        self,
        file_writer_class: type[SceneFileWriter] = SceneFileWriter,
        camera_class: type[Camera] | None = None,
        camera: Camera | None = None,
        skip_animations: bool = False,
        *,
        _raster_settings: _RasterSettings | None = None,
    ) -> None:
        if camera is not None and camera_class is not None:
            raise ValueError("Pass either camera or camera_class, not both.")
        self._file_writer_class = file_writer_class
        if camera is None:
            camera = Camera() if camera_class is None else camera_class()
        self.camera = camera
        self._initialize_execution(skip_animations)
        self._raster_settings = _raster_settings or _RasterSettings(
            pixel_width=config.pixel_width,
            pixel_height=config.pixel_height,
            base_pixel_width=config.pixel_width,
            base_pixel_height=config.pixel_height,
        )
        # Image files read for this renderer, shared by all its targets.
        self._image_files = _ImageFiles()
        self._vk_device = None
        self._target: SkiaRenderTarget | None = None
        # Targets of MultiCamera views, by the id of their ImageMobjectFromCamera,
        # and the views drawn in the current frame.
        self._sub_targets: dict[int, SkiaRenderTarget] = {}
        self._occlusion_culling = True
        self._drawn_views: set[int] = set()
        # A finished scene whose final state is drawn on the first pixel read.
        self._final_scene: Scene | None = None
        self._render_all_mobjects = False
        self._snapshot_renderer: SkiaRenderer | None = None
        self.window = None
        self._state = RendererState.UNOPENED

    @property
    def state(self) -> RendererState:
        """Where the renderer is in its life (read-only)."""
        return self._state

    def _ensure_open(self) -> None:
        """Check that the renderer is not closed."""
        if not self._state.accepts_work:
            raise RuntimeError(f"The renderer is {self._state.value}.")

    def init_scene(self, scene: Scene, session_spec: RenderSessionSpec) -> None:
        self._ensure_open()
        if hasattr(self, "_scene"):
            raise RuntimeError("This renderer is already bound to a Scene.")
        self._scene = scene
        if session_spec.presentation.live_preview:
            from .window import VulkanWindow
            from .window_settings import _WindowSettings

            # Create the offscreen device first so the preview can ask wgpu
            # for the same adapter instead of silently opening a second GPU.
            self._ensure_device()
            self.window = VulkanWindow(
                self, _settings=_WindowSettings.from_config(config)
            )

    @property
    def file_writer(self) -> SceneFileWriter:
        """Return the scene manager's file writer, creating it if needed."""
        return self._scene._get_manager().file_writer

    def play(
        self,
        scene: Scene,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> None:
        """Delegate playback to the scene's manager; prefer Scene.play for new code."""
        scene._get_manager().play(*args, **kwargs)

    # -- device and targets ------------------------------------------------

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

    def _get_target(self) -> SkiaRenderTarget:
        self._ensure_open()
        self._state = RendererState.OPEN
        if self._target is None:
            self._target = SkiaRenderTarget(
                self._raster_settings, self._image_files, device=self._ensure_device()
            )
            self._target.set_occlusion_culling(self._occlusion_culling)
        return self._target

    def memory_usage(self) -> GpuMemory:
        """GPU memory in use now (see :class:`GpuMemory`)."""
        return self._get_target().memory_usage()

    def set_occlusion_culling(self, enabled: bool) -> None:
        """Whether depth pieces that cannot be seen are left out of drawing,
        for every target, now and later, snapshot renderers included (see
        :meth:`SkiaRenderTarget.set_occlusion_culling`); on by default."""
        self._occlusion_culling = bool(enabled)
        for target in [self._target, *self._sub_targets.values()]:
            if target is not None:
                target.set_occlusion_culling(enabled)
        if self._snapshot_renderer is not None:
            self._snapshot_renderer.set_occlusion_culling(enabled)

    def take_draw_stats(self) -> dict[str, float]:
        """See :meth:`SkiaRenderTarget.take_draw_stats`: summed over the main
        target and, while an evaluation draws through one, the snapshot
        renderer."""
        stats = self._get_target().take_draw_stats()
        if self._snapshot_renderer is not None:
            for key, value in self._snapshot_renderer.take_draw_stats().items():
                stats[key] = stats.get(key, 0.0) + value
        return stats

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
            int(
                parent_settings.pixel_height
                * image_mobject.height
                / parent_camera.frame_height
            ),
        )
        pixel_width = max(
            1,
            int(
                parent_settings.pixel_width
                * image_mobject.width
                / parent_camera.frame_width
            ),
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
            target = SkiaRenderTarget(
                parent_settings.resized(
                    pixel_width=pixel_width, pixel_height=pixel_height
                ),
                self._image_files,
                device=self._ensure_device(),
            )
            target.set_occlusion_culling(self._occlusion_culling)
            self._sub_targets[key] = target
        return target

    # -- drawing -----------------------------------------------------------

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
            raise RuntimeError("Camera views cannot contain a composition cycle.")
        next_stack = (*camera_stack, camera_id)
        mobject_list = list(mobjects)
        # Finished views of this camera, by the id of their image mobject and
        # of their camera: copies of a registered view show their camera's.
        views: dict[int, SkiaRenderTarget] = {}

        if isinstance(camera, MultiCamera):
            for image_mobject in camera.image_mobjects_from_cameras:
                sub_target = self._sub_target_for(
                    image_mobject, parent_camera=camera, parent_target=target
                )
                sub_target.reset(image_mobject.camera)
                self._render_camera(
                    camera=image_mobject.camera,
                    target=sub_target,
                    mobjects=mobject_list,
                    include_submobjects=include_submobjects,
                    excluded_mobjects=list_update(
                        list(excluded_mobjects or []), [image_mobject]
                    ),
                    camera_stack=next_stack,
                )
                # The finished surface stays on the GPU; the parent samples it.
                views[id(image_mobject)] = sub_target
                views[id(image_mobject.camera)] = sub_target
                self._drawn_views.add(id(image_mobject))

        def resolve_image(
            image_mobject: ImageMobjectFromCamera,
        ) -> SkiaRenderTarget | None:
            return views.get(id(image_mobject)) or views.get(id(image_mobject.camera))

        SkiaDrawingContext(
            camera=camera, target=target, image_resolver=resolve_image
        ).draw(
            mobject_list,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )

    def _draw_frame(
        self,
        *,
        camera: Camera,
        mobjects: Iterable[Mobject],
        include_submobjects: bool = True,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> None:
        self._drawn_views.clear()
        try:
            self._render_camera(
                camera=camera,
                target=self._get_target(),
                mobjects=mobjects,
                include_submobjects=include_submobjects,
                excluded_mobjects=excluded_mobjects,
                camera_stack=(),
            )
        finally:
            # Retire the targets of removed views and of views whose drawing
            # failed before it finished.
            for key in self._sub_targets.keys() - self._drawn_views:
                self._sub_targets.pop(key).close()

    def update_frame(
        self,
        scene: Scene,
        mobjects: Iterable[Mobject] | None = None,
        include_submobjects: bool = True,
        ignore_skipping: bool = True,
        *,
        excluded_mobjects: list[Mobject] | None = None,
    ) -> None:
        """Draw the scene's mobjects, from the play's static layer if it has one.

        ``excluded_mobjects`` and their families are left out of the frame.
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

    def render_mobjects(
        self, mobjects: Iterable[Mobject], *, camera: Camera | None = None
    ) -> None:
        """Draw the supplied mobjects on a cleared frame."""
        self._ensure_open()
        render_camera = self.camera if camera is None else camera
        self._final_scene = None
        self._get_target().reset(render_camera)
        self._draw_frame(camera=render_camera, mobjects=mobjects)

    def render(
        self,
        scene: Scene,
        time: float,
        moving_mobjects: Iterable[Mobject] | None = None,
    ) -> None:
        if self._render_all_mobjects:
            moving_mobjects = None
        # Manager advances animation time and sends the pixels to the writer.
        self.update_frame(scene, moving_mobjects)

    # -- plays ---------------------------------------------------------------

    def _animation_cache_identity(self, scene: Scene) -> tuple[str, Any]:
        return "vulkan", ()

    def _start_animation(self) -> None:
        self._ensure_open()

    def _prepare_animation(self, scene: Scene) -> None:
        self.save_static_frame_data(scene, scene.static_mobjects)

    def _present_frame(self, scene: Scene, frame_offset: float) -> None:
        if self.window is not None:
            self.window.swap_buffers()

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None:
        if self.window is not None:
            self.window.swap_buffers()

    def save_static_frame_data(
        self, scene: Scene, static_mobjects: Iterable[Mobject]
    ) -> None:
        """Draw the play's static mobjects once and keep them on the GPU."""
        self._ensure_open()
        self.discard_static_frame()
        # A nested camera view can show any moving mobject, so such a play
        # draws everything every frame.
        self._render_all_mobjects = (
            isinstance(self.camera, MultiCamera)
            and bool(self.camera.image_mobjects_from_cameras)
            and bool(scene.moving_mobjects)
        ) or not self._static_layer_keeps_draw_order(scene)
        if self._render_all_mobjects or not static_mobjects:
            return
        self.update_frame(scene, mobjects=static_mobjects)
        self._get_target().capture_static_layer()

    def _static_layer_keeps_draw_order(self, scene: Scene) -> bool:
        """Whether drawing the static mobjects first and the moving ones on top
        gives the frame's own drawing order.

        The static layer is under every moving mobject, so it may only hold
        mobjects that draw before all of them.  In a depth-sorting camera,
        moving 3D mobjects interleave with static ones by depth (and flat
        mobjects draw after every 3D one), so no static layer is kept; with
        ``use_z_index``, a static mobject may draw above a moving one.
        """
        moving = {id(mobject) for mobject in scene.moving_mobjects}
        if not moving:
            return True
        if isinstance(self.camera, ThreeDCamera) and any(
            mobject.depth_test or getattr(mobject, "shade_in_3d", False)
            for mobject in scene.moving_mobjects
        ):
            return False
        order = self.camera.get_mobjects_to_display(
            list_update(scene.mobjects, scene.foreground_mobjects)
        )
        seen_moving = False
        for mobject in order:
            if id(mobject) in moving:
                seen_moving = True
            elif seen_moving:
                return False
        return True

    def discard_static_frame(self) -> None:
        """Forget the static background of the finished play."""
        if self._target is not None and not self._target._closed:
            self._target.clear_static_layer()

    def update_skipping_status(self) -> None:
        self._scene._get_manager()._update_skipping_status()

    def scene_finished(self, scene: Scene) -> None:
        """Finish the video, and write the still a still-only scene asks for.

        An interactive scene with no animations leaves only the live preview:
        it does not fall back to writing a PNG.
        """
        self._ensure_open()
        output = self.file_writer.output_spec
        if self.num_plays and (output.is_video or output.is_image_sequence):
            self.file_writer.finish()
        self.discard_static_frame()
        # Skipped and cached plays do not draw, so the surface may not show the
        # final state.  Draw it when the pixels are read, not for every scene.
        self._final_scene = scene
        if output.is_still or (
            not self.num_plays
            and output.fallback_to_still
            and not getattr(scene, "interactive_mode", False)
        ):
            self.file_writer.save_image(self.get_frame())

    # -- frames --------------------------------------------------------------

    def _draw_final_state(self) -> None:
        scene, self._final_scene = self._final_scene, None
        if scene is not None:
            self.update_frame(scene)

    def get_frame(self) -> RGBAPixelArray:
        """Read the current image back as RGBA, with row zero at the top."""
        self._draw_final_state()
        return self._get_target().read_pixels()

    def request_yuv420_frame(self) -> Callable[[], YUV420Frame] | None:
        """Start reading the current image as YUV 4:2:0, converted on the
        GPU; see :meth:`.SkiaRenderTarget.request_yuv420`.

        Only opaque frames convert: with a translucent background, alpha
        decides what the encoder keeps, so the RGBA frame is read instead.
        """
        if self.camera.background_opacity < 1:
            return None
        self._draw_final_state()
        return self._get_target().request_yuv420()

    def get_image(self) -> Image.Image:
        """Return the current image as a PIL image."""
        return Image.fromarray(self.get_frame())

    def add_frame(self, frame: RGBAPixelArray, num_frames: int = 1) -> None:
        self._ensure_open()
        self._scene._get_manager()._legacy_add_frame(frame, num_frames)

    def freeze_current_frame(self, duration: float) -> None:
        self._ensure_open()
        manager = self._scene._get_manager()
        # An evaluation draws frames only when they are requested, without a target.
        manager._legacy_add_frame(
            self.get_frame() if manager._mode.draws_added_frames else None,
            frame_count(duration, self._scene.session_spec.frame_rate),
        )

    def show_frame(self, scene: Scene) -> None:
        self.update_frame(scene, ignore_skipping=True)
        self.get_image().show()

    def _new_snapshot_renderer(self) -> SkiaRenderer:
        """A renderer for snapshots that shares this renderer's Vulkan device."""
        renderer = SkiaRenderer(
            camera=self.camera, _raster_settings=self._raster_settings
        )
        renderer._vk_device = self._ensure_device()
        renderer._occlusion_culling = self._occlusion_culling
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

    # -- life ----------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def open(self) -> None:
        """Check that the renderer can draw.

        Targets are created on the first draw, so there is nothing to allocate
        here; callers can open the renderer before :meth:`.Scene.setup` runs.
        """
        self._ensure_open()
        self._state = RendererState.OPEN

    def close(self) -> None:
        """Close the preview window and release the targets.

        If releasing fails, for example on an interruption, the renderer is
        ``CLOSE_FAILED`` and ``close()`` can be called again.
        """
        if self._state is RendererState.CLOSED:
            return
        self._state = RendererState.CLOSING
        # Release everything before raising a failure or interruption. A target
        # that did not close stays, so that calling close() again closes it.
        failures: list[BaseException] = []

        def release(target: SkiaRenderTarget) -> bool:
            try:
                target.close()
            except BaseException as error:
                failures.append(error)
                return False
            return True

        if self.window is not None:
            try:
                self.window.close()
                self.window = None
            except BaseException as error:
                failures.append(error)
        if self._target is not None:
            release(self._target)
        for key, target in list(self._sub_targets.items()):
            if release(target):
                del self._sub_targets[key]
        self._drawn_views.clear()
        self._image_files.clear()
        self._render_all_mobjects = False
        self._final_scene = None
        if failures:
            self._state = RendererState.CLOSE_FAILED
            for secondary in failures[1:]:
                logger.error("Additional renderer cleanup failure", exc_info=secondary)
            raise failures[0]
        self._state = RendererState.CLOSED
