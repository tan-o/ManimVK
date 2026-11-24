"""Vulkan preview window backed by the native ``manim_vk`` extension."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ... import __version__, config, logger
from .window_settings import _WindowSettings

if TYPE_CHECKING:
    from .renderer import SkiaRenderer

__all__ = ["VulkanWindow"]


class VulkanWindow:
    """A native Vulkan swapchain window fed with the renderer's readback frame."""

    def __init__(
        self,
        renderer: SkiaRenderer,
        window_size: str | tuple[int, ...] | None = None,
        *,
        _settings: _WindowSettings | None = None,
        **kwargs: Any,
    ) -> None:
        import manim_vk

        self._settings = (
            _WindowSettings.from_config(config) if _settings is None else _settings
        )
        if window_size is None:
            window_size = self._settings.size
        if isinstance(window_size, tuple):
            if len(window_size) != 2:
                raise ValueError("window_size must be a 2-tuple or 'width,height'")
            size = (int(window_size[0]), int(window_size[1]))
        elif window_size == "default":
            size = (
                int(self._settings.frame_width * config.pixel_width / config.frame_width),
                int(config.pixel_height),
            )
        elif isinstance(window_size, str) and "," in window_size:
            size = tuple(int(part) for part in window_size.split(","))
            if len(size) != 2:
                raise ValueError("window_size must be a 2-tuple or 'width,height'")
        else:
            raise ValueError(
                "window_size must be a tuple (width, height), 'default', or 'width,height'"
            )
        self.renderer = renderer
        self.width, self.height = size
        device = getattr(renderer, "_vk_device", None)
        adapter_name = ""
        if device is not None:
            try:
                adapter_name = device.device_name()
            except Exception:
                adapter_name = ""
        self._native = manim_vk.NativeWindow(
            self.width,
            self.height,
            f"Manim Community {__version__}",
            bool(self._settings.fullscreen),
            str(self._settings.position or ""),
            max(0, int(self._settings.monitor)),
            adapter_name,
        )
        self.title = f"Manim Community {__version__}"
        self._closed = False
        self._adapter_logged = False
        logger.debug("Vulkan preview window created (%dx%d)", self.width, self.height)

    @property
    def is_closing(self) -> bool:
        if self._closed:
            return True
        try:
            return bool(self._native.is_closing())
        except Exception:
            return True

    def swap_buffers(self) -> None:
        """Present the renderer's latest top-down RGBA frame."""
        if self._closed or self._native.is_closing():
            error = self._native.error()
            if error and not self._closed:
                self._closed = True
                raise RuntimeError(error)
            return
        if not self._adapter_logged:
            adapter = self._native.adapter_name()
            if adapter:
                logger.info("Vulkan preview adapter: %s", adapter)
                device = getattr(self.renderer, "_vk_device", None)
                if device is not None:
                    try:
                        render_device = device.device_name()
                    except Exception:
                        render_device = ""
                    if render_device and adapter != render_device:
                        logger.warning(
                            "Vulkan preview adapter (%s) differs from the offscreen "
                            "render device (%s); dual-GPU systems may present on a "
                            "different adapter.",
                            adapter,
                            render_device,
                        )
                self._adapter_logged = True
        frame = self.renderer.get_frame()
        self._native.present_rgba(frame.tobytes())

    def get_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._native.close()
        except Exception:
            logger.exception("Failed to close Vulkan preview window")

    destroy = close

    def __enter__(self) -> "VulkanWindow":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
