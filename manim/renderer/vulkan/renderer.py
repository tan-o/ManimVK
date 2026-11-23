"""Compatibility name for the Vulkan renderer.

The Vulkan backend is implemented by :class:`~manim.renderer.skia.SkiaRenderer`,
which draws classic Manim mobjects through Skia's Ganesh Vulkan backend.  This
module keeps ``VulkanRenderer`` importable for code that used the old name,
without retaining the former OpenGL-family bridge.
"""

from manim.renderer.skia.renderer import SkiaRenderer

VulkanRenderer = SkiaRenderer

__all__ = ["VulkanRenderer"]
