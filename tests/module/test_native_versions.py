from __future__ import annotations

import importlib

import manim_vk
import pytest

import manim._native


def test_compiled_modules_match_this_manim():
    importlib.reload(manim._native)


def test_a_stale_build_stops_the_import(monkeypatch):
    monkeypatch.setattr(manim_vk, "API_VERSION", 0, raising=False)
    with pytest.raises(ImportError, match="maturin develop -r` in native/manim_vk"):
        importlib.reload(manim._native)
