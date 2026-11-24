import numpy as np
import pytest

from manim import ImageMobject


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_invert_image(dtype):
    rng = np.random.default_rng()
    array = (255 * rng.random((10, 10, 4))).astype(dtype)
    image = ImageMobject(array, pixel_array_dtype=dtype, invert=True)
    assert image.pixel_array.dtype == dtype

    array[:, :, :3] = np.iinfo(dtype).max - array[:, :, :3]
    assert np.allclose(array, image.pixel_array)


def _image(seed=0, shape=(6, 5, 4)):
    rng = np.random.default_rng(seed)
    return ImageMobject(rng.integers(0, 255, shape, dtype=np.uint8))


def test_writing_into_pixel_array_changes_the_texture_key():
    image = _image()
    key = image.texture_key
    image.pixel_array[0, 0, 0] = 1
    assert image.pixel_array[0, 0, 0] == 1
    assert image.texture_key != key

    # Through a view and an in-place operator, as Manim Community scripts do.
    key = image.texture_key
    alpha = image.pixel_array[:, :, 3]
    alpha //= 2
    assert image.texture_key != key

    key = image.texture_key
    image.pixel_array = np.zeros((6, 5, 4), dtype=np.uint8)
    assert image.texture_key != key


def test_reading_pixels_keeps_the_texture_key():
    image = _image()
    key = image.texture_key
    _ = image.pixel_array * 2
    image.set_opacity(0.5)
    image.get_pixel_array()
    assert image.texture_key == key


def test_copies_have_their_own_pixels_and_share_the_texture_until_written():
    image = _image()
    copy = image.copy()
    assert copy.pixel_array is not image.pixel_array
    np.testing.assert_array_equal(copy.pixel_array, image.pixel_array)
    assert copy.texture_key == image.texture_key
    copy.pixel_array[0, 0] = 7
    assert copy.texture_key != image.texture_key
    assert not (image.pixel_array[0, 0] == 7).all()


def test_opacity_is_applied_when_drawn():
    image = _image()
    source, key = image.pixel_array, image.texture_key
    image.set_opacity(0.5)
    assert image.pixel_array is source
    assert image.texture_key == key
    expected = source.astype(np.float64)
    expected[:, :, 3] *= 0.5
    np.testing.assert_array_equal(image.get_pixel_array(), expected.astype(np.uint8))


def test_fade_interpolates_opacity_only():
    target = _image()
    start = target.copy().fade(1)
    moving = start.copy()
    moving.interpolate_color(start, target, 0.25)
    assert moving.get_image_mix() is None
    assert moving.texture_key == target.texture_key
    assert moving.get_opacity() == 0.25


def test_transform_between_images_mixes_them():
    first, second = _image(1), _image(2)
    moving = first.copy()
    moving.interpolate_color(first, second, 0.3)
    pixels, key, alpha = moving.get_image_mix()
    assert pixels is second.pixel_array
    assert key == second.texture_key
    assert alpha == 0.3
    expected = (0.7 * first.pixel_array.view(np.ndarray) + 0.3 * second.pixel_array.view(np.ndarray)).astype(
        np.uint8
    )
    np.testing.assert_array_equal(moving.get_pixel_array(), expected)
    moving.interpolate_color(first, second, 1)
    assert moving.get_image_mix() is None
    assert moving.texture_key == second.texture_key


def test_set_color_replaces_the_texture():
    image = _image()
    key = image.texture_key
    image.set_color("#ff0000")
    assert image.texture_key != key
    assert (image.pixel_array[:, :, :3] == (255, 0, 0)).all()
