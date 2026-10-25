from __future__ import annotations

import pytest

from PIL import Image
from PIL.Image import Transpose

from . import helper
from .helper import assert_image_equal

HOPPER = {
    mode: helper.hopper(mode).crop((0, 0, 121, 127)).copy()
    for mode in ["L", "RGB", "I;16", "I;16L", "I;16B"]
}


@pytest.mark.parametrize("mode", HOPPER)
def test_flip_left_right(mode: str) -> None:
    im = HOPPER[mode]
    out = im.transpose(Transpose.FLIP_LEFT_RIGHT)
    assert out.mode == mode
    assert out.size == im.size

    x, y = im.size
    assert im.getpixel((1, 1)) == out.getpixel((x - 2, 1))
    assert im.getpixel((x - 2, 1)) == out.getpixel((1, 1))
    assert im.getpixel((1, y - 2)) == out.getpixel((x - 2, y - 2))
    assert im.getpixel((x - 2, y - 2)) == out.getpixel((1, y - 2))


@pytest.mark.parametrize("mode", HOPPER)
def test_flip_top_bottom(mode: str) -> None:
    im = HOPPER[mode]
    out = im.transpose(Transpose.FLIP_TOP_BOTTOM)
    assert out.mode == mode
    assert out.size == im.size

    x, y = im.size
    assert im.getpixel((1, 1)) == out.getpixel((1, y - 2))
    assert im.getpixel((x - 2, 1)) == out.getpixel((x - 2, y - 2))
    assert im.getpixel((1, y - 2)) == out.getpixel((1, 1))
    assert im.getpixel((x - 2, y - 2)) == out.getpixel((x - 2, 1))


@pytest.mark.parametrize("mode", HOPPER)
def test_rotate_90(mode: str) -> None:
    im = HOPPER[mode]
    out = im.transpose(Transpose.ROTATE_90)
    assert out.mode == mode
    assert out.size == im.size[::-1]

    x, y = im.size
    assert im.getpixel((1, 1)) == out.getpixel((1, x - 2))
    assert im.getpixel((x - 2, 1)) == out.getpixel((1, 1))
    assert im.getpixel((1, y - 2)) == out.getpixel((y - 2, x - 2))
    assert im.getpixel((x - 2, y - 2)) == out.getpixel((y - 2, 1))


@pytest.mark.parametrize("mode", HOPPER)
def test_rotate_180(mode: str) -> None:
    im = HOPPER[mode]
    out = im.transpose(Transpose.ROTATE_180)
    assert out.mode == mode
    assert out.size == im.size

    x, y = im.size
    assert im.getpixel((1, 1)) == out.getpixel((x - 2, y - 2))
    assert im.getpixel((x - 2, 1)) == out.getpixel((1, y - 2))
    assert im.getpixel((1, y - 2)) == out.getpixel((x - 2, 1))
    assert im.getpixel((x - 2, y - 2)) == out.getpixel((1, 1))


@pytest.mark.parametrize("mode", HOPPER)
def test_rotate_270(mode: str) -> None:
    im = HOPPER[mode]
    out = im.transpose(Transpose.ROTATE_270)
    assert out.mode == mode
    assert out.size == im.size[::-1]

    x, y = im.size
    assert im.getpixel((1, 1)) == out.getpixel((y - 2, 1))
    assert im.getpixel((x - 2, 1)) == out.getpixel((y - 2, x - 2))
    assert im.getpixel((1, y - 2)) == out.getpixel((1, 1))
    assert im.getpixel((x - 2, y - 2)) == out.getpixel((1, x - 2))


@pytest.mark.parametrize("mode", HOPPER)
def test_transpose(mode: str) -> None:
    im = HOPPER[mode]
    out = im.transpose(Transpose.TRANSPOSE)
    assert out.mode == mode
    assert out.size == im.size[::-1]

    x, y = im.size
    assert im.getpixel((1, 1)) == out.getpixel((1, 1))
    assert im.getpixel((x - 2, 1)) == out.getpixel((1, x - 2))
    assert im.getpixel((1, y - 2)) == out.getpixel((y - 2, 1))
    assert im.getpixel((x - 2, y - 2)) == out.getpixel((y - 2, x - 2))


@pytest.mark.parametrize("mode", HOPPER)
def test_tranverse(mode: str) -> None:
    im = HOPPER[mode]
    out = im.transpose(Transpose.TRANSVERSE)
    assert out.mode == mode
    assert out.size == im.size[::-1]

    x, y = im.size
    assert im.getpixel((1, 1)) == out.getpixel((y - 2, x - 2))
    assert im.getpixel((x - 2, 1)) == out.getpixel((y - 2, 1))
    assert im.getpixel((1, y - 2)) == out.getpixel((1, x - 2))
    assert im.getpixel((x - 2, y - 2)) == out.getpixel((1, 1))


@pytest.mark.parametrize("mode", HOPPER)
def test_roundtrip(mode: str) -> None:
    im = HOPPER[mode]

    def transpose(first: Transpose, second: Transpose) -> Image.Image:
        return im.transpose(first).transpose(second)

    assert_image_equal(
        im, transpose(Transpose.FLIP_LEFT_RIGHT, Transpose.FLIP_LEFT_RIGHT)
    )
    assert_image_equal(
        im, transpose(Transpose.FLIP_TOP_BOTTOM, Transpose.FLIP_TOP_BOTTOM)
    )
    assert_image_equal(im, transpose(Transpose.ROTATE_90, Transpose.ROTATE_270))
    assert_image_equal(im, transpose(Transpose.ROTATE_180, Transpose.ROTATE_180))
    assert_image_equal(
        im.transpose(Transpose.TRANSPOSE),
        transpose(Transpose.ROTATE_90, Transpose.FLIP_TOP_BOTTOM),
    )
    assert_image_equal(
        im.transpose(Transpose.TRANSPOSE),
        transpose(Transpose.ROTATE_270, Transpose.FLIP_LEFT_RIGHT),
    )
    assert_image_equal(
        im.transpose(Transpose.TRANSVERSE),
        transpose(Transpose.ROTATE_90, Transpose.FLIP_LEFT_RIGHT),
    )
    assert_image_equal(
        im.transpose(Transpose.TRANSVERSE),
        transpose(Transpose.ROTATE_270, Transpose.FLIP_TOP_BOTTOM),
    )
    assert_image_equal(
        im.transpose(Transpose.TRANSVERSE),
        transpose(Transpose.ROTATE_180, Transpose.TRANSPOSE),
    )


def _pixels(im: Image.Image) -> memoryview:
    # The image's pixels as items of its pixel size
    data = memoryview(im.tobytes())
    if im.mode == "L":
        return data.cast("B")
    if im.mode == "I;16":
        return data.cast("H")
    return data.cast("I")


@pytest.mark.parametrize("mode", ("L", "RGBA", "I", "I;16"))
@pytest.mark.parametrize(
    "size", ((1, 1), (3, 5), (8, 8), (37, 29), (530, 13), (11, 523), (520, 515))
)
def test_rotations_match_columns(mode: str, size: tuple[int, int]) -> None:
    # Pixels that differ, in blocks and at the edges, across chunks
    xsize, ysize = size
    data = bytes((i * 7 + i // 251) & 255 for i in range(xsize * ysize * 4))
    im = Image.frombytes(
        mode, size, data[: xsize * ysize * {"L": 1, "I;16": 2}.get(mode, 4)]
    )
    pixels = _pixels(im)

    def column(x: int, reverse: bool) -> bytes:
        values = pixels[x::xsize]
        return values[::-1].tobytes() if reverse else values.tobytes()

    expected = {
        Transpose.TRANSPOSE: (False, False),
        Transpose.ROTATE_90: (True, False),
        Transpose.ROTATE_270: (False, True),
        Transpose.TRANSVERSE: (True, True),
    }
    for method, (flip_x, flip_y) in expected.items():
        out = im.transpose(method)
        assert out.size == (ysize, xsize)
        assert out.tobytes() == b"".join(
            column(xsize - 1 - x if flip_x else x, flip_y) for x in range(xsize)
        )


@pytest.mark.parametrize("mode", ("L", "RGBA", "I", "I;16"))
@pytest.mark.parametrize("size", ((1, 1), (3, 2), (15, 3), (16, 2), (17, 4), (37, 29)))
def test_flips_match_rows(mode: str, size: tuple[int, int]) -> None:
    xsize, ysize = size
    data = bytes((i * 7 + i // 251) & 255 for i in range(xsize * ysize * 4))
    im = Image.frombytes(
        mode, size, data[: xsize * ysize * {"L": 1, "I;16": 2}.get(mode, 4)]
    )
    pixels = _pixels(im)
    rows = [pixels[y * xsize : (y + 1) * xsize][::-1].tobytes() for y in range(ysize)]

    assert im.transpose(Transpose.FLIP_LEFT_RIGHT).tobytes() == b"".join(rows)
    assert im.transpose(Transpose.ROTATE_180).tobytes() == b"".join(rows[::-1])
