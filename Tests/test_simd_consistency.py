from __future__ import annotations

import random

import pytest

from PIL import Image

# SIMD kernels process several pixels at once. The pixels left over at the
# end of a row must get exactly the same arithmetic, so identical input
# pixels give identical output pixels wherever they are in the row.

WIDTHS = (1, 3, 5, 7, 9, 17)


def _colors(bands: int, count: int = 40) -> list[tuple[int, ...]]:
    rnd = random.Random(bands)
    return [tuple(rnd.randrange(256) for _ in range(bands)) for _ in range(count)]


def _assert_flat(im: Image.Image) -> None:
    pixels = list(im.get_flattened_data())
    assert pixels == [pixels[0]] * len(pixels)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize(
    "mode_in, mode_out",
    (
        ("RGB", "L"),
        ("RGB", "LA"),
        ("RGB", "La"),
        ("RGB", "I"),
        ("RGBA", "LA"),
        ("RGBA", "RGBa"),
        ("RGBa", "RGBA"),
    ),
)
def test_convert(width: int, mode_in: str, mode_out: str) -> None:
    bands = len(mode_in)
    for color in _colors(bands):
        im = Image.new(mode_in, (width, 2), color)
        _assert_flat(im.convert(mode_out))


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("mode", ("RGBA", "LA"))
def test_alpha_composite(width: int, mode: str) -> None:
    bands = len(mode)
    colors = _colors(bands)
    for dst, src in zip(colors, reversed(colors)):
        im_dst = Image.new(mode, (width, 2), dst)
        im_src = Image.new(mode, (width, 2), src)
        _assert_flat(Image.alpha_composite(im_dst, im_src))
