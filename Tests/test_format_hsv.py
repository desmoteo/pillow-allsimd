from __future__ import annotations

import array
import colorsys
import itertools
import math

import pytest

from PIL import Image

from .helper import assert_image_similar, hopper
from .simd_shapes import X86_32

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable


def int_to_float(i: int) -> float:
    return i / 255


def tuple_to_ints(tp: tuple[float, float, float]) -> tuple[int, int, int]:
    x, y, z = tp
    return int(x * 255.0), int(y * 255.0), int(z * 255.0)


def test_sanity() -> None:
    Image.new("HSV", (100, 100))


def linear_gradient() -> Image.Image:
    im = Image.linear_gradient(mode="L")
    im90 = im.rotate(90)

    px, h = im.size

    r = Image.new("L", (px * 3, h))
    g = r.copy()
    b = r.copy()

    r.paste(im, (0, 0))
    r.paste(im90, (px, 0))

    g.paste(im90, (0, 0))
    g.paste(im, (2 * px, 0))

    b.paste(im, (px, 0))
    b.paste(im90, (2 * px, 0))

    return Image.merge("RGB", (r, g, b))


def to_xxx_colorsys(
    im: Image.Image,
    func: Callable[[float, float, float], tuple[float, float, float]],
    mode: str,
) -> Image.Image:
    # convert the hard way using the library colorsys routines.

    r, g, b = im.split()

    conv_func = int_to_float

    converted = [
        tuple_to_ints(func(conv_func(_r), conv_func(_g), conv_func(_b)))
        for (_r, _g, _b) in itertools.zip_longest(r.tobytes(), g.tobytes(), b.tobytes())
    ]

    new_bytes = b"".join(
        bytes(chr(h) + chr(s) + chr(v), "latin-1") for (h, s, v) in converted
    )

    hsv = Image.frombytes(mode, r.size, new_bytes)

    return hsv


def to_hsv_colorsys(im: Image.Image) -> Image.Image:
    return to_xxx_colorsys(im, colorsys.rgb_to_hsv, "HSV")


def to_rgb_colorsys(im: Image.Image) -> Image.Image:
    return to_xxx_colorsys(im, colorsys.hsv_to_rgb, "RGB")


def test_linear_gradient() -> None:
    src = linear_gradient().resize((3 * 32, 32), Image.Resampling.BILINEAR)
    im = src.convert("HSV")
    comparable = to_hsv_colorsys(src)

    assert_image_similar(
        im.getchannel(0), comparable.getchannel(0), 1, "Hue conversion is wrong"
    )
    assert_image_similar(
        im.getchannel(1),
        comparable.getchannel(1),
        1,
        "Saturation conversion is wrong",
    )
    assert_image_similar(
        im.getchannel(2), comparable.getchannel(2), 1, "Value conversion is wrong"
    )

    comparable = src
    im = im.convert("RGB")

    assert_image_similar(
        im.getchannel(0), comparable.getchannel(0), 3, "R conversion is wrong"
    )
    assert_image_similar(
        im.getchannel(1), comparable.getchannel(1), 3, "G conversion is wrong"
    )
    assert_image_similar(
        im.getchannel(2), comparable.getchannel(2), 3, "B conversion is wrong"
    )


def test_convert() -> None:
    im = hopper("RGB").convert("HSV")
    comparable = to_hsv_colorsys(hopper("RGB"))

    assert_image_similar(
        im.getchannel(0), comparable.getchannel(0), 1, "Hue conversion is wrong"
    )
    assert_image_similar(
        im.getchannel(1),
        comparable.getchannel(1),
        1,
        "Saturation conversion is wrong",
    )
    assert_image_similar(
        im.getchannel(2), comparable.getchannel(2), 1, "Value conversion is wrong"
    )


def test_hsv_to_rgb() -> None:
    comparable = to_hsv_colorsys(hopper("RGB"))
    converted = comparable.convert("RGB")
    comparable = to_rgb_colorsys(comparable)

    assert_image_similar(
        converted.getchannel(0),
        comparable.getchannel(0),
        3,
        "R conversion is wrong",
    )
    assert_image_similar(
        converted.getchannel(1),
        comparable.getchannel(1),
        3,
        "G conversion is wrong",
    )
    assert_image_similar(
        converted.getchannel(2),
        comparable.getchannel(2),
        3,
        "B conversion is wrong",
    )


def _f32(v: float) -> float:
    # Rounds to single precision. For +, -, * and / of single precision
    # values, rounding the double result gives the single precision result.
    return array.array("f", [v])[0]


def _rgb2hsv_reference(r: int, g: int, b: int) -> tuple[int, int, int]:
    # The float and double steps of rgb2hsv_row in Convert.c.
    maxc, minc = max(r, g, b), min(r, g, b)
    if maxc == minc:
        return 0, 0, maxc
    cr = float(maxc - minc)
    s = _f32(cr / maxc)
    rc, gc, bc = (_f32((maxc - c) / cr) for c in (r, g, b))
    if r == maxc:
        h = _f32(bc - gc)
    elif g == maxc:
        h = _f32(2.0 + rc - bc)
    else:
        h = _f32(4.0 + gc - rc)
    h = _f32(math.fmod(h / 6.0 + 1.0, 1.0))
    return int(h * 255.0), int(s * 255.0), maxc


@pytest.mark.skipif(
    X86_32, reason="x87 builds round Pillow's float steps differently from IEEE doubles"
)
def test_rgb_to_hsv_exact() -> None:
    # Every value of each channel against others nearby and far, so that
    # near-ties and all hue cases are covered, and a coarse grid. 79 pixels
    # per row exercise the last pixels of a row too.
    colors: list[tuple[int, int, int]] = [
        c
        for v in range(256)
        for w in range(0, 256, 15)
        for c in ((v, w, v ^ 1), (w, v, (v + 7) % 256), ((w * 3) % 256, v, v))
    ]
    grid = [*range(0, 256, 9), 254, 255]
    colors += [(r, g, b) for r in grid for g in grid for b in grid]
    width = 79
    colors += [(0, 0, 0)] * (-len(colors) % width)
    im = Image.new("RGB", (width, len(colors) // width))
    im.putdata(colors)
    expected = [_rgb2hsv_reference(*c) for c in colors]
    with Image.fast_mode(False):
        assert list(im.convert("HSV").get_flattened_data()) == expected
