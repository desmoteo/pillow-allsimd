from __future__ import annotations

import random

import pytest

from PIL import Image, ImageFilter

from .simd_shapes import SHAPES

# SIMD kernels process several pixels at once. The pixels left over at the
# end of a row must get exactly the same arithmetic, so identical input
# pixels give identical output pixels wherever they are in the row.

WIDTHS = (1, 3, 5, 7, 9, 17)
# rows of those widths, and the six test shapes
SIZES = [(w, 2) for w in WIDTHS] + list(SHAPES.values())


def _colors(bands: int, count: int = 40) -> list[tuple[int, ...]]:
    rnd = random.Random(bands)
    return [tuple(rnd.randrange(256) for _ in range(bands)) for _ in range(count)]


def _assert_flat(im: Image.Image) -> None:
    pixels = list(im.get_flattened_data())
    assert pixels == [pixels[0]] * len(pixels)


@pytest.mark.parametrize("size", SIZES)
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
def test_convert(size: tuple[int, int], mode_in: str, mode_out: str) -> None:
    bands = len(mode_in)
    for color in _colors(bands):
        im = Image.new(mode_in, size, color)
        _assert_flat(im.convert(mode_out))


def test_unpremultiply_every_value() -> None:
    # Every colour value with every alpha: the vector kernels (a reciprocal
    # and a correction on x86, one division per pixel on arm64) give the
    # scalar code's 255 * c / alpha, truncated and clipped
    data = bytes(
        v for a in range(256) for c in range(256) for v in (c, 255 - c, c ^ 85, a)
    )
    out = Image.frombytes("RGBa", (256, 256), data).convert("RGBA").tobytes()
    expected = bytes(
        v
        if i % 4 == 3 or data[i - i % 4 + 3] in (0, 255)
        else min(255, 255 * v // data[i - i % 4 + 3])
        for i, v in enumerate(data)
    )
    assert out == expected


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("mode", ("RGBA", "LA"))
def test_alpha_composite(size: tuple[int, int], mode: str) -> None:
    bands = len(mode)
    colors = _colors(bands)
    for dst, src in zip(colors, reversed(colors)):
        im_dst = Image.new(mode, size, dst)
        im_src = Image.new(mode, size, src)
        _assert_flat(Image.alpha_composite(im_dst, im_src))


@pytest.mark.parametrize(
    "width", (5, 6, 7, 9, 17, 21, 23, *(w for w, _ in SHAPES.values()))
)
@pytest.mark.parametrize("mode", ("L", "LA", "RGB", "RGBA", "CMYK"))
@pytest.mark.parametrize("size", (3, 5))
def test_filter(width: int, mode: str, size: int) -> None:
    rnd = random.Random(size)
    kernels: list[list[float]] = [
        # Small coefficients take the integer kernels, large ones the float.
        [rnd.randint(-3, 5) for _ in range(size * size)],
        [rnd.uniform(-1000, 1000) for _ in range(size * size)],
    ]
    for kernel in kernels:
        scale = sum(kernel) or 1
        flt = ImageFilter.Kernel((size, size), kernel, scale=scale, offset=3)
        for color in _colors(len(mode), 8):
            im = Image.new(mode, (width, size + 2), color)
            out = im.filter(flt)
            border = size // 2
            _assert_flat(out.crop((border, border, width - border, border + 2)))


@pytest.mark.parametrize("mode", ("L", "LA", "RGB", "RGBA", "CMYK"))
@pytest.mark.parametrize("radius", (0.4, 1, 2, 3, 4, (2, 0.5), (1, 3)))
def test_box_blur_fused_matches_passes(
    mode: str, radius: float | tuple[float, float]
) -> None:
    # one pass each way through a ring of rows: the same as a horizontal
    # pass, then a vertical one
    rx, ry = radius if isinstance(radius, tuple) else (radius, radius)
    for size in (*SHAPES.values(), (1, 1), (3, 40), (40, 3)):
        rnd = random.Random(f"{mode}{size}")
        n = size[0] * size[1] * (1 if mode == "L" else 4)
        im = Image.frombytes(mode, size, rnd.randbytes(n))
        both = im.filter(ImageFilter.BoxBlur(radius))
        h = im._new(im.im.box_blur((rx, 0), 1))
        v = h._new(h.im.box_blur((0, ry), 1))
        assert both.tobytes() == v.tobytes()


@pytest.mark.parametrize("mode", ("L", "RGB", "RGBA", "CMYK"))
def test_point_table(mode: str) -> None:
    # table lookups (NEON tbl) against the table itself
    rnd = random.Random(mode)
    bands = len(mode)
    table = [rnd.randrange(256) for _ in range(256 * bands)]
    for size in SHAPES.values():
        im = Image.frombytes(mode, size, rnd.randbytes(size[0] * size[1] * bands))
        out = im.point(table).tobytes()
        src = im.tobytes()
        assert out == bytes(table[(i % bands) * 256 + v] for i, v in enumerate(src))
