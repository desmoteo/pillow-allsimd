from __future__ import annotations

import platform
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
        (
            v
            if i % 4 == 3 or data[i - i % 4 + 3] in (0, 255)
            else min(255, 255 * v // data[i - i % 4 + 3])
        )
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


@pytest.mark.skipif(
    platform.machine().lower() in ("x86_64", "amd64", "i386", "i686"),
    reason="the x86 kernels take a reciprocal estimate (CHANGES.SIMD.rst)",
)
def test_alpha_composite_every_alpha() -> None:
    # Every source alpha over every destination alpha: the scalar code's
    # integers, which NEON takes from a table of its divisions
    def div255(v: int) -> int:
        return ((v >> 8) + v) >> 8

    src = [
        (sa * 7 % 256, (sa + da * 3) % 256, da * 11 % 256, sa)
        for sa in range(256)
        for da in range(256)
    ]
    dst = [
        (da * 5 % 256, (sa * 13 + 1) % 256, sa ^ da, da)
        for sa in range(256)
        for da in range(256)
    ]
    expected: list[int] = []
    for s, d in zip(src, dst):
        if s[3] == 0:
            expected += d
            continue
        outa255 = s[3] * 255 + d[3] * (255 - s[3])
        coef1 = s[3] * 255 * 255 * 128 // outa255
        coef2 = 255 * 128 - coef1
        expected += [
            div255(s[c] * coef1 + d[c] * coef2 + (0x80 << 7)) >> 7 for c in range(3)
        ]
        expected.append(div255(outa255 + 0x80))
    im_src = Image.frombytes("RGBA", (256, 256), bytes(v for p in src for v in p))
    im_dst = Image.frombytes("RGBA", (256, 256), bytes(v for p in dst for v in p))
    assert Image.alpha_composite(im_dst, im_src).tobytes() == bytes(expected)


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


@pytest.mark.parametrize("mode", ("RGB", "RGBA"))
@pytest.mark.parametrize(
    "resample", (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
)
@pytest.mark.parametrize(
    "method", (Image.Transform.AFFINE, Image.Transform.PERSPECTIVE)
)
def test_transform_pairs_match_single_pixels(
    mode: str, resample: Image.Resampling, method: Image.Transform
) -> None:
    # On arm64, pairs of pixels whose source pixels are inside the image go
    # through filters for two; the others, and any output one pixel wide,
    # one at a time. Coefficients with few binary digits give each pixel
    # the same coordinates in a column of its own, shifted by k.
    a, b, c, d, e, f = 0.8125, 0.3125, -3.25, -0.25, 0.9375, 2.5
    for size in SHAPES.values():
        rnd = random.Random(f"{mode}{size}")
        im = Image.frombytes(mode, size, rnd.randbytes(size[0] * size[1] * len(mode)))
        out_size = (size[0] + 6, size[1] + 4)

        def coeffs(k: int) -> tuple[float, ...]:
            if method == Image.Transform.AFFINE:
                return (a, b, c + a * k, d, e, f + d * k)
            return (a, b, c + a * k, d, e, f + d * k, 0, 2**-10)

        full = im.transform(out_size, method, coeffs(0), resample)
        for k in range(out_size[0]):
            column = im.transform((1, out_size[1]), method, coeffs(k), resample)
            expected = full.crop((k, 0, k + 1, out_size[1]))
            assert column.tobytes() == expected.tobytes(), k


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


def _alpha_image(mode: str, size: tuple[int, int]) -> Image.Image:
    # every alpha value, with colours both below and above it
    rnd = random.Random(size[0] * 7 + size[1])
    n = size[0] * size[1] * 4
    data = bytes(rnd.randrange(256) if i % 4 != 3 else (i // 4) % 256 for i in range(n))
    return Image.frombytes("RGBA", size, data).convert(mode)


@pytest.mark.parametrize("mode", ("RGBA", "LA"))
@pytest.mark.parametrize(
    "resample",
    (
        Image.Resampling.BOX,
        Image.Resampling.BILINEAR,
        Image.Resampling.HAMMING,
        Image.Resampling.BICUBIC,
        Image.Resampling.LANCZOS,
        Image.Resampling.MKS2013,
        Image.Resampling.MKS2021,
    ),
)
@pytest.mark.parametrize(
    "size, box",
    (
        ((61, 37), None),  # upscale: horizontal, then vertical
        ((13, 9), None),  # downscale
        ((37, 5), None),  # vertical first, a 2x larger height reduction
        ((23, 17), None),  # horizontal only
        ((41, 11), None),  # vertical only
        ((10, 6), (3, 2, 13, 8)),  # an integer box of the output's size: a crop
        ((12, 7), (2.5, 1.25, 30.5, 15.75)),
    ),
)
def test_resize_alpha_matches_premultiplied_conversions(
    mode: str,
    resample: Image.Resampling,
    size: tuple[int, int],
    box: tuple[float, float, float, float] | None,
) -> None:
    # RGBA and LA images are premultiplied inside the resize passes; the
    # pixels must be those of converting, resizing and converting back
    im = _alpha_image(mode, (41, 17))
    premultiplied = im.convert({"LA": "La", "RGBA": "RGBa"}[mode])
    with Image.fast_mode(False):
        for reducing_gap in (None, 2.0):
            expected = premultiplied.resize(size, resample, box, reducing_gap)
            result = im.resize(size, resample, box, reducing_gap)
            assert result.mode == mode
            assert result.tobytes() == expected.convert(mode).tobytes()
