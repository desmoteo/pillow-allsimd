"""Pillow-AllSIMD: the fast octree quantizer reads the rows of an RGB or
RGBA image and writes the rows of the result; other modes go through an
array of pixels, as every mode did. Both give the same palette and the
same indices."""

from __future__ import annotations

import hashlib
import random

import pytest

from PIL import Image

from .helper import hopper

FASTOCTREE = Image.Quantize.FASTOCTREE


def result(im: Image.Image, colors: int) -> tuple[bytes, bytes]:
    q = im.quantize(colors, method=FASTOCTREE)
    palette = q.getpalette("RGBA" if im.mode == "RGBA" else "RGB")
    assert palette is not None
    return bytes(palette), q.tobytes()


def with_alpha(size: tuple[int, int], seed: int, transparent: int) -> Image.Image:
    """Random RGBA pixels, none transparent but those at every
    transparent-th place from the third row on (0: none at all)."""
    rnd = random.Random(seed)
    data = bytearray(rnd.randbytes(size[0] * size[1] * 4))
    for i in range(3, len(data), 4):
        data[i] |= 1
    if transparent:
        for p in range(2 * size[0] + 1, size[0] * size[1], transparent):
            data[p * 4 + 3] = 0
    return Image.frombytes("RGBA", size, bytes(data))


@pytest.mark.parametrize("colors", (1, 2, 17, 64, 255, 256))
@pytest.mark.parametrize("size", ((128, 128), (1, 1), (1, 97), (97, 1), (33, 7)))
def test_rows_as_the_array_of_pixels(size: tuple[int, int], colors: int) -> None:
    # A P image goes through the array, its RGB conversion through the rows:
    # the same pixels, so the same palette and indices.
    p = hopper("RGB").resize(size).quantize(256)
    assert result(p.convert("RGB"), colors) == result(p, colors)


def test_rows_fourth_byte_is_not_an_alpha() -> None:
    # a colour matrix leaves the fourth byte of each RGB pixel as it found it
    im = hopper("RGB")
    dirty = im.convert("RGB", (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0))
    assert result(dirty, 100) == result(im, 100)


@pytest.mark.parametrize("transparent", (0, 1, 3, 50, 10**9))
@pytest.mark.parametrize("colors", (2, 31, 256))
def test_transparent_pixels_take_the_first_one_s_colour(
    transparent: int, colors: int
) -> None:
    # Every fully transparent pixel is quantized as the first of them: an
    # image where they already have its colour gives the same result.
    im = with_alpha((67, 23), transparent, transparent)
    data = bytearray(im.tobytes())
    first = None
    for i in range(0, len(data), 4):
        if data[i + 3] == 0:
            if first is None:
                first = bytes(data[i : i + 3])
            data[i : i + 3] = first
    recoloured = Image.frombytes("RGBA", im.size, bytes(data))
    palette, indices = result(im, colors)
    assert (palette, indices) == result(recoloured, colors)
    if first is not None:
        # and they share an index
        alpha = im.getchannel("A").tobytes()
        assert len({indices[i] for i, a in enumerate(alpha) if a == 0}) == 1


def test_known_results() -> None:
    # The palettes and indices of Pillow 13.0.0's fast octree.
    rgb = hopper("RGB")
    rgba = hopper("RGBA")
    rgba.putalpha(hopper("L").point(lambda v: 0 if v < 40 else v))

    def digest(im: Image.Image, colors: int) -> str:
        palette, indices = result(im, colors)
        return hashlib.sha256(palette + indices).hexdigest()[:16]

    assert digest(rgb, 256) == "9c26499639818044"
    assert digest(rgb, 9) == "86234f900f9a4a2c"
    assert digest(rgba, 256) == "c02f7001edac1cef"
    assert digest(rgba, 5) == "27a8bd46dbbf63e1"


def test_more_pixels_than_the_sums_hold() -> None:
    # The rows' sums are 32-bit and are moved to the cube every 2**24
    # pixels. All of these pixels fall in one bucket and their red is 255:
    # kept in 32 bits to the end, its sum would overflow, and the palette
    # entry, which is the sum over the count, would not be 255.
    size = (4200, 4012)
    assert size[0] * size[1] * 255 > 2**32
    p = Image.new("P", size)
    p.putpalette(bytes(v for i in range(256) for v in (255, 250, 240 + i % 16)))
    rows = bytes(range(256)) * 18
    p.frombytes(b"".join(rows[y % 256 :][: size[0]] for y in range(size[1])))
    palette, indices = result(p.convert("RGB"), 256)
    assert (palette, indices) == result(p, 256)
    assert palette[:2] == bytes((255, 250))
