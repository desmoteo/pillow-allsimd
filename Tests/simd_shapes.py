"""The images the Pillow-SIMD tests run on: six shapes, two landscape, two
square and two portrait, of different sizes and proportions, each as noise
(the steepest gradients) and as a photograph. Odd widths and heights reach
the remainders of the SIMD loops, and every edge is tried along rows and
along columns."""

from __future__ import annotations

import functools
import platform
import random
import struct

import pytest

from PIL import Image

from .helper import hopper

SHAPES = {
    "landscape-wide": (301, 120),
    "landscape": (160, 97),
    "square": (64, 64),
    "square-odd": (129, 129),
    "portrait": (97, 160),
    "portrait-tall": (75, 211),
}

# @shapes runs a test once per shape, as its `shape` argument
shapes = pytest.mark.parametrize("shape", list(SHAPES.values()), ids=list(SHAPES))

# 32-bit x86: scalar builds there do double arithmetic on the x87 unit,
# with 80-bit intermediates (the SIMD builds use SSE), so Pillow's float
# steps can round differently from IEEE doubles
X86_32 = struct.calcsize("P") == 4 and platform.machine().lower() in (
    "x86_64",
    "amd64",
    "i386",
    "i686",
    "x86",
)

# bytes per pixel of tobytes(), where not 4
_BYTES = {"1": 1, "L": 1, "P": 1, "LA": 2, "La": 2, "RGB": 3, "YCbCr": 3, "HSV": 3}


def scaled(shape: tuple[int, int], k: float) -> tuple[int, int]:
    return round(shape[0] * k), round(shape[1] * k)


def noise(mode: str, size: tuple[int, int], seed: int = 1) -> Image.Image:
    rnd = random.Random(seed)
    n = size[0] * size[1] * _BYTES.get(mode, 4)
    return Image.frombytes(mode, size, rnd.randbytes(n))


@functools.lru_cache(maxsize=64)
def _photo(mode: str, size: tuple[int, int]) -> Image.Image:
    im = hopper("RGB").resize(size, Image.Resampling.BICUBIC)
    if mode in ("1", "I;16"):
        return im.convert("L").convert(mode)
    return im.convert(mode)


def photo(mode: str, size: tuple[int, int]) -> Image.Image:
    """hopper resized to size, in mode: smooth where noise is not."""
    return _photo(mode, size).copy()


def images(mode: str, size: tuple[int, int], seed: int = 1) -> list[Image.Image]:
    return [noise(mode, size, seed), photo(mode, size)]
