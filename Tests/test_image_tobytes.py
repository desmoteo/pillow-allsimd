from __future__ import annotations

import pytest

from PIL import Image

from .helper import hopper


def test_sanity() -> None:
    data = hopper().tobytes()
    assert isinstance(data, bytes)


@pytest.mark.parametrize(
    "width", (1, 3, 4, 5, 6, 7, 8, 15, 16, 17, 21, 22, 23, 47, 48, 49)
)
def test_rgb_roundtrip(width: int) -> None:
    # Packing and unpacking RGB work several pixels at a time, then one.
    data = bytes(range(256)) * ((width * 2 * 3) // 256 + 1)
    data = data[: width * 2 * 3]
    im = Image.frombytes("RGB", (width, 2), data)
    assert im.tobytes() == data
    assert [im.getpixel((x, 1)) for x in range(width)] == [
        tuple(data[(width + x) * 3 : (width + x) * 3 + 3]) for x in range(width)
    ]
