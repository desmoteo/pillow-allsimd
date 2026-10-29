from __future__ import annotations

import pytest

from PIL import Image

from .helper import assert_image_equal, hopper


def test_sanity() -> None:
    im = hopper()

    with pytest.raises(ValueError):
        im.point(list(range(256)))
    im.point(list(range(256)) * 3)
    im.point(lambda x: x)
    im.point(lambda x: x * 1.2)

    im = im.convert("I")
    with pytest.raises(ValueError):
        im.point(list(range(256)))
    im.point(lambda x: x * 1)
    im.point(lambda x: x + 1)
    im.point(lambda x: x - 1)
    im.point(lambda x: x * 1 + 1)
    im.point(lambda x: 0.1 + 0.2 * x)
    im.point(lambda x: -x)
    im.point(lambda x: x - 0.5)
    im.point(lambda x: 1 - x / 2)
    im.point(lambda x: (2 + x) / 3)
    im.point(lambda x: 0.5)
    im.point(lambda x: x / 1)
    im.point(lambda x: x + x)
    with pytest.raises(TypeError):
        im.point(lambda x: x * x)
    with pytest.raises(TypeError):
        im.point(lambda x: x / x)
    with pytest.raises(TypeError):
        im.point(lambda x: 1 / x)
    with pytest.raises(TypeError):
        im.point(lambda x: x // 2)


def test_16bit_lut() -> None:
    """Tests for 16 bit -> 8 bit lut for converting I->L images
    see https://github.com/python-pillow/Pillow/issues/440
    """
    im = hopper("I")
    im.point(list(range(256)) * 256, "L")


def test_f_lut() -> None:
    """Tests for floating point lut of 8bit gray image"""
    im = hopper("L")
    lut = [0.5 * float(x) for x in range(256)]

    out = im.point(lut, "F")

    int_lut = [x // 2 for x in range(256)]
    assert_image_equal(out.convert("L"), im.point(int_lut, "L"))


def test_f_mode() -> None:
    im = hopper("F")
    with pytest.raises(ValueError):
        im.point([])


def test_overstated_length() -> None:
    # shouldn't segfault
    # see https://github.com/python-pillow/Pillow/issues/9892

    class OverstatedLengthSequence:
        def __len__(self) -> int:
            return 256

        def __getitem__(self, index: int) -> int:
            if index >= 8:
                raise IndexError
            return index

    im = Image.new("L", (4, 4))
    with pytest.raises(ValueError):
        im.point(OverstatedLengthSequence(), "F")  # type: ignore[arg-type]


def test_raising_length() -> None:
    class RaisingLengthSequence:
        def __len__(self) -> int:
            raise RuntimeError

        def __getitem__(self, index: int) -> float:
            return index

    im = Image.new("L", (1, 1))
    with pytest.raises(RuntimeError):
        im.point(RaisingLengthSequence(), "F")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ("L", "LA", "RGB", "RGBA"))
@pytest.mark.parametrize("width", (256, 1, 15, 17, 33))
def test_byte_operation_tables(mode: str, width: int) -> None:
    # Tables equal to one operation on every byte (invert, posterize,
    # solarize) take a faster path; tables differing in one entry don't.
    bands = len(mode)
    data = bytes((i * 37 + i // 256) % 256 for i in range(width * 16 * bands))
    im = Image.frombytes(mode, (width, 16), data)
    tables = [[i ^ c for i in range(256)] for c in (0, 1, 255, 0x5A)]
    tables += [[i & (256 - 2**b) for i in range(256)] for b in range(9)]
    tables += [[i if i < t else 255 - i for i in range(256)] for t in (0, 1, 128, 255)]
    near = [i ^ 255 for i in range(256)]
    near[200] = 7
    tables.append(near)
    for table in tables:
        out = im.point(table * bands)
        expected = bytes(table[v] for v in data)
        assert out.tobytes() == expected
