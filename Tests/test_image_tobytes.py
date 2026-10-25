from __future__ import annotations

import pytest

from PIL import Image, ImageFile

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


@pytest.mark.parametrize("failure", (None, "setimage", "encode", "status"))
def test_encoder_cleanup(failure: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup_called = False

    class TestPyEncoder(ImageFile.PyEncoder):
        def encode(self, bufsize: int) -> tuple[int, int, bytes]:
            if failure == "encode":
                raise ValueError(failure)
            if failure == "status":
                return (0, -2, b"")
            return (6, 1, b"pixels")

        def setimage(
            self,
            im: Image.core.ImagingCore,
            extents: tuple[int, int, int, int] | None = None,
        ) -> None:
            if failure == "setimage":
                raise ValueError(failure)
            return super().setimage(im, extents)

        def cleanup(self) -> None:
            nonlocal cleanup_called
            cleanup_called = True

    im = Image.new("RGB", (1, 1))

    monkeypatch.setattr(Image, "ENCODERS", {"raw": TestPyEncoder})
    if failure in ("setimage", "encode"):
        with pytest.raises(ValueError, match=failure):
            im.tobytes()
    elif failure == "status":
        with pytest.raises(RuntimeError, match="encoder error -2 in tobytes"):
            im.tobytes()
    else:
        assert im.tobytes() == b"pixels"

    assert cleanup_called
