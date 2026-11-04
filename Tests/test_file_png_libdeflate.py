"""Pillow-AllSIMD: PNG saving through libdeflate gives the same pixels as
through zlib, at every level, and falls back to zlib where it must."""

from __future__ import annotations

from io import BytesIO

import pytest

from PIL import Image, features

from .helper import assert_image_equal, hopper, skip_unless_feature

pytestmark = skip_unless_feature("libdeflate")


def save(im: Image.Image, **options: object) -> bytes:
    out = BytesIO()
    im.save(out, "PNG", **options)
    return out.getvalue()


def test_version() -> None:
    version = features.version("libdeflate")
    assert version is not None
    assert version[0].isdigit()


@pytest.mark.parametrize("mode", ("L", "LA", "RGB", "RGBA", "P", "1", "I;16"))
@pytest.mark.parametrize("compress_level", (0, 1, 6, 9))
def test_same_pixels_as_zlib(mode: str, compress_level: int) -> None:
    im = hopper(mode)
    # compress_type names a zlib strategy, which only zlib's path honours
    zlib_bytes = save(im, compress_level=compress_level, compress_type=1)
    libdeflate_bytes = save(im, compress_level=compress_level)
    with Image.open(BytesIO(libdeflate_bytes)) as reloaded:
        assert_image_equal(reloaded, im)
    with Image.open(BytesIO(zlib_bytes)) as reloaded:
        assert_image_equal(reloaded, im)
    if compress_level == 0:
        # stored blocks in both: within the block overhead of each other
        assert abs(len(libdeflate_bytes) - len(zlib_bytes)) < len(zlib_bytes) // 50
    else:
        # the point of libdeflate: no larger than zlib's within 10%
        assert len(libdeflate_bytes) <= len(zlib_bytes) * 11 // 10


def test_optimize_smaller() -> None:
    im = hopper("RGB")
    assert len(save(im, optimize=True)) <= len(save(im))


def test_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    im = hopper("RGBA")
    reference = save(im)
    # a zlib strategy: zlib's path, the same pixels
    with Image.open(BytesIO(save(im, compress_type=2))) as reloaded:
        assert_image_equal(reloaded, im)
    # a preset dictionary: zlib's path too (the file needs the dictionary
    # to decode, so only that it is written and differs is checked)
    assert save(im, dictionary=b"\0" * 64) != reference
    # the size cap, in bytes: 0 turns libdeflate off, a row's worth too
    for cap in ("0", "100"):
        monkeypatch.setenv("PILLOW_LIBDEFLATE_MAX", cap)
        capped = save(im)
        assert capped != reference
        with Image.open(BytesIO(capped)) as reloaded:
            assert_image_equal(reloaded, im)
    monkeypatch.setenv("PILLOW_LIBDEFLATE_MAX", str(1 << 30))
    assert save(im) == reference


def test_small_buffer_and_large_image() -> None:
    # the stream is handed out in pieces smaller than itself
    im = Image.effect_noise((1024, 768), 60).convert("RGBA")
    with Image.open(BytesIO(save(im))) as reloaded:
        assert_image_equal(reloaded, im)
    # one row, and a one-pixel image
    for size in ((300, 1), (1, 1), (1, 300)):
        tiny = Image.effect_noise(size, 60)
        with Image.open(BytesIO(save(tiny))) as reloaded:
            assert_image_equal(reloaded, tiny)
