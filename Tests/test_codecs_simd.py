from __future__ import annotations

import io
import random
import struct
import zlib

import pytest

from PIL import Image, ImageFile, WebPImagePlugin, features

from .helper import hopper, skip_unless_feature
from .simd_shapes import SHAPES, photo

# Codec paths that Pillow-AllSIMD changes and that must give the same
# pixels: PNG unfiltering, WebP still images decoded straight into the
# image, JPEG decoded to 4-byte pixels by libjpeg-turbo.


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _png(size: tuple[int, int], color_type: int, rows: list[bytes]) -> bytes:
    ihdr = struct.pack(">IIBBBBB", size[0], size[1], 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + _chunk(b"IEND", b"")
    )


def _unfilter(rows: list[bytes], bpp: int) -> bytes:
    # the PNG specification's filters, undone byte by byte
    out = []
    prev = bytearray(len(rows[0]) - 1)
    for row in rows:
        f, x = row[0], bytearray(row[1:])
        for i in range(len(x)):
            a = x[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 1:
                x[i] = (x[i] + a) & 255
            elif f == 2:
                x[i] = (x[i] + b) & 255
            elif f == 3:
                x[i] = (x[i] + (a + b) // 2) & 255
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
                x[i] = (x[i] + pred) & 255
        out.append(bytes(x))
        prev = x
    return b"".join(out)


@skip_unless_feature("zlib")
@pytest.mark.parametrize(
    "mode, color_type", (("L", 0), ("LA", 4), ("RGB", 2), ("RGBA", 6))
)
@pytest.mark.parametrize("size", (*SHAPES.values(), (1, 1), (5, 3), (17, 2)))
def test_png_every_filter(mode: str, color_type: int, size: tuple[int, int]) -> None:
    rnd = random.Random(f"{mode}{size}")
    bpp = len(mode)
    w, h = size
    rows = [bytes([rnd.randrange(5)]) + rnd.randbytes(w * bpp) for _ in range(h)]
    with Image.open(io.BytesIO(_png(size, color_type, rows))) as im:
        assert im.mode == mode
        assert im.tobytes() == _unfilter(rows, bpp)


@skip_unless_feature("webp")
@pytest.mark.parametrize(
    "path",
    (
        "Tests/images/hopper.webp",
        "Tests/images/transparent.webp",
        "Tests/images/flower.webp",
    ),
)
def test_webp_still_matches_the_frame_decoder(path: str) -> None:
    from PIL import _webp

    with open(path, "rb") as f:
        data = f.read()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        assert not im.tile  # decoded straight into the image
        decoder = _webp.WebPAnimDecoder(data)
        rawmode = decoder.get_info()[4]
        raw, timestamp = decoder.get_next()
        assert (
            im.tobytes()
            == Image.frombytes(im.mode, im.size, raw, "raw", rawmode).tobytes()
        )
        assert im.info["duration"] == timestamp
        assert im.info["timestamp"] == 0


@skip_unless_feature("webp")
@pytest.mark.parametrize("mode", ("RGB", "RGBA"))
def test_webp_still_shapes(mode: str) -> None:
    for size in SHAPES.values():
        src = photo(mode, size)
        if mode == "RGBA":
            src.putalpha(photo("L", size).transpose(Image.Transpose.ROTATE_180))
        for kw in ({"quality": 80}, {"lossless": True, "exact": True}):
            b = io.BytesIO()
            src.save(b, "WEBP", **kw)
            with Image.open(b) as im:
                im.load()
                assert not im.tile
                assert (im.mode, im.size) == (mode, size)
                if kw.get("lossless"):
                    assert im.tobytes() == src.tobytes()


@skip_unless_feature("webp")
def test_webp_animated_frames() -> None:
    # animations keep the frame decoder
    frames = [hopper("RGB"), hopper("RGB").rotate(90)]
    b = io.BytesIO()
    frames[0].save(b, "WEBP", save_all=True, append_images=frames[1:], lossless=True)
    with Image.open(b) as im:
        assert isinstance(im, WebPImagePlugin.WebPImageFile)
        assert im.is_animated
        for k, frame in enumerate(frames):
            im.seek(k)
            assert im.tobytes() == frame.tobytes()


@skip_unless_feature("jpg")
@pytest.mark.parametrize("size", SHAPES.values())
def test_jpeg_rgbx_decode_matches_rgb(size: tuple[int, int]) -> None:
    b = io.BytesIO()
    photo("RGB", size).save(b, "JPEG", quality=90)
    with Image.open(io.BytesIO(b.getvalue())) as im:
        expected_rawmode = "RGBX" if features.check_feature("libjpeg_turbo") else "RGB"
        args = im.tile[0].args
        assert isinstance(args, tuple)
        assert args[0] == expected_rawmode
        im.load()
        fast = im.tobytes()
    with Image.open(io.BytesIO(b.getvalue())) as im:
        tile = im.tile[0]
        im.tile = [ImageFile._Tile(tile[0], tile[1], tile[2], ("RGB", ""))]
        im.load()
        assert im.tobytes() == fast


@skip_unless_feature("jpg")
def test_jpeg_draft_from_rgbx() -> None:
    b = io.BytesIO()
    photo("RGB", (301, 120)).save(b, "JPEG")
    for mode in ("L", "YCbCr"):
        with Image.open(io.BytesIO(b.getvalue())) as im:
            im.draft(mode, (80, 30))
            im.load()
            assert im.mode == mode
            assert im.size == (151, 60)
