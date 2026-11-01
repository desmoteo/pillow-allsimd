"""Pillow-AllSIMD: PNG saving through libdeflate gives the same pixels as
through zlib, at every level, and falls back to zlib where it must; PNG
loading inflates the image data in one piece with libdeflate and gives the
pixels, and for damaged files the errors, of zlib's stream."""

from __future__ import annotations

from io import BytesIO

import pytest

from PIL import Image, ImageFile, PngImagePlugin, features

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


def chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """The chunks of a PNG file, as (type, data)."""
    found = []
    pos = 8
    while pos < len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        found.append((data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]))
        pos += 12 + length
    return found


def with_image_data(data: bytes, pieces: list[bytes]) -> bytes:
    """The PNG file with its image data replaced by these IDAT chunks."""
    out = BytesIO()
    out.write(data[:8])
    written = False
    for cid, content in chunks(data):
        if cid != b"IDAT":
            PngImagePlugin.putchunk(out, cid, content)
        elif not written:
            for piece in pieces:
                PngImagePlugin.putchunk(out, b"IDAT", piece)
            written = True
    return out.getvalue()


def image_data(data: bytes) -> bytes:
    return b"".join(content for cid, content in chunks(data) if cid == b"IDAT")


def load(data: bytes) -> Image.Image | str:
    """The decoded image, or the text of the error that loading raised."""
    try:
        with Image.open(BytesIO(data)) as im:
            im.load()
            return im.copy()
    except (OSError, SyntaxError, ValueError) as e:
        return f"{type(e).__name__}: {e}"


def load_both(
    data: bytes, monkeypatch: pytest.MonkeyPatch
) -> tuple[Image.Image | str, Image.Image | str]:
    """Loaded in one piece, and with zlib's stream (the cap at 0)."""
    with monkeypatch.context() as m:
        m.delenv("PILLOW_LIBDEFLATE_MAX", raising=False)
        one_piece = load(data)
        m.setenv("PILLOW_LIBDEFLATE_MAX", "0")
        stream = load(data)
    return one_piece, stream


def assert_same(a: Image.Image | str, b: Image.Image | str) -> None:
    if isinstance(a, str) or isinstance(b, str):
        assert a == b
    else:
        assert_image_equal(a, b)


@pytest.mark.parametrize("mode", ("L", "LA", "RGB", "RGBA", "P", "1", "I;16"))
def test_decode_same_pixels_as_stream(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    im = hopper(mode)
    one_piece, stream = load_both(save(im), monkeypatch)
    assert not isinstance(one_piece, str)
    assert_image_equal(one_piece, im)
    assert_same(one_piece, stream)


@pytest.mark.parametrize(
    "path",
    (
        "Tests/images/pil123p.png",  # interlaced, palette
        "Tests/images/pil123rgba.png",  # interlaced, RGBA
        "Tests/images/hopper.png",
        "Tests/images/padded_idat.png",  # bytes after the stream's end
        "Tests/images/apng/split_fdat.png",  # an animation's first frame
    ),
)
def test_decode_files(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with open(path, "rb") as fp:
        data = fp.read()
    one_piece, stream = load_both(data, monkeypatch)
    assert_same(one_piece, stream)


@pytest.mark.parametrize("size", (1, 7, 1000, 100000))
def test_decode_chunked(size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # the image data in many chunks, with an empty one among them
    im = hopper("RGB")
    data = save(im)
    stream = image_data(data)
    pieces = [stream[i : i + size] for i in range(0, len(stream), size)]
    pieces.insert(1, b"")
    one_piece, by_stream = load_both(with_image_data(data, pieces), monkeypatch)
    assert not isinstance(one_piece, str)
    assert_image_equal(one_piece, im)
    assert_same(one_piece, by_stream)


@pytest.mark.parametrize("announced", (1, 10, 5000))
@pytest.mark.parametrize("block", (7, 65536))
def test_decode_more_data_than_announced(
    announced: int, block: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the decoder is given a length below the stream's: what it gathered
    # and the rest go through zlib's stream
    im = hopper("RGB")
    data = save(im)
    monkeypatch.setattr(ImageFile, "MAXBLOCK", block)
    monkeypatch.setattr(
        PngImagePlugin.PngImageFile, "_idat_stream_size", lambda self: announced
    )
    loaded = load(data)
    assert not isinstance(loaded, str)
    assert_image_equal(loaded, im)


def test_decode_damaged(monkeypatch: pytest.MonkeyPatch) -> None:
    im = hopper("RGB")
    data = save(im)
    stream = image_data(data)
    damaged = bytearray(stream)
    damaged[len(damaged) // 2] ^= 0x10
    wrong_checksum = bytearray(stream)
    wrong_checksum[-1] ^= 0x01
    cases = {
        # a stream that ends early, in chunks that are whole
        "short": [stream[: len(stream) // 2]],
        "damaged": [bytes(damaged)],
        "checksum": [bytes(wrong_checksum)],
        # rows beyond the image's
        "long": [image_data(save(Image.new("RGB", (im.width, im.height + 3), "red")))],
        "empty": [b""],
    }
    for name, pieces in cases.items():
        one_piece, stream_result = load_both(with_image_data(data, pieces), monkeypatch)
        assert_same(one_piece, stream_result)
        if name == "long":
            assert not isinstance(one_piece, str)
        if name in ("short", "empty"):
            assert isinstance(one_piece, str)
            assert "truncated" in one_piece

    # a file that ends inside the image data: the rows that arrived,
    # with LOAD_TRUNCATED_IMAGES, as before
    cut = data[: len(data) - len(stream) // 2]
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    one_piece, stream_result = load_both(cut, monkeypatch)
    assert not isinstance(one_piece, str)
    assert_same(one_piece, stream_result)
    assert one_piece.getpixel((0, 0)) == im.getpixel((0, 0))


def test_decode_position_restored() -> None:
    # walking the chunk headers leaves the file where it was, and the
    # chunks after the image data are still read
    info = PngImagePlugin.PngInfo()
    info.add(b"tEXt", b"after\0the image", after_idat=True)
    data = save(hopper("RGB"), pnginfo=info)
    with Image.open(BytesIO(data)) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.fp is not None
        position = im.fp.tell()
        assert im._idat_stream_size() == len(image_data(data))
        assert im.fp.tell() == position
        im.load()
        assert im.text["after"] == "the image"
