from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import threading

import pytest

from PIL import Image, ImageFilter, features

from .helper import hopper

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def restore_fast_mode() -> Generator[None]:
    previous = Image.get_fast()
    yield
    Image.set_fast(previous)


def test_feature() -> None:
    assert features.check_feature("fast") is True
    assert features.version_feature("fast") is None


def test_off_by_default() -> None:
    env = {k: v for k, v in os.environ.items() if k != "PILLOW_SIMD_FAST"}
    out = subprocess.check_output(
        [sys.executable, "-c", "from PIL import Image; print(Image.get_fast())"],
        env=env,
    )
    assert out.strip() == b"False"


def test_set_get() -> None:
    Image.set_fast(True)
    assert Image.get_fast() is True
    Image.set_fast(False)
    assert Image.get_fast() is False
    Image.set_fast()
    assert Image.get_fast() is True


@pytest.mark.parametrize(
    "value, expected",
    (
        ("1", True),
        ("true", True),
        ("ON", True),
        (" yes ", True),
        ("0", False),
        ("false", False),
        ("Off", False),
        ("no", False),
        ("", False),
    ),
)
def test_env(value: str, expected: bool) -> None:
    Image.set_fast(not expected)
    Image._apply_env_variables({"PILLOW_SIMD_FAST": value})
    assert Image.get_fast() is expected


def test_env_invalid() -> None:
    Image.set_fast(True)
    with pytest.warns(UserWarning, match="PILLOW_SIMD_FAST"):
        Image._apply_env_variables({"PILLOW_SIMD_FAST": "maybe"})
    assert Image.get_fast() is True


def test_env_unset() -> None:
    Image.set_fast(True)
    Image._apply_env_variables({})
    assert Image.get_fast() is True


def test_env_at_import() -> None:
    env = dict(os.environ, PILLOW_SIMD_FAST="1")
    out = subprocess.check_output(
        [sys.executable, "-c", "from PIL import Image; print(Image.get_fast())"],
        env=env,
    )
    assert out.strip() == b"True"


def test_context_manager() -> None:
    Image.set_fast(False)
    with Image.fast_mode():
        assert Image.get_fast() is True
        with Image.fast_mode(False):
            assert Image.get_fast() is False
        assert Image.get_fast() is True
    assert Image.get_fast() is False


def test_context_manager_restores_after_exception() -> None:
    Image.set_fast(False)
    with pytest.raises(ZeroDivisionError):
        with Image.fast_mode():
            1 / 0
    assert Image.get_fast() is False


def test_toggle_while_processing() -> None:
    # The flag is read once per operation; toggling it from another thread
    # while operations run must neither crash nor leave it in a bad state.
    im = hopper("RGB")
    stop = threading.Event()

    def toggle() -> None:
        while not stop.is_set():
            Image.set_fast(True)
            Image.set_fast(False)

    t = threading.Thread(target=toggle)
    t.start()
    try:
        for _ in range(20):
            im.filter(ImageFilter.GaussianBlur(2))
            im.convert("HSV")
            im.convert("YCbCr")
            im.rotate(30, Image.Resampling.BILINEAR)
    finally:
        stop.set()
        t.join()
    assert Image.get_fast() in (True, False)


# Gaussian blur (GaussianFast.c): Pillow's three boxes as one kernel with
# 8-bit weights, the bands near the edges from the exact blur.

GAUSSIAN_MODES = ("L", "LA", "La", "RGB", "RGBA", "RGBa", "RGBX", "CMYK")
BANDS = {"L": 1, "LA": 2, "La": 2, "RGB": 3, "RGBA": 4, "RGBa": 4, "RGBX": 4}


def _noise(mode: str, size: tuple[int, int], seed: int = 1) -> Image.Image:
    rnd = random.Random(seed)
    return Image.frombytes(
        mode, size, rnd.randbytes(size[0] * size[1] * BANDS.get(mode, 4))
    )


def _blur(
    im: Image.Image, radius: float | tuple[float, float], fast: bool
) -> Image.Image:
    with Image.fast_mode(fast):
        return im.filter(ImageFilter.GaussianBlur(radius))


def _diff(a: Image.Image, b: Image.Image) -> tuple[int, float]:
    d = [abs(x - y) for x, y in zip(a.tobytes(), b.tobytes())]
    return max(d), sum(d) / len(d)


@pytest.mark.parametrize("mode", GAUSSIAN_MODES)
@pytest.mark.parametrize("radius", (0.3, 1, 2.5, (2, 0), (0, 3), (1.5, 4), 6))
def test_gaussian_close_to_exact(
    mode: str, radius: float | tuple[float, float]
) -> None:
    im = _noise(mode, (160, 120))
    exact = _blur(im, radius, False)
    fast = _blur(im, radius, True)
    assert (fast.mode, fast.size) == (exact.mode, exact.size)
    largest, mean = _diff(fast, exact)
    assert largest <= 3
    assert mean <= 1


def test_gaussian_uses_the_fast_kernel() -> None:
    im = _noise("RGB", (160, 120))
    assert _blur(im, 1, True).tobytes() != _blur(im, 1, False).tobytes()


def test_gaussian_edges_exact() -> None:
    # sigma 2: boxes of radius 1.375, so the three reach 6 pixels; the bands
    # within that of the edges are those of the exact blur
    im = _noise("RGB", (160, 120), 2)
    exact = _blur(im, 2, False)
    fast = _blur(im, 2, True)
    w, h = im.size
    for box in ((0, 0, w, 6), (0, h - 6, w, h), (0, 0, 6, h), (w - 6, 0, w, h)):
        assert fast.crop(box).tobytes() == exact.crop(box).tobytes()


def test_gaussian_closer_to_true_gaussian() -> None:
    # Against a true Gaussian, computed here in floats, the fast result is
    # at least as close as the exact one: one rounding instead of three.
    size, sigma, r = (48, 40), 1.5, 6
    im = _noise("L", size, 3)
    k = [math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-r, r + 1)]
    k = [v / sum(k) for v in k]
    w, h = size
    px = list(im.tobytes())

    def at(x: int, y: int) -> int:
        return px[min(max(y, 0), h - 1) * w + min(max(x, 0), w - 1)]

    rows = [
        [sum(k[i] * at(x + i - r, y) for i in range(2 * r + 1)) for x in range(w)]
        for y in range(h)
    ]
    ref = [
        sum(k[i] * rows[min(max(y + i - r, 0), h - 1)][x] for i in range(2 * r + 1))
        for y in range(h)
        for x in range(w)
    ]

    def error(out: Image.Image) -> float:
        values = out.tobytes()
        inner = [
            abs(values[y * w + x] - math.floor(ref[y * w + x] + 0.5))
            for y in range(r, h - r)
            for x in range(r, w - r)
        ]
        return sum(inner) / len(inner)

    assert error(_blur(im, sigma, True)) <= error(_blur(im, sigma, False))


@pytest.mark.parametrize("mode", ("L", "RGB", "RGBA"))
def test_gaussian_uniform(mode: str) -> None:
    colors: dict[str, int | tuple[int, ...]] = {
        "L": 77,
        "RGB": (10, 200, 37),
        "RGBA": (1, 2, 254, 128),
    }
    color = colors[mode]
    im = Image.new(mode, (150, 100), color)
    for radius in (0.3, 1, 2.5, 6):
        assert _blur(im, radius, True).tobytes() == im.tobytes()


@pytest.mark.parametrize("size", ((1, 1), (1, 40), (40, 1), (3, 3), (20, 9)))
def test_gaussian_small_images_exact(size: tuple[int, int]) -> None:
    # too small for exact edge bands: blurred exactly
    im = _noise("RGB", size)
    for radius in (0.5, 2, 6):
        assert _blur(im, radius, True).tobytes() == _blur(im, radius, False).tobytes()


def test_gaussian_large_sigma_exact() -> None:
    im = _noise("RGB", (160, 120))
    assert _blur(im, 6.5, True).tobytes() == _blur(im, 6.5, False).tobytes()


def test_gaussian_other_passes_exact() -> None:
    im = _noise("RGB", (160, 120))
    with Image.fast_mode():
        fast = im.im.gaussian_blur((2.0, 2.0), 1)
    exact = im.im.gaussian_blur((2.0, 2.0), 1)
    assert Image.Image()._new(fast).tobytes() == Image.Image()._new(exact).tobytes()


@pytest.mark.parametrize("mode", ("I", "F", "I;16", "1"))
def test_gaussian_unsupported_modes_unchanged(mode: str) -> None:
    im = hopper(mode)
    with Image.fast_mode():
        try:
            fast: object = im.filter(ImageFilter.GaussianBlur(1)).tobytes()
        except ValueError as e:
            fast = type(e)
    try:
        exact: object = im.filter(ImageFilter.GaussianBlur(1)).tobytes()
    except ValueError as e:
        exact = type(e)
    assert fast == exact


def test_unsharp_mask_close_to_exact() -> None:
    im = _noise("RGB", (160, 120))
    flt = ImageFilter.UnsharpMask(2, 150, 3)
    with Image.fast_mode():
        fast = im.filter(flt)
    largest, mean = _diff(fast, im.filter(flt))
    assert largest <= 6
    assert mean <= 1
