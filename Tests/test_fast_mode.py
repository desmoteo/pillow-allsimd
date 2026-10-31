from __future__ import annotations

import math
import os
import subprocess
import sys
import threading

import pytest

from PIL import Image, ImageChops, ImageFilter, features

from .helper import hopper
from .simd_shapes import images, noise, photo, scaled, shapes

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable, Generator


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


# Every kernel below is tested on the six shapes of simd_shapes (two
# landscape, two square, two portrait), as noise and as a photograph.


def _diff(a: Image.Image, b: Image.Image) -> tuple[int, float]:
    # the largest and the mean difference of the bytes
    x, y = a.tobytes(), b.tobytes()
    assert len(x) == len(y)
    d = ImageChops.difference(
        Image.frombytes("L", (len(x), 1), x), Image.frombytes("L", (len(y), 1), y)
    ).histogram()
    return max(i for i, n in enumerate(d) if n), sum(
        i * n for i, n in enumerate(d)
    ) / len(x)


# Gaussian blur (GaussianFast.c): Pillow's three boxes as one kernel with
# 8-bit weights, the bands near the edges from the exact blur.

GAUSSIAN_MODES = ("L", "LA", "La", "RGB", "RGBA", "RGBa", "RGBX", "CMYK")


def _blur(
    im: Image.Image, radius: float | tuple[float, float], fast: bool
) -> Image.Image:
    with Image.fast_mode(fast):
        return im.filter(ImageFilter.GaussianBlur(radius))


@shapes
@pytest.mark.parametrize("mode", GAUSSIAN_MODES)
@pytest.mark.parametrize("radius", (0.3, 1, 2.5, (2, 0), (0, 3), (1.5, 4), 6))
def test_gaussian_close_to_exact(
    mode: str, radius: float | tuple[float, float], shape: tuple[int, int]
) -> None:
    for im in images(mode, shape):
        exact = _blur(im, radius, False)
        fast = _blur(im, radius, True)
        assert (fast.mode, fast.size) == (exact.mode, exact.size)
        largest, mean = _diff(fast, exact)
        assert largest <= 3
        assert mean <= 1


@shapes
def test_gaussian_uses_the_fast_kernel(shape: tuple[int, int]) -> None:
    im = noise("RGB", shape)
    assert _blur(im, 1, True).tobytes() != _blur(im, 1, False).tobytes()


@shapes
def test_gaussian_edges_exact(shape: tuple[int, int]) -> None:
    # sigma 2: boxes of radius 1.375, so the three reach 6 pixels; the bands
    # within that of the edges are those of the exact blur
    for im in images("RGB", shape, 2):
        exact = _blur(im, 2, False)
        fast = _blur(im, 2, True)
        w, h = im.size
        for box in ((0, 0, w, 6), (0, h - 6, w, h), (0, 0, 6, h), (w - 6, 0, w, h)):
            assert fast.crop(box).tobytes() == exact.crop(box).tobytes()


@shapes
def test_gaussian_closer_to_true_gaussian(shape: tuple[int, int]) -> None:
    # Against a true Gaussian, computed here in floats, the fast result is
    # at least as close as the exact one: one rounding instead of three.
    sigma, r = 1.5, 6
    k = [math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-r, r + 1)]
    k = [v / sum(k) for v in k]
    w, h = shape
    for im in images("L", shape, 3):
        px = list(im.tobytes())
        rows = [
            [
                sum(
                    k[i] * px[y * w + min(max(x + i - r, 0), w - 1)]
                    for i in range(2 * r + 1)
                )
                for x in range(w)
            ]
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


@shapes
@pytest.mark.parametrize("mode", ("L", "RGB", "RGBA"))
def test_gaussian_uniform(mode: str, shape: tuple[int, int]) -> None:
    colors: dict[str, int | tuple[int, ...]] = {
        "L": 77,
        "RGB": (10, 200, 37),
        "RGBA": (1, 2, 254, 128),
    }
    im = Image.new(mode, shape, colors[mode])
    for radius in (0.3, 1, 2.5, 6):
        assert _blur(im, radius, True).tobytes() == im.tobytes()


@pytest.mark.parametrize("size", ((1, 1), (1, 40), (40, 1), (3, 3), (20, 9)))
def test_gaussian_small_images_exact(size: tuple[int, int]) -> None:
    # too small for exact edge bands: blurred exactly
    im = noise("RGB", size)
    for radius in (0.5, 2, 6):
        assert _blur(im, radius, True).tobytes() == _blur(im, radius, False).tobytes()


@shapes
def test_gaussian_large_sigma_exact(shape: tuple[int, int]) -> None:
    for im in images("RGB", shape):
        assert _blur(im, 6.5, True).tobytes() == _blur(im, 6.5, False).tobytes()


@shapes
def test_gaussian_other_passes_exact(shape: tuple[int, int]) -> None:
    im = noise("RGB", shape)
    with Image.fast_mode():
        fast = im.im.gaussian_blur((2.0, 2.0), 1)
    exact = im.im.gaussian_blur((2.0, 2.0), 1)
    assert Image.Image()._new(fast).tobytes() == Image.Image()._new(exact).tobytes()


@shapes
@pytest.mark.parametrize("mode", ("I", "F", "I;16", "1"))
def test_gaussian_unsupported_modes_unchanged(
    mode: str, shape: tuple[int, int]
) -> None:
    im = photo(mode, shape)
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


@shapes
def test_unsharp_mask_close_to_exact(shape: tuple[int, int]) -> None:
    flt = ImageFilter.UnsharpMask(2, 150, 3)
    for im in images("RGB", shape):
        with Image.fast_mode():
            fast = im.filter(flt)
        largest, mean = _diff(fast, im.filter(flt))
        # the blur's difference, sharpened by 150%
        assert largest <= 8
        assert mean <= 1


# RGB <-> YCbCr (ConvertYCbCr.c): 14-bit coefficients, one rounding.


def _lattice(step: int = 3) -> Image.Image:
    # RGB values in steps of 3: 86 ** 3 pixels, every 8-bit triple's
    # neighbourhood
    values = range(0, 256, step)
    data = bytes(c for r in values for g in values for b in values for c in (r, g, b))
    n = len(values)
    return Image.frombytes("RGB", (n * n, n), data)


def _ycbcr_reference(r: int, g: int, b: int) -> tuple[int, int, int]:
    def rnd(v: float) -> int:
        return min(255, max(0, math.floor(v + 0.5)))

    return (
        rnd(0.299 * r + 0.587 * g + 0.114 * b),
        rnd(128 - 0.168736 * r - 0.331264 * g + 0.5 * b),
        rnd(128 + 0.5 * r - 0.418688 * g - 0.081312 * b),
    )


def _rgb_reference(y: int, cb: int, cr: int) -> tuple[int, int, int]:
    def rnd(v: float) -> int:
        return min(255, max(0, math.floor(v + 0.5)))

    cb -= 128
    cr -= 128
    return (
        rnd(y + 1.402 * cr),
        rnd(y - 0.344136 * cb - 0.714136 * cr),
        rnd(y + 1.772 * cb),
    )


@pytest.mark.parametrize("mode", ("RGB", "RGBA", "RGBX"))
def test_ycbcr_lattice_close_to_exact(mode: str) -> None:
    im = _lattice().convert(mode)
    with Image.fast_mode():
        fast = im.convert("YCbCr")
    largest, mean = _diff(fast, im.convert("YCbCr"))
    assert largest <= 1
    assert mean < 0.5
    assert fast.tobytes() != im.convert("YCbCr").tobytes()


def test_ycbcr_lattice_inverse_close_to_exact() -> None:
    im = Image.frombytes("YCbCr", _lattice().size, _lattice().tobytes())
    with Image.fast_mode():
        fast = im.convert("RGB")
    largest, mean = _diff(fast, im.convert("RGB"))
    assert largest <= 1
    assert mean < 0.5


@shapes
@pytest.mark.parametrize("mode", ("RGB", "RGBA", "RGBX"))
def test_ycbcr_close_to_exact(mode: str, shape: tuple[int, int]) -> None:
    for im in images(mode, shape):
        with Image.fast_mode():
            fast = im.convert("YCbCr")
        largest, mean = _diff(fast, im.convert("YCbCr"))
        assert largest <= 1
        assert mean < 0.5
        ycc = Image.frombytes("YCbCr", shape, im.convert("RGB").tobytes())
        with Image.fast_mode():
            fast = ycc.convert("RGB")
        largest, mean = _diff(fast, ycc.convert("RGB"))
        assert largest <= 1
        assert mean < 0.5


@shapes
def test_ycbcr_accuracy(shape: tuple[int, int]) -> None:
    # against the float formulas, the fast result is off by 1 at most, and
    # less often than the exact tables
    for rgb in images("RGB", shape, 5):
        data = rgb.tobytes()
        pixels = [tuple(data[i : i + 3]) for i in range(0, len(data), 3)]
        ycc = Image.frombytes("YCbCr", shape, data)
        for convert, im, ref in (
            ("YCbCr", rgb, _ycbcr_reference),
            ("RGB", ycc, _rgb_reference),
        ):
            expected = bytes(c for p in pixels for c in ref(*p))
            with Image.fast_mode():
                fast = im.convert(convert).tobytes()
            exact = im.convert(convert).tobytes()
            off = [abs(a - b) for a, b in zip(fast, expected)]
            assert max(off) <= 1
            exact_off = sum(a != b for a, b in zip(exact, expected))
            assert sum(v != 0 for v in off) * 10 < exact_off


@shapes
def test_ycbcr_other_conversions_exact(shape: tuple[int, int]) -> None:
    # palette images convert through the exact code
    im = photo("P", shape)
    with Image.fast_mode():
        fast = im.convert("YCbCr")
    assert fast.tobytes() == im.convert("YCbCr").tobytes()


# RGB -> HSV (Convert.c, rgb2hsv_fast): the formulas' truncation, exact.


def _hsv_floor(r: int, g: int, b: int) -> tuple[int, int, int]:
    v = max(r, g, b)
    d = v - min(r, g, b)
    if not d:
        return 0, 0, v
    if r == v:
        n = g - b
    elif g == v:
        n = 2 * d + b - r
    else:
        n = 4 * d + r - g
    if n < 0:
        n += 6 * d
    return 255 * n // (6 * d), 255 * d // v, v


def _hsv_floors(im: Image.Image) -> bytes:
    raw = im.convert("RGB").tobytes()
    return bytes(
        c
        for i in range(0, len(raw), 3)
        for c in _hsv_floor(raw[i], raw[i + 1], raw[i + 2])
    )


@pytest.mark.parametrize("mode", ("RGB", "RGBA", "RGBX"))
def test_hsv_lattice_exact_floors(mode: str) -> None:
    im = _lattice()
    with Image.fast_mode():
        fast = im.convert(mode).convert("HSV")
    assert fast.tobytes() == _hsv_floors(im)


@shapes
@pytest.mark.parametrize("mode", ("RGB", "RGBA", "RGBX"))
def test_hsv_exact_floors(mode: str, shape: tuple[int, int]) -> None:
    for im in images(mode, shape):
        with Image.fast_mode():
            fast = im.convert("HSV")
        assert fast.tobytes() == _hsv_floors(im)


@shapes
def test_hsv_close_to_exact(shape: tuple[int, int]) -> None:
    for im in [_lattice()] + images("RGB", shape):
        with Image.fast_mode():
            fast = im.convert("HSV")
        exact = im.convert("HSV")
        largest, _ = _diff(fast, exact)
        assert largest <= 1
        # saturation and value are the same; some hues are 1 above
        for k in (1, 2):
            assert fast.getchannel(k).tobytes() == exact.getchannel(k).tobytes()


# Affine, perspective and quad transforms (TransformFast.c): fixed-point
# positions in 1/256 pixel, integer filters with one rounding.

TRANSFORM_MODES = ("L", "RGB", "RGBa", "RGBX", "CMYK")


def _transforms(w: int, h: int) -> list[tuple[str, Callable[..., Image.Image]]]:
    T = Image.Transform
    return [
        ("rotate", lambda im, r: im.rotate(30, r)),
        ("rotate expand", lambda im, r: im.rotate(211.7, r, expand=True)),
        (
            "scale",
            lambda im, r: im.transform(
                (w * 2, h * 2), T.AFFINE, (0.5, 0, 0, 0, 0.5, 0), r
            ),
        ),
        (
            "affine",
            lambda im, r: im.transform(
                (w + 7, h + 3), T.AFFINE, (0.7, 0.2, -1.5, -0.1, 1.3, 2.25), r
            ),
        ),
        (
            "perspective",
            lambda im, r: im.transform(
                (w, h),
                T.PERSPECTIVE,
                (1.1, 0.1, -0.1 * w, 0, 1.05, 0, 0.13 / w, 0.19 / h),
                r,
            ),
        ),
        (
            "quad",
            lambda im, r: im.transform(
                (w, h), T.QUAD, (0, 0, 0, h, w * 1.1, h * 0.9, w * 0.8, 0), r
            ),
        ),
    ]


def _inside(
    fn: Callable[..., Image.Image], shape: tuple[int, int], fast: bool
) -> bytes:
    # which pixels a transform takes from inside the source: a constant
    # image stays constant there and is 0 outside
    with Image.fast_mode(fast):
        return fn(Image.new("L", shape, 200), Image.Resampling.BILINEAR).tobytes()


@shapes
@pytest.mark.parametrize("mode", TRANSFORM_MODES)
@pytest.mark.parametrize(
    "resample", (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
)
def test_transform_close_to_exact(
    mode: str, resample: Image.Resampling, shape: tuple[int, int]
) -> None:
    # Within 2 levels, but for pixels whose source position is on the
    # source's edge (rotating 75x211 by 30 degrees maps pixel (37, 30) to x
    # 0 exactly): rounded differently, one path may take them as inside and
    # the other as outside.
    w, h = shape
    for im in images(mode, shape):
        for name, fn in _transforms(w, h):
            exact = fn(im, resample)
            with Image.fast_mode():
                fast = fn(im, resample)
            assert (fast.mode, fast.size) == (exact.mode, exact.size), name
            assert _diff(fast, exact)[1] < 0.6, name
            ins = _inside(fn, shape, False), _inside(fn, shape, True)
            flips = [i for i, (a, b) in enumerate(zip(*ins)) if (a == 0) != (b == 0)]
            assert len(flips) <= 4, name
            f, e = fast.tobytes(), exact.tobytes()
            n = len(f) // len(ins[0])
            for i in set(range(len(ins[0]))) - set(flips):
                assert (
                    max(
                        abs(x - y)
                        for x, y in zip(f[i * n : i * n + n], e[i * n : i * n + n])
                    )
                    <= 2
                ), name


@shapes
def test_transform_uses_the_fast_kernel(shape: tuple[int, int]) -> None:
    im = noise("RGB", shape)
    with Image.fast_mode():
        fast = im.rotate(30, Image.Resampling.BILINEAR)
    assert fast.tobytes() != im.rotate(30, Image.Resampling.BILINEAR).tobytes()


@shapes
def test_transform_accuracy(shape: tuple[int, int]) -> None:
    # Against bilinear interpolation in floats at the exact positions, the
    # fast result is off by 1 at most, and far less often than the exact
    # path, which truncates: on photos about 15 times less often, on noise,
    # whose steep gradients show the 1/256 pixel positions most, 5 times.
    _check_bilinear_accuracy(photo("L", shape), 10)
    _check_bilinear_accuracy(noise("L", shape, 4), 4)


def _check_bilinear_accuracy(im: Image.Image, ratio: int) -> None:
    w, h = im.size
    px = im.tobytes()
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    data = (c, s, w / 2 - c * w / 2 - s * h / 2, -s, c, h / 2 + s * w / 2 - c * h / 2)
    exact = im.transform(
        (w, h), Image.Transform.AFFINE, data, Image.Resampling.BILINEAR
    ).tobytes()
    with Image.fast_mode():
        fast = im.transform(
            (w, h), Image.Transform.AFFINE, data, Image.Resampling.BILINEAR
        ).tobytes()
    fast_off = []
    exact_off = 0
    for y in range(h):
        for x in range(w):
            xs = data[0] * (x + 0.5) + data[1] * (y + 0.5) + data[2] - 0.5
            ys = data[3] * (x + 0.5) + data[4] * (y + 0.5) + data[5] - 0.5
            if not (0 <= xs < w - 1 and 0 <= ys < h - 1):
                continue
            ix, iy = int(xs), int(ys)
            fx, fy = xs - ix, ys - iy
            p = [px[(iy + j) * w + ix + i] for j in (0, 1) for i in (0, 1)]
            top = p[0] * (1 - fx) + p[1] * fx
            bottom = p[2] * (1 - fx) + p[3] * fx
            ref = math.floor(top * (1 - fy) + bottom * fy + 0.5)
            fast_off.append(abs(fast[y * w + x] - ref))
            exact_off += exact[y * w + x] != ref
    assert max(fast_off) <= 1
    assert sum(v != 0 for v in fast_off) * ratio < exact_off


@shapes
def test_transform_fill_exact(shape: tuple[int, int]) -> None:
    # the pixels outside the source are the exact path's
    w, h = shape
    for im in images("RGB", shape):
        exact = im.rotate(30, Image.Resampling.BICUBIC, fillcolor=(255, 0, 128))
        with Image.fast_mode():
            fast = im.rotate(30, Image.Resampling.BICUBIC, fillcolor=(255, 0, 128))
        for xy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            assert fast.getpixel(xy) == exact.getpixel(xy) == (255, 0, 128)


@pytest.mark.parametrize("size", ((1, 1), (2, 5), (5, 2), (3, 3)))
def test_transform_small_images(size: tuple[int, int]) -> None:
    im = noise("RGB", size)
    for resample in (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC):
        for name, fn in _transforms(*size):
            exact = fn(im, resample)
            with Image.fast_mode():
                fast = fn(im, resample)
            assert _diff(fast, exact)[0] <= 2, name


@shapes
def test_transform_mesh_close_to_exact(shape: tuple[int, int]) -> None:
    w, h = shape
    m = w // 2
    mesh = [
        ((0, 0, m, h), (0, 0, 0, h, m * 1.25, h * 0.92, m * 0.94, 2)),
        ((m, 0, w, h), (m * 0.94, 2, m * 1.25, h * 0.92, w, h, w, 0)),
    ]
    for im in images("RGB", shape):
        exact = im.transform(
            shape, Image.Transform.MESH, mesh, Image.Resampling.BILINEAR
        )
        with Image.fast_mode():
            fast = im.transform(
                shape, Image.Transform.MESH, mesh, Image.Resampling.BILINEAR
            )
        assert _diff(fast, exact)[0] <= 2


@shapes
@pytest.mark.parametrize(
    "method, data",
    (
        (Image.Transform.AFFINE, (1e12, 0, 0, 0, 1, 0)),
        (Image.Transform.AFFINE, (1, 0, -1e15, 0, 1, 0)),
        (Image.Transform.PERSPECTIVE, (1, 0, 0, 0, 1, 0, 1e9, 0)),
        (Image.Transform.QUAD, (0, 0, 0, 1e12, 1e12, 1e12, 1e12, 0)),
    ),
)
def test_transform_out_of_range_exact(
    method: Image.Transform, data: tuple[float, ...], shape: tuple[int, int]
) -> None:
    # positions the fixed point can't hold go to the exact path
    im = noise("RGB", shape)
    exact = im.transform(shape, method, data, Image.Resampling.BILINEAR)
    with Image.fast_mode():
        fast = im.transform(shape, method, data, Image.Resampling.BILINEAR)
    assert fast.tobytes() == exact.tobytes()


@shapes
@pytest.mark.parametrize("mode", ("I", "F", "LA", "1", "P"))
def test_transform_other_modes_exact(mode: str, shape: tuple[int, int]) -> None:
    im = photo(mode, shape)
    for resample in (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC):
        exact = im.rotate(30, resample)
        with Image.fast_mode():
            fast = im.rotate(30, resample)
        assert fast.tobytes() == exact.tobytes()


@shapes
def test_transform_nearest_exact(shape: tuple[int, int]) -> None:
    for im in images("RGB", shape):
        for name, fn in _transforms(*shape):
            exact = fn(im, Image.Resampling.NEAREST)
            with Image.fast_mode():
                fast = fn(im, Image.Resampling.NEAREST)
            assert fast.tobytes() == exact.tobytes(), name


# Blend (Blend.c): (in2 - in1) * alpha in 2^-15 units, rounded half up.


def _pairs(mode: str, shape: tuple[int, int]) -> list[tuple[Image.Image, Image.Image]]:
    p = photo(mode, shape)
    return [
        (noise(mode, shape, 1), noise(mode, shape, 2)),
        (p, p.transpose(Image.Transpose.ROTATE_180)),
    ]


@shapes
@pytest.mark.parametrize("mode", ("L", "LA", "RGB", "RGBA", "CMYK"))
def test_blend_close_to_exact(mode: str, shape: tuple[int, int]) -> None:
    for a, b in _pairs(mode, shape):
        for alpha in (0.001, 0.25, 0.5, 0.77, 0.99999):
            exact = Image.blend(a, b, alpha)
            with Image.fast_mode():
                fast = Image.blend(a, b, alpha)
            assert _diff(fast, exact)[0] <= 1


@shapes
def test_blend_accuracy(shape: tuple[int, int]) -> None:
    # the fast result is the float result rounded, but for values within
    # 0.005 of a half; the exact one truncates
    for a, b in _pairs("RGB", shape):
        for alpha in (0.3, 0.71, 0.9):
            values = [x + alpha * (y - x) for x, y in zip(a.tobytes(), b.tobytes())]
            exact = Image.blend(a, b, alpha).tobytes()
            with Image.fast_mode():
                fast = Image.blend(a, b, alpha).tobytes()
            # alpha in 2^-15 units moves a value by 0.004 at most
            assert max(abs(f - v) for f, v in zip(fast, values)) < 0.505
            far = sum(abs(e - v) >= 0.505 for e, v in zip(exact, values))
            assert far > len(values) / 4


@shapes
@pytest.mark.parametrize("alpha", (-0.5, 0, 1, 1.5))
def test_blend_other_alphas_exact(alpha: float, shape: tuple[int, int]) -> None:
    for a, b in _pairs("RGB", shape):
        with Image.fast_mode():
            fast = Image.blend(a, b, alpha)
        assert fast.tobytes() == Image.blend(a, b, alpha).tobytes()


# Large downscales (Image.py, resize): reducing_gap 3.0 when none is given,
# for the filters it makes faster. The sources are the shapes at 8 times
# their size (up to 2408x960), the results 1/6 to 1/16 of that.

FAST_DOWNSCALE_FILTERS = (
    Image.Resampling.BICUBIC,
    Image.Resampling.LANCZOS,
    Image.Resampling.MKS2013,
    Image.Resampling.MKS2021,
)


@shapes
@pytest.mark.parametrize("mode", ("L", "RGB", "RGBA", "I", "F"))
@pytest.mark.parametrize("resample", FAST_DOWNSCALE_FILTERS)
def test_downscale_reducing_gap(
    mode: str, resample: Image.Resampling, shape: tuple[int, int]
) -> None:
    im = photo(mode, scaled(shape, 8))
    for k in (8 / 6, 1, 0.5):
        size = scaled(shape, k)
        with Image.fast_mode():
            fast = im.resize(size, resample)
        assert fast.tobytes() == im.resize(size, resample, reducing_gap=3.0).tobytes()


@shapes
@pytest.mark.parametrize("resample", FAST_DOWNSCALE_FILTERS)
def test_downscale_close_to_exact(
    resample: Image.Resampling, shape: tuple[int, int]
) -> None:
    im = photo("RGB", scaled(shape, 8))
    for k in (8 / 6, 1, 0.5):
        size = scaled(shape, k)
        with Image.fast_mode():
            fast = im.resize(size, resample)
        exact = im.resize(size, resample)
        largest, mean = _diff(fast, exact)
        assert mean < 0.6
        # within 2 levels at photo sizes, up to 4 inside 32x32 results; the
        # edges differ more, where reduce() averages partial boxes
        box = (2, 2, size[0] - 2, size[1] - 2)
        assert _diff(fast.crop(box), exact.crop(box))[0] <= 4
        assert largest <= 10


@shapes
def test_downscale_small_factors_exact(shape: tuple[int, int]) -> None:
    # below 6x, reducing_gap 3.0 doesn't reduce
    im = photo("RGB", scaled(shape, 4))
    for size in (scaled(shape, 2), scaled(shape, 0.8), (shape[0] * 4, shape[1])):
        with Image.fast_mode():
            fast = im.resize(size, Image.Resampling.LANCZOS)
        assert fast.tobytes() == im.resize(size, Image.Resampling.LANCZOS).tobytes()


@shapes
@pytest.mark.parametrize(
    "kwargs",
    (
        {"resample": Image.Resampling.LANCZOS, "reducing_gap": 2.0},
        {"resample": Image.Resampling.NEAREST},
        {"resample": Image.Resampling.BILINEAR},
        {"resample": Image.Resampling.BOX},
        {"resample": Image.Resampling.HAMMING},
    ),
)
def test_downscale_other_arguments_exact(
    kwargs: dict[str, object], shape: tuple[int, int]
) -> None:
    im = photo("RGB", scaled(shape, 8))
    size = scaled(shape, 0.5)
    exact = im.resize(size, **kwargs)  # type: ignore[arg-type]
    with Image.fast_mode():
        fast = im.resize(size, **kwargs)  # type: ignore[arg-type]
    assert fast.tobytes() == exact.tobytes()


@shapes
def test_downscale_i16_exact(shape: tuple[int, int]) -> None:
    # reduce() doesn't take I;16 images
    im = photo("I;16", scaled(shape, 8))
    size = scaled(shape, 0.5)
    exact = im.resize(size, Image.Resampling.LANCZOS)
    with Image.fast_mode():
        fast = im.resize(size, Image.Resampling.LANCZOS)
    assert fast.tobytes() == exact.tobytes()
