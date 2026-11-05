"""Every SIMD tier of one build gives the same pixels (Pillow-AllSIMD).

A dispatch build holds several tiers (SSE4.1, AVX2, AVX-VNNI) and chooses
one when it loads; the pixels must not depend on the CPU, so the tiers are
compared with each other here, operation by operation, through
``Image.core.set_simd()``. The scalar table is compared too, outside the
areas where ``docs/allsimd-kernels.rst`` says the SIMD kernels differ from
the portable code on purpose.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from .helper import hopper, skip_unless_feature
from .simd_shapes import SHAPES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _usable(tier: str) -> bool:
    """Whether this CPU can run the tier (a wheel holds tiers it may lack)."""
    active = Image.core.simd_kernels
    try:
        Image.core.set_simd(tier)
    except ValueError:
        return False
    finally:
        Image.core.set_simd(active)
    return True


TIERS = tuple(t for t in getattr(Image.core, "simd_tiers", ()) if _usable(t))
SIMD_TIERS = [t for t in TIERS if t != "scalar"]

pytestmark = pytest.mark.skipif(
    len(TIERS) < 2, reason="one kernel table in this build: nothing to compare"
)

# Where the SIMD kernels give other pixels than the portable code, by
# design (docs/allsimd-kernels.rst, "Ground rules"): the x86 alpha
# composite's reciprocal, the x86 colour LUT's own precision, the kernel
# filters whose coefficients are inexact at 13 bits and the float kernels,
# RGB to L's 15-bit weights. The SIMD tiers of one build agree everywhere.
SCALAR_DIFFERS = {
    "alpha_composite",
    "colorlut3",
    "colorlut4",
    "filter DETAIL",
    "filter SMOOTH_MORE",
    "filter float 3x3",
    "filter float 5x5",
    "filter float 3x3 L",
    "convert RGB L",
    "convert RGB LA",
    "convert RGB I",
    "convert RGBA LA",
}


def _inputs(size: tuple[int, int]) -> dict[str, Image.Image]:
    rgb = hopper("RGB").resize(size, Image.Resampling.NEAREST)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(rgb.getchannel(0))
    return {
        "RGB": rgb,
        "RGBA": rgba,
        "RGBa": rgba.convert("RGBa"),
        "L": rgb.getchannel(1),
        "YCbCr": rgb.convert("YCbCr"),
    }


def _operations(
    im: dict[str, Image.Image],
) -> Iterator[tuple[str, Callable[[], Image.Image]]]:
    rgb, rgba, rgba_p, L, ycc = im["RGB"], im["RGBA"], im["RGBa"], im["L"], im["YCbCr"]
    w, h = rgb.size
    small = (max(1, w * 2 // 5), max(1, h * 2 // 5))
    large = (w * 3 // 2 + 1, h * 3 // 2 + 1)
    R = Image.Resampling
    k3 = ImageFilter.Kernel((3, 3), [-0.7, -1, -0.7, -1, 5.5, -1, -0.7, -1, -0.7], 1, 2)
    k5 = ImageFilter.Kernel((5, 5), [0.3] * 12 + [-2.3] + [0.3] * 12, 1, 2)
    lut3 = ImageFilter.Color3DLUT.generate(9, lambda r, g, b: (g, b, r))
    lut4 = ImageFilter.Color3DLUT.generate(
        9, lambda r, g, b: (g, b, r, 1 - r), channels=4
    )
    yield "resize RGB down bicubic", lambda: rgb.resize(small, R.BICUBIC)
    yield "resize RGB up lanczos", lambda: rgb.resize(large, R.LANCZOS)
    yield "resize RGBA down bilinear", lambda: rgba.resize(small, R.BILINEAR)
    yield "resize L down bicubic", lambda: L.resize(small, R.BICUBIC)
    yield "resize L up lanczos", lambda: L.resize(large, R.LANCZOS)
    yield "resize F up bicubic", lambda: L.convert("F").resize(large, R.BICUBIC)
    yield "resize I;16 down", lambda: L.convert("I;16").resize(small, R.BICUBIC)
    yield "reduce 2", lambda: rgb.reduce(2)
    yield "reduce 3x5", lambda: rgb.reduce((3, 5))
    yield "reduce L 4", lambda: L.reduce(4)
    yield "gaussian 1", lambda: rgb.filter(ImageFilter.GaussianBlur(1))
    yield "gaussian 2.5", lambda: rgb.filter(ImageFilter.GaussianBlur(2.5))
    yield "gaussian L 7", lambda: L.filter(ImageFilter.GaussianBlur(7))
    yield "box 3", lambda: rgb.filter(ImageFilter.BoxBlur(3))
    yield "box L 0.4", lambda: L.filter(ImageFilter.BoxBlur(0.4))
    yield "unsharp", lambda: rgb.filter(ImageFilter.UnsharpMask(2, 150, 3))
    yield "filter SHARPEN", lambda: rgb.filter(ImageFilter.SHARPEN)
    yield "filter DETAIL", lambda: rgb.filter(ImageFilter.DETAIL)
    yield "filter SMOOTH_MORE", lambda: rgb.filter(ImageFilter.SMOOTH_MORE)
    yield "filter FIND_EDGES L", lambda: L.filter(ImageFilter.FIND_EDGES)
    # integer kernels with a power-of-two scale: small coefficients on arm64
    yield "filter EMBOSS", lambda: rgb.filter(ImageFilter.EMBOSS)
    yield "filter CONTOUR L", lambda: L.filter(ImageFilter.CONTOUR)
    yield "filter EDGE_ENHANCE", lambda: rgb.filter(ImageFilter.EDGE_ENHANCE)
    yield "filter BLUR", lambda: rgb.filter(ImageFilter.BLUR)
    yield "filter BLUR L", lambda: L.filter(ImageFilter.BLUR)
    k4 = ImageFilter.Kernel((3, 3), [1, -2, 1, -2, 8, -2, 1, -2, 1], 4, 7)
    yield "filter integer /4 +7", lambda: rgb.filter(k4)
    yield "filter float 3x3", lambda: rgb.filter(k3)
    yield "filter float 5x5", lambda: rgb.filter(k5)
    yield "filter float 3x3 L", lambda: L.filter(k3)
    yield "median 3", lambda: rgb.filter(ImageFilter.MedianFilter(3))
    yield "median 5 L", lambda: L.filter(ImageFilter.MedianFilter(5))
    yield "min 3", lambda: rgb.filter(ImageFilter.MinFilter(3))
    yield "max 7 L", lambda: L.filter(ImageFilter.MaxFilter(7))
    yield "rank 5 of 9", lambda: L.filter(ImageFilter.RankFilter(3, 5))
    yield "mode 3", lambda: L.filter(ImageFilter.ModeFilter(3))
    yield "convert RGB L", lambda: rgb.convert("L")
    yield "convert RGB LA", lambda: rgb.convert("LA")
    yield "convert RGB I", lambda: rgb.convert("I")
    yield "convert RGBA LA", lambda: rgba.convert("LA")
    yield "convert RGB HSV", lambda: rgb.convert("HSV")
    yield "convert RGB YCbCr", lambda: rgb.convert("YCbCr")
    yield "convert YCbCr RGB", lambda: ycc.convert("RGB")
    yield "convert RGBA RGBa", lambda: rgba.convert("RGBa")
    yield "convert RGBa RGBA", lambda: rgba_p.convert("RGBA")
    yield "convert L RGB", lambda: L.convert("RGB")
    yield "convert RGB RGBA", lambda: rgb.convert("RGBA")
    yield "convert RGB CMYK", lambda: rgb.convert("CMYK")
    yield "alpha_composite", lambda: Image.alpha_composite(
        rgba, rgba.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    )
    yield "blend", lambda: Image.blend(
        rgb, rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT), 0.3
    )
    yield "multiply", lambda: ImageChops.multiply(rgb, rgb.rotate(180))
    yield "screen", lambda: ImageChops.screen(rgb, rgb.rotate(180))
    yield "add", lambda: ImageChops.add(rgb, rgb.rotate(180))
    yield "subtract", lambda: ImageChops.subtract(rgb, rgb.rotate(180))
    yield "invert", lambda: ImageOps.invert(rgb)
    yield "posterize", lambda: ImageOps.posterize(rgb, 3)
    yield "solarize L", lambda: ImageOps.solarize(L, 100)
    yield "point RGB", lambda: rgb.point(lambda v: (v * 7 + 3) % 256)
    yield "point L", lambda: L.point(lambda v: 255 - v // 2)
    yield "colorlut3", lambda: rgb.filter(lut3)
    yield "colorlut4", lambda: rgba.filter(lut4)
    yield "getchannel", lambda: rgba.getchannel(2)
    yield "split merge", lambda: Image.merge("RGBA", rgba.split())
    yield "merge RGB", lambda: Image.merge("RGB", rgb.split())
    yield "transpose", lambda: rgb.transpose(Image.Transpose.TRANSPOSE)
    yield "rotate 90", lambda: rgb.transpose(Image.Transpose.ROTATE_90)
    yield "rotate 270 L", lambda: L.transpose(Image.Transpose.ROTATE_270)
    yield "flip LR", lambda: rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    yield "rotate 180 L", lambda: L.transpose(Image.Transpose.ROTATE_180)
    yield "rotate 30 bilinear", lambda: rgb.rotate(30, R.BILINEAR)
    yield "rotate 30 bicubic", lambda: rgb.rotate(30, R.BICUBIC)
    yield "rotate 30 bicubic RGBA", lambda: rgba.rotate(30, R.BICUBIC)
    yield "rotate 30 bilinear L", lambda: L.rotate(30, R.BILINEAR)
    yield "perspective bicubic", lambda: rgb.transform(
        rgb.size,
        Image.Transform.PERSPECTIVE,
        (1.0, 0.1, -3, 0.05, 1.0, -2, 0.001, 0.0005),
        R.BICUBIC,
    )
    yield "quad bilinear", lambda: rgb.transform(
        rgb.size,
        Image.Transform.QUAD,
        (0, 0, 2, h - 1, w - 1, h - 3, w - 2, 1),
        R.BILINEAR,
    )
    yield "paste mask", lambda: _paste(rgb, L)
    yield "paste mask RGBA", lambda: _paste(rgba, L)
    yield "draw", lambda: _draw(rgba)
    yield "png", lambda: _png(rgb)
    yield "tobytes BGR", lambda: Image.frombytes(
        "RGB", rgb.size, rgb.tobytes("raw", "BGR"), "raw", "BGR"
    )
    yield "frombytes RGBX", lambda: Image.frombytes(
        "RGB", rgb.size, rgb.tobytes("raw", "RGBX"), "raw", "RGBX"
    )
    yield "quantize", lambda: rgb.quantize(64).convert("RGB")


def _paste(im: Image.Image, mask: Image.Image) -> Image.Image:
    out = im.copy()
    out.paste(im.rotate(180), (0, 0), mask)
    return out


def _draw(im: Image.Image) -> Image.Image:
    out = im.copy()
    d = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    d.line((0, 0, w - 1, h - 1), fill=(255, 0, 0, 128), width=3)
    d.rectangle((1, 1, w // 2, h // 2), fill=(0, 255, 0, 200))
    d.polygon([(w // 2, 0), (w - 1, h // 2), (w // 2, h - 1)], fill=(0, 0, 255, 90))
    return out


def _png(im: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    out = Image.open(io.BytesIO(buf.getvalue()))
    out.load()
    return out


def _outputs(tier: str) -> dict[str, bytes]:
    Image.core.set_simd(tier)
    out: dict[str, bytes] = {}
    for shape in ("landscape", "portrait-tall"):
        size = SHAPES[shape]
        inputs = _inputs(size)
        for fast in (False, True):
            Image.set_fast(fast)
            for name, op in _operations(inputs):
                result = op()
                out[f"{shape} {'fast' if fast else 'exact'} {name}"] = result.tobytes()
    return out


@pytest.fixture(scope="module")
def outputs() -> Iterator[dict[str, dict[str, bytes]]]:
    active = Image.core.simd_kernels
    was_fast = Image.get_fast()
    try:
        yield {tier: _outputs(tier) for tier in TIERS}
    finally:
        Image.set_fast(was_fast)
        Image.core.set_simd(active)


def _compare(a: dict[str, bytes], b: dict[str, bytes], skip: set[str]) -> list[str]:
    return [
        name
        for name in a
        if a[name] != b[name] and not any(name.endswith(s) for s in skip)
    ]


@pytest.mark.skipif(len(SIMD_TIERS) < 2, reason="one SIMD tier in this build")
def test_simd_tiers_give_the_same_pixels(outputs: dict[str, dict[str, bytes]]) -> None:
    first = SIMD_TIERS[0]
    for tier in SIMD_TIERS[1:]:
        different = _compare(outputs[first], outputs[tier], set())
        assert not different, f"{first} and {tier} differ: {different}"


@pytest.mark.skipif(not SIMD_TIERS, reason="no SIMD tier in this build")
def test_scalar_agrees_outside_the_documented_areas(
    outputs: dict[str, dict[str, bytes]],
) -> None:
    tier = SIMD_TIERS[0]
    different = _compare(outputs["scalar"], outputs[tier], SCALAR_DIFFERS)
    assert not different, f"scalar and {tier} differ: {different}"


@skip_unless_feature("simd")
def test_simd_feature() -> None:
    assert Image.core.simd_kernels in TIERS
    assert TIERS[0] == "scalar"
    with pytest.raises(ValueError):
        Image.core.set_simd("no such tier")
