#!/usr/bin/env python3
"""Hashes of the outputs of each SIMD-accelerated area, to compare builds.

Prints one MD5 per area (quantize, resample F/I/I;16, transposes, raw
pack/unpack, YCbCr, PNG filters, masked paste, alpha_composite, point and
ImageEnhance, ImageDraw, rank filters, transforms, blur and unsharp mask,
8-bit resize and reduce, chops and conversions, unpremultiply, and the
fast-mode kernels), each over many sizes and modes. Builds whose kernels
are exact give the same hashes: SIMD and scalar on one architecture, and
x86 and arm64 except where arm64 compilers fuse multiply-adds (resample
F/I/I;16, transforms). The x86 alpha_composite kernel is known to differ
by 1 from the portable one.

Inputs are made with frombytes only, since RGB to L conversions and RGB
resizes differ between SIMD and scalar builds. Run from the repository root
(for Tests/images), with the build on PYTHONPATH (not with python -I):

    PYTHONPATH=build-dir python3 checks/simd_hashes.py
"""

from __future__ import annotations

import hashlib
import io
import math
import random
import struct
from collections.abc import Callable, Iterator

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

R = Image.Resampling
T = Image.Transpose
Update = Callable[[bytes], None]
results: dict[str, str] = {}


def section(
    name: str,
) -> Callable[[Callable[[Update], None]], Callable[[Update], None]]:
    def wrap(f: Callable[[Update], None]) -> Callable[[Update], None]:
        h = hashlib.md5()
        f(h.update)
        results[name] = h.hexdigest()
        return f

    return wrap


# Inputs are made with frombytes only: conversions (RGB to L) and RGB
# resizes are known to differ between SIMD and scalar builds.
BYTES = {
    "1": 1,
    "L": 1,
    "P": 1,
    "LA": 2,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "YCbCr": 3,
    "I": 4,
    "F": 4,
    "I;16": 2,
}


def noise(mode: str, size: tuple[int, int], seed: int) -> Image.Image:
    rnd = random.Random(seed)
    w, hh = size
    if mode == "1":
        return Image.frombytes("1", size, rnd.randbytes((w + 7) // 8 * hh))
    if mode == "F":
        n = w * hh
        return Image.frombytes(
            "F", size, struct.pack(f"{n}f", *(rnd.uniform(-9, 9) for _ in range(n)))
        )
    return Image.frombytes(mode, size, rnd.randbytes(w * hh * BYTES[mode]))


def smooth(mode: str, size: tuple[int, int]) -> Image.Image:
    w, hh = size
    n = BYTES[mode]
    data = bytes(
        (x * 255 // max(1, w - 1) + y * 3 + c * 40 + (x * y) % 7) & 255
        for y in range(hh)
        for x in range(w)
        for c in range(n)
    )
    return Image.frombytes(mode, size, data)


hopper = Image.open("Tests/images/hopper.ppm").convert("RGB")


@section("quantize")
def _(upd: Update) -> None:
    rnd = random.Random(2)
    w, hh = 160, 100
    nz = Image.frombytes("RGB", (w, hh), rnd.randbytes(w * hh * 3))
    smooth = nz.resize((10, 6)).resize((w, hh), R.BICUBIC)
    few = Image.frombytes(
        "RGB", (w, hh), bytes(rnd.choice((0, 64, 128, 255)) for _ in range(w * hh * 3))
    )
    gray = Image.linear_gradient("L").resize((w, hh)).convert("RGB")
    for im in (nz, smooth, few, gray, hopper):
        for n in (2, 7, 16, 64, 255, 256):
            for method in (Image.Quantize.MEDIANCUT, Image.Quantize.MAXCOVERAGE):
                q = im.quantize(n, method)
                upd(q.tobytes() + bytes(q.getpalette() or []))
            q = im.convert("P", palette=Image.Palette.ADAPTIVE, colors=n)
            upd(q.tobytes() + bytes(q.getpalette() or []))


@section("resample F/I/I;16")
def _(upd: Update) -> None:
    rnd = random.Random(7)

    def images(w: int, hgt: int) -> Iterator[Image.Image]:
        n = w * hgt
        f = [rnd.uniform(-1e3, 1e3) for _ in range(n)]
        for i in rnd.sample(range(n), min(n, 12)):
            f[i] = rnd.choice([math.inf, -math.inf, math.nan, -0.0, 1e38, -1e38])
        yield Image.frombytes("F", (w, hgt), struct.pack(f"{n}f", *f))
        iv = [
            (
                rnd.randint(-(2**31), 2**31 - 1)
                if i % 7 == 0
                else rnd.randint(-70000, 70000)
            )
            for i in range(n)
        ]
        yield Image.frombytes("I", (w, hgt), struct.pack(f"{n}i", *iv))
        u = [rnd.choice([0, 65535, rnd.randint(0, 65535)]) for _ in range(n)]
        for mode, fmt in (("I;16", "<"), ("I;16B", ">"), ("I;16N", "=")):
            yield Image.frombytes(mode, (w, hgt), struct.pack(f"{fmt}{n}H", *u))

    for w, hgt in ((1, 1), (7, 9), (37, 23), (130, 67)):
        for im in images(w, hgt):
            for size in (
                (w * 2 + 1, hgt),
                (w, hgt * 3),
                (max(1, w // 3), max(1, hgt // 2)),
                (w + 3, hgt + 5),
            ):
                for flt in (R.BOX, R.BILINEAR, R.BICUBIC, R.LANCZOS, R.MKS2021):
                    upd(im.resize(size, flt).tobytes())
                    upd(
                        im.resize(
                            size, flt, box=(0.5, 0.25, w - 0.3, hgt - 0.1)
                        ).tobytes()
                    )


@section("transpose")
def _(upd: Update) -> None:
    for mode in ("1", "L", "P", "LA", "RGB", "RGBA", "I", "F", "I;16"):
        for size in (
            (1, 1),
            (3, 5),
            (8, 8),
            (15, 17),
            (37, 29),
            (530, 13),
            (11, 523),
            (520, 515),
        ):
            im = noise(mode, size, size[0] * 31 + size[1])
            for t in T:
                upd(im.transpose(t).tobytes())


@section("pack/unpack")
def _(upd: Update) -> None:
    rnd = random.Random(5)
    unpack = {
        "RGB": ("RGB", "BGR", "BGRX", "XRGB", "XBGR", "BGXR", "RGBX"),
        "RGBA": ("RGBA", "BGR", "BGRA", "ABGR", "ARGB", "BGAR", "RGB"),
        "RGBX": ("BGR", "BGRX", "XRGB", "XBGR"),
    }
    pack = {
        "RGB": ("RGB", "BGR", "BGRX", "XRGB", "XBGR", "RGBX"),
        "RGBA": ("RGBA", "BGR", "BGRA", "ABGR", "RGB"),
        "RGBX": ("BGR", "BGRX", "XBGR"),
    }
    for w in list(range(1, 70)) + [1280]:
        for mode, raws in unpack.items():
            for raw in raws:
                n = w * 3 * len(raw)
                try:
                    im = Image.frombytes(mode, (w, 3), rnd.randbytes(n), "raw", raw)
                except ValueError:
                    upd(b"unsupported")
                    continue
                upd(im.tobytes())
                for praw in pack[mode]:
                    try:
                        upd(im.tobytes("raw", praw))
                    except ValueError:
                        upd(b"unsupported")


@section("ycbcr")
def _(upd: Update) -> None:
    for size in ((1, 1), (5, 3), (33, 17), (257, 100)):
        rgb = noise("RGB", size, size[0])
        y = rgb.convert("YCbCr")
        upd(y.tobytes())
        upd(y.convert("RGB").tobytes())
        upd(noise("RGBA", size, 9).convert("YCbCr").convert("RGB").tobytes())
        # any YCbCr, out of the RGB range too, which clips
        upd(noise("YCbCr", size, 11).convert("RGB").tobytes())


@section("png filters")
def _(upd: Update) -> None:
    for mode in ("L", "LA", "RGB", "RGBA", "P", "I;16"):
        for size in ((1, 1), (7, 5), (63, 21), (200, 50)):
            im = smooth(mode, size) if mode != "P" else smooth("L", size).convert("P")
            for level in (1, 6):
                out = io.BytesIO()
                im.save(out, "PNG", compress_level=level)
                upd(out.getvalue())


@section("paste/composite")
def _(upd: Update) -> None:
    for size in ((1, 1), (17, 9), (64, 33), (301, 120)):
        base_rgb = noise("RGB", size, 1)
        base_rgba = noise("RGBA", size, 2)
        mask = noise("L", size, 3)
        for base in (base_rgb, base_rgba, noise("L", size, 4), noise("LA", size, 6)):
            out = base.copy()
            out.paste((10, 200, 30, 128)[: len(base.getbands())], (0, 0), mask)
            upd(out.tobytes())
            out = base.copy()
            out.paste(noise(base.mode, size, 5), (0, 0), mask)
            upd(out.tobytes())
        drawn = base_rgb.copy()
        d = ImageDraw.Draw(drawn)
        d.bitmap(
            (1, 0),
            mask.point(lambda v: 255 if v > 100 else 0).convert("1"),
            fill=(1, 2, 3),
        )
        upd(drawn.tobytes())
        over = noise("RGBA", size, 8)
        upd(Image.composite(base_rgba, over, mask).tobytes())
        upd(Image.blend(base_rgba, over, 0.3).tobytes())


@section("alpha_composite (x86 SIMD vs scalar: known)")
def _(upd: Update) -> None:
    for size in ((1, 1), (17, 9), (64, 33), (301, 120)):
        base, over = noise("RGBA", size, 2), noise("RGBA", size, 8)
        upd(Image.alpha_composite(base, over).tobytes())
        for dest in ((0, 0), (3, 2), (-5, -1)):
            out = base.copy()
            out.alpha_composite(over, dest)
            upd(out.tobytes())


@section("point/enhance")
def _(upd: Update) -> None:
    for mode in ("L", "RGB", "RGBA"):
        im = noise(mode, (131, 47), 11)
        rgbish = im if mode != "RGBA" else noise("RGB", (131, 47), 12)
        upd(ImageOps.invert(rgbish).tobytes())  # posterize, solarize: point() tables
        for bits in (1, 3, 7):
            upd(ImageOps.posterize(rgbish, bits).tobytes())
        for t in (0, 1, 128, 255):
            upd(ImageOps.solarize(rgbish, t).tobytes())
        for f in (0.0, 0.5, 1.3, 2.0):
            if mode == "L":  # others take the mean of an L conversion
                upd(ImageEnhance.Contrast(im).enhance(f).tobytes())
            upd(ImageEnhance.Brightness(im).enhance(f).tobytes())
        upd(im.point(lambda v: v ^ 0x5A).tobytes())
        upd(im.point(lambda v: v & 0xF0).tobytes())


@section("draw")
def _(upd: Update) -> None:
    rnd = random.Random(13)
    for mode in ("L", "RGB", "RGBA"):
        im = noise(mode, (203, 151), 14)
        d = ImageDraw.Draw(im)
        for i in range(40):
            x, y = rnd.uniform(-20, 210), rnd.uniform(-20, 160)
            box = (x, y, x + rnd.uniform(1, 90), y + rnd.uniform(1, 70))
            rgba = (
                rnd.randrange(256),
                rnd.randrange(256),
                rnd.randrange(256),
                rnd.randrange(256),
            )
            ink: int | tuple[int, ...] = (
                rgba[: len(im.getbands())] if mode != "L" else rgba[0]
            )
            d.rectangle(box, fill=ink)
            d.ellipse(box, outline=ink, width=3)
            d.pieslice(box, i * 17 % 360, i * 31 % 360 + 20, fill=ink)
            d.arc(box, i * 7, i * 7 + 200, fill=ink, width=2)
            d.rounded_rectangle(box, 5, outline=ink)
            d.polygon(
                [(rnd.uniform(0, 200), rnd.uniform(0, 150)) for _ in range(6)], fill=ink
            )
        upd(im.tobytes())
        rgb = noise("RGB", (203, 151), 15)
        rgba_ink = ImageDraw.Draw(rgb, "RGBA")
        for i in range(30):
            x, y = rnd.uniform(0, 200), rnd.uniform(0, 150)
            rgba_ink.rectangle(
                (x, y, x + 40, y + 30), fill=(200, 10, 90, rnd.randrange(256))
            )
            rgba_ink.ellipse(
                (x, y, x + 50, y + 20), fill=(20, 210, 90, rnd.randrange(256))
            )
        upd(rgb.tobytes())
        flood = im.point(lambda v: 255 if v > 128 else 0) if mode == "L" else im.copy()
        ImageDraw.floodfill(
            flood, (10, 10), 7 if mode == "L" else (7, 8, 9)[: len(im.getbands())]
        )
        upd(flood.tobytes())


@section("rank filters")
def _(upd: Update) -> None:
    for mode in ("L", "RGB"):
        im = noise(mode, (97, 61), 17)
        for size in (3, 5, 7, 9, 11):
            upd(im.filter(ImageFilter.MedianFilter(size)).tobytes())
            upd(im.filter(ImageFilter.MinFilter(size)).tobytes())
            upd(im.filter(ImageFilter.MaxFilter(size)).tobytes())
            upd(im.filter(ImageFilter.RankFilter(size, size * size // 3)).tobytes())


@section("transform")
def _(upd: Update) -> None:
    for mode in ("L", "LA", "RGB", "RGBA", "I", "F"):
        for size in ((1, 1), (9, 7), (61, 37)):
            im = noise(mode, size, size[0] + 3)
            w, hh = size
            for flt in (R.NEAREST, R.BILINEAR, R.BICUBIC):
                for angle in (0, 30, 90.5, 200):
                    upd(im.rotate(angle, flt).tobytes())
                    upd(im.rotate(angle, flt, expand=True).tobytes())
                upd(
                    im.transform(
                        (w + 5, hh + 3),
                        Image.Transform.PERSPECTIVE,
                        (1.1, 0.1, -2, 0.05, 1.05, 1, 0.001, 0.002),
                        flt,
                    ).tobytes()
                )
                upd(
                    im.transform(
                        size,
                        Image.Transform.QUAD,
                        (0, 0, 2, hh - 1, w, hh, w - 3, 1),
                        flt,
                    ).tobytes()
                )


@section("blur/unsharp")
def _(upd: Update) -> None:
    for mode in ("L", "LA", "RGB", "RGBA"):
        for size in ((1, 1), (5, 9), (67, 23), (203, 71)):
            im = noise(mode, size, size[1])
            for r in (0.4, 1, 2, 2.5, 3, 3.6, 4, 7, 40):
                upd(im.filter(ImageFilter.BoxBlur(r)).tobytes())
                upd(im.filter(ImageFilter.GaussianBlur(r)).tobytes())
                upd(im.filter(ImageFilter.GaussianBlur((r, r / 3))).tobytes())
            if mode in ("L", "RGB"):
                for args in ((2, 150, 3), (1, 300, 0), (5, 80, 10)):
                    upd(im.filter(ImageFilter.UnsharpMask(*args)).tobytes())


@section("fast: gaussian")
def _(upd: Update) -> None:
    # Fast mode (Image.set_fast): its kernels are exact integer arithmetic
    # too, so every build gives the same hash.
    if not hasattr(Image, "set_fast"):
        return
    previous = Image.get_fast()
    Image.set_fast(True)
    try:
        for mode in ("L", "LA", "RGB", "RGBA", "CMYK"):
            for size in ((1, 1), (2, 31), (5, 9), (67, 23), (203, 71)):
                im = noise(mode, size, size[0] + size[1])
                for r in (0.2, 0.5, 1, 1.5, 2.2, 3, 4.7, 6):
                    upd(im.filter(ImageFilter.GaussianBlur(r)).tobytes())
                    upd(im.filter(ImageFilter.GaussianBlur((r, 0))).tobytes())
                    upd(im.filter(ImageFilter.GaussianBlur((0.7, r))).tobytes())
                if mode in ("L", "RGB"):
                    upd(im.filter(ImageFilter.UnsharpMask(2, 150, 3)).tobytes())
    finally:
        Image.set_fast(previous)


@section("resize 8-bit")
def _(upd: Update) -> None:
    for mode in ("L", "LA", "RGB"):
        for w, hh in ((1, 1), (13, 7), (97, 61), (301, 17)):
            im = noise(mode, (w, hh), w)
            for size in (
                (w * 2 + 1, hh),
                (w, hh * 3),
                (max(1, w // 3), max(1, hh // 2)),
            ):
                for flt in (R.BILINEAR, R.BICUBIC, R.LANCZOS):
                    upd(im.resize(size, flt).tobytes())
            for n in (2, 3, (2, 1), (1, 3)):
                upd(im.reduce(n).tobytes())


@section("chops/convert")
def _(upd: Update) -> None:
    from PIL import ImageChops

    for size in ((1, 1), (35, 9), (130, 31)):
        a, b = noise("RGB", size, 21), noise("RGB", size, 22)
        for scale, offset in ((1.0, 0), (1.0, -30), (2.0, 7), (0.6, 0)):
            upd(ImageChops.add(a, b, scale, offset).tobytes())
            upd(ImageChops.subtract(a, b, scale, offset).tobytes())
        for f in (
            ImageChops.lighter,
            ImageChops.darker,
            ImageChops.difference,
            ImageChops.multiply,
            ImageChops.screen,
        ):
            upd(f(a, b).tobytes())
        lum = noise("L", size, 23)
        for mode in ("RGB", "RGBA", "LA"):
            upd(lum.convert(mode).tobytes())
        for mode in ("RGBA", "CMYK", "HSV"):
            upd(a.convert(mode).tobytes())
        upd(noise("RGBA", size, 24).convert("RGB").tobytes())


@section("unpremultiply")
def _(upd: Update) -> None:
    data = bytes(
        (c, a)[k == 3] for a in range(256) for c in range(256) for k in range(4)
    )
    im = Image.frombytes("RGBa", (256, 256), data)
    upd(im.convert("RGBA").tobytes())
    upd(Image.frombytes("RGBA", (256, 256), data).convert("RGBa").tobytes())


for name, digest in results.items():
    print(f"{name:20} {digest}")
