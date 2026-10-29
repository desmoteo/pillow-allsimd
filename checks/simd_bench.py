#!/usr/bin/env python3
"""Benchmark and compare several Pillow builds on the same machine.

Each build is given as NAME=PATH, where PATH is a directory containing the
PIL package (as made by ``pip install --target PATH``), or ``-`` for the
Pillow importable by default. Every build runs in its own subprocess, and
the rounds are interleaved so drifting clock speeds affect all builds alike.
The best time of all rounds is reported.

    python3 checks/simd_bench.py --variant upstream=- --variant simd=/tmp/simd \\
        --baseline upstream --rounds 5 --check --json results.json

``--check`` also compares the output of every operation on small odd-sized
images between the builds and reports differences against the baseline.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 1234


def operations(
    Image: Any, ImageChops: Any, ImageFilter: Any, w: int, h: int
) -> dict[str, Callable[[], Any]]:
    """Name -> zero-argument callable returning an image (or list of images)."""
    import random

    rnd = random.Random(SEED)
    rgb = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3))
    rgb2 = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3))
    rgb_bytes = rgb2.tobytes()
    L = Image.frombytes("L", (w, h), rnd.randbytes(w * h))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.frombytes("L", (w, h), rnd.randbytes(w * h)))
    rgba2 = rgba.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    rgba_pm = rgba.convert("RGBa")
    lut = ImageFilter.Color3DLUT.generate(17, lambda r, g, b: (g, b, r))
    bands = rgb.split()
    small, big = (max(1, w // 4), max(1, h // 4)), (w * 3 // 2, h * 3 // 2)
    R = Image.Resampling
    return {
        "resize RGB 1/4 bicubic": lambda: rgb.resize(small, R.BICUBIC),
        "resize RGB x1.5 lanczos": lambda: rgb.resize(big, R.LANCZOS),
        "resize RGBA 1/4 bicubic": lambda: rgba.resize(small, R.BICUBIC),
        "resize L 1/4 bicubic": lambda: L.resize(small, R.BICUBIC),
        "resize L x1.5 lanczos": lambda: L.resize(big, R.LANCZOS),
        "reduce RGB 2": lambda: rgb.reduce(2),
        "reduce RGB (1, 2)": lambda: rgb.reduce((1, 2)),
        "reduce RGB (2, 1)": lambda: rgb.reduce((2, 1)),
        "reduce RGB 3": lambda: rgb.reduce(3),
        "reduce RGB 4": lambda: rgb.reduce(4),
        "reduce RGB 5": lambda: rgb.reduce(5),
        "reduce RGB 8": lambda: rgb.reduce(8),
        "reduce L 2": lambda: L.reduce(2),
        "reduce L 4": lambda: L.reduce(4),
        "gaussian RGB r=5": lambda: rgb.filter(ImageFilter.GaussianBlur(5)),
        "gaussian L r=5": lambda: L.filter(ImageFilter.GaussianBlur(5)),
        "box RGB r=2": lambda: rgb.filter(ImageFilter.BoxBlur(2)),
        "box L r=2": lambda: L.filter(ImageFilter.BoxBlur(2)),
        "unsharp RGB": lambda: rgb.filter(ImageFilter.UnsharpMask()),
        "filter RGB 3x3 SHARPEN": lambda: rgb.filter(ImageFilter.SHARPEN),
        "filter RGB 5x5 SMOOTH_MORE": lambda: rgb.filter(ImageFilter.SMOOTH_MORE),
        "filter L 3x3 SHARPEN": lambda: L.filter(ImageFilter.SHARPEN),
        "filter L 5x5 SMOOTH_MORE": lambda: L.filter(ImageFilter.SMOOTH_MORE),
        "filter RGB 3x3 FIND_EDGES": lambda: rgb.filter(ImageFilter.FIND_EDGES),
        "filter L 3x3 CONTOUR": lambda: L.filter(ImageFilter.CONTOUR),
        "convert RGB->L": lambda: rgb.convert("L"),
        "convert RGB->LA": lambda: rgb.convert("LA"),
        "convert RGBA->RGBa": lambda: rgba.convert("RGBa"),
        "convert RGBa->RGBA": lambda: rgba_pm.convert("RGBA"),
        "alpha_composite": lambda: Image.alpha_composite(rgba, rgba2),
        "split RGBA": lambda: rgba.split(),
        "merge RGB": lambda: Image.merge("RGB", bands),
        "colorlut RGB 17": lambda: rgb.filter(lut),
        "blend RGB": lambda: Image.blend(rgb, rgb2, 0.3),
        "chops multiply": lambda: ImageChops.multiply(rgb, rgb2),
        "chops screen": lambda: ImageChops.screen(rgb, rgb2),
        "chops multiply L": lambda: ImageChops.multiply(L, L),
        "median 3 L": lambda: L.filter(ImageFilter.MedianFilter(3)),
        "max 3 RGB": lambda: rgb.filter(ImageFilter.MaxFilter(3)),
        "histogram RGB": lambda: rgb.histogram(),
        "chops offset RGB": lambda: ImageChops.offset(rgb, 100, 50),
        "frombytes RGB": lambda: Image.frombytes("RGB", (w, h), rgb_bytes),
        "tobytes RGB": lambda: rgb.tobytes(),
    }


def extra_operations(
    Image: Any, ImageChops: Any, ImageFilter: Any, w: int, h: int
) -> dict[str, Callable[[], Any]]:
    """Common operations with no SIMD kernel, to find where one would pay."""
    import importlib
    import random

    # The PIL that import_pil() put first on the path.
    ImageOps = importlib.import_module("PIL.ImageOps")
    rnd = random.Random(SEED)
    rgb = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3))
    rgb2 = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3))
    L = Image.frombytes("L", (w, h), rnd.randbytes(w * h))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.frombytes("L", (w, h), rnd.randbytes(w * h)))
    ycc = rgb.convert("YCbCr")
    # A palette image with large flat areas, as from a real quantised photo.
    P = rgb.resize((max(1, w // 10), max(1, h // 10))).quantize(64).resize((w, h))
    F = L.convert("F")
    I16 = L.convert("I;16")
    I32 = L.convert("I")
    mask = L.point(lambda v: 255 if v > 128 else 0)
    # 4 << 13 overflows the 16-bit integer kernels: these take the float ones.
    float5 = ImageFilter.Kernel((5, 5), [-1] * 12 + [24] + [-1] * 12, 1)
    small = (max(1, w // 4), max(1, h // 4))
    R, T = Image.Resampling, Image.Transform

    def paste(im: Any, src: Any, m: Any) -> Any:
        out = im.copy()
        out.paste(src, (0, 0), m)
        return out

    def thumbnail() -> Any:
        out = rgb.copy()
        out.thumbnail((max(1, w // 8), max(1, h // 8)))
        return out

    def putalpha() -> Any:
        out = rgb.convert("RGBA")
        out.putalpha(L)
        return out

    ops = {
        "rotate RGB 30 nearest": lambda: rgb.rotate(30, R.NEAREST),
        "rotate RGB 30 bilinear": lambda: rgb.rotate(30, R.BILINEAR),
        "rotate RGB 30 bicubic": lambda: rgb.rotate(30, R.BICUBIC),
        "perspective RGBA bilinear": lambda: rgba.transform(
            (w, h), T.PERSPECTIVE, (1.1, 0.1, -5, 0.05, 0.9, 3, 1e-5, 2e-5), R.BILINEAR
        ),
        "transpose RGB ROTATE_90": lambda: rgb.transpose(Image.Transpose.ROTATE_90),
        "transpose RGB FLIP_LR": lambda: rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "transpose L TRANSPOSE": lambda: L.transpose(Image.Transpose.TRANSPOSE),
        "resize RGB 1/4 nearest": lambda: rgb.resize(small, R.NEAREST),
        "resize F 1/4 bicubic": lambda: F.resize(small, R.BICUBIC),
        "resize I;16 1/4 bicubic": lambda: I16.resize(small, R.BICUBIC),
        "resize F x1.5 lanczos": lambda: F.resize((w * 3 // 2, h * 3 // 2), R.LANCZOS),
        "resize I;16 x1.5 lanczos": lambda: I16.resize(
            (w * 3 // 2, h * 3 // 2), R.LANCZOS
        ),
        "resize I 1/4 bicubic": lambda: I32.resize(small, R.BICUBIC),
        "convert RGB->RGBA": lambda: rgb.convert("RGBA"),
        "convert RGBA->RGB": lambda: rgba.convert("RGB"),
        "convert L->RGB": lambda: L.convert("RGB"),
        "convert P->RGB": lambda: P.convert("RGB"),
        "convert RGB->YCbCr": lambda: rgb.convert("YCbCr"),
        "convert YCbCr->RGB": lambda: ycc.convert("RGB"),
        "convert RGB->HSV": lambda: rgb.convert("HSV"),
        "convert RGB->CMYK": lambda: rgb.convert("CMYK"),
        "convert L->F": lambda: L.convert("F"),
        "convert L->1 dither": lambda: L.convert("1"),
        "convert RGB->P adaptive": lambda: rgb.convert(
            "P", palette=Image.Palette.ADAPTIVE
        ),
        "quantize RGB fastoctree": lambda: rgb.quantize(256, Image.Quantize.FASTOCTREE),
        "paste RGB mask 1-bit L": lambda: paste(rgb, rgb2, mask),
        "composite RGB mask L": lambda: Image.composite(rgb, rgb2, L),
        "paste RGBA over RGB": lambda: paste(rgb, rgba, rgba),
        "putalpha RGBA": putalpha,
        "point RGB lut": lambda: rgb.point([255 - i for i in range(256)] * 3),
        "autocontrast RGB": lambda: ImageOps.autocontrast(rgb, cutoff=1),
        "equalize RGB": lambda: ImageOps.equalize(rgb),
        "getextrema RGB": lambda: [v for band in rgb.getextrema() for v in band],
        "chops add": lambda: ImageChops.add(rgb, rgb2),
        "chops difference": lambda: ImageChops.difference(rgb, rgb2),
        "chops lighter": lambda: ImageChops.lighter(rgb, rgb2),
        "filter RGB 5x5 float": lambda: rgb.filter(float5),
        "filter L 5x5 float": lambda: L.filter(float5),
        "filter RGB EDGE_ENHANCE": lambda: rgb.filter(ImageFilter.EDGE_ENHANCE),
        "mode 3 L": lambda: L.filter(ImageFilter.ModeFilter(3)),
        "expand RGB 20": lambda: ImageOps.expand(rgb, 20),
        "thumbnail RGB 1/8": thumbnail,
    }
    if hasattr(R, "MKS2021"):  # Pillow 13.0 and newer
        # Vertical pass first: the height shrinks most.
        ops.update(
            {
                "resize RGB 1/4 MKS2013": lambda: rgb.resize(small, R.MKS2013),
                "resize RGB 1/4 MKS2021": lambda: rgb.resize(small, R.MKS2021),
                "resize L 1/4 MKS2021": lambda: L.resize(small, R.MKS2021),
                "resize RGB 1/8 height lanczos": lambda: rgb.resize(
                    (w, max(1, h // 8)), R.LANCZOS
                ),
            }
        )
    return ops


def api_operations(
    Image: Any, ImageChops: Any, ImageFilter: Any, w: int, h: int
) -> dict[str, Callable[[], Any]]:
    """The rest of the public API: drawing, text, ImageOps, ImageEnhance,
    ImageMath, ImageMorph, ImageStat, other Image methods and the codecs.
    Shapes and text use fixed counts with coordinates scaled to the image,
    so their cost doesn't grow with the pixel count."""
    import importlib
    import io
    import random

    def mod(name: str) -> Any:
        # The PIL that import_pil() put first on the path.
        return importlib.import_module("PIL." + name)

    ImageDraw, ImageFont, ImageOps = mod("ImageDraw"), mod("ImageFont"), mod("ImageOps")
    ImageEnhance, ImageMath, ImageMorph = (
        mod("ImageEnhance"),
        mod("ImageMath"),
        mod("ImageMorph"),
    )
    ImageStat, features = mod("ImageStat"), mod("features")
    rnd = random.Random(SEED)
    rgb = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3))
    rgb2 = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3))
    L = Image.frombytes("L", (w, h), rnd.randbytes(w * h))
    L2 = Image.frombytes("L", (w, h), rnd.randbytes(w * h))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(L)
    R, T = Image.Resampling, Image.Transform
    # Smooth areas with a little noise, closer to a photo than noise is:
    # the codecs and palette code behave very differently on noise.
    small = (max(1, w // 16), max(1, h // 16))
    photo = rgb.resize(small, R.BOX).resize((w, h), R.BICUBIC)
    photo = ImageChops.add(photo, rgb.point(lambda v: v // 32), offset=-4)
    photo_L = photo.convert("L")
    photo_P = photo.quantize(256)
    binary = L.point(lambda v: 255 if v > 200 else 0)
    # Sized fonts need FreeType; the text operations are skipped without it.
    freetype = features.check("freetype2")
    font = ImageFont.load_default(max(8, h // 30)) if freetype else None
    sx, sy = w / 1000, h / 1000

    def pts(n: int, k: int) -> list[list[tuple[float, float]]]:
        return [
            [(rnd.uniform(0, w), rnd.uniform(0, h)) for _ in range(k)] for _ in range(n)
        ]

    def boxes(n: int, size: float) -> list[tuple[float, float, float, float]]:
        out = []
        for _ in range(n):
            x, y = rnd.uniform(0, w), rnd.uniform(0, h)
            bw, bh = rnd.uniform(4, size * sx), rnd.uniform(4, size * sy)
            out.append((x, y, x + bw, y + bh))
        return out

    lines2, lines_w, polys = pts(2000, 2), pts(500, 2), pts(300, 8)
    rects, ellipses, arcs = boxes(1000, 200), boxes(500, 200), boxes(300, 200)
    points = [(rnd.randrange(w), rnd.randrange(h)) for _ in range(100000)]
    text_pos = [(rnd.uniform(0, w * 0.6), rnd.uniform(0, h)) for _ in range(100)]
    words = "The quick brown fox jumps over the lazy dog 0123456789"

    def draw(mode: str, fn: Callable[[Any], None], im_mode: str = "RGB") -> Any:
        def run() -> Any:
            im = Image.new(im_mode, (w, h), (20, 40, 60, 255)[: len(im_mode)])
            fn(ImageDraw.Draw(im, mode or None))
            return im

        return run

    def lines(d: Any) -> None:
        for p in lines2:
            d.line(p, fill=(255, 128, 0))

    def wide_lines(d: Any) -> None:
        for p in lines_w:
            d.line(p, fill=(255, 128, 0), width=5)

    def rectangles(d: Any) -> None:
        for b in rects:
            d.rectangle(b, fill=(0, 200, 100))

    def rect_outlines(d: Any) -> None:
        for b in rects:
            d.rectangle(b, outline=(0, 200, 100), width=3)

    def ellipses_fill(d: Any) -> None:
        for b in ellipses:
            d.ellipse(b, fill=(200, 0, 100))

    def ellipse_outlines(d: Any) -> None:
        for b in ellipses:
            d.ellipse(b, outline=(200, 0, 100), width=4)

    def polygons(d: Any) -> None:
        for p in polys:
            d.polygon(p, fill=(90, 90, 250))

    def pieslices(d: Any) -> None:
        for i, b in enumerate(arcs):
            d.pieslice(b, i % 360, (i * 7) % 360 + 30, fill=(250, 250, 0))

    def arcs_w(d: Any) -> None:
        for i, b in enumerate(arcs):
            d.arc(b, i % 360, (i * 7) % 360 + 30, fill=(250, 250, 0), width=3)

    def rounded(d: Any) -> None:
        for b in arcs:
            d.rounded_rectangle(b, radius=12, fill=(10, 220, 220))

    def point_cloud(d: Any) -> None:
        d.point(points, fill=(255, 255, 255))

    def text(d: Any) -> None:
        for p in text_pos:
            d.text(p, words, fill=(255, 255, 255), font=font)

    def text_stroke(d: Any) -> None:
        for p in text_pos[:30]:
            d.text(p, words, fill=(255, 255, 255), font=font, stroke_width=2)

    def blend_ellipses(d: Any) -> None:
        for b in ellipses:
            d.ellipse(b, fill=(200, 0, 100, 128))

    def floodfill() -> Any:
        im = Image.new("RGB", (w, h), (0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, w // 4, h // 4), outline=(255, 255, 255))
        ImageDraw.floodfill(im, (w // 8, h // 8), (255, 0, 0))
        return im

    def textbbox() -> list[int]:
        d = ImageDraw.Draw(Image.new("L", (1, 1)))
        return [int(v) for p in text_pos for v in d.textbbox(p, words, font=font)]

    def pixel_access() -> list[int]:
        px = rgb.load()
        return [px[x, y][0] for x, y in points[:20000]]

    def stat() -> list[int]:
        s = ImageStat.Stat(rgb)
        return [int(v * 1000) for v in s.mean + s.stddev + s.median]

    def encode(im: Any, fmt: str, **kw: Any) -> Callable[[], bytes]:
        def run() -> bytes:
            buf = io.BytesIO()
            im.save(buf, fmt, **kw)
            return buf.getvalue()

        return run

    def decode(make: Callable[[], bytes], **kw: Any) -> Callable[[], Any]:
        # Encoded on the first call, which every harness makes untimed.
        data: list[bytes] = []

        def run() -> Any:
            if not data:
                data.append(make())
            with Image.open(io.BytesIO(data[0])) as im:
                if kw:
                    im.draft(*kw["draft"])
                im.load()
                return im.copy()

        return run

    morph = ImageMorph.MorphOp(op_name="dilation8")
    matrix = (0.3, 0.5, 0.2, 0)
    tone = ImageOps.colorize
    quad = (0, 0, 0, h, w, h * 0.9, w * 0.95, 0)
    qw, qh = w // 4, h // 4
    mesh = [
        ((x, y, x + qw, y + qh), (x + 3, y, x, y + qh, x + qw, y + qh - 3, x + qw, y))
        for x in range(0, w - qw + 1, qw)
        for y in range(0, h - qh + 1, qh)
    ]
    chops_rgb = {
        n: getattr(ImageChops, n)
        for n in (
            "subtract",
            "add_modulo",
            "subtract_modulo",
            "darker",
            "soft_light",
            "hard_light",
            "overlay",
        )
    }
    ops: dict[str, Callable[[], Any]] = {
        "draw lines 1px": draw("", lines),
        "draw lines width 5": draw("", wide_lines),
        "draw rectangles filled": draw("", rectangles),
        "draw rectangles outline 3": draw("", rect_outlines),
        "draw ellipses filled": draw("", ellipses_fill),
        "draw ellipses outline 4": draw("", ellipse_outlines),
        "draw polygons filled": draw("", polygons),
        "draw pieslices": draw("", pieslices),
        "draw arcs width 3": draw("", arcs_w),
        "draw rounded rectangles": draw("", rounded),
        "draw points 100k": draw("", point_cloud),
        # ImageDraw blends RGBA ink only onto RGB images.
        "draw ellipses RGBA ink on RGB": draw("RGBA", blend_ellipses, "RGB"),
        "draw floodfill 1/16 area": floodfill,
        "ImageOps grayscale": lambda: ImageOps.grayscale(rgb),
        "ImageOps invert": lambda: ImageOps.invert(rgb),
        "ImageOps mirror": lambda: ImageOps.mirror(rgb),
        "ImageOps posterize 3": lambda: ImageOps.posterize(rgb, 3),
        "ImageOps solarize": lambda: ImageOps.solarize(rgb, 128),
        "ImageOps colorize L": lambda: tone(L, (0, 0, 80), (255, 240, 200)),
        "ImageOps fit 1/2": lambda: ImageOps.fit(rgb, (w // 2, h // 3)),
        "ImageOps pad": lambda: ImageOps.pad(rgb, (w, w)),
        "ImageOps contain 1/3": lambda: ImageOps.contain(rgb, (w // 3, h // 3)),
        "ImageOps scale 0.6": lambda: ImageOps.scale(rgb, 0.6),
        "ImageEnhance color": lambda: ImageEnhance.Color(rgb).enhance(1.4),
        "ImageEnhance contrast": lambda: ImageEnhance.Contrast(rgb).enhance(1.4),
        "ImageEnhance brightness": lambda: ImageEnhance.Brightness(rgb).enhance(1.4),
        "ImageEnhance sharpness": lambda: ImageEnhance.Sharpness(rgb).enhance(2.0),
        "ImageMath lambda_eval L": lambda: ImageMath.lambda_eval(
            lambda a: a["convert"](a["a"] * 2 + a["b"] - 100, "L"), a=L, b=L2
        ),
        "ImageMorph dilation8": lambda: morph.apply(binary)[1],
        "ImageStat mean stddev median": stat,
        "crop + copy": lambda: rgb.crop((w // 8, h // 8, w * 7 // 8, h * 7 // 8)),
        "paste RGB no mask": lambda: (lambda o: (o.paste(rgb2, (5, 5)), o)[1])(
            rgb.copy()
        ),
        "point L function": lambda: L.point(lambda v: v * 0.7 + 20),
        "point I scale offset": lambda: L.convert("I").point(lambda v: v * 3 + 7),
        "convert RGB->L matrix": lambda: rgb.convert("L", matrix),
        "convert RGB->P web dither": lambda: photo.convert("P"),
        "convert RGB->1 dither": lambda: photo.convert("1"),
        "convert L->I": lambda: L.convert("I"),
        "convert I->L": lambda: L.convert("I").convert("L"),
        "convert L->I;16": lambda: L.convert("I;16"),
        "convert P->RGBA": lambda: photo_P.convert("RGBA"),
        "quantize mediancut": lambda: photo.quantize(256, Image.Quantize.MEDIANCUT),
        "quantize maxcoverage": lambda: photo.quantize(256, Image.Quantize.MAXCOVERAGE),
        "getcolors palette image": lambda: [
            c for n, c in photo_P.getcolors(256) or [] for c in (n, c)
        ],
        "getbbox RGBA": lambda: list(rgba.getbbox() or ()),
        "entropy RGB": lambda: [int(rgb.entropy() * 1e6)],
        "histogram L masked": lambda: L.histogram(binary),
        "effect_spread 5": lambda: L.effect_spread(5),
        "linear_gradient": lambda: Image.linear_gradient("L"),
        "radial_gradient": lambda: Image.radial_gradient("L"),
        "rotate RGBA 45 expand bicubic": lambda: rgba.rotate(
            45, R.BICUBIC, expand=True
        ),
        "transform RGB quad bilinear": lambda: rgb.transform(
            (w, h), T.QUAD, quad, R.BILINEAR
        ),
        "transform RGB mesh bilinear": lambda: rgb.transform(
            (w, h), T.MESH, mesh, R.BILINEAR
        ),
        "transform RGB extent bicubic": lambda: rgb.transform(
            (w // 2, h // 2), T.EXTENT, (10, 10, w - 10, h - 10), R.BICUBIC
        ),
        "resize RGB x1.5 nearest": lambda: rgb.resize((w * 3 // 2, h * 3 // 2), 0),
        "resize RGB 1/3 box": lambda: rgb.resize((w // 3, h // 3), R.BOX),
        "resize RGB 1/2 hamming": lambda: rgb.resize((w // 2, h // 2), R.HAMMING),
        "resize RGB 1/2 bilinear": lambda: rgb.resize((w // 2, h // 2), R.BILINEAR),
        "transpose RGB TRANSVERSE": lambda: rgb.transpose(Image.Transpose.TRANSVERSE),
        "transpose L ROTATE_90": lambda: L.transpose(Image.Transpose.ROTATE_90),
        "transpose RGB FLIP_TB": lambda: rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        "transpose RGB ROTATE_270": lambda: rgb.transpose(Image.Transpose.ROTATE_270),
        "transpose RGB ROTATE_180": lambda: rgb.transpose(Image.Transpose.ROTATE_180),
        "transpose RGB TRANSPOSE": lambda: rgb.transpose(Image.Transpose.TRANSPOSE),
        "transpose L FLIP_LR": lambda: L.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "alpha_composite dest offset": lambda: (
            lambda o: (o.alpha_composite(rgba, (w // 4, h // 4)), o)[1]
        )(rgba.copy()),
        "blend RGBA": lambda: Image.blend(rgba, rgba.transpose(0), 0.3),
        "composite RGBA mask": lambda: Image.composite(rgba, rgb2.convert("RGBA"), L),
        "getchannel": lambda: rgb.getchannel(1),
        "merge RGBA": lambda: Image.merge("RGBA", rgba.split()),
        "tobytes RGBA BGRA": lambda: rgba.tobytes("raw", "BGRA"),
        "tobytes RGB BGR": lambda: rgb.tobytes("raw", "BGR"),
        "frombytes RGB BGR": lambda b=rgb.tobytes(): Image.frombytes(
            "RGB", (w, h), b, "raw", "BGR"
        ),
        "frombytes RGBA BGRA": lambda b=rgba.tobytes(): Image.frombytes(
            "RGBA", (w, h), b, "raw", "BGRA"
        ),
        "pixel access 20k": pixel_access,
        "get_flattened_data L": lambda: list(L.get_flattened_data())[:: w + 1],
        "chops invert": lambda: ImageChops.invert(rgb),
        "chops logical_and 1": lambda: ImageChops.logical_and(
            binary.convert("1"), L2.convert("1")
        ),
        "filter RGB BLUR": lambda: rgb.filter(ImageFilter.BLUR),
        "filter RGB DETAIL": lambda: rgb.filter(ImageFilter.DETAIL),
        "filter RGB EMBOSS": lambda: rgb.filter(ImageFilter.EMBOSS),
        "filter RGB SMOOTH": lambda: rgb.filter(ImageFilter.SMOOTH),
        "median 5 L": lambda: L.filter(ImageFilter.MedianFilter(5)),
        "median 3 RGB": lambda: rgb.filter(ImageFilter.MedianFilter(3)),
        "min 5 L": lambda: L.filter(ImageFilter.MinFilter(5)),
        "rank 7 L": lambda: L.filter(ImageFilter.RankFilter(7, 20)),
        "gaussian RGB r=20": lambda: rgb.filter(ImageFilter.GaussianBlur(20)),
        "gaussian RGB r=0.6": lambda: rgb.filter(ImageFilter.GaussianBlur(0.6)),
        "box RGB r=(10, 1)": lambda: rgb.filter(ImageFilter.BoxBlur((10, 1))),
        "unsharp L": lambda: L.filter(ImageFilter.UnsharpMask(2, 150, 3)),
    }
    if freetype:
        ops["draw text 100 lines"] = draw("", text)
        ops["draw text stroke 30 lines"] = draw("", text_stroke)
        ops["font textbbox 100"] = textbbox
    for n, f in chops_rgb.items():
        ops[f"chops {n}"] = (lambda f: lambda: f(rgb, rgb2))(f)
    if features.check("libimagequant"):
        ops["quantize libimagequant"] = lambda: photo.quantize(
            256, Image.Quantize.LIBIMAGEQUANT
        )
    codecs: dict[str, tuple[str | None, dict[str, Any]]] = {
        "PNG": ("zlib", {}),
        "PNG level 1": ("zlib", {"compress_level": 1}),
        "JPEG q85": ("jpg", {"quality": 85}),
        "WEBP q80": ("webp", {"quality": 80}),
        "TIFF raw": ("libtiff", {}),
        "TIFF LZW": ("libtiff", {"compression": "tiff_lzw"}),
        "TIFF deflate": ("libtiff", {"compression": "tiff_adobe_deflate"}),
        "BMP": (None, {}),
        "PPM": (None, {}),
    }
    for name, (feature, kw) in codecs.items():
        if feature and not features.check(feature):
            continue
        fmt = name.split()[0]
        ops[f"encode {name} RGB"] = encode(photo, fmt, **kw)
        ops[f"decode {name} RGB"] = decode(encode(photo, fmt, **kw))
    if features.check("zlib"):
        ops["encode PNG RGBA"] = encode(photo.convert("RGBA"), "PNG")
        ops["decode PNG RGBA"] = decode(encode(photo.convert("RGBA"), "PNG"))
        ops["encode PNG P"] = encode(photo_P, "PNG")
        ops["encode PNG L"] = encode(photo_L, "PNG")
    if features.check("jpg"):
        jpeg = encode(photo, "JPEG", quality=85)
        ops["decode JPEG RGB draft 1/4"] = decode(jpeg, draft=("RGB", small))
        ops["encode JPEG L"] = encode(photo_L, "JPEG", quality=85)
    ops["encode GIF P"] = encode(photo_P, "GIF")
    ops["decode GIF P"] = decode(encode(photo_P, "GIF"))
    return ops


SETS = {
    "simd": (operations,),
    "extra": (extra_operations,),
    "api": (api_operations,),
    "all": (operations, extra_operations),
    "full": (operations, extra_operations, api_operations),
}


def all_operations(
    which: str, Image: Any, ImageChops: Any, ImageFilter: Any, w: int, h: int
) -> dict[str, Callable[[], Any]]:
    ops: dict[str, Callable[[], Any]] = {}
    for fn in SETS[which]:
        ops.update(fn(Image, ImageChops, ImageFilter, w, h))
    return ops


def import_pil(path: str) -> tuple[Any, Any, Any, str, str]:
    if path != "-":
        sys.path.insert(0, path)
    from PIL import Image, ImageChops, ImageFilter, __version__
    from PIL import __file__ as pil_file

    return Image, ImageChops, ImageFilter, pil_file, __version__


def worker(
    path: str, mode: str, size: tuple[int, int], only: list[str], which: str
) -> None:
    """Runs in a subprocess; prints one JSON object."""
    Image, ImageChops, ImageFilter, pil_file, version = import_pil(path)
    ops = all_operations(which, Image, ImageChops, ImageFilter, *size)
    if only:
        ops = {k: v for k, v in ops.items() if any(o in k for o in only)}
    result: dict[str, Any] = {"pil": pil_file, "version": version, "ops": {}}
    for name, op in ops.items():
        if mode == "time":
            op()  # warm-up
            best = float("inf")
            count = 0
            start = time.perf_counter()
            while count < 3 or time.perf_counter() - start < 0.3:
                t = time.perf_counter()
                op()
                best = min(best, time.perf_counter() - t)
                count += 1
            result["ops"][name] = best * 1000
        else:
            out = op()
            if isinstance(out, bytes):
                data = out
            else:
                items = out if isinstance(out, (list, tuple)) else [out]
                # Images, or the counts of a histogram as 4-byte values.
                data = b"".join(
                    v.to_bytes(4, "little") if isinstance(v, int) else v.tobytes()
                    for v in items
                )
            result["ops"][name] = base64.b64encode(data).decode()
            # Histogram counts and other numbers are compared as 4-byte values.
            ints = not isinstance(out, bytes) and all(isinstance(v, int) for v in items)
            result.setdefault("itemsize", {})[name] = 4 if ints else 1
    print(json.dumps(result))


def run_worker(
    path: str,
    mode: str,
    size: tuple[int, int],
    only: list[str],
    cpu: int | None,
    which: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-I",
        os.path.abspath(__file__),
        "--worker",
        path,
        "--mode",
        mode,
        "--size",
        f"{size[0]}x{size[1]}",
        "--set",
        which,
    ]
    for o in only:
        cmd += ["--only", o]
    if cpu is not None and sys.platform.startswith("linux") and shutil.which("taskset"):
        cmd = ["taskset", "-c", str(cpu)] + cmd
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def machine_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
    }
    try:
        if sys.platform.startswith("linux"):
            info["cpu"] = subprocess.run(
                ["lscpu"], capture_output=True, text=True
            ).stdout
        elif sys.platform == "darwin":
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()
    except OSError:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--baseline", help="variant to compare against (default: first)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--size", default="2560x1600", help="benchmark image size")
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        help="run only operations whose name contains this (repeatable)",
    )
    ap.add_argument(
        "--set",
        choices=sorted(SETS),
        default="simd",
        help="simd: operations with SIMD kernels (default); extra: common "
        "operations without; all: both",
    )
    ap.add_argument("--cpu", type=int, help="pin to this CPU (Linux, needs taskset)")
    ap.add_argument("--check", action="store_true", help="also compare outputs")
    ap.add_argument("--json", help="write all results to this file")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="time", help=argparse.SUPPRESS)
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    size = (w, h)

    if args.worker:
        worker(args.worker, args.mode, size, args.only, args.set)
        return 0

    if not args.variant:
        ap.error("give at least one --variant NAME=PATH")
    variants = dict(v.split("=", 1) for v in args.variant)
    names = list(variants)
    baseline = args.baseline or names[0]
    if baseline not in variants:
        ap.error(f"unknown baseline {baseline}")

    report: dict[str, Any] = {
        "machine": machine_info(),
        "size": args.size,
        "variants": {},
    }
    timings: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    for r in range(args.rounds):
        for n in names:
            res = run_worker(variants[n], "time", size, args.only, args.cpu, args.set)
            report["variants"][n] = {"pil": res["pil"], "version": res["version"]}
            for op, ms in res["ops"].items():
                timings[n].setdefault(op, []).append(ms)
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr)

    best = {n: {op: min(v) for op, v in timings[n].items()} for n in names}
    report["timings_ms"] = timings
    report["best_ms"] = best

    print(
        f"\nMachine: {report['machine']['platform']}, image {args.size}, "
        f"best of {args.rounds} rounds, times in ms"
    )
    for n in names:
        variant = report["variants"][n]
        print(f"  {n}: Pillow {variant['version']} from {variant['pil']}")
    others = [n for n in names if n != baseline]
    width = max([11] + [len(n) + 2 for n in names])
    swidth = max([len(baseline) + 6] + [len(n) + 4 for n in others])
    print()
    print(
        f"{'':28s}"
        + " " * (width * len(names))
        + "".join(f"{'vs ' + baseline:>{swidth}s}" for _ in others)
    )
    print(
        f"{'operation':28s}"
        + "".join(f"{n:>{width}s}" for n in names)
        + "".join(f"{n:>{swidth}s}" for n in others)
    )
    # Operations some builds lack, such as filters added in a later Pillow.
    all_ops = list(dict.fromkeys(op for n in names for op in best[n]))
    skipped = [op for op in all_ops if any(op not in best[n] for n in names)]
    for op in best[baseline]:
        if op in skipped:
            continue
        row = f"{op:28s}" + "".join(f"{best[n][op]:{width}.2f}" for n in names)
        row += "".join(
            f"{best[baseline][op] / best[n][op]:{swidth - 1}.2f}x" for n in others
        )
        print(row)

    if skipped:
        print(f"\nNot in every build, skipped: {', '.join(skipped)}")

    if args.check:
        checks = {
            n: run_worker(variants[n], "check", (67, 23), args.only, None, args.set)
            for n in names
        }
        outputs = {n: r["ops"] for n, r in checks.items()}
        itemsize = checks[baseline].get("itemsize", {})
        report["check"] = {}
        print(f"\nOutput differences vs {baseline} (67x23 images):")
        for n in others:
            diffs: dict[str, dict[str, int]] = {}
            for op, b64 in outputs[baseline].items():
                if op not in outputs[n]:
                    continue
                a, b = base64.b64decode(b64), base64.b64decode(outputs[n][op])
                if a != b:
                    size = itemsize.get(op, 1)
                    av = [
                        int.from_bytes(a[i : i + size], "little")
                        for i in range(0, len(a), size)
                    ]
                    bv = [
                        int.from_bytes(b[i : i + size], "little")
                        for i in range(0, len(b), size)
                    ]
                    dv = (
                        [abs(x - y) for x, y in zip(av, bv)]
                        if len(av) == len(bv)
                        else [-1]
                    )
                    diffs[op] = {"max": max(dv), "count": sum(1 for v in dv if v)}
            report["check"][n] = diffs
            print(
                f"  {n}: "
                + ("identical" if not diffs else f"{len(diffs)} operations differ")
            )
            for op, diff in diffs.items():
                print(f"    {op:30s} max {diff['max']:4d}  values {diff['count']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
