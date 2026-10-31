"""Where the time goes on the Python side of Pillow-AllSIMD.

Run from the repository root: python3 checks/python_overhead/overhead.py
[json-out]. Images come from Tests/images (PY2C_IMAGES overrides). Each case reports the
median time per call; "split" cases also report, under cProfile, how much of
that time is in Pillow's C calls, in other C (zlib, file I/O) and in Python.
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import time

from PIL import (
    Image,
    ImageChops,
    ImageColor,
    ImageDraw,
    ImageFilter,
    ImageOps,
    ImageSequence,
    ImageStat,
)

HERE = os.environ.get("PY2C_IMAGES", "Tests/images")
RESULTS: dict[str, dict[str, float]] = {}


def bench(fn, min_time=0.3, repeat=7):
    n = 1
    while True:
        t = time.perf_counter()
        for _ in range(n):
            fn()
        dt = time.perf_counter() - t
        if dt > min_time / repeat:
            break
        n *= 2
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        times.append((time.perf_counter() - t) / n)
    return statistics.median(times)


PILLOW_C = (
    "ImagingCore",
    "ImagingDecoder",
    "ImagingEncoder",
    "ImagingDraw",
    "ImagingFont",
    "PIL._imaging",
    "_imaging",
    "_webp",
    "_avif",
    "_imagingcms",
    "_imagingft",
    "Font' objects",
    "PixelAccess",
    "WebPAnim",
    "WebPEncoder",
    "AvifDecoder",
    "AvifEncoder",
)
OTHER_C = ("zlib", "BytesIO", "BufferedReader", "BufferedWriter", "FileIO", "mmap")


def split(fn, n):
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        fn()
    pr.disable()
    st = pstats.Stats(pr)
    pil = other = py = 0.0
    for (filename, _, name), (_, _, tt, _, _) in st.stats.items():
        if filename == "~":
            if any(k in name for k in PILLOW_C):
                pil += tt
            elif any(k in name for k in OTHER_C) or "read" in name or "write" in name:
                other += tt
            else:
                py += tt  # builtins called by Python code (len, isinstance...)
        else:
            py += tt
    total = pil + other + py
    return pil / total, other / total, py / total


def case(name, fn, do_split=False, ref=None):
    t = bench(fn)
    if ref is not None:
        # interleaved, best of three, against clock drift
        tr = bench(ref)
        for _ in range(2):
            t = min(t, bench(fn))
            tr = min(tr, bench(ref))
    row = {"us": t * 1e6}
    msg = f"{name:48s} {t * 1e6:11.1f} us"
    if ref is not None:
        row["core_us"] = tr * 1e6
        row["py_share"] = 1 - tr / t
        msg += f"   core {tr * 1e6:10.1f} us   python {100 * (1 - tr / t):5.1f}%"
    if do_split:
        n = max(1, int(0.3 / t))
        pil, other, py = split(fn, n)
        row.update(pil=pil, other=other, py=py)
        msg += (
            f"   [cProfile: pillow C {100 * pil:4.1f}%, "
            f"other C {100 * other:4.1f}%, python {100 * py:4.1f}%]"
        )
    RESULTS[name] = row
    print(msg, flush=True)


def photo(size):
    base = Image.open(os.path.join(HERE, "hopper.jpg")).convert("RGB")
    im = base.resize(size, Image.Resampling.BICUBIC)
    noise = Image.effect_noise(size, 24).convert("RGB")
    return Image.blend(im, noise, 0.15)


def encoded(im, fmt, **kw):
    b = io.BytesIO()
    im.save(b, fmt, **kw)
    return b.getvalue()


def main():
    small = photo((64, 64))
    thumb = photo((256, 256))
    big = photo((1920, 1080))
    big_rgba = big.convert("RGBA")
    big_l = big.convert("L")

    print("== 1. Per-call overhead of Image methods (Image.method vs the C call)")
    for label, im in (("64x64", small), ("256x256", thumb), ("1920x1080", big)):
        core = im.im
        w, h = im.size
        case(
            f"resize {label} -> half, bilinear",
            lambda im=im, w=w, h=h: im.resize(
                (w // 2, h // 2), Image.Resampling.BILINEAR
            ),
            ref=lambda core=core, w=w, h=h: core.resize(
                (w // 2, h // 2), 2, (0, 0, w, h)
            ),
        )
        case(
            f"convert RGB->L {label}",
            lambda im=im: im.convert("L"),
            ref=lambda core=core: core.convert("L", 0),
        )
        case(
            f"crop {label} (quarter)",
            lambda im=im, w=w, h=h: im.crop((0, 0, w // 2, h // 2)),
            ref=lambda core=core, w=w, h=h: core.crop((0, 0, w // 2, h // 2)),
            do_split=True,
        )
        case(
            f"transpose FLIP_LEFT_RIGHT {label}",
            lambda im=im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            ref=lambda core=core: core.transpose(0),
        )
        case(
            f"GaussianBlur(2) {label}",
            lambda im=im: im.filter(ImageFilter.GaussianBlur(2)),
            ref=lambda core=core: core.gaussian_blur((2, 2)),
        )
        case(
            f"point(256-entry list) {label}",
            lambda im=im: im.point(list(range(256)) * 3),
            ref=lambda core=core: core.point(list(range(256)) * 3, None),
        )
        case(
            f"Image.new RGB {label}",
            lambda w=w, h=h: Image.new("RGB", (w, h), (10, 20, 30)),
            ref=lambda w=w, h=h: Image.core.fill("RGB", (w, h), (10, 20, 30)),
        )
        case(
            f"copy {label}", lambda im=im: im.copy(), ref=lambda core=core: core.copy()
        )

    print("\n== 2. Pure-Python algorithms and wrappers")
    case(
        "ImageStat mean+stddev 1920x1080 RGB",
        lambda: ImageStat.Stat(big).stddev,
        do_split=True,
    )
    case(
        "ImageStat mean+stddev 64x64 RGB",
        lambda: ImageStat.Stat(small).stddev,
        do_split=True,
    )
    case(
        "ImageOps.autocontrast 1920x1080 RGB",
        lambda: ImageOps.autocontrast(big),
        do_split=True,
    )
    case(
        "ImageOps.equalize 1920x1080 RGB", lambda: ImageOps.equalize(big), do_split=True
    )
    case(
        "ImageOps.autocontrast 64x64 RGB",
        lambda: ImageOps.autocontrast(small),
        do_split=True,
    )
    case(
        "point(lambda) 1920x1080 L",
        lambda: big_l.point(lambda v: 255 - v),
        do_split=True,
    )
    i16 = big_l.convert("I;16")
    case(
        "point(lambda) 1920x1080 I;16",
        lambda: i16.point(lambda v: v * 2),
        do_split=True,
    )
    case(
        "ImageChops.difference 1920x1080",
        lambda: ImageChops.difference(big, big),
        do_split=True,
    )
    case(
        "ImageOps.fit 1920x1080 -> 224x224",
        lambda: ImageOps.fit(big, (224, 224)),
        do_split=True,
    )
    case(
        "thumbnail 1920x1080 -> 256 (copy first)",
        lambda: big.copy().thumbnail((256, 256)),
        do_split=True,
    )
    case("ImageColor.getrgb('#a0b0c0')", lambda: ImageColor.getrgb("#a0b0c0"))
    case(
        "ImageColor.getrgb('rgb(10%,20%,30%)')",
        lambda: ImageColor.getrgb("rgb(10%,20%,30%)"),
    )
    case(
        "getpixel x1000 64x64",
        lambda: [small.getpixel((i & 63, i >> 6)) for i in range(1000)],
        do_split=True,
    )
    px = small.load()
    case(
        "PixelAccess [] x1000 64x64", lambda: [px[i & 63, i >> 6] for i in range(1000)]
    )
    case("get_flattened_data() 256x256", lambda: thumb.get_flattened_data())
    case("tobytes 1920x1080 RGBA", lambda: big_rgba.tobytes(), do_split=True)

    import numpy as np

    arr = np.asarray(big)
    case("np.asarray(im) 1920x1080 RGB", lambda: np.asarray(big), do_split=True)
    case("Image.fromarray 1920x1080 RGB", lambda: Image.fromarray(arr), do_split=True)
    case("np.asarray(im) 64x64 RGB", lambda: np.asarray(small), do_split=True)

    print("\n== 3. ImageDraw: per-primitive overhead")
    canvas = Image.new("RGB", (512, 512))

    def draw_rects():
        d = ImageDraw.Draw(canvas)
        for i in range(1000):
            d.rectangle((i % 500, i % 400, i % 500 + 8, i % 400 + 8), fill=(255, 0, 0))

    def draw_rects_str():
        d = ImageDraw.Draw(canvas)
        for i in range(1000):
            d.rectangle((i % 500, i % 400, i % 500 + 8, i % 400 + 8), fill="red")

    def draw_lines():
        d = ImageDraw.Draw(canvas)
        for i in range(1000):
            d.line((i % 500, 0, 511 - i % 500, 511), fill=(0, 255, 0), width=1)

    def draw_points():
        d = ImageDraw.Draw(canvas)
        for i in range(1000):
            d.point((i % 512, i // 2 % 512), fill=(0, 0, 255))

    def draw_text():
        d = ImageDraw.Draw(canvas)
        for i in range(100):
            d.text((i % 400, i * 5 % 480), "Hello, world", fill=(255, 255, 255))

    def core_rects():
        d = Image.core.draw(canvas.im, 0)
        for i in range(1000):
            d.draw_rectangle(
                (i % 500, i % 400, i % 500 + 8, i % 400 + 8), 0x0000FF, 1, 0
            )

    case("1000 rectangles fill=(r,g,b)", draw_rects, do_split=True, ref=core_rects)
    case("1000 rectangles fill='red'", draw_rects_str, do_split=True, ref=core_rects)
    case("1000 lines width=1", draw_lines, do_split=True)
    case("1000 points", draw_points, do_split=True)
    case("100 text() default font", draw_text, do_split=True)

    print("\n== 4. Open, decode and save (format plugins)")
    data = {
        "png 64x64": encoded(small, "PNG"),
        "png 1920x1080": encoded(big, "PNG"),
        "jpeg 64x64": encoded(small, "JPEG", quality=90),
        "jpeg 256x256": encoded(thumb, "JPEG", quality=90),
        "jpeg 1920x1080": encoded(big, "JPEG", quality=90),
        "jpeg 256x256 +exif": encoded(
            thumb,
            "JPEG",
            quality=90,
            exif=Image.open(os.path.join(HERE, "hopper_orientation_6.jpg")).getexif(),
        ),
        "webp 256x256": encoded(thumb, "WEBP", quality=90),
        "tiff 256x256": encoded(thumb, "TIFF"),
        "tiff 1920x1080 deflate": encoded(
            big, "TIFF", compression="tiff_adobe_deflate"
        ),
        "gif 256x256": encoded(thumb.convert("P"), "GIF"),
        "bmp 256x256": encoded(thumb, "BMP"),
        "ppm 256x256": encoded(thumb, "PPM"),
    }
    for k, v in data.items():
        case(
            f"open only (header) {k}",
            lambda v=v: Image.open(io.BytesIO(v)),
            do_split=True,
        )
        case(
            f"open+load {k}",
            lambda v=v: Image.open(io.BytesIO(v)).load(),
            do_split=True,
        )
    case(
        "open+load+exif_transpose jpeg 256x256 +exif",
        lambda: ImageOps.exif_transpose(
            Image.open(io.BytesIO(data["jpeg 256x256 +exif"]))
        ),
        do_split=True,
    )
    case(
        "open+getexif jpeg 256x256 +exif",
        lambda: Image.open(io.BytesIO(data["jpeg 256x256 +exif"])).getexif(),
        do_split=True,
    )
    # PNG written with many small IDAT chunks, as some encoders do
    png_small_idat = encoded(big, "PNG")
    import struct
    import zlib

    def rechunk(png, size):
        out = [png[:8]]
        pos = 8
        idat = b""
        while pos < len(png):
            n, t = struct.unpack(">I4s", png[pos : pos + 8])
            body = png[pos + 8 : pos + 8 + n]
            if t == b"IDAT":
                idat += body
            else:
                if t == b"IEND":
                    for i in range(0, len(idat), size):
                        c = idat[i : i + size]
                        out.append(
                            struct.pack(">I", len(c))
                            + b"IDAT"
                            + c
                            + struct.pack(">I", zlib.crc32(b"IDAT" + c))
                        )
                out.append(png[pos : pos + 12 + n])
            pos += 12 + n
        return b"".join(out)

    png8k = rechunk(png_small_idat, 8192)
    case(
        "open+load png 1920x1080, 8 KiB IDAT chunks",
        lambda: Image.open(io.BytesIO(png8k)).load(),
        do_split=True,
    )

    for fmt, kw in (
        ("PNG", {}),
        ("JPEG", {"quality": 90}),
        ("WEBP", {"quality": 90}),
        ("TIFF", {}),
        ("GIF", {}),
        ("BMP", {}),
    ):
        for label, im in (("64x64", small), ("1920x1080", big)):
            src = im.convert("P") if fmt == "GIF" else im
            case(
                f"save {fmt} {label}",
                lambda src=src, fmt=fmt, kw=kw: src.save(io.BytesIO(), fmt, **kw),
                do_split=True,
            )

    print("\n== 5. Pipelines")
    jpg_exif = data["jpeg 256x256 +exif"]
    jpg_big = data["jpeg 1920x1080"]

    def ml_loader():
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(jpg_exif)))
        im = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        return np.asarray(im)

    def thumb_service():
        im = Image.open(io.BytesIO(jpg_big))
        im.thumbnail((256, 256))
        im.save(io.BytesIO(), "JPEG", quality=85)

    def thumb_small():
        im = Image.open(io.BytesIO(data["jpeg 256x256"]))
        im.thumbnail((64, 64))
        im.save(io.BytesIO(), "JPEG", quality=85)

    def tiles():
        return [
            big.crop((x, y, x + 64, y + 64)).resize((32, 32)).convert("L")
            for y in range(0, 1080 - 64, 64)
            for x in range(0, 1920 - 64, 64)
        ]

    case(
        "ML loader: jpeg 256 +exif -> transpose -> 224 -> numpy",
        ml_loader,
        do_split=True,
    )
    case(
        "thumbnail service: jpeg 1920x1080 -> 256 -> jpeg", thumb_service, do_split=True
    )
    case("thumbnail service: jpeg 256x256 -> 64 -> jpeg", thumb_small, do_split=True)
    case("tiling: 464 crops 64x64 -> resize 32 -> L", tiles, do_split=True)

    print("\n== 6. Multi-frame formats")
    gif = open(os.path.join(HERE, "iss634.gif"), "rb").read()
    case(
        "GIF 245x245 x42: iterate all frames",
        lambda: [f.load() for f in ImageSequence.Iterator(Image.open(io.BytesIO(gif)))],
        do_split=True,
    )
    case(
        "GIF 245x245 x42: n_frames",
        lambda: Image.open(io.BytesIO(gif)).n_frames,
        do_split=True,
    )
    frames = [
        f.convert("RGB") for f in ImageSequence.Iterator(Image.open(io.BytesIO(gif)))
    ]
    case(
        "GIF save 42 frames (RGB in, optimize)",
        lambda: frames[0].save(
            io.BytesIO(), "GIF", save_all=True, append_images=frames[1:]
        ),
        do_split=True,
    )
    pages = [thumb.rotate(i * 10) for i in range(20)]
    tiff = io.BytesIO()
    pages[0].save(tiff, "TIFF", save_all=True, append_images=pages[1:])
    tiff = tiff.getvalue()
    case(
        "TIFF 256x256 x20: iterate all pages",
        lambda: [
            f.load() for f in ImageSequence.Iterator(Image.open(io.BytesIO(tiff)))
        ],
        do_split=True,
    )
    case(
        "TIFF save 20 pages 256x256",
        lambda: pages[0].save(
            io.BytesIO(), "TIFF", save_all=True, append_images=pages[1:]
        ),
        do_split=True,
    )

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(RESULTS, f, indent=1)


main()
