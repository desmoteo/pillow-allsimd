"""Thread scaling of per-image pipelines: the C parts release the GIL, the
Python parts don't."""

from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from PIL import Image, ImageOps

IMAGES = os.environ.get("PY2C_IMAGES", "Tests/images")

base = Image.open(os.path.join(IMAGES, "hopper.jpg")).convert("RGB")


def photo(size):
    im = base.resize(size, Image.Resampling.BICUBIC)
    return Image.blend(im, Image.effect_noise(size, 24).convert("RGB"), 0.15)


ex = Image.open(os.path.join(IMAGES, "hopper_orientation_6.jpg")).getexif()


def enc(im, **k):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=90, **k)
    return b.getvalue()


j256 = enc(photo((256, 256)), exif=ex)
j1080 = enc(photo((1920, 1080)))


def ml(d=j256):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(d)))
    return np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))


def thumb(d=j1080):
    im = Image.open(io.BytesIO(d))
    im.thumbnail((256, 256))
    im.save(io.BytesIO(), "JPEG", quality=85)


big = photo((1920, 1080))


def tiles():
    for y in range(0, 1080 - 64, 64):
        for x in range(0, 1920 - 64, 64):
            big.crop((x, y, x + 64, y + 64)).resize((32, 32)).convert("L")


ncpu = len(os.sched_getaffinity(0))
for name, fn, n in (
    ("ML loader jpeg 256", ml, 3000),
    ("thumbnail jpeg 1080p", thumb, 300),
    ("tiling 464 tiles", tiles, 60),
):
    base_rate = None
    for th in (1, 2, 4):
        if th > ncpu:
            break
        best = 0
        for _ in range(3):
            with ThreadPoolExecutor(th) as pool:
                t = time.perf_counter()
                list(pool.map(lambda _: fn(), range(n)))
                dt = time.perf_counter() - t
            best = max(best, n / dt)
        base_rate = base_rate or best
        print(
            f"{name:22s} threads {th}: {best:8.1f} /s  "
            f"speedup {best / base_rate:4.2f}x",
            flush=True,
        )
