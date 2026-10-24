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
from collections.abc import Callable
from typing import Any

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

    return {
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


SETS = {
    "simd": (operations,),
    "extra": (extra_operations,),
    "all": (operations, extra_operations),
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
    for op in best[baseline]:
        row = f"{op:28s}" + "".join(f"{best[n][op]:{width}.2f}" for n in names)
        row += "".join(
            f"{best[baseline][op] / best[n][op]:{swidth - 1}.2f}x" for n in others
        )
        print(row)

    if args.check:
        outputs = {
            n: run_worker(variants[n], "check", (67, 23), args.only, None, args.set)[
                "ops"
            ]
            for n in names
        }
        report["check"] = {}
        print(f"\nOutput differences vs {baseline} (67x23 images):")
        for n in others:
            diffs: dict[str, dict[str, int]] = {}
            for op, b64 in outputs[baseline].items():
                a, b = base64.b64decode(b64), base64.b64decode(outputs[n][op])
                if a != b:
                    dv = (
                        [abs(x - y) for x, y in zip(a, b)] if len(a) == len(b) else [-1]
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
