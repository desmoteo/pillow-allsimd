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

SEED = 1234


def operations(Image, ImageChops, ImageFilter, w: int, h: int) -> dict:
    """Name -> zero-argument callable returning an image (or list of images)."""
    import random

    rnd = random.Random(SEED)
    rgb = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3))
    rgb2 = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3))
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
        "reduce L 2": lambda: L.reduce(2),
        "reduce L 4": lambda: L.reduce(4),
        "gaussian RGB r=5": lambda: rgb.filter(ImageFilter.GaussianBlur(5)),
        "gaussian L r=5": lambda: L.filter(ImageFilter.GaussianBlur(5)),
        "box RGB r=2": lambda: rgb.filter(ImageFilter.BoxBlur(2)),
        "unsharp RGB": lambda: rgb.filter(ImageFilter.UnsharpMask()),
        "filter RGB 3x3 SHARPEN": lambda: rgb.filter(ImageFilter.SHARPEN),
        "filter RGB 5x5 SMOOTH_MORE": lambda: rgb.filter(ImageFilter.SMOOTH_MORE),
        "filter L 3x3 SHARPEN": lambda: L.filter(ImageFilter.SHARPEN),
        "filter L 5x5 SMOOTH_MORE": lambda: L.filter(ImageFilter.SMOOTH_MORE),
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
    }


def import_pil(path: str):
    if path != "-":
        sys.path.insert(0, path)
    from PIL import Image, ImageChops, ImageFilter, __file__ as pil_file, __version__

    return Image, ImageChops, ImageFilter, pil_file, __version__


def worker(path: str, mode: str, size: tuple[int, int], only: list[str]) -> None:
    """Runs in a subprocess; prints one JSON object."""
    Image, ImageChops, ImageFilter, pil_file, version = import_pil(path)
    ops = operations(Image, ImageChops, ImageFilter, *size)
    if only:
        ops = {k: v for k, v in ops.items() if any(o in k for o in only)}
    result: dict = {"pil": pil_file, "version": version, "ops": {}}
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
            images = out if isinstance(out, (list, tuple)) else [out]
            data = b"".join(im.tobytes() for im in images)
            result["ops"][name] = base64.b64encode(data).decode()
    print(json.dumps(result))


def run_worker(path: str, mode: str, size: tuple[int, int], only: list[str], cpu):
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--worker", path,
           "--mode", mode, "--size", f"{size[0]}x{size[1]}"]
    for o in only:
        cmd += ["--only", o]
    if cpu is not None and sys.platform.startswith("linux") and shutil.which("taskset"):
        cmd = ["taskset", "-c", str(cpu)] + cmd
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def machine_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "cpus": os.cpu_count()}
    try:
        if sys.platform.startswith("linux"):
            info["cpu"] = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        elif sys.platform == "darwin":
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
    except OSError:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--baseline", help="variant to compare against (default: first)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--size", default="2560x1600", help="benchmark image size")
    ap.add_argument("--only", action="append", default=[],
                    help="run only operations whose name contains this (repeatable)")
    ap.add_argument("--cpu", type=int, help="pin to this CPU (Linux, needs taskset)")
    ap.add_argument("--check", action="store_true", help="also compare outputs")
    ap.add_argument("--json", help="write all results to this file")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="time", help=argparse.SUPPRESS)
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))

    if args.worker:
        worker(args.worker, args.mode, size, args.only)
        return 0

    if not args.variant:
        ap.error("give at least one --variant NAME=PATH")
    variants = dict(v.split("=", 1) for v in args.variant)
    names = list(variants)
    baseline = args.baseline or names[0]
    if baseline not in variants:
        ap.error(f"unknown baseline {baseline}")

    report: dict = {"machine": machine_info(), "size": args.size, "variants": {}}
    timings: dict = {n: {} for n in names}
    for r in range(args.rounds):
        for n in names:
            res = run_worker(variants[n], "time", size, args.only, args.cpu)
            report["variants"][n] = {"pil": res["pil"], "version": res["version"]}
            for op, ms in res["ops"].items():
                timings[n].setdefault(op, []).append(ms)
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr)

    best = {n: {op: min(v) for op, v in timings[n].items()} for n in names}
    report["timings_ms"] = timings
    report["best_ms"] = best

    print(f"\nMachine: {report['machine']['platform']}, image {args.size}, "
          f"best of {args.rounds} rounds, times in ms")
    for n in names:
        print(f"  {n}: Pillow {report['variants'][n]['version']} from {report['variants'][n]['pil']}")
    others = [n for n in names if n != baseline]
    width = max([11] + [len(n) + 2 for n in names])
    swidth = max([len(baseline) + 6] + [len(n) + 4 for n in others])
    print()
    print(f"{'':28s}" + " " * (width * len(names))
          + "".join(f"{'vs ' + baseline:>{swidth}s}" for _ in others))
    print(f"{'operation':28s}" + "".join(f"{n:>{width}s}" for n in names)
          + "".join(f"{n:>{swidth}s}" for n in others))
    for op in best[baseline]:
        row = f"{op:28s}" + "".join(f"{best[n][op]:{width}.2f}" for n in names)
        row += "".join(f"{best[baseline][op] / best[n][op]:{swidth - 1}.2f}x" for n in others)
        print(row)

    if args.check:
        outputs = {n: run_worker(variants[n], "check", (67, 23), args.only, None)["ops"]
                   for n in names}
        report["check"] = {}
        print(f"\nOutput differences vs {baseline} (67x23 images):")
        for n in others:
            diffs = {}
            for op, b64 in outputs[baseline].items():
                a, b = base64.b64decode(b64), base64.b64decode(outputs[n][op])
                if a != b:
                    d = [abs(x - y) for x, y in zip(a, b)] if len(a) == len(b) else [-1]
                    diffs[op] = {"max": max(d), "count": sum(1 for v in d if v)}
            report["check"][n] = diffs
            print(f"  {n}: " + ("identical" if not diffs else f"{len(diffs)} operations differ"))
            for op, d in diffs.items():
                print(f"    {op:30s} max {d['max']:4d}  values {d['count']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
