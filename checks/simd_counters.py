#!/usr/bin/env python3
"""Hardware counters per run of each benchmark case, read with
perf_event_open on this thread around every run: user-mode cycles,
instructions, last-level cache misses and branch misses. They count only
while the process runs, so other load on the machine reaches them only
through cache and predictor pollution, and the minimum over many runs
removes most of that: a measurement for a machine that is not idle, and
the stage-2 evidence of design/OPTIMIZATION_METHOD.md (instructions per
cycle and misses per instruction say whether a loop waits on memory).

Run with the build to measure on PYTHONPATH, for SECS seconds per case
(at least three runs), one line per case on stdout: name, runs, min
cycles, median cycles, min instructions, best ms, min LLC misses, min
branch misses::

    PYTHONPATH=/tmp/build PILLOW_ALLSIMD_ISA=sse4 \\
        python3 checks/simd_counters.py 1 > counters/sse4-1.txt

Needs perf_event_paranoid <= 2, or CAP_PERFMON; in Docker, run as root
with --cap-add PERFMON (a capability added to the container doesn't reach
a --user). ``--table DIR`` reads DIR/<name>-<round>.txt, takes the minimum
over the rounds, and prints a Markdown table per name (best ms, millions
of cycles, instructions per cycle, misses per thousand instructions) with
each name's cycles against ``--baseline`` (default: ``scalar`` if present,
else the first name), the geometric mean, and the cases slower than the
baseline by more than ``--threshold`` (for an old/new comparison, name the
files old-N.txt and new-N.txt and pass ``--baseline old``).

The cases are those of the perf-profiling loops in bench-results plus one
or more per kernel group of SIMDKernels.h, on a 2560x1600 image.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import math
import os
import pathlib
import platform
import struct
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# perf_event_open's syscall number per architecture (241 is asm-generic).
SYSCALL = {
    "x86_64": 298,
    "i686": 336,
    "i386": 336,
    "aarch64": 241,
    "riscv64": 241,
    "ppc64le": 319,
    "ppc64": 319,
    "s390x": 331,
    "armv7l": 364,
}
# PERF_TYPE_HARDWARE configs
HW_CPU_CYCLES, HW_INSTRUCTIONS, HW_CACHE_MISSES, HW_BRANCH_MISSES = 0, 1, 3, 5
COLUMNS = 8


def counter(config: int) -> int:
    """Opens a user-mode hardware counter on this thread; returns its fd."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    # struct perf_event_attr up to PERF_ATTR_SIZE_VER0 (64 bytes):
    # type, size, config, sample_period, sample_type, read_format,
    # flags (exclude_kernel | exclude_hv), wakeup_events, bp_type, bp_addr
    attr = struct.pack("=IIQQQQQIIQ", 0, 64, config, 0, 0, 0, 0x60, 0, 0, 0)
    buf = ctypes.create_string_buffer(attr, 64)
    fd = libc.syscall(
        ctypes.c_long(SYSCALL[platform.machine()]),
        buf,
        ctypes.c_long(0),  # this process
        ctypes.c_long(-1),  # any CPU
        ctypes.c_long(-1),  # no group
        ctypes.c_ulong(0),
    )
    if fd < 0:
        err = ctypes.get_errno()
        msg = (
            "perf_event_open refused: set kernel.perf_event_paranoid to 2 or"
            " lower, or run as root in a container with --cap-add PERFMON"
        )
        raise OSError(err, f"{msg} ({os.strerror(err)})")
    return int(fd)


def read(fd: int) -> int:
    return int(struct.unpack("Q", os.read(fd, 8))[0])


def make_cases() -> dict[str, Callable[[], object]]:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

    w, h = 2560, 1600
    im = Image.radial_gradient("L").resize((w, h)).convert("RGB")
    im = Image.merge(
        "RGB",
        [
            im.getchannel(0),
            im.getchannel(0).rotate(90, expand=0),
            Image.effect_noise((w, h), 40),
        ],
    )
    rgba = im.convert("RGBA")
    rgba.putalpha(im.getchannel(2))
    lum = im.getchannel(1)
    small = im.reduce(2)
    ycbcr = im.convert("YCbCr")
    premul = rgba.convert("RGBa")
    im2 = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    rgba2 = rgba.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    rs = Image.Resampling
    tr = Image.Transpose
    coeffs = (0.9, 0.05, -40, 0.03, 1.02, -30, 0.00002, 0.00003)
    lut_l = bytes(range(256))[::-1]
    lut_rgb = bytes(range(256)) + bytes(range(256))[::-1] + bytes(range(256))
    lut3d = ImageFilter.Color3DLUT.generate(17, lambda r, g, b: (b, g, r))
    kernel5 = ImageFilter.Kernel((5, 5), [1] * 25, scale=25)
    sharpen_int = ImageFilter.Kernel(
        (3, 3), [-1, -1, -1, -1, 9, -1, -1, -1, -1], scale=1
    )
    scratch = im.copy()
    draw = ImageDraw.Draw(scratch)
    draw_rgba = ImageDraw.Draw(im.copy(), "RGBA")  # RGBA ink blended onto RGB
    poly = [(100, 100), (2400, 300), (2000, 1500), (300, 1200)]
    png = io.BytesIO()
    im.save(png, "PNG", compress_level=1)
    png_bytes = png.getvalue()
    raw_rgb = im.tobytes()
    mask = lum.point(lambda v: 255 if v > 128 else 0)

    def exact(f: Callable[[], object]) -> Callable[[], object]:
        def g() -> object:
            with Image.fast_mode(False):
                return f()

        return g

    def png_save() -> None:
        im.save(io.BytesIO(), "PNG", compress_level=1)

    def png_load() -> None:
        Image.open(io.BytesIO(png_bytes)).load()

    def paste() -> None:
        scratch.paste(im2, (0, 0), mask)

    return {
        "rot-rgba-bicubic": lambda: rgba.rotate(30, rs.BICUBIC),
        "rot-rgb-bicubic": lambda: im.rotate(30, rs.BICUBIC),
        "rot-rgb-bilinear": lambda: im.rotate(30, rs.BILINEAR),
        "persp-rgb-bilinear": lambda: im.transform(
            im.size, Image.Transform.PERSPECTIVE, coeffs, rs.BILINEAR
        ),
        "ycbcr2rgb": lambda: ycbcr.convert("RGB"),
        "rgb2ycbcr": lambda: im.convert("YCbCr"),
        "rgb2hsv": lambda: im.convert("HSV"),
        "up-rgba-bilinear": lambda: rgba.resize((5478, 3424), rs.BILINEAR),
        "rgb2rgba": lambda: im.convert("RGBA"),
        "rgba2rgb": lambda: rgba.convert("RGB"),
        "premul": lambda: rgba.convert("RGBa"),
        "transpose-l": lambda: lum.transpose(tr.ROTATE_90),
        "composite": lambda: Image.alpha_composite(rgba2, rgba),
        "min3": lambda: im.filter(ImageFilter.MinFilter(3)),
        "median3": lambda: im.filter(ImageFilter.MedianFilter(3)),
        "box2": lambda: im.filter(ImageFilter.BoxBlur(2)),
        "box10": lambda: im.filter(ImageFilter.BoxBlur(10)),
        "l2rgb": lambda: lum.convert("RGB"),
        "invert": lambda: ImageOps.invert(im),
        "reduce4": lambda: im.reduce(4),
        "reduce2": lambda: im.reduce(2),
        "up-rgb-bilinear": lambda: im.resize((5478, 3424), rs.BILINEAR),
        "h-rgb-bilinear": lambda: im.resize((2048, 1600), rs.BILINEAR),
        "h-rgb-bicubic": lambda: im.resize((2048, 1600), rs.BICUBIC),
        "down-rgb-bicubic": lambda: im.resize((2048, 1280), rs.BICUBIC),
        "rot-rgb-bicubic-exact": exact(lambda: im.rotate(30, rs.BICUBIC)),
        "rot-rgb-bilinear-exact": exact(lambda: im.rotate(30, rs.BILINEAR)),
        "down-l-bicubic": lambda: lum.resize((2048, 1280), rs.BICUBIC),
        "reduce4-l": lambda: lum.reduce(4),
        "rgb2l": lambda: im.convert("L"),
        "unpremul": lambda: premul.convert("RGBA"),
        "lut3d": lambda: im.filter(lut3d),
        "transpose-rgb": lambda: im.transpose(tr.ROTATE_90),
        "flip-rgb": lambda: im.transpose(tr.FLIP_TOP_BOTTOM),
        "box2-l": lambda: lum.filter(ImageFilter.BoxBlur(2)),
        "gauss2": lambda: im.filter(ImageFilter.GaussianBlur(2)),
        "gauss2-exact": exact(lambda: im.filter(ImageFilter.GaussianBlur(2))),
        "unsharp": lambda: im.filter(ImageFilter.UnsharpMask(2)),
        "sharpen-f3": lambda: im.filter(ImageFilter.SHARPEN),
        "sharpen-i3": lambda: im.filter(sharpen_int),
        "sharpen-i3-l": lambda: lum.filter(sharpen_int),
        "kernel5": lambda: im.filter(kernel5),
        "blend": lambda: Image.blend(im, im2, 0.3),
        "chops-multiply": lambda: ImageChops.multiply(im, im2),
        "chops-add": lambda: ImageChops.add(im, im2),
        "png-save": png_save,
        "png-load": png_load,
        "tobytes-rgb": lambda: im.tobytes(),
        "tobytes-bgr": lambda: im.tobytes("raw", "BGR"),
        "tobytes-bgra": lambda: rgba.tobytes("raw", "BGRA"),
        "frombytes-rgb": lambda: Image.frombytes("RGB", im.size, raw_rgb),
        "point-l": lambda: lum.point(lut_l),
        "point-rgb": lambda: im.point(lut_rgb),
        "split": lambda: im.split(),
        "merge": lambda: Image.merge("RGB", (lum, lum, lum)),
        "paste-mask": paste,
        "draw-polygon": lambda: draw.polygon(poly, fill=(10, 200, 30)),
        "draw-polygon-rgba": lambda: draw_rgba.polygon(poly, fill=(10, 200, 30, 128)),
        "quantize": lambda: small.quantize(64),
        "median5": lambda: im.filter(ImageFilter.MedianFilter(5)),
        "max5": lambda: im.filter(ImageFilter.MaxFilter(5)),
    }


def measure(f: Callable[[], object], secs: float, fds: list[int]) -> list[float]:
    """Runs f for secs seconds (at least three runs); returns the row."""
    f()
    cyc: list[int] = []
    ins: list[int] = []
    llc: list[int] = []
    brm: list[int] = []
    wall: list[float] = []
    cfd, ifd, lfd, bfd = fds
    end = time.perf_counter() + secs
    while time.perf_counter() < end or len(cyc) < 3:
        b0, l0, i0, c0 = read(bfd), read(lfd), read(ifd), read(cfd)
        t0 = time.perf_counter()
        f()
        t1 = time.perf_counter()
        c1, i1, l1, b1 = read(cfd), read(ifd), read(lfd), read(bfd)
        cyc.append(c1 - c0)
        ins.append(i1 - i0)
        llc.append(l1 - l0)
        brm.append(b1 - b0)
        wall.append(t1 - t0)
    cyc.sort()
    return [
        len(cyc),
        cyc[0],
        cyc[len(cyc) // 2],
        min(ins),
        min(wall) * 1e3,
        min(llc),
        min(brm),
    ]


def run(secs: float, names: list[str]) -> None:
    fds = [
        counter(HW_CPU_CYCLES),
        counter(HW_INSTRUCTIONS),
        counter(HW_CACHE_MISSES),
        counter(HW_BRANCH_MISSES),
    ]
    cases = make_cases()
    for name in names or list(cases):
        n, cmin, cmed, imin, ms, lmin, bmin = measure(cases[name], secs, fds)
        print(
            f"{name}\t{n}\t{cmin}\t{cmed}\t{imin}\t{ms:.3f}\t{lmin}\t{bmin}",
            flush=True,
        )


def load(files: list[pathlib.Path]) -> dict[str, list[float]]:
    """The minimum per case over the files: cycles, instructions, ms,
    LLC misses, branch misses (the last two 0 in older six-column files)."""
    best: dict[str, list[float]] = {}
    for f in files:
        for line in f.read_text().splitlines():
            p = line.split("\t")
            if len(p) < 6:
                continue
            p += ["0"] * (COLUMNS - len(p))
            row = [float(p[2]), float(p[4]), float(p[5]), float(p[6]), float(p[7])]
            if p[0] in best:
                best[p[0]] = [min(a, b) for a, b in zip(best[p[0]], row)]
            else:
                best[p[0]] = row
    return best


def table(out: pathlib.Path, baseline: str | None, threshold: float) -> None:
    groups: dict[str, list[pathlib.Path]] = {}
    for f in sorted(out.glob("*-*.txt")):
        groups.setdefault(f.name.rsplit("-", 1)[0], []).append(f)
    if not groups:
        sys.exit(f"no <name>-<round>.txt files in {out}")
    data = {name: load(files) for name, files in groups.items()}
    tiers = ("scalar", "sse4", "avx2", "avxvnni", "neon")
    order: list[str] = [t for t in tiers if t in data]
    order += [t for t in data if t not in order]
    if baseline is None:
        baseline = "scalar" if "scalar" in data else order[0]
    base = data[baseline]
    print(f"Baseline {baseline}; the minimum per case over the rounds")
    for name in order:
        rows = data[name]
        ghz = sorted(c / (ms * 1e6) for c, _i, ms, _l, _b in rows.values())
        rounds = len(groups[name])
        print(
            f"\n### {name} ({len(rows)} cases, {rounds} rounds,"
            f" clock {ghz[len(ghz) // 2]:.2f} GHz)\n"
        )
        head = "| case | ms | Mcycles | IPC | LLC/Ki | br/Ki |"
        if name != baseline:
            head += f" vs {baseline} |"
        print(head)
        print("|" + "---:|" * (head.count("|") - 1))
        ratios = []
        slower = []
        for case, (cyc, ins, ms, llc, brm) in rows.items():
            line = (
                f"| {case} | {ms:.2f} | {cyc / 1e6:.1f} | {ins / cyc:.2f}"
                f" | {1e3 * llc / ins:.2f} | {1e3 * brm / ins:.2f} |"
            )
            if name != baseline and case in base:
                r = base[case][0] / cyc
                ratios.append(r)
                if r < 1 - threshold:
                    slower.append((case, r))
                line += f" {r:.2f}x |"
            print(line)
        if ratios:
            gm = math.exp(sum(map(math.log, ratios)) / len(ratios))
            print(
                f"\nGeometric mean of {baseline}'s cycles over {name}'s:"
                f" {gm:.3f}x (min {min(ratios):.3f}x, max {max(ratios):.3f}x)"
            )
            if slower:
                print(f"\nCases more than {threshold:.0%} slower than {baseline}:")
                for case, r in sorted(slower, key=lambda t: t[1]):
                    print(f"- {case}: {r:.3f}x")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("secs", nargs="?", type=float, help="seconds per case")
    ap.add_argument("cases", nargs="*", help="cases to run (default: all)")
    ap.add_argument("--list", action="store_true", help="list the cases")
    ap.add_argument("--table", metavar="DIR", help="summarize DIR/<name>-<round>.txt")
    ap.add_argument("--baseline", help="the name the others are compared with")
    ap.add_argument("--threshold", type=float, default=0.03)
    args = ap.parse_args()
    if args.list:
        print("\n".join(make_cases()))
    elif args.table:
        table(pathlib.Path(args.table), args.baseline, args.threshold)
    elif args.secs is None:
        ap.error("SECS or --table DIR is required")
    else:
        run(args.secs, args.cases)


if __name__ == "__main__":
    main()
