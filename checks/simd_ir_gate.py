#!/usr/bin/env python3
"""Fail when an operation executes more instructions than its baseline.

Timings need a quiet machine, which CI is not. Instruction counts under
callgrind are the same on every run, and three kinds of regression show
in them at once: a kernel that fell back to the scalar loop, a call per
pixel or per span where there was one per row, and a library call inside
a loop. They are not time (a vector instruction counts as one, a cache
miss as nothing), so a count that holds says little; one that rises says
where to look.

A fixed list of operations (GATED below), one or more per kernel group and
per-call path, runs under callgrind (``simd_callgrind.py``) in every tier
of a dispatch build (``-C simd=dispatch``; ``PILLOW_ALLSIMD_ISA`` selects
the tier). What is compared is the count inside Pillow's own extension
modules, which the machine's C library, Python and codec libraries do not
move; for the operations in WHOLE it is the whole count, because there
the question is which library path was taken.

    python3 checks/simd_ir_gate.py --build /path/to/build
    python3 checks/simd_ir_gate.py --build /path/to/build --update

``--build`` is a directory holding the ``PIL`` package (``pip install
--target``), or ``-`` for the installed one. The first form exits 1 if a
count is more than the tolerance above ``simd_ir_baseline.json``; the
second rewrites the baseline, and the commit that does so says why.

Counts belong to a compiler. The baseline records the one it was made
with; with another the differences are printed as a warning and the exit
status is 0 unless ``--strict`` is given. ``checks/simd_ir_gate.sh``
builds and runs in the container image of the baseline, as CI does, and
with ``--update`` rewrites it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "simd_ir_baseline.json")
CALLGRIND = os.path.join(HERE, "simd_callgrind.py")

# Operations of simd_bench.py's "full" set. Every kernel group has at least
# one, and so have the paths that pay per call: drawing spans, the
# quantizers, pasting, the codecs' row loops.
GATED = [
    "resize RGB 1/4 bicubic",
    "resize RGB x1.5 lanczos",
    "resize RGBA 1/4 bicubic",
    "resize L 1/4 bicubic",
    "resize I;16 1/4 bicubic",
    "resize F 1/4 bicubic",
    "reduce RGB 2",
    "reduce RGB 3",
    "reduce L 4",
    "gaussian RGB r=5",
    "box RGB r=2",
    "unsharp RGB",
    "filter RGB 3x3 SHARPEN",
    "filter RGB 5x5 SMOOTH_MORE",
    "filter L 3x3 CONTOUR",
    "filter RGB 5x5 float",
    "median 3 L",
    "median 5 L",
    "max 3 RGB",
    "mode 3 L",
    "convert RGB->L",
    "convert RGBA->RGBa",
    "convert RGBa->RGBA",
    "convert RGB->YCbCr",
    "convert YCbCr->RGB",
    "convert RGB->HSV",
    "convert L->1 dither",
    "convert RGB->P adaptive",
    "alpha_composite",
    "blend RGB",
    "chops multiply",
    "chops add",
    "composite RGB mask L",
    "paste RGB mask 1-bit L",
    "paste RGBA over RGB",
    "split RGBA",
    "merge RGB",
    "colorlut RGB 17",
    "point RGB lut",
    "rotate RGB 30 nearest",
    "rotate RGB 30 bilinear",
    "rotate RGB 30 bicubic",
    "perspective RGBA bilinear",
    "transpose RGB ROTATE_90",
    "transpose RGB FLIP_LR",
    "histogram RGB",
    "histogram L masked",
    "getextrema RGB",
    "frombytes RGB",
    "tobytes RGB",
    "quantize RGB fastoctree",
    "quantize mediancut",
    "draw lines 1px",
    "draw lines width 5",
    "draw rectangles filled",
    "draw ellipses filled",
    "draw polygons filled",
    "encode PNG RGB",
    "decode PNG RGB",
]

# Compared on the whole count, libraries included: with libdeflate the
# compressing and inflating leave zlib, and Pillow's own part barely moves.
WHOLE = {"encode PNG RGB", "decode PNG RGB"}

# Not gated: JPEG and WebP decoding and getbbox() on an image without a
# transparent border run some ten thousand instructions of Pillow's own, of
# which 3% is a changed argument check.


def environment() -> dict[str, str]:
    def first_line(cmd: list[str]) -> str:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout
        except OSError:
            return "missing"
        return out.splitlines()[0].strip() if out else "missing"

    return {
        "compiler": first_line(["cc", "--version"]),
        "valgrind": first_line(["valgrind", "--version"]),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def active_tier(build: str, tier: str) -> str:
    """The tier the build runs with PILLOW_ALLSIMD_ISA set to this one."""
    code = (
        "import sys\n"
        "if sys.argv[1] != '-': sys.path.insert(0, sys.argv[1])\n"
        "from PIL import Image\n"
        "print(getattr(Image.core, 'simd_kernels', 'none'))\n"
    )
    env = {**os.environ, "PILLOW_ALLSIMD_ISA": tier}
    out = subprocess.run(
        [sys.executable, "-c", code, build], capture_output=True, text=True, env=env
    )
    return out.stdout.strip() or out.stderr.strip().splitlines()[-1]


def measure(build: str, tier: str, args: argparse.Namespace) -> dict[str, int]:
    out = os.path.join(args.out, tier)
    cmd = [sys.executable, CALLGRIND, "--variant", f"{tier}={build}", "--set", "full",
           "--size", args.size, "--out", out, "--jobs", str(args.jobs),
           "--top", "0"]  # fmt: skip
    for op in GATED:
        cmd += ["--only", op]
    env = {**os.environ, "PILLOW_ALLSIMD_ISA": tier}
    with open(os.path.join(args.out, f"{tier}.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    with open(os.path.join(out, "totals.json")) as fh:
        totals = json.load(fh)
    counts = {}
    for op in GATED:
        t = totals.get(op, {}).get(tier)
        if not t:
            msg = f"{tier}: no callgrind result for {op!r} (see {out}/*.log)"
            raise SystemExit(msg)
        counts[op] = t["Ir"] if op in WHOLE else t["own"]
    return counts


def annotate(kind: str, text: str) -> None:
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::{kind}::{text}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--build", default="-", help="directory holding PIL, or -")
    ap.add_argument("--tiers", default="scalar,sse4,avx2")
    ap.add_argument("--size", default="640x400", help="image size (the baseline's)")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--tolerance", type=float, default=3.0, help="percent")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", help="directory for the dumps (default: temporary)")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline")
    ap.add_argument(
        "--strict", action="store_true", help="fail with another compiler too"
    )
    args = ap.parse_args()
    tiers = args.tiers.split(",")
    if not args.out:
        args.out = tempfile.mkdtemp(prefix="simd-ir-")
    os.makedirs(args.out, exist_ok=True)

    for tier in tiers:
        active = active_tier(args.build, tier)
        if active != tier:
            print(
                f"the build runs {active!r} when asked for {tier!r}: this needs a "
                "dispatch build (-C simd=dispatch) and a CPU with that tier"
            )
            return 2

    env = environment()
    counts = {}
    for tier in tiers:
        print(f"{tier}: {len(GATED)} operations under callgrind", flush=True)
        counts[tier] = measure(args.build, tier, args)

    if args.update:
        with open(args.baseline, "w") as fh:
            json.dump(
                {"environment": env, "size": args.size, "counts": counts},
                fh,
                indent=1,
                sort_keys=True,
            )
            fh.write("\n")
        print(f"wrote {args.baseline}")
        return 0

    with open(args.baseline) as fh:
        base = json.load(fh)
    if base["size"] != args.size:
        print(f"the baseline is for {base['size']}, not {args.size}")
        return 2
    same = all(base["environment"][k] == env[k] for k in ("compiler", "machine"))
    worse, better, missing = [], [], []
    for tier in tiers:
        for op in GATED:
            old = base["counts"].get(tier, {}).get(op)
            new = counts[tier][op]
            if old is None:
                missing.append((tier, op))
                continue
            change = 100.0 * (new - old) / max(old, 1)
            if change > args.tolerance:
                worse.append((change, tier, op, old, new))
            elif change < -args.tolerance:
                better.append((change, tier, op, old, new))

    def table(rows: list[tuple[float, str, str, int, int]]) -> None:
        for change, tier, op, old, new in sorted(rows, reverse=True):
            print(f"  {change:+7.1f}%  {tier:8s} {op:30s} {old:>13,} -> {new:>13,}")

    total = len(tiers) * len(GATED)
    print(f"\n{total - len(worse) - len(better) - len(missing)} of {total} counts "
          f"within {args.tolerance:g}% of the baseline")  # fmt: skip
    if better:
        print(f"\nFewer instructions ({len(better)}); --update records them:")
        table(better)
    if missing:
        print(f"\nNot in the baseline ({len(missing)}); --update adds them:")
        for tier, op in missing:
            print(f"  {tier:8s} {op}")
    if worse:
        print(f"\nMore instructions ({len(worse)}):")
        table(worse)
    if not same:
        text = (
            "instruction counts were not compared strictly: the baseline is "
            f"from {base['environment']['compiler']!r} on "
            f"{base['environment']['machine']}, this is {env['compiler']!r} on "
            f"{env['machine']}; refresh it with --update"
        )
        print("\n" + text)
        annotate("warning", text)
        return 1 if args.strict and (worse or missing) else 0
    if worse or missing:
        annotate(
            "error",
            f"{len(worse)} operations execute more instructions than the baseline "
            f"and {len(missing)} are missing from it (checks/simd_ir_gate.py)",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
