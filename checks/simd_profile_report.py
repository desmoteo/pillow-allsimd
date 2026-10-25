#!/usr/bin/env python3
"""Combine callgrind dumps, native timings and allocation counts into a
Markdown report that ranks operations by the kind of work left in them.

    python3 checks/simd_profile_report.py --callgrind cg-out \\
        --times times.json --allocs allocs.json --scalar scalar --simd avx2 \\
        --size 1280x800 --ghz 5.4 > report.md

Inputs:
- ``--callgrind``: the output directory of ``simd_callgrind.py``.
- ``--times``: ``simd_bench.py --json`` at the same image size; a variant
  named ``upstream`` is shown as the reference when present.
- ``--allocs``: ``simd_allocs.py --json`` (optional).

"IPC" is instructions per cycle estimated from callgrind's instruction
count and the native time at ``--ghz``; it is only a guide, since the
clock varies and the native build may differ slightly from the profiled
one. Valgrind's branch model has no history and its cache model no
prefetcher, so rates are relative signals, not cycle counts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))

PILLOW_OBJECTS = ("_imaging", "_webp", "_avif")
PYTHON = re.compile(r"python|libpython", re.IGNORECASE)
LIBC = re.compile(r"^(libc|ld-linux|libm)[.-]")


def load_callgrind() -> Any:
    spec = importlib.util.spec_from_file_location(
        "simd_callgrind", os.path.join(HERE, "simd_callgrind.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owner(row: dict[str, Any]) -> str:
    obj, fn = row.get("object", ""), row["function"]
    path = fn.rsplit(":", 1)[0]
    # Intrinsics inlined into Pillow's kernels are attributed to the
    # compiler's headers, often without an object name.
    if (
        obj.startswith(PILLOW_OBJECTS)
        or "src/libImaging/" in path
        or re.search(r"(^|/)src/[_a-z]+\.c$", path)
        or path.endswith("intrin.h")
    ):
        return "pillow"
    if PYTHON.search(obj) or re.search(r"/(Python|Objects|Modules|Include)/", path):
        return "python"
    if LIBC.match(obj) or re.search(r"(^|:)__mem|(^|:)mem(cpy|set|move)", fn):
        return "libc"
    return "library"


def category(op: str) -> str:
    rules = (
        ("Drawing and text", ("draw ", "font ")),
        ("Codecs", ("encode ", "decode ")),
        (
            "Resampling and geometry",
            ("resize", "reduce", "rotate", "transform", "transpose", "perspective",
             "thumbnail", "crop", "expand", "ImageOps fit", "ImageOps pad",
             "ImageOps contain", "ImageOps scale", "ImageOps mirror"),
        ),  # fmt: skip
        (
            "Filters",
            ("filter", "gaussian", "box ", "unsharp", "median", "max ", "min ",
             "rank ", "mode ", "ImageEnhance sharpness", "ImageMorph",
             "effect_spread"),
        ),  # fmt: skip
        (
            "Conversion and palettes",
            ("convert", "quantize", "getcolors", "colorlut", "ImageOps grayscale",
             "ImageOps colorize", "ImageOps posterize"),
        ),  # fmt: skip
        (
            "Compositing and channel operations",
            ("chops", "blend", "composite", "alpha_composite", "paste", "split",
             "merge", "getchannel", "putalpha", "ImageOps invert",
             "ImageOps solarize", "ImageEnhance", "ImageMath"),
        ),  # fmt: skip
    )
    for name, prefixes in rules:
        if any(p in op for p in prefixes):
            return name
    return "Statistics, point operations and pixel access"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--callgrind", required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--allocs")
    ap.add_argument("--scalar", default="scalar")
    ap.add_argument("--simd", default="avx2")
    ap.add_argument("--size", default="1280x800")
    ap.add_argument("--ghz", type=float, default=5.4)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", help="also write the per-operation data here")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    pixels = w * h
    cg = load_callgrind()
    times = json.load(open(args.times))["best_ms"]
    allocs = json.load(open(args.allocs)) if args.allocs else {}
    up = times.get("upstream", {})

    rows: list[dict[str, Any]] = []
    ops = list(times[args.simd])
    for op in ops:
        r: dict[str, Any] = {"op": op, "category": category(op)}
        for v in (args.scalar, args.simd):
            t = cg.totals(args.callgrind, v, op)
            if not t:
                break
            fns = cg.functions(args.callgrind, v, op, 100000)
            d = cg.derived(t, pixels)
            share: dict[str, float] = {}
            # Inlined code is split by source file; merge it per function.
            merged: dict[str, dict[str, Any]] = {}
            for f in fns:
                share[owner(f)] = share.get(owner(f), 0) + f["Ir"] / max(t["Ir"], 1)
                name = f["function"].rsplit(":", 1)[-1]
                m = merged.setdefault(name, {"function": name, "owner": owner(f)})
                for e, c in f.items():
                    if isinstance(c, int):
                        m[e] = m.get(e, 0) + c
            by_cost = sorted(merged.values(), key=lambda m: -m["Ir"])
            ms = times[v].get(op)
            r[v] = {
                **d,
                "Ir": t["Ir"],
                "Bcm/px": (t["Bcm"] + t["Bim"]) / pixels,
                "ms": ms,
                "IPC": t["Ir"] / (ms * 1e-3 * args.ghz * 1e9) if ms else None,
                "share": share,
                "functions": by_cost[: args.top],
            }
        else:
            r["upstream ms"] = up.get(op)
            r["allocs"] = allocs.get(args.simd, {}).get(op)
            rows.append(r)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)

    s, v = args.scalar, args.simd

    def top_pillow(r: dict[str, Any]) -> str:
        for f in r[v]["functions"]:
            if f["owner"] == "pillow":
                return f"{f['function']} {100 * f['Ir'] / r[v]['Ir']:.0f}%"
        return "-"

    def fmt(x: float | None, spec: str = ".2f") -> str:
        return "-" if x is None else format(x, spec)

    print(f"# Profile of {len(rows)} operations at {args.size}\n")
    print(
        "Columns: native ms (13.0 wheel, scalar, SIMD); instructions per pixel "
        "(scalar, SIMD); SIMD/scalar instruction ratio; estimated IPC of the "
        "SIMD build; conditional+indirect mispredicts per pixel and conditional "
        "mispredict rate; L1 and last-level misses per 1000 instructions; share "
        "of the SIMD build's instructions in Pillow's C code, the Python "
        "interpreter, other libraries and libc; Pillow's hottest function; "
        "allocations per call.\n"
    )
    for cat in dict.fromkeys(r["category"] for r in rows):
        print(f"## {cat}\n")
        print(
            "| Operation | 13.0 ms | scalar ms | SIMD ms | Ir/px s | Ir/px v | v/s "
            "| IPC | Bm/px | Bcm% | D1m/kIr | LLm/kIr | C/Py/lib/libc % "
            "| hottest Pillow function | allocs |"
        )
        print("|---|" + "---:|" * 12 + "---|---:|")
        for r in rows:
            if r["category"] != cat:
                continue
            a, b = r[s], r[v]
            sh = b["share"]
            mix = "/".join(
                f"{100 * sh.get(k, 0):.0f}"
                for k in ("pillow", "python", "library", "libc")
            )
            al = r["allocs"]
            ratio = b["Ir"] / max(a["Ir"], 1)
            print(
                f"| {r['op']} | {fmt(r['upstream ms'])} | {fmt(a['ms'])} "
                f"| {fmt(b['ms'])} | {a['Ir/px']:.1f} | {b['Ir/px']:.1f} | {ratio:.2f} "
                f"| {fmt(b['IPC'], '.1f')} | {b['Bcm/px']:.2f} | {b['Bcm%']:.1f} "
                f"| {b['D1m/kIr']:.1f} | {b['LLm/kIr']:.2f} | {mix} | {top_pillow(r)} "
                f"| {al['calls'] if al else '-'} |"
            )
        print()


if __name__ == "__main__":
    main()
