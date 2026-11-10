#!/usr/bin/env python3
"""
Writes the benchmark tables and figures of the documentation from the result
files of the benchmark repository (pillow-allsimd-perf), into marked regions
of PyPI.rst, README.md, docs/reference/fast_mode.rst and
docs/allsimd-kernels.rst (design/GENERATED_TABLES_PLAN.md):

    python3 checks/simd_tables.py --data ../pillow-allsimd-perf/docs/data \\
        --commit $(git -C ../pillow-allsimd-perf rev-parse HEAD)
    python3 checks/simd_tables.py --check

A region starts with a comment naming it, this script and the commit of the
benchmark repository its numbers come from, and ends with another::

    .. generated: kernels-resize (checks/simd_tables.py, pillow-allsimd-perf@<sha>)

    <the table>

    .. generated-end

(``<!-- generated: ... -->`` and ``<!-- generated-end -->`` in Markdown.)
Everything between the markers is this script's; the text around them is
hand-written. ``--data`` and ``--commit`` rewrite every region from that
checkout of the benchmark repository and stamp its commit; ``--check``
fetches the data at the commit each region names and fails if any region
would change, so a table edited by hand, or a marker bumped without
regenerating, fails the lint workflow.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent))
from simd_results import BASE, Machine, fetch, load

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).parent.parent
Machines = dict[str, Machine]

# The variants of the published runs
SCALAR = "branch-scalar"
SSE4 = "branch-sse4"
AVX2 = "branch-avx2"
AVXVNNI = "branch-avxvnni"
AVX2_FAST = "branch-avx2-fast"
NEON = "branch-neon"
NEON_FAST = "branch-neon-fast"
PILLOW_SIMD = "pillow-simd-12.1.1.post0-avx2"
OPENCV = "opencv"
VIPS = "vips"


# --- formatting -------------------------------------------------------------


def ms1(x: float) -> str:
    """Milliseconds with one decimal; whole, with a thousands separator, from
    100 up."""
    return f"{x:,.0f}" if x >= 100 else f"{x:.1f}"


def sig3(x: float) -> str:
    """Three significant digits (whole from 100 up)."""
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def times(x: float, digits: int = 1, sign: str = "x") -> str:
    """A speed-up: one decimal, whole from 100 up."""
    if x >= 100:
        digits = 0
    return f"{x:.{digits}f}{sign}"


def list_table(header: list[str], rows: list[list[str]]) -> str:
    out = [".. list-table::", "   :header-rows: 1", ""]
    for row in [header, *rows]:
        for i, cell in enumerate(row):
            out.append(f"   {'* ' if i == 0 else '  '}- {cell}".rstrip())
    return "\n".join(out)


def simple_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    rule = "  ".join("=" * w for w in widths)
    lines = [rule, "  ".join(c.ljust(w) for c, w in zip(header, widths)).rstrip(), rule]
    lines.extend(
        "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows
    )
    lines.append(rule)
    return "\n".join(lines)


# --- the "Numbers" tables of the kernels page -------------------------------


@dataclass(frozen=True)
class Column:
    header: str
    kind: str  # "ms" or "ratio"
    machine: str
    variant: str

    def cell(self, m: Machines, case: str, digits: int) -> str:
        machine = m[self.machine]
        if self.kind == "ms":
            return ms1(machine.ms(self.variant, case))
        return times(machine.speedup(self.variant, case), digits)


MS_X86 = Column("Pillow 13 (x86, ms)", "ms", "x86", BASE)
MS_ARM = Column("Pillow 13 (arm64, ms)", "ms", "arm64", BASE)
PORTABLE_X86 = Column("Portable", "ratio", "x86", SCALAR)
SSE4_X86 = Column("SSE4", "ratio", "x86", SSE4)
AVX2_X86 = Column("AVX2", "ratio", "x86", AVX2)
AVXVNNI_X86 = Column("AVX-VNNI", "ratio", "x86", AVXVNNI)
NEON_ARM = Column("NEON", "ratio", "arm64", NEON)

EXACT = [MS_X86, SSE4_X86, AVX2_X86, MS_ARM, NEON_ARM]
WITH_PORTABLE = [MS_X86, PORTABLE_X86, SSE4_X86, AVX2_X86, MS_ARM, NEON_ARM]


@dataclass(frozen=True)
class Numbers:
    """One "Numbers" table: the first header cell, the columns, the rows as
    (label, case id), and the decimals of the speed-ups."""

    first: str
    columns: list[Column]
    rows: list[tuple[str, str]]
    digits: int = 1

    def render(self, m: Machines) -> str:
        header = [self.first, *(c.header for c in self.columns)]
        rows = [
            [label, *(c.cell(m, case, self.digits) for c in self.columns)]
            for label, case in self.rows
        ]
        return list_table(header, rows)


NUMBERS = {
    "kernels-resize": Numbers(
        "Resize",
        EXACT,
        [
            ("To 320×200, bilinear", "scale/RGB/320x200/bilinear"),
            ("To 320×200, bicubic", "scale/RGB/320x200/bicubic"),
            ("To 320×200, Lanczos", "scale/RGB/320x200/lanczos"),
            ("To 2048×1280, bicubic", "scale/RGB/2048x1280/bicubic"),
            ("To 5478×3424, bicubic", "scale/RGB/5478x3424/bicubic"),
            ("RGBA to 320×200, bicubic", "scale/RGBA/320x200/bicubic"),
            ("L to 320×200, bicubic", "scale/L/320x200/bicubic"),
            ("Reduce by 2", "reduce/RGB/2"),
        ],
    ),
    "kernels-blur": Numbers(
        "Blur",
        EXACT,
        [
            ("Gaussian RGB, σ 1", "blur/gaussian/RGB/1"),
            ("Gaussian RGB, σ 10", "blur/gaussian/RGB/10"),
            ("Gaussian RGB, σ 30", "blur/gaussian/RGB/30"),
            ("Gaussian RGBA, σ 10", "blur/gaussian/RGBA/10"),
            ("Gaussian L, σ 10", "blur/gaussian/L/10"),
            ("Box RGB, radius 2", "blur/box/RGB/2"),
            ("Box RGB, radius 10", "blur/box/RGB/10"),
            ("Unsharp mask RGB", "unsharp/RGB"),
            ("Unsharp mask L", "unsharp/L"),
        ],
    ),
    "kernels-filters": Numbers(
        "Filter",
        [MS_X86, SSE4_X86, AVX2_X86, AVXVNNI_X86, MS_ARM, NEON_ARM],
        [
            ("RGB 3×3 SMOOTH", "filter/RGB/SMOOTH"),
            ("RGB 3×3 SHARPEN", "filter/RGB/SHARPEN"),
            ("RGB 3×3 EDGE_ENHANCE", "filter/RGB/EDGE_ENHANCE"),
            ("RGB 3×3 FIND_EDGES", "filter/RGB/FIND_EDGES"),
            ("RGB 5×5 SMOOTH_MORE", "filter/RGB/SMOOTH_MORE"),
            ("L 3×3 SHARPEN", "filter/L/SHARPEN"),
            ("L 5×5 SMOOTH_MORE", "filter/L/SMOOTH_MORE"),
        ],
    ),
    "kernels-rank": Numbers(
        "Filter",
        WITH_PORTABLE,
        [
            ("Median 3×3 RGB", "rank/median3/RGB"),
            ("Median 5×5 RGB", "rank/median5/RGB"),
            ("Median 3×3 L", "rank/median3/L"),
            ("Minimum 3×3 RGB", "rank/min3/RGB"),
            ("Maximum 3×3 RGB", "rank/max3/RGB"),
        ],
        digits=0,
    ),
    "kernels-conversions": Numbers(
        "Conversion",
        EXACT,
        [
            ("RGB to L", "convert/RGB/L"),
            ("L to RGB", "convert/L/RGB"),
            ("RGB to RGBA", "convert/RGB/RGBA"),
            ("RGB to CMYK", "convert/RGB/CMYK"),
            ("RGBA to RGBa", "convert/RGBA/RGBa"),
            ("RGBa to RGBA", "convert/RGBa/RGBA"),
            ("RGB to HSV", "convert/RGB/HSV"),
            ("RGB to YCbCr", "convert/RGB/YCbCr"),
            ("YCbCr to RGB", "convert/YCbCr/RGB"),
        ],
    ),
    "kernels-compositing": Numbers(
        "Operation",
        EXACT,
        [
            ("alpha_composite RGBA", "composite/alpha"),
            ("blend RGB, 50%", "composite/blend"),
            ("composite RGB, L mask", "composite/mask"),
            ("ImageChops.multiply RGB", "chops/RGB/multiply"),
            ("ImageChops.screen RGB", "chops/RGB/screen"),
            ("ImageChops.add RGB", "chops/RGB/add"),
            ("ImageChops.lighter RGB", "chops/RGB/lighter"),
            ("ImageOps.invert RGB", "point/RGB/invert"),
            ("point() RGB, a table", "point/RGB/table"),
            ("point() L, a table", "point/L/table"),
            ("Color3DLUT RGB, 16³", "lut/RGB/16"),
            ("Color3DLUT RGBA, 16³", "lut/RGBA/16"),
        ],
    ),
    "kernels-geometry": Numbers(
        "Operation",
        EXACT,
        [
            ("FLIP_LEFT_RIGHT RGB", "transpose/RGB/FLIP_LEFT_RIGHT"),
            ("ROTATE_180 RGB", "transpose/RGB/ROTATE_180"),
            ("ROTATE_90 RGB", "transpose/RGB/ROTATE_90"),
            ("ROTATE_270 RGB", "transpose/RGB/ROTATE_270"),
            ("TRANSPOSE RGB", "transpose/RGB/TRANSPOSE"),
            ("ROTATE_90 L", "transpose/L/ROTATE_90"),
            ("rotate 30°, nearest, RGB", "rotate/RGB/30/nearest"),
            ("rotate 30°, bilinear, RGB", "rotate/RGB/30/bilinear"),
            ("rotate 30°, bicubic, RGB", "rotate/RGB/30/bicubic"),
            ("rotate 30°, bicubic, RGBA", "rotate/RGBA/30/bicubic"),
            ("perspective, bilinear, RGB", "rotate/RGB/perspective/bilinear"),
        ],
    ),
    "kernels-drawing": Numbers(
        "Drawing",
        WITH_PORTABLE,
        [
            ("200 filled polygons", "draw/polygons"),
            ("200 filled ellipses", "draw/ellipses"),
            ("500 filled rectangles", "draw/rectangles"),
            ("500 lines, 5 px wide", "draw/lines"),
            ("500 boxes, 2 px outline", "draw/outline/rectangles"),
            ("100 ROI polygons, 2 px outline", "draw/outline/polygons"),
            ("100 lines of text", "draw/text"),
        ],
    ),
    "kernels-codecs": Numbers(
        "Operation",
        WITH_PORTABLE,
        [
            ("Encode PNG, level 6", "codec/encode/png"),
            ("Decode PNG", "codec/decode/png"),
            ("Decode WebP", "codec/decode/webp"),
            ("Decode JPEG", "codec/decode/jpeg"),
            ("Median cut, 256 colours", "quantize/mediancut"),
        ],
    ),
}


# --- Appendix A: the geometric means ----------------------------------------

APPENDIX_A = [
    ("Portable (no SIMD)", SCALAR, SCALAR),
    ("SSE4", SSE4, None),
    ("AVX2", AVX2, None),
    ("AVX-VNNI", AVXVNNI, None),
    ("NEON", None, NEON),
    ("Fast mode (AVX2, NEON)", AVX2_FAST, NEON_FAST),
    ("Pillow-SIMD 12.1.1 (AVX2)", PILLOW_SIMD, None),
    ("OpenCV 5.0 (single-threaded)", OPENCV, OPENCV),
    ("libvips 8.16", VIPS, VIPS),
]


def appendix_a(m: Machines) -> str:
    def cell(machine: str, variant: str | None) -> str:
        return f"{m[machine].geomean(variant)[0]:.2f}x" if variant else ""

    rows = [[label, cell("x86", x), cell("arm64", a)] for label, x, a in APPENDIX_A]
    return list_table(
        ["Build", "x86-64 (Core Ultra 9 285H)", "arm64 (Jetson Orin)"], rows
    )


# --- fast mode: the fast kernels against the exact ones ---------------------

FAST_KERNELS = [
    ("Gaussian blur, radius 1", "blur/gaussian/RGB/1"),
    ("Unsharp mask, radius 2", "unsharp/RGB"),
    ("RGB to YCbCr", "convert/RGB/YCbCr"),
    ("YCbCr to RGB", "convert/YCbCr/RGB"),
    ("RGB to HSV", "convert/RGB/HSV"),
    ("rotate 30°, bilinear", "rotate/RGB/30/bilinear"),
    ("rotate 30°, bicubic", "rotate/RGB/30/bicubic"),
    ("rotate 30°, bicubic, RGBA", "rotate/RGBA/30/bicubic"),
    ("perspective, bilinear", "rotate/RGB/perspective/bilinear"),
    ("blend, 50%", "composite/blend"),
    ("resize to 5478×3424, bilinear", "scale/RGB/5478x3424/bilinear"),
    ("resize to 5478×3424, bicubic", "scale/RGB/5478x3424/bicubic"),
    ("resize RGBA to 5478×3424, bilinear", "scale/RGBA/5478x3424/bilinear"),
    ("resize to 2048×1280, bicubic", "scale/RGB/2048x1280/bicubic"),
    ("resize to 320×200, Lanczos (``reducing_gap``)", "scale/RGB/320x200/lanczos"),
    ("resize to 26×16, Lanczos (``reducing_gap``)", "scale/RGB/26x16/lanczos"),
]


def fast_over_exact(m: Machines, machine: str, case: str) -> float:
    exact, fast = (AVX2, AVX2_FAST) if machine == "x86" else (NEON, NEON_FAST)
    return m[machine].best(exact, case) / m[machine].best(fast, case)


def fast_mode_speedups(m: Machines) -> str:
    rows = [
        [
            label,
            f"{fast_over_exact(m, 'x86', case):.2f}x",
            f"{fast_over_exact(m, 'arm64', case):.2f}x",
        ]
        for label, case in FAST_KERNELS
    ]
    return list_table(["Operation", "x86-64 (AVX2)", "arm64 (NEON)"], rows)


def readme_gaussian(m: Machines) -> str:
    x = fast_over_exact(m, "x86", "blur/gaussian/RGB/1")
    a = fast_over_exact(m, "arm64", "blur/gaussian/RGB/1")
    return f"{x:.1f}x at radius 1 on x86-64, {a:.1f}x on arm64"


# --- the PyPI description ---------------------------------------------------

PYPI_ROWS = [
    ("Resize to 320×200, bicubic", "scale/RGB/320x200/bicubic"),
    ("Resize RGBA to 2048×1280, Lanczos", "scale/RGBA/2048x1280/lanczos"),
    ("Enlarge to 5478×3424, bicubic", "scale/RGB/5478x3424/bicubic"),
    ("Gaussian blur, radius 1", "blur/gaussian/RGB/1"),
    ("Gaussian blur, radius 10", "blur/gaussian/RGB/10"),
    ("Unsharp mask", "unsharp/RGB"),
    ("SHARPEN filter", "filter/RGB/SHARPEN"),
    ("Median filter, 3×3", "rank/median3/RGB"),
    ("RGB to HSV", "convert/RGB/HSV"),
    ("Unpremultiply alpha", "convert/RGBa/RGBA"),
    ("Rotate 30°, bicubic", "rotate/RGB/30/bicubic"),
    ("Alpha composite", "composite/alpha"),
    ("Draw 100 lines of text", "draw/text"),
    ("Decode JPEG", "codec/decode/jpeg"),
    ("Encode PNG", "codec/encode/png"),
    ("JPEG load, rotate, resize, blur, save", "cycle/3"),
]
PYPI_FAST = {"x86": AVX2_FAST, "arm64": NEON_FAST}
PYPI_EXACT = {"x86": AVX2, "arm64": NEON}


def pypi_benchmarks(m: Machines) -> str:
    def cell(machine: str, case: str) -> str:
        mc = m[machine]
        variant = PYPI_FAST[machine]
        return (
            f"{sig3(mc.ms(BASE, case))} → {sig3(mc.ms(variant, case))} ms, "
            f"**{times(mc.speedup(variant, case), sign='×')}**"
        )

    rows = [
        [label, cell("x86", case), cell("arm64", case)] for label, case in PYPI_ROWS
    ]
    return simple_table(["Operation", "x86-64, AVX2", "arm64, NEON"], rows)


def pypi_figures(m: Machines) -> str:
    """The substitutions the description's prose uses."""
    x, a = m["x86"], m["arm64"]
    fast_x, cases = x.geomean(AVX2_FAST)
    fast_a, cases_a = a.geomean(NEON_FAST)
    if cases != cases_a or cases != len(x.cases) or cases != len(a.cases):
        msg = f"the machines ran different case sets: {cases}, {cases_a}"
        raise SystemExit(msg)
    cv_x, cv_cases = x.faster_than(AVX2_FAST, OPENCV)
    cv_a, cv_cases_a = a.faster_than(NEON_FAST, OPENCV)
    if cv_cases != cv_cases_a:
        msg = f"OpenCV ran different case sets: {cv_cases}, {cv_cases_a}"
        raise SystemExit(msg)
    if x.date != a.date:
        msg = f"the machines ran on different days: {x.date}, {a.date}"
        raise SystemExit(msg)
    subs = {
        "fast-x86": f"{fast_x:.1f}",
        "fast-arm64": f"{fast_a:.1f}",
        "cases": str(cases),
        "vs-pillow-simd": f"{fast_x / x.geomean(PILLOW_SIMD)[0]:.1f}",
        "opencv-cases": str(cv_cases),
        "opencv-x86": str(cv_x),
        "opencv-arm64": str(cv_a),
        "exact-x86": f"{x.geomean(AVX2)[0]:.1f}",
        "exact-arm64": f"{a.geomean(NEON)[0]:.1f}",
        "date": x.date,
    }
    return "\n".join(f".. |{k}| replace:: {v}" for k, v in subs.items())


def pypi_headline(m: Machines) -> str:
    """The bold first sentence: whole numbers (a substitution can't sit inside
    inline markup)."""
    x = m["x86"].geomean(AVX2_FAST)[0]
    a = m["arm64"].geomean(NEON_FAST)[0]
    return (
        f"**Pillow, {x:.0f} times as fast on x86-64 and {a:.0f} times as fast "
        "on arm64.**"
    )


def readme_headline(m: Machines) -> str:
    """The README's first paragraph: the speed-ups that the PyPI page
    states, in Markdown."""
    x, a = m["x86"], m["arm64"]
    fast_x, cases = x.geomean(AVX2_FAST)
    fast_a = a.geomean(NEON_FAST)[0]
    cv_x, cv_cases = x.faster_than(AVX2_FAST, OPENCV)
    cv_a = a.faster_than(NEON_FAST, OPENCV)[0]
    return (
        f"**Pillow 13, {fast_x:.1f}× as fast on x86-64 and {fast_a:.1f}× as "
        f"fast on arm64**, as the geometric mean of {cases} operations on one "
        f"thread ({x.geomean(AVX2)[0]:.1f}× and {a.geomean(NEON)[0]:.1f}× with "
        f"Pillow's exact arithmetic). That is "
        f"{fast_x / x.geomean(PILLOW_SIMD)[0]:.1f}× as fast as Pillow-SIMD, "
        f"and faster than OpenCV in {cv_x} of the {cv_cases} operations both "
        f"have on x86-64, {cv_a} on arm64."
    )


# --- the regions ------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    file: str
    render: Callable[[Machines], str]
    inline: bool = False  # the markers on the content's line (Markdown prose)


REGIONS: dict[str, Region] = {
    "pypi-figures": Region("PyPI.rst", pypi_figures),
    "pypi-headline": Region("PyPI.rst", pypi_headline),
    "pypi-benchmarks": Region("PyPI.rst", pypi_benchmarks),
    "readme-headline": Region("README.md", readme_headline),
    "readme-gaussian-fast": Region("README.md", readme_gaussian, inline=True),
    "fast-mode-speedups": Region("docs/reference/fast_mode.rst", fast_mode_speedups),
    "kernels-appendix-a": Region("docs/allsimd-kernels.rst", appendix_a),
    **{
        name: Region("docs/allsimd-kernels.rst", table.render)
        for name, table in NUMBERS.items()
    },
}

MARKER = re.compile(
    r"(?P<start>(?:\.\. |<!-- )generated: (?P<name>[\w-]+) "
    r"\(checks/simd_tables\.py, pillow-allsimd-perf@(?P<commit>[0-9a-f]{7,40})\)"
    r"(?: -->)?)"
    r"(?P<body>.*?)"
    r"(?P<end>(?:\.\. |<!-- )generated-end(?: -->)?)",
    re.DOTALL,
)


def rewrite(text: str, data: Callable[[str], Machines], commit: str | None) -> str:
    """The text with every region regenerated, from the data of the commit
    it names, or of `commit` when given (stamped into the marker)."""
    seen: set[str] = set()

    def sub(match: re.Match[str]) -> str:
        name = match["name"]
        if name not in REGIONS:
            msg = f"unknown region {name!r}"
            raise SystemExit(msg)
        if name in seen:
            msg = f"region {name!r} appears twice"
            raise SystemExit(msg)
        seen.add(name)
        region = REGIONS[name]
        at = commit or match["commit"]
        content = region.render(data(at))
        start = match["start"].replace(f"@{match['commit']}", f"@{at}")
        body = content if region.inline else f"\n\n{content}\n\n"
        return f"{start}{body}{match['end']}"

    return MARKER.sub(sub, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, help="docs/data of a checkout")
    parser.add_argument("--commit", help="the commit of that checkout")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fetch the data at the commit each region names; fail on a change",
    )
    args = parser.parse_args()
    if args.check == bool(args.data) or bool(args.data) != bool(args.commit):
        parser.error("either --data DIR --commit SHA, or --check")

    cache: dict[str, Machines] = {}

    def data(commit: str) -> Machines:
        if commit not in cache:
            cache[commit] = load(args.data if args.data else fetch(commit))
        return cache[commit]

    files = sorted({r.file for r in REGIONS.values()})
    changed = []
    found: set[str] = set()
    for file in files:
        path = ROOT / file
        before = path.read_text(encoding="utf-8")
        found |= {m["name"] for m in MARKER.finditer(before)}
        after = rewrite(before, data, args.commit)
        if after != before:
            changed.append(file)
            if not args.check:
                path.write_text(after, encoding="utf-8")
    missing = set(REGIONS) - found
    if missing:
        msg = f"regions without markers: {', '.join(sorted(missing))}"
        raise SystemExit(msg)
    if args.check:
        if changed:
            msg = "generated regions out of date in " + ", ".join(changed)
            raise SystemExit(msg)
        print(f"generated regions up to date in {', '.join(files)}")
    else:
        print("rewrote " + (", ".join(changed) if changed else "nothing"))


if __name__ == "__main__":
    main()
