#!/usr/bin/env python3
"""
Draws the speed-up chart of the PyPI description (PyPI.rst) as an SVG, from
the result files of the benchmark repository (pillow-allsimd-perf):

    python3 checks/simd_speedups_svg.py ../pillow-allsimd-perf/docs/data \
        docs/resources/allsimd-speedups.svg

Each bar is Pillow 13.0.0's best time over the branch build's, in fast mode
(the default): AVX2 on x86-64, NEON on arm64.
"""

from __future__ import annotations

import math
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simd_results import load as load_results

MACHINES = [
    ("x86", "branch-avx2-fast", "x86-64, AVX2 (Intel Core Ultra 9 285H)"),
    ("arm64", "branch-neon-fast", "arm64, NEON (NVIDIA Jetson Orin)"),
]
ROWS = [
    ("scale/RGB/320x200/bicubic", "Resize to 320×200, bicubic"),
    ("scale/RGBA/2048x1280/lanczos", "Resize RGBA to 2048×1280, Lanczos"),
    ("blur/gaussian/RGB/1", "Gaussian blur, radius 1"),
    ("unsharp/RGB", "Unsharp mask"),
    ("filter/RGB/SHARPEN", "SHARPEN filter"),
    ("convert/RGB/HSV", "RGB to HSV"),
    ("rotate/RGB/30/bicubic", "Rotate 30°, bicubic"),
    ("composite/alpha", "Alpha composite"),
    ("draw/text", "Draw 100 lines of text"),
    ("cycle/3", "JPEG load, 3 operations, save"),
]
OFF_SCALE = ("rank/median3/RGB", "3×3 median filter")

WIDTH = 760
LABEL_RIGHT = 262
PLOT_LEFT = 274
PLOT_RIGHT = WIDTH - 56
BAR = 10
PAIR_GAP = 2
ROW = 36
TOP = 104

STYLE = """
.surface { fill: #fcfcfb; }
.title { fill: #0b0b0b; font-size: 17px; font-weight: 600; }
.ink { fill: #0b0b0b; }
.secondary { fill: #52514e; }
.muted { fill: #898781; }
.grid { stroke: #e1e0d9; stroke-width: 1; }
.axis { stroke: #c3c2b7; stroke-width: 1; }
.s1 { fill: #2a78d6; }
.s2 { fill: #eb6834; }
@media (prefers-color-scheme: dark) {
  .surface { fill: #1a1a19; }
  .title, .ink { fill: #f0efec; }
  .secondary { fill: #c3c2b7; }
  .grid { stroke: #2c2c2a; }
  .axis { stroke: #383835; }
  .s1 { fill: #3987e5; }
  .s2 { fill: #d95926; }
}
text { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica,
  Arial, sans-serif; }
"""


def load(data_dir: Path) -> list[dict[str, float]]:
    """Speed-ups per case, one dict per machine, with "geomean" over all cases."""
    machines = load_results(data_dir)
    out = []
    for name, variant, _ in MACHINES:
        m = machines[name]
        speedups = {
            c: m.speedup(variant, c) for c in m.cases if c in m.results(variant)
        }
        speedups["geomean"], _ = m.geomean(variant)
        speedups["count"] = len(m.cases)
        out.append(speedups)
    return out


def bar(x0: float, y: float, length: float, cls: str) -> str:
    """A bar square at the baseline with a 4px rounded end."""
    r = min(4.0, length, BAR / 2)
    x1 = x0 + length
    return (
        f'<path class="{cls}" d="M{x0:.1f},{y:.1f}H{x1 - r:.1f}'
        f"A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f}V{y + BAR - r:.1f}"
        f'A{r},{r} 0 0 1 {x1 - r:.1f},{y + BAR:.1f}H{x0:.1f}Z"/>'
    )


def draw(speedups: list[dict[str, float]]) -> str:
    rows = list(ROWS)
    rows.append(("geomean", f"All {int(speedups[0]['count'])} operations, geomean"))
    top = max(s[k] for s in speedups for k, _ in rows)
    scale_max = math.ceil(top / 5) * 5
    px = (PLOT_RIGHT - PLOT_LEFT) / scale_max

    height = TOP + ROW * len(rows) + 8 + 30 + 40
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{height}" viewBox="0 0 {WIDTH} {height}" role="img">'
        ),
        "<title>Pillow-AllSIMD speed-up over Pillow 13.0.0</title>",
        f"<style>{STYLE}</style>",
        f'<rect class="surface" width="{WIDTH}" height="{height}" rx="8"/>',
        (
            '<text class="title" x="24" y="34">'
            "Pillow-AllSIMD: times faster than Pillow 13.0.0</text>"
        ),
        (
            '<text class="secondary" x="24" y="56" font-size="13">'
            "One thread, a 2560×1600 image, fast mode (the default)</text>"
        ),
    ]
    # legend
    x: float = 24
    for i, (_, _, name) in enumerate(MACHINES):
        parts.append(
            f'<rect class="s{i + 1}" x="{x}" y="72" width="12" height="12" rx="3"/>'
        )
        parts.append(
            f'<text class="ink" x="{x + 18}" y="82.5" font-size="13">'
            f"{escape(name)}</text>"
        )
        x += 18 + 7.2 * len(name) + 28

    plot_bottom = TOP + ROW * len(rows)
    for tick in range(0, scale_max + 1, 5):
        gx = PLOT_LEFT + tick * px
        cls = "axis" if tick == 0 else "grid"
        parts.append(
            f'<line class="{cls}" x1="{gx:.1f}" y1="{TOP - 8}" '
            f'x2="{gx:.1f}" y2="{plot_bottom}"/>'
        )
        parts.append(
            f'<text class="muted" x="{gx:.1f}" y="{plot_bottom + 18}" '
            f'font-size="12" text-anchor="middle">{tick}×</text>'
        )

    for r, (key, label) in enumerate(rows):
        y = TOP + r * ROW + (ROW - 2 * BAR - PAIR_GAP) / 2 - 4
        if key == "geomean":
            parts.append(
                f'<line class="axis" x1="24" y1="{y - 6:.1f}" '
                f'x2="{PLOT_RIGHT}" y2="{y - 6:.1f}"/>'
            )
        weight = ' font-weight="600"' if key == "geomean" else ""
        parts.append(
            f'<text class="ink" x="{LABEL_RIGHT}" y="{y + BAR + 5:.1f}" '
            f'font-size="13" text-anchor="end"{weight}>{escape(label)}</text>'
        )
        for i, s in enumerate(speedups):
            by = y + i * (BAR + PAIR_GAP)
            length = s[key] * px
            parts.append(bar(PLOT_LEFT, by, length, f"s{i + 1}"))
            parts.append(
                f'<text class="secondary" x="{PLOT_LEFT + length + 5:.1f}" '
                f'y="{by + BAR - 1:.1f}" font-size="11"{weight}>{s[key]:.1f}×</text>'
            )

    key, label = OFF_SCALE
    note = (
        f"Off the scale: a {label}, {speedups[0][key]:.0f}× on x86-64 and "
        f"{speedups[1][key]:.0f}× on arm64."
    )
    parts.append(
        f'<text class="secondary" x="24" y="{plot_bottom + 46}" font-size="12">'
        f"{escape(note)}</text>"
    )
    parts.append(
        f'<text class="muted" x="24" y="{plot_bottom + 64}" font-size="12">'
        "Best of 3 rounds; Python 3.12, GCC 14. "
        "desmoteo.github.io/pillow-allsimd-perf</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} DATA_DIR OUT.svg")
    Path(sys.argv[2]).write_text(draw(load(Path(sys.argv[1]))), encoding="utf-8")


if __name__ == "__main__":
    main()
