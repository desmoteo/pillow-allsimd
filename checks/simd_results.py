"""
The result files of the benchmark repository (pillow-allsimd-perf,
``docs/data/*.json``), for the scripts that draw and write the published
numbers: ``checks/simd_speedups_svg.py`` (the chart of the PyPI description)
and ``checks/simd_tables.py`` (the tables and figures of the documentation).

A machine's file holds, per variant (``pillow-13.0.0``, ``branch-avx2``,
``branch-neon-fast``, ``opencv``...), the best time of each case. A speed-up
is the baseline's best time over the variant's; a geometric mean is over the
cases both have, as the benchmark page computes it (``docs/app.js``).
"""

from __future__ import annotations

import json
import math
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

BASE = "pillow-13.0.0"
REPOSITORY = "desmoteo/pillow-allsimd-perf"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}"

# The two machines of the published runs, by the short name the scripts use
FILES = {
    "x86": "x86-core-ultra-9-285h.json",
    "arm64": "arm64-jetson-orin.json",
}


class Machine:
    """One machine's results."""

    def __init__(self, name: str, data: dict[str, Any]) -> None:
        self.name = name
        self.data = data
        self.title: str = data["title"]
        self.date: str = data["date"]
        self.cases: dict[str, dict[str, Any]] = data["cases"]
        self.variants: dict[str, dict[str, Any]] = data["variants"]

    def results(self, variant: str) -> dict[str, dict[str, Any]]:
        return self.variants[variant]["results"]

    def best(self, variant: str, case: str) -> float:
        """The best time of a case, in seconds."""
        return self.results(variant)[case]["best"]

    def ms(self, variant: str, case: str) -> float:
        return self.best(variant, case) * 1000

    def speedup(self, variant: str, case: str, base: str = BASE) -> float:
        """How many times faster than the baseline the variant ran the case."""
        return self.best(base, case) / self.best(variant, case)

    def geomean(
        self, variant: str, base: str = BASE, cases: list[str] | None = None
    ) -> tuple[float, int]:
        """The geometric mean of the speed-ups over the cases both have, and
        how many that is."""
        res = self.results(variant)
        ref = self.results(base)
        ids = [c for c in (cases or self.cases) if c in res and c in ref]
        if not ids:
            return math.nan, 0
        s = sum(math.log(ref[c]["best"] / res[c]["best"]) for c in ids)
        return math.exp(s / len(ids)), len(ids)

    def faster_than(self, variant: str, other: str) -> tuple[int, int]:
        """How many of the cases both have the variant ran faster, and how many
        cases that is."""
        res = self.results(variant)
        oth = self.results(other)
        ids = [c for c in self.cases if c in res and c in oth]
        return sum(res[c]["best"] < oth[c]["best"] for c in ids), len(ids)


def load(data_dir: Path) -> dict[str, Machine]:
    """The machines, from the data directory of a checkout of the benchmark
    repository."""
    return {
        name: Machine(name, json.loads((data_dir / file).read_text()))
        for name, file in FILES.items()
    }


def fetch(commit: str) -> Path:
    """Download the data files at a commit of the benchmark repository into a
    temporary directory, and return it."""
    out = Path(tempfile.mkdtemp(prefix=f"pillow-allsimd-perf-{commit[:12]}-"))
    for file in FILES.values():
        url = f"{RAW}/{commit}/docs/data/{file}"
        with urllib.request.urlopen(url, timeout=60) as response:
            (out / file).write_bytes(response.read())
    return out
