#!/usr/bin/env python3
"""Count instructions, cache misses and branch mispredicts per operation
with callgrind, for several Pillow builds.

Uses the operations of ``simd_bench.py``. Only the operation itself is
measured: each run starts callgrind with instrumentation off, switches it
on for a warm-up call, which fills the simulated caches but isn't counted,
then collects for exactly one more call, through a small helper library
built from ``valgrind/callgrind.h``.

    python3 checks/simd_callgrind.py --variant scalar=/tmp/scalar \\
        --variant sse4=/tmp/sse4 --out /tmp/cg --jobs 12

Builds should carry debug info for line-level annotation, for example
``CFLAGS="-O2 -g -DNDEBUG" pip install --target ...``. Setting CFLAGS
replaces Python's own flags, so keep -O2 in it. Then:

    callgrind_annotate --auto=yes /tmp/cg/sse4/<operation>.<pid>.1

Valgrind's branch predictor model has no history, so short loops count
a mispredict at every exit; its cache model has no prefetcher. Use the
instruction counts first and the miss counts as a guide only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

TYPE_CHECKING = False
if TYPE_CHECKING:
    from types import ModuleType

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "simd_bench.py")
CONTROL_C = """
#include <valgrind/callgrind.h>
void vg_instrument(void) { CALLGRIND_START_INSTRUMENTATION; }
void vg_start(void) {
    CALLGRIND_ZERO_STATS;
    CALLGRIND_TOGGLE_COLLECT;
}
void vg_stop(void) {
    CALLGRIND_TOGGLE_COLLECT;
    CALLGRIND_STOP_INSTRUMENTATION;
}
void vg_dump(const char *name) { CALLGRIND_DUMP_STATS_AT(name); }
"""


def load_bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("simd_bench", BENCH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slug(op: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", op)


def worker(path: str, op: str, control: str, size: tuple[int, int], which: str) -> None:
    sb = load_bench()
    Image, ImageChops, ImageFilter, _, _ = sb.import_pil(path)
    fn = sb.all_operations(which, Image, ImageChops, ImageFilter, *size)[op]
    vg = ctypes.CDLL(control)
    fn()  # lazy imports and tables stay out of the count
    vg.vg_instrument()
    fn()  # warms the simulated caches; collection is still off
    vg.vg_start()
    fn()
    vg.vg_stop()
    vg.vg_dump(op.encode())


def run(
    name: str, path: str, op: str, control: str, args: argparse.Namespace
) -> tuple[str, str, int]:
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    if args.resume and totals(args.out, name, op):
        return name, op, 0
    # Dumps of an earlier run would be read as this run's results.
    for old in glob.glob(os.path.join(d, slug(op) + ".*")):
        if not old.endswith(".log"):
            os.remove(old)
    cmd = [
        "valgrind", "--tool=callgrind", "--instr-atstart=no", "--collect-atstart=no",
        "--cache-sim=yes", "--branch-sim=yes", *args.vg_arg,
        f"--callgrind-out-file={d}/{slug(op)}.%p",
        sys.executable, "-I", os.path.abspath(__file__),
        "--worker", path, "--op", op, "--control", control, "--size", args.size,
        "--set", args.set,
    ]  # fmt: skip
    with open(os.path.join(d, slug(op) + ".log"), "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return name, op, rc


def totals(out: str, name: str, op: str) -> dict[str, int] | None:
    # The dump made by the client request is the one numbered .1.
    for f in glob.glob(os.path.join(out, name, slug(op) + ".*.1")):
        events = values = None
        with open(f) as fh:
            for line in fh:
                if line.startswith("events:"):
                    events = line.split()[1:]
                elif line.startswith("totals:"):
                    values = [int(v) for v in line.split()[1:]]
        if events and values:
            return dict(zip(events, values))
    return None


def functions(out: str, name: str, op: str, top: int) -> list[dict[str, Any]]:
    """The most expensive functions of a dump by their own (exclusive) cost."""
    files = glob.glob(os.path.join(out, name, slug(op) + ".*.1"))
    if not files:
        return []
    cmd = ["callgrind_annotate", "--inclusive=no", "--threshold=100",
           "--show-percs=no", "--auto=no", files[0]]  # fmt: skip
    text = subprocess.run(cmd, capture_output=True, text=True).stdout
    events: list[str] = []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("Events shown:"):
            events = line.split()[2:]
            continue
        parts = line.split()
        if not events or len(parts) <= len(events) or "PROGRAM" in line:
            continue
        values = parts[: len(events)]
        if not all(re.fullmatch(r"[\d,]+|\.", v) for v in values):
            continue
        costs = {e: int(v.replace(",", "")) if v != "." else 0
                 for e, v in zip(events, values)}  # fmt: skip
        fn = " ".join(parts[len(events) :])
        obj = re.search(r" \[(.*)\]$", fn)
        rows.append(
            {
                "function": re.sub(r" \[.*\]$", "", fn),
                "object": os.path.basename(obj.group(1)) if obj else "",
                **costs,
            }
        )
        if len(rows) == top:
            break
    return rows


def derived(t: dict[str, int], pixels: int) -> dict[str, float]:
    """Per-pixel and per-1000-instruction rates of one dump's totals."""
    ir = max(t["Ir"], 1)
    d1 = t["D1mr"] + t["D1mw"]
    ll = t["DLmr"] + t["DLmw"] + t["ILmr"]
    bm = t["Bcm"] + t["Bim"]
    return {
        "Ir/px": t["Ir"] / pixels,
        "mem/Ir": (t["Dr"] + t["Dw"]) / ir,
        "Bc/kIr": 1000 * t["Bc"] / ir,
        "Bcm%": 100 * t["Bcm"] / max(t["Bc"], 1),
        "Bim%": 100 * t["Bim"] / max(t["Bi"], 1),
        "D1m/kIr": 1000 * d1 / ir,
        "LLm/kIr": 1000 * ll / ir,
        # KCachegrind's cycle estimate: Ir + 10 Bm + 10 L1m + 100 LLm.
        "CEst/px": (t["Ir"] + 10 * (bm + d1 + t["I1mr"]) + 100 * ll) / pixels,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--only", action="append", default=[], help="operation name")
    ap.add_argument("--size", default="2560x1600")
    ap.add_argument("--set", default="simd", help="operation set of simd_bench.py")
    ap.add_argument("--out", default="callgrind-out")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument(
        "--vg-arg",
        action="append",
        default=[],
        help="extra valgrind argument, e.g. --vg-arg=--D1=49152,12,64",
    )
    ap.add_argument(
        "--top", type=int, default=8, help="functions to keep per dump (default 8)"
    )
    ap.add_argument(
        "--resume", action="store_true", help="keep the dumps of an earlier run"
    )
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--op", help=argparse.SUPPRESS)
    ap.add_argument("--control", help=argparse.SUPPRESS)
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    size = (w, h)

    if args.worker:
        worker(args.worker, args.op, args.control, size, args.set)
        return

    variants = [v.split("=", 1) for v in args.variant] or [["default", "-"]]
    os.makedirs(args.out, exist_ok=True)
    control = os.path.join(os.path.abspath(args.out), "vgctl.so")
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as src:
        src.write(CONTROL_C)
    subprocess.run(
        ["cc", "-O2", "-shared", "-fPIC", "-o", control, src.name], check=True
    )
    os.unlink(src.name)

    sb = load_bench()
    Image, ImageChops, ImageFilter, _, _ = sb.import_pil(variants[0][1])
    ops = [o for o in sb.all_operations(args.set, Image, ImageChops, ImageFilter, 8, 8)
           if not args.only or o in args.only]  # fmt: skip

    jobs = [(n, p, o) for o in ops for n, p in variants]
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run, n, p, o, control, args) for n, p, o in jobs]
        for fut in concurrent.futures.as_completed(futures):
            name, op, rc = fut.result()
            print(
                f"{name:10s} {op:30s} {'ok' if rc == 0 else f'exit {rc}'}", flush=True
            )

    names = [n for n, _ in variants]
    results: dict[str, dict[str, Any]] = {}
    for o in ops:
        results[o] = {}
        for n in names:
            t = totals(args.out, n, o)
            results[o][n] = t and {
                **t,
                "derived": derived(t, w * h),
                "functions": functions(args.out, n, o, args.top),
            }
    with open(os.path.join(args.out, "totals.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    print("\nInstructions (millions)")
    print(f"{'operation':30s}" + "".join(f"{n:>12s}" for n in names))
    for o in ops:
        row = []
        for n in names:
            t = results[o][n]
            row.append(t["Ir"] / 1e6 if t else float("nan"))
        print(f"{o:30s}" + "".join(f"{v:12.1f}" for v in row))

    print("\nRates (see derived() for the definitions)")
    keys = ("Ir/px", "Bc/kIr", "Bcm%", "D1m/kIr", "LLm/kIr", "CEst/px")
    print(f"{'operation':30s}{'variant':>10s}" + "".join(f"{k:>9s}" for k in keys))
    for o in ops:
        for n in names:
            t = results[o][n]
            if t:
                d = t["derived"]
                print(f"{o:30s}{n:>10s}" + "".join(f"{d[k]:9.1f}" for k in keys))

    print("\nMost expensive function (share of instructions, of mispredicts)")
    for o in ops:
        for n in names:
            t = results[o][n]
            if t and t["functions"]:
                f = t["functions"][0]
                ir = 100 * f["Ir"] / max(t["Ir"], 1)
                bm = 100 * (f["Bcm"] + f["Bim"]) / max(t["Bcm"] + t["Bim"], 1)
                print(f"{o:30s}{n:>10s}  {ir:5.1f}% {bm:5.1f}%  {f['function']}")


if __name__ == "__main__":
    main()
