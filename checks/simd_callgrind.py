#!/usr/bin/env python3
"""Count instructions, cache misses and branch mispredicts per operation
with callgrind, for several Pillow builds.

Uses the operations of ``simd_bench.py``. Only the operation itself is
measured: each run starts callgrind with instrumentation off, does one
warm-up call, then switches instrumentation on for exactly one more call
through a small helper library built from ``valgrind/callgrind.h``.

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
from types import ModuleType

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "simd_bench.py")
CONTROL_C = """
#include <valgrind/callgrind.h>
void vg_start(void) {
    CALLGRIND_ZERO_STATS;
    CALLGRIND_START_INSTRUMENTATION;
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


def worker(path: str, op: str, control: str, size: tuple[int, int]) -> None:
    sb = load_bench()
    Image, ImageChops, ImageFilter, _, _ = sb.import_pil(path)
    fn = sb.operations(Image, ImageChops, ImageFilter, *size)[op]
    vg = ctypes.CDLL(control)
    fn()  # lazy imports and tables stay out of the count
    vg.vg_start()
    fn()
    vg.vg_stop()
    vg.vg_dump(op.encode())


def run(
    name: str, path: str, op: str, control: str, args: argparse.Namespace
) -> tuple[str, str, int]:
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    cmd = [
        "valgrind", "--tool=callgrind", "--instr-atstart=no", "--collect-atstart=no",
        "--cache-sim=yes", "--branch-sim=yes",
        f"--callgrind-out-file={d}/{slug(op)}.%p",
        sys.executable, "-I", os.path.abspath(__file__),
        "--worker", path, "--op", op, "--control", control, "--size", args.size,
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--only", action="append", default=[], help="operation name")
    ap.add_argument("--size", default="2560x1600")
    ap.add_argument("--out", default="callgrind-out")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--op", help=argparse.SUPPRESS)
    ap.add_argument("--control", help=argparse.SUPPRESS)
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    size = (w, h)

    if args.worker:
        worker(args.worker, args.op, args.control, size)
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
    ops = [o for o in sb.operations(Image, ImageChops, ImageFilter, 8, 8)
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
    results = {o: {n: totals(args.out, n, o) for n in names} for o in ops}
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

    print("\nMisses and mispredicts (millions)")
    header = ("variant", "D1 miss", "LL miss", "branch mp")
    print(f"{'operation':30s}" + "".join(f"{h:>11s}" for h in header))
    for o in ops:
        for n in names:
            t = results[o][n]
            if t:
                d1 = (t["D1mr"] + t["D1mw"]) / 1e6
                ll = (t["DLmr"] + t["DLmw"]) / 1e6
                mp = (t["Bcm"] + t["Bim"]) / 1e6
                print(f"{o:30s}{n:>11s}{d1:11.2f}{ll:11.2f}{mp:11.2f}")


if __name__ == "__main__":
    main()
