#!/usr/bin/env python3
"""Count heap allocations per operation, for several Pillow builds.

Uses the operations of ``simd_bench.py``. Each operation runs natively with
a small LD_PRELOAD library that counts malloc, calloc and realloc calls,
the bytes they request, and the peak of the bytes in use, during one call
after a warm-up call. Python's small-object allocator doesn't use malloc,
so the counts are mostly Pillow's own buffers and Python's large objects.

    python3 checks/simd_allocs.py --variant avx2=/tmp/avx2 --set full \\
        --size 1280x800 --json allocs.json

Linux with glibc only.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "simd_bench.py")
COUNTER_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <malloc.h>
#include <stddef.h>

static void *(*real_malloc)(size_t), *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static void (*real_free)(void *);
static __thread int busy;
static long calls, bytes, live, peak, base;
/* dlsym can call calloc before real_calloc is known. */
static char boot[4096];
static size_t boot_used;

static void init(void) {
    busy = 1;
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_free = dlsym(RTLD_NEXT, "free");
    busy = 0;
}
static void count(void *p, size_t n) {
    if (p && !busy) {
        calls++;
        bytes += n;
        live += malloc_usable_size(p);
        if (live > peak) peak = live;
    }
}
void *malloc(size_t n) {
    if (!real_malloc) init();
    void *p = real_malloc(n);
    count(p, n);
    return p;
}
void *calloc(size_t k, size_t n) {
    if (!real_calloc) {
        if (busy) {  /* inside dlsym */
            void *p = boot + boot_used;
            boot_used += (k * n + 15) & ~(size_t)15;
            return p;
        }
        init();
    }
    void *p = real_calloc(k, n);
    count(p, k * n);
    return p;
}
void *realloc(void *q, size_t n) {
    if (!real_realloc) init();
    if (q && !busy) live -= malloc_usable_size(q);
    void *p = real_realloc(q, n);
    count(p, n);
    return p;
}
void free(void *p) {
    if ((char *)p >= boot && (char *)p < boot + sizeof(boot)) return;
    if (!real_free) init();
    if (p && !busy) live -= malloc_usable_size(p);
    real_free(p);
}
void alloc_reset(void) { calls = bytes = 0; base = peak = live; }
void alloc_read(long *out) { out[0] = calls; out[1] = bytes; out[2] = peak - base; }
"""


def load_bench() -> Any:
    spec = importlib.util.spec_from_file_location("simd_bench", BENCH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(path: str, lib: str, size: tuple[int, int], which: str) -> None:
    sb = load_bench()
    Image, ImageChops, ImageFilter, _, _ = sb.import_pil(path)
    ops = sb.all_operations(which, Image, ImageChops, ImageFilter, *size)
    counter = ctypes.CDLL(lib)
    out = (ctypes.c_long * 3)()
    result = {}
    for name, fn in ops.items():
        fn()  # lazy imports, tables and caches stay out of the count
        counter.alloc_reset()
        fn()
        counter.alloc_read(out)
        result[name] = {"calls": out[0], "bytes": out[1], "peak": out[2]}
    print(json.dumps(result))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--size", default="1280x800")
    ap.add_argument("--set", default="full", help="operation set of simd_bench.py")
    ap.add_argument("--json", help="write the results to this file")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--lib", help=argparse.SUPPRESS)
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    if args.worker:
        worker(args.worker, args.lib, (w, h), args.set)
        return

    variants = [v.split("=", 1) for v in args.variant] or [["default", "-"]]
    with tempfile.TemporaryDirectory() as tmp:
        src, lib = os.path.join(tmp, "count.c"), os.path.join(tmp, "count.so")
        with open(src, "w") as fh:
            fh.write(COUNTER_C)
        subprocess.run(
            ["cc", "-O2", "-shared", "-fPIC", "-o", lib, src, "-ldl"], check=True
        )
        results = {}
        for name, path in variants:
            cmd = [sys.executable, "-I", os.path.abspath(__file__), "--worker", path,
                   "--lib", lib, "--size", args.size, "--set", args.set]  # fmt: skip
            env = {**os.environ, "LD_PRELOAD": lib}
            out = subprocess.run(
                cmd, env=env, check=True, capture_output=True, text=True
            ).stdout
            results[name] = json.loads(out.strip().splitlines()[-1])

    names = [n for n, _ in variants]
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    print(f"Allocations per call ({args.size}): calls, MB requested, MB peak")
    print(f"{'operation':34s}" + "".join(f"{n:>27s}" for n in names))
    for op in results[names[0]]:
        row = ""
        for n in names:
            r = results[n].get(op)
            row += (
                f"{r['calls']:9d}{r['bytes'] / 1e6:9.1f}{r['peak'] / 1e6:9.1f}"
                if r
                else f"{'-':>27s}"
            )
        print(f"{op:34s}{row}")


if __name__ == "__main__":
    main()
