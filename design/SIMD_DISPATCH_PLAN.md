# Plan: companion files for the SIMD kernels, and run-time dispatch on x86

Status: planned 2026-11-04, for review findings 3 and 11 of
[REVIEW_2026-11.md](REVIEW_2026-11.md). Phase A (the companion files and the
kernel table) first, then phase B (the tiers and the dispatch), then the
wheels. Each phase is a series of small commits, each verified against the
hashes and the test suite; the code's pixels never change.

## What the two findings have in common

Finding 3: the SIMD code of 23 operations is inlined in Pillow's own files
under `#if defined(IMAGING_SIMD_*)`, and those files conflict at every merge
of a Pillow release. Finding 11: x86 users compile from source, because a
wheel can hold only one instruction set today, and an SSE4.1 wheel would
silently lose AVX2.

One structure answers both. The SIMD kernels move out of the scalar files
into companion files, as the resample and filter kernels already are
(`ResampleSIMD*.c`, `FilterSIMD_*.c`, `ReduceSIMD.c`), and every companion
is compiled into **one translation unit per instruction-set tier**, whose
only exported symbol is a table of kernel pointers. The scalar file keeps
its driver and its portable loop, and calls the kernel through the table of
the tier chosen when `_imaging` loads. On arm64 there is one tier (NEON);
on x86 a wheel holds SSE4.1, AVX2 and AVX-VNNI tiers and a scalar fallback.

Nothing in the kernels changes: the same intrinsics, the same loops, the
same tails. The pixels of every build stay those documented in
[docs/allsimd-kernels.rst](../docs/allsimd-kernels.rst), "Ground rules".

## Where the inline code is

Lines inside `#if ... IMAGING_SIMD` blocks, and how many of the blocks sit
inside a function body (the ones that need an entry point) rather than at
file scope (whole helper functions, which just move):

| File | SIMD lines / total | Blocks: file scope / in a function |
|---|---|---|
| Convert.c | 886 / 2,783 | 5 / 13 |
| Geometry.c | 849 / 2,222 | 2 / 6 |
| ColorLUT.c | 836 / 839 | 1 / 5 |
| TransformFast.c | 671 / 1,097 | 0 / 15 |
| BoxBlur.c | 557 / 1,523 | 1 / 14 |
| ConvertYCbCr.c | 546 / 1,007 | 4 / 4 |
| AlphaComposite.c | 350 / 544 | 1 / 5 |
| ZipEncode.c | 159 / 575 | 1 / 4 |
| Point.c | 131 / 496 | 0 / 11 |
| Chops.c | 107 / 482 | 0 / 4 |
| Paste.c | 99 / 793 | 1 / 1 |
| Bands.c | 95 / 472 | 0 / 6 |
| GaussianFast.c | 82 / 391 | 0 / 2 |
| Draw.c | 77 / 2,346 | 0 / 4 |
| UnsharpMask.c | 72 / 151 | 0 / 2 |
| Quant.c | 64 / 1,652 | 1 / 2 |
| Filter.c | 63 / 708 | 3 / 1 |
| ResampleFast.c | 63 / 320 | 0 / 2 |
| Blend.c | 43 / 167 | 0 / 2 |
| Unpack.c | 41 / 1,904 | 0 / 2 |
| ZipDecode.c | 37 / 336 | 0 / 3 |
| Pack.c | 35 / 720 | 0 / 2 |
| RankFilter.c | 22 / 724 | 1 / 0 |

Already companions, included from the scalar file today:
`ResampleSIMDHorizontalConv.c` (1,282 lines, included twice, the second
time with `RESAMPLE_OUT16` for fast mode), `ResampleSIMDVerticalConv.c`
(398), `ResampleSIMD32bpc.c` (494), `ReduceSIMD.c` (954), the eight
`FilterSIMD_*.c` (about 1,900 together). The shared helpers are
`ImagingSIMD.h` (the ISA macros, `mm_dpwssd`, `permute4`, the widening
loads) and `ImagingPx4.h` (one pixel as four 32-bit lanes).

## The design

### Files

- `src/libImaging/<Area>SIMD.c`, one per scalar file above: the SSE4,
  AVX2 and NEON code of that area, both ISAs in one file under `#if`, as
  the `FilterSIMD_*.c` files have them (the page explains them side by
  side too). Every function in a companion is `static`.
- `src/libImaging/SIMDKernels.c`: the one translation unit. It includes
  `ImagingSIMD.h`, then every companion, then defines the tier's table
  (below). On a single-tier build (arm64, `-C simd=sse4|avx2|avxvnni|
  native`, and `disable`, whose table is empty) it is compiled as it is.
- `src/libImaging/SIMDKernels_sse4.c`, `SIMDKernels_avx2.c`,
  `SIMDKernels_avxvnni.c`: three-line stubs that define
  `IMAGING_SIMD_TIER` and include `SIMDKernels.c`. The dispatch build
  compiles the stubs instead of `SIMDKernels.c`, each with its own flags.
- `src/libImaging/SIMDDispatch.c`: the CPU detection, the choice of the
  tier, the `ImagingSIMD` pointer, and the scalar table (every pointer
  NULL).
- `src/libImaging/ImagingSIMD.h` gains the `ImagingSIMDKernels` struct,
  the `IMAGING_SIMD_CALL` macro and the tier-macro logic: with
  `IMAGING_SIMD_TIER` defined, the ISA macros follow the tier, not the
  compiler's `__AVX2__` (MSVC compiles AVX2 intrinsics without `/arch`, and
  GCC's `#pragma GCC target` would redefine `__AVX2__` but clang's would
  not; the explicit macro is the same on every compiler). A tier that the
  compiler can't generate code for is a `#error` on GCC and clang.

Why one translation unit per tier and not one per companion: the same
source compiled three times needs three object files, which setuptools
names after the source, so each tier needs a stub file per source; with
the unity file that is three stubs in all, and the kernels can stay
`static`, so no symbol needs a tier suffix. The cost is three large
compilations (about 10,000 lines of intrinsics each) that run in parallel
with the rest of the module under pybind11's `ParallelCompile`.

### The table

```c
typedef struct {
    const char *name; /* "scalar", "sse4", "avx2", "avxvnni", "neon" */
    struct {
        int (*rgb2l)(UINT8 *out, const UINT8 *in, int xsize);
        ...
    } convert;
    struct { ... } blur;
    ...
} ImagingSIMDKernels;

extern const ImagingSIMDKernels *ImagingSIMD; /* the active tier */

#define IMAGING_SIMD_CALL(kernel, ...) \
    (ImagingSIMD->kernel ? ImagingSIMD->kernel(__VA_ARGS__) : 0)
```

Grouped by area, in the order of the kernels page. A tier's table is a
designated initializer in `SIMDKernels.c`; an entry whose ISA block is
compiled out stays NULL, so the scalar fallback is the absence of a
kernel, not a copy of the portable code.

The calling convention, which most inline blocks already follow: a row
kernel takes the scalar loop's arguments, does the pixels the vector loop
can, and **returns how many it did**; the scalar file finishes the row
from there (`x = IMAGING_SIMD_CALL(convert.rgb2l, out, in, xsize); for (;
x < xsize; x++) ...`). A kernel that handles a whole operation (a
transform tile, a resample pass, a filter) returns 1 for done, 0 for not
handled, and the scalar file falls through to its own code. A kernel is
called once per row or per image, never per pixel, so the indirect call
costs nothing measurable; this is checked on the benchmark (below).

`ImagingSIMD` is initialized statically to the scalar table, so a kernel
called before `ImagingSIMDInit()` runs the portable code rather than a
NULL pointer; `setup_module()` in `_imaging.c` calls the init. The
pointer is set once at import and read afterwards; `set_simd()` (below)
swaps it for tests and benchmarks, which is benign for an operation in
flight on another thread (it keeps using the entries it loaded).

### Build

`-C simd=` keeps every option it has, with one change of default:

- `auto` (the default): **on x86, `dispatch`**; on arm64 NEON; portable C
  elsewhere. A source build on x86 then gets AVX2 without being asked,
  which is what the README tells users to pass `-C simd=avx2` for today.
- `dispatch`: the three x86 tiers and the scalar fallback; the common
  code (everything but the tier units) is compiled for plain x86-64, as
  Pillow's own wheels are, plus `-ftree-vectorize` as now.
- `sse4`, `avx2`, `avxvnni`, `native`, `native512`, `neon`: one tier, the
  whole module compiled with that ISA as today, for benchmarks and for
  machines one controls. These builds check the CPU at import and raise
  `ImportError` with the tier's name when it is missing (the second item
  of the old "Later" list in [PLAN.md](PLAN.md)).
- `disable`: the scalar table only.

Per-file flags: `build_extensions()` in `setup.py` wraps the compiler's
`_compile` (what `ParallelCompile` calls per source) to add `-msse4.1`,
`-msse4.1 -mavx2` or `-msse4.1 -mavx2 -mavxvnni` when the source is one
of the three stubs. MSVC needs no per-file flags and has no `_compile`:
its intrinsics compile without `/arch`, and the stub's macro selects the
tier. The summary line becomes `--- SIMD kernels: dispatch (SSE4.1, AVX2,
AVX-VNNI)` and the release workflow's check reads it per platform.

### Run time

- `SIMDDispatch.c`: `cpuid` leaf 1 (SSE4.1, OSXSAVE), `xgetbv` (the OS
  saves the YMM state), leaf 7 (AVX2), leaf 7 subleaf 1 (AVX-VNNI), with
  `<cpuid.h>` on GCC and clang and `__cpuidex`/`_xgetbv` on MSVC. No FMA
  bit: the kernels use no FMA intrinsics (checked). The highest built tier
  the CPU supports wins.
- `PILLOW_ALLSIMD_ISA=scalar|sse4|avx2|avxvnni` caps the tier at import
  (a request above the CPU falls to the best it has), for CI and A/B runs.
- `_imaging.simd_kernels` (the active tier's name), `_imaging.simd_tiers`
  (the tiers built), `_imaging.set_simd(name)` (tests and benchmarks: an
  A/B between tiers without rebuilding), and `features.version("simd")`
  so `python3 -m PIL` reports it.

### Exactness

The contract gains a clause: **every tier of one build gives the same
pixels**, so a wheel's result doesn't depend on the CPU. Today the SSE4
and AVX2 builds agree in every area of `checks/simd_hashes.py` and on
the filters, except the 4-byte float kernel filter, whose AVX2 summation
order differs from SSE4's in 0.19% of bytes by one level (the page,
"Kernel filters"). That is fixed first, in phase B's first step: the
AVX2 kernel sums in the SSE4 order (the 1-byte AVX2 kernel already does),
or, if that costs speed, the AVX2 tier uses the SSE4 kernel for it.

The check: `Tests/test_simd_tiers.py` runs every operation of the
consistency suite on every built tier through `set_simd()` and compares
the SIMD tiers with each other byte for byte, and with the scalar tier
except in the documented areas. One process, one machine, so the Linux
x86 CI job covers SSE4 and AVX2 (the runners have AVX2; AVX-VNNI is
verified on the development machine before each release, as NEON is on
the Jetson, and the tier is skipped with a notice where the CPU lacks it).

### Measurements that gate the phases

1. After phase A, single-tier builds: the benchmark (`bench-results/
   build-variant.sh`, `-O3`, taskset 6-15, turbo off) at SSE4, AVX2 and
   NEON against the builds before the move. Expected: within noise.
2. After phase B: the dispatch build with each tier active against the
   single-tier builds of the same ISA. The difference is the common code
   compiled for plain x86-64 instead of the tier's ISA: the portable loops
   that compilers vectorize (`CONVERT_PIXELS`, the sorting networks, the
   premultiply tail, the box blur's vertical sums). Where a loop loses
   measurably, it moves into the companion as a kernel of its own, which
   the table makes a one-step change.
3. Wheel size: the AVX2 build's `_imaging` is 3.8 MB against 2.6 MB
   scalar at `-O3`, so the three tiers add about 3 MB uncompressed.
   Acceptable; recorded in the release notes.

## Phases and steps

### Phase A: companion files and the kernel table (finding 3)

- [x] A1. `ImagingSIMDKernels`, `ImagingSIMD`, `IMAGING_SIMD_CALL`,
      `SIMDKernels.c` with an empty table, `SIMDDispatch.c` with the
      static scalar table and `ImagingSIMDInit()` that selects the one
      compiled tier; `setup.py` lists the new sources. No kernel moves
      yet; builds and hashes unchanged. Done 2026-11-04 (69bf0883f).
- [x] A2. The companions that already exist join `SIMDKernels.c`:
      `Resample.c` stops including `ResampleSIMD*.c` and `ResampleFast.c`'s
      SIMD loops, `Reduce.c` stops including `ReduceSIMD.c`, `Filter.c`
      stops including `FilterSIMD_*.c`; each gets table entries. Shared
      helpers (`clip8()`, the coefficient tables) move to a header where
      both sides need them. Done 2026-11-04, three commits; the portable
      resample kernels moved out of the companions into
      `ResampleHorizontal.c` and `Resample.c`.
- [x] A3 to A12. One scalar file per commit, largest first: Convert.c and
      ConvertYCbCr.c, Geometry.c, ColorLUT.c, TransformFast.c, BoxBlur.c
      with UnsharpMask.c and GaussianFast.c, AlphaComposite.c with
      Blend.c and Chops.c, ZipEncode.c with ZipDecode.c, Pack.c and
      Unpack.c, Point.c with Bands.c and Paste.c, Draw.c, Quant.c,
      RankFilter.c. Each commit: the SIMD code moves verbatim into
      `<Area>SIMD.c`, the scalar file gets the call, the hashes of the
      SSE4, AVX2 and scalar builds are the same as before the commit, and
      the SIMD tests pass (`test_simd_consistency`, `test_image_filter`,
      `test_image_resample`, `test_image_convert`, `test_imagechops`, the
      file's own tests). NEON is checked on the Jetson at the end of the
      phase with `checks/simd_test_remote.sh` and the hashes, and in CI on
      the arm64 runners at every push. Done 2026-11-04 (0238a3998 to
      35a821700, plus the `permute4` shuffle of `ImagingSIMD.h`, the last
      SIMD code a scalar file compiled itself); the whole test suite
      passes on the three x86 builds. Two lessons: the fast transforms,
      first called per tile row of 64 pixels, cost the SSE4 build 14% in
      calls and vector constants set up again on each call, so they are
      called per region (`TransformFast.h`, instantiated once scalar and
      once with the vector loops); and the one translation unit found
      one name clash (`interleave4`), fixed with a per-area prefix.
      Quick timings of each move are within noise except the SSE4 box
      blur's direct passes, 3 to 5% slower, to confirm in measurement 1.
      The Jetson was unreachable on 2026-11-04; NEON was built and tested
      under QEMU instead, and the Jetson run is still to do.
- [x] A13. Measurement 1. The kernels page's Appendix B (file map) and the
      code maps of the sections list the companions; CHANGES.SIMD.rst.
      Done 2026-11-04. The measurement (`checks/simd_bench.py`, 43
      operations, best of 5 interleaved rounds, the `-O3` Docker builds of
      b8fb92778 and 770e78be3, `bench-results/ab-phaseA-{avx2,sse4}.txt`):
      outputs identical; AVX2 between 0.93x and 1.08x with no pattern
      (premultiply 0.93x, colour LUT 1.06x, max filter 1.07x); SSE4 between
      0.87x and 1.08x, most within 3%, the blurs and the 5x5 filter at
      0.96-0.97x and the reduces by 8 at 0.87x (0.50 to 0.57 ms, a
      whole-image kernel called once, so layout rather than calls). Within
      the noise of single runs; the published benchmark after phase B will
      settle it.

### Phase B: the tiers (finding 11)

- [x] B1. The 4-byte float kernel filter: AVX2 sums in the SSE4 order;
      measure; the page's "Kernel filters" section loses its exception.
      Done 2026-11-04 (68a594fed): the AVX2 kernel was 1.6x slower than
      the SSE4 one and rounded 0.2% of the bytes differently, so the AVX2
      tier uses the SSE4 kernel; the AVX2 and SSE4 hashes now agree in
      every area.
- [x] B2. The tier macro in `ImagingSIMD.h`, the three stubs, `-C
      simd=dispatch` with the per-file flags, `auto` → `dispatch` on x86.
      A dispatch build on the development machine: all three tiers
      present, the hashes of each tier (through `PILLOW_ALLSIMD_ISA`)
      equal to the single-tier builds'.
      Done 2026-11-04 (4d517eca8). The per-file flags go through a wrapper
      of the compiler's per-source `_compile` in `setup.py`; the common
      files get `IMAGING_SIMD_DISPATCH` and no instruction-set flag. On
      the Core Ultra 9 285H the build has the four tables, `avxvnni`
      active, `_imaging.so` 6.2 MB, and each tier's hashes equal the
      single-tier builds' (the AVX-VNNI tier's equal the AVX2 build's).
      Compile time: a tier's unit takes about two seconds at `-O2` with
      GCC 13, so the units stay whole.
- [x] B3. The CPU detection, `ImagingSIMDInit()` choosing the tier, the
      env variable, `set_simd()`, `simd_kernels`, `simd_tiers`, the
      `features` entry, the import check of single-tier builds.
      Done 2026-11-04 (4d517eca8): `cpuid` leaf 1 (SSE4.1; OSXSAVE and AVX
      with XCR0 bits 1 and 2 through `xgetbv`), leaf 7 (AVX2), leaf 7
      sub-leaf 1 (AVX-VNNI); `features.check("simd")`; `pilinfo` prints
      the active tier and the tiers built.
- [x] B4. `Tests/test_simd_tiers.py`; the x86 CI jobs run it (`auto` is
      now the dispatch build) and `checks/simd_hashes.py` per tier;
      `test-windows.yml` builds the dispatch build with MSVC.
      Done 2026-11-04 (4d517eca8): 75 operations, two shapes, exact and
      fast, per tier; `test.yml` runs the whole suite on the dispatch
      build under `PILLOW_ALLSIMD_ISA` set to `sse4`, `avx2` and
      `scalar` besides the runner's best tier. `test-windows.yml`
      installs with no `simd` option, so it builds the dispatch build
      (SSE4.1 and AVX2 tiers) with MSVC; its first run after the push is
      the check that MSVC compiles it.
- [x] B5. Measurement 2; loops moved into companions where it says so.
      AVX2 done 2026-11-04 (`bench-results/ab-phaseB-avx2.txt`, the `-O3`
      Docker builds of 4d517eca8, `disp` with `PILLOW_ALLSIMD_ISA=avx2`
      against `one-avx2`): outputs identical, 0.96x to 1.08x with no
      pattern (premultiply 1.08x, unpremultiply 0.96x, both about 1 ms),
      so the common code compiled for plain x86-64 costs nothing
      measurable. SSE4 done the same day (`ab-phaseB-sse4.txt`,
      `PILLOW_ALLSIMD_ISA=sse4` against `one-sse4`): outputs identical,
      0.96x to 1.10x (the multiply and screen chops 0.96x at 1.6 ms,
      merge 1.10x). No loop moves: the hot loops are all in the
      companions already, and the scalar tails are too short to show.
- [x] B6. Docs: README (installation: x86 wheels, the `simd` option list,
      the "Why the SIMD code isn't in Pillow" section's claim about
      `-mavx2`, now answered), `docs/allsimd-kernels.rst` (ground rules'
      build model, the exactness clause, Appendix B and C), PyPI.rst,
      `docs/reference/fast_mode.rst` where it names builds,
      CHANGES.SIMD.rst, a release note.
      Done 2026-11-04. `fast_mode.rst` names no build, so it is unchanged;
      the release note is the CHANGES.SIMD.rst entry (the fork's
      changelog; `docs/releasenotes/` is Pillow's).

### Phase C: x86 Linux wheels

- [x] C1. `release_plan.py`: `manylinux_2_28 x86_64` and `musllinux
      x86_64` in the Linux platforms; the check step expects the dispatch
      summary line on x86 and NEON on arm64; `check_wheel.py` asserts the
      tiers. The wheel tests run the suite on the runner's best tier and
      `test_simd_tiers.py` covers the rest.
      Done 2026-11-04: each platform entry carries its `kernels` line and
      the check step greps for it; `check_wheel.py` expects the `simd`
      feature and the tiers of the machine (four on x86-64, three with
      MSVC, two on arm64); `test_simd_tiers.py` compares only the tiers
      the CPU can run, since a wheel holds tiers the runner may lack. The
      expected count of a Linux release is 29 files (the source package
      and 4 x 7 wheels). The install docs (README, `PyPI.rst`,
      `docs/allsimd.rst`) say Linux x86-64 and arm64 get wheels.
      Checked before CI with cibuildwheel 4.3.1 on this machine, from a
      clean clone (`cibuildwheel --only cp312-manylinux_x86_64`; the
      working tree has ignored test-image directories that break the
      wheel test's `mv`, and needs `wheels/multibuild` checked out):
      the wheel builds with `dispatch (SSE4.1, AVX2, AVX-VNNI)`, 9.2 MB
      after auditwheel, and in it the self-test, `check_wheel.py` (the
      four tiers) and the full suite pass (8,124 passed, 106 skipped)
      under the AVX-VNNI tier.
- [ ] C2. Release 13.0.0.post3 as `allsimd-13.0.0.post3-linux`; verify
      on PyPI; the perf page gets a `wheel` variant run (the dispatch
      build's AVX2 tier) next to `branch-avx2`.

### Phase D: the other platforms, each with its own go/no-go

- [ ] D1. macOS x86_64 wheels with the macOS release (`macos-26-intel`
      runner, as upstream's `wheels.yml` uses), once the arm64 macOS
      wheels exist.
- [ ] D2. Windows x64: `test-windows.yml` green with the dispatch build is
      the gate; then `windows-latest` with `cibw_arch: AMD64` in the
      release matrix (upstream's `winbuild/build_prepare.py` builds the
      dependencies). Windows arm64 (MSVC, NEON) stays untested until
      a runner is available.

## Risks and what to do about them

- **Static helpers with the same name in two companions**: they all live
  in one translation unit now. The compiler reports the collision; the
  rule is a per-area prefix for file-scope helpers.
- **Shared code between a scalar file and its companion** (a lookup
  table, an inline clip): it moves to a small header, never duplicated.
- **Compile time**: three units of about 10,000 lines each at `-O2` or
  `-O3`. Measured on the first dispatch build; if a unit takes more than
  about a minute, the companions split into two units per tier (resample
  and filters in one, the rest in the other).
- **The common code at plain x86-64 in wheels**: measurement 2 decides,
  loop by loop.
- **`-C simd=native`** still compiles one tier with `-march=native` and
  `-ffp-contract=off`; unchanged.
- **Other architectures** (`design/OTHER_ARCHITECTURES.md`): the table
  with every entry NULL; `SIMDKernels.c` compiles to the empty table.
