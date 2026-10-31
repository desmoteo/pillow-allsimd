# Plan: fast approximate kernels beside the bit-exact ones

Status: decided 2026-10-31 (names, scope, phase 1). Phase 0 (the switch)
and phase 1 (the Gaussian blur, YCbCr, HSV and the transforms) done; see
[SIMD_FAST_KERNELS.md](SIMD_FAST_KERNELS.md). Evidence and measurements are in
[OTHER_LIBRARIES_KERNELS.md](OTHER_LIBRARIES_KERNELS.md).

## Goal

Pillow-SIMD's contract today is that every build (SSE4, AVX2, AVX-VNNI,
NEON, portable C) produces the pixels Pillow produces. That contract stays
the default. Beside it, an **opt-in fast mode** lets an operation use a
kernel that is faster and whose result may differ from Pillow's by a small,
documented amount, the way OpenCV's `ALGO_HINT_APPROX`, libvips' vector
paths and KleidiCV do. Users choose at run time, with an environment
variable or a library call, through one Boolean flag.

Two findings shape the design:

- **Fast does not have to mean less accurate.** Several of Pillow's results
  are biased by its arithmetic: the bilinear transform filter and the HSV
  conversion truncate instead of rounding, the YCbCr tables floor, the
  Gaussian is three box passes. Against float references of the same
  operation, about half the values of these results are one level off
  (41% to 50%). OpenCV's kernels for the same operations are 1.2x to 4.6x
  faster and, apart from HSV hue, closer to the reference. The flag
  therefore chooses between **"identical to Pillow"** and **"fast, within
  a documented error"**, not between correct and incorrect.
- **Much of the gap needs no approximation at all.** Fusing passes,
  16-bit running sums and folding premultiplication into the resize keep
  every pixel identical. Those changes go into the default path for
  everyone (listed in [PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md)); the flag
  is only for kernels whose arithmetic or algorithm differs.

## The switch

### Interface

| Where | Proposed name | Behaviour |
|---|---|---|
| Environment | `PILLOW_SIMD_FAST` | `1`, `true`, `on`, `yes` enable; `0`, `false`, `off`, `no` and the empty string disable; anything else warns and is ignored. Read once when `PIL.Image` is imported, in `_apply_env_variables()`, like `PILLOW_BLOCKS_MAX`. |
| Python | `PIL.Image.set_fast(enabled: bool = True) -> None`, `PIL.Image.get_fast() -> bool` | Process-wide. |
| Python | `PIL.Image.fast_mode(enabled: bool = True)` | Context manager: sets the flag, restores the previous value on exit. Documented as process-wide (it changes the flag for every thread while active). |
| Feature test | `PIL.features.check_feature("fast")` | True in Pillow-SIMD builds that have the switch (`_imaging.HAVE_FAST`), so code can detect it without catching AttributeError under upstream Pillow. |
| C core | `_imaging.set_fast(int)`, `_imaging.get_fast()` | What the Python functions call. |
| libImaging | `ImagingSetFast(int)`, `ImagingGetFast(void)` | A global word, written with release semantics and read with acquire (GCC/Clang `__atomic`, MSVC interlocked intrinsics), safe under free-threaded Python. |

The stored value is a bit set with one bit used today ("all fast kernels").
A later per-family selection (`PILLOW_SIMD_FAST=blur,resize`) would
then need no ABI change. Not part of this plan.

### Semantics

- **Default off.** Nothing changes for users who don't opt in; the exact
  path stays the reference for every test that exists today.
- **Read once per operation.** Each operation's C entry point reads the flag
  before it releases the GIL and passes the value down, so one call never
  mixes kernels, and a change from another thread applies from the next call.
- **Deterministic.** A fast kernel's result depends only on its input and
  arguments, and is the same on every build and architecture of a given
  version. Each fast kernel has a portable C implementation that defines its
  result; the SSE4, AVX2 and NEON versions must match it bit for bit, the
  discipline the exact kernels follow now. Instructions without an exact
  portable equivalent (FP16 arithmetic) are therefore out; integer
  instructions such as `SDOT` are in, since C can reproduce them exactly.
- **Not stable across versions.** Fast results may change in a later
  release, when a kernel improves; the exact path never does (beyond what
  upstream Pillow changes). Documented in the release notes.
- **Bounded.** Each fast kernel documents two numbers, measured on the
  benchmark images and checked by tests: the largest difference from the
  exact result, and its error against a float reference of the operation,
  which should not be worse than the exact path's.
- **Same shape.** The same output mode, size and border behaviour as the
  exact path; only values differ.

## Kernel catalogue

Ordered by expected gain over effort. Speeds are measured with the
equivalent OpenCV or KleidiCV kernel on the benchmark machines (2560x1600,
one core), the target for a Pillow-SIMD implementation. Errors are from the
same measurements. "Ref." is the error against the float reference: maximum
/ share of values that differ.

| # | Operation | Fast kernel | Today (x86 / Jetson ms) | Target | Ref. today | Ref. fast | Notes |
|---|---|---|---|---|---|---|---|
| 1 | GaussianBlur, small σ | a true separable Gaussian, ksize 2⌈3σ⌉+1, 8-bit weights summing to 256, vertical then horizontal through one 8-bit row (KleidiCV's scheme), one sweep | 17.9 / 51.5 (σ 1) | 7.8 / 11.2 | 2 / 49% | 1 / 12.5% | Direct taps grow with σ while box passes don't: use it below a measured crossover σ (6, measured), the box passes above. **As built**, the kernel is Pillow's three boxes convolved, not a sampled Gaussian (which under-blurs at small σ and differs by up to 25 levels at sharp edges), and the bands near the edges come from the exact blur; within 3 levels of the exact result, see SIMD_FAST_KERNELS.md. A variant in 8.8 fixed point (OpenCV's) is closer to the reference (0.4%) and 2.3x faster on x86, but only 1.2x on NEON. Unsharp mask inherits it. |
| 2 | RGB↔YCbCr | 14-bit coefficients, one rounding (BT.601 full range) | 3.2 / 17.4 to YCbCr, 2.0 / 12.2 back | 2.6 / 10.4, ~2.0 / 5.7 | 1 / 47% | 1 / 3% to 11% | 16-bit multiplies (pmaddwd, `vmull`/`vqdmulh`). Image.convert only; the JPEG codec uses libjpeg's own conversion. |
| 3 | RGB→HSV | reciprocal tables (255·2^12/v, 255·2^12/(6·diff)) and integer arithmetic, rounded | 9.0 / 35.6 | 3.6 / 13.6 | 1 / 50% hue, 24% s, v | 1 / 49% hue, 0.4% s, v | Gathers on AVX2 (vpgatherdd) or byte-table lookups; NEON `vqtbl4q` over split tables. |
| 4 | Affine, perspective, quad transforms (bilinear, bicubic) | float32 coordinates and weights, 8 (AVX2) or 4 (NEON) output pixels per step, all-inside fast path, rounding at the end | 36 / 110 (rotate bilinear), 91 / 266 (bicubic) | 24 / 40, 62 / 80 | 1 / 41% | 1 / 0% (OpenCV accurate) | Float32 is reproducible in portable C if the operations and their order are fixed and FMA contraction is disabled (`-ffp-contract=off` for these files, or explicit `fmaf` everywhere). |
| 5 | Resize | the two passes fused, the horizontal result kept in 8.8 fixed point instead of 8 bits; on arm64, 8-bit weights through `SDOT` with the 128-offset trick for large kernels | 25.7 / 93.7 (bilinear up), 3.2 / 18.1 (Lanczos to 320x200) | 25 / 70; ~3 / ~7 (estimate) | 1 / 18% | 1 / 13% | Fusion alone (class A) comes first, in the default path; this item is the precision change on top. The Lanczos target is an estimate: OpenCV's 3.6 / 6.7 ms is box averaging, not the same filter. |
| 6 | Unpremultiply (RGBa→RGBA, and in RGBA resize and transforms) | multiply by a 16-bit reciprocal table instead of dividing | 2.2 / 11.1 (RGBa→RGBA) | ~1.5 / ~4 (estimate: the cost of the premultiply, 1.3 / 3.1) | | | Only where the division is the cost, on arm64 mostly. OpenCV is slower here (3.9 / 19.0), so there is no measured target. |
| 7 | Blend | 8-bit alpha in integer arithmetic, rounded | 3.1 / 13.0 | 1.8 / 11.9 | | | Small; batch with #6. |
| 8 | Large downscales | shrink with a box by an integer factor first, then convolve the remaining ≤2x (libvips' default `gap`, Pillow's `reducing_gap=2.0`) when the caller gives no `reducing_gap` | 3.2 / 18.1 (8x Lanczos) | ~1 / ~4 (estimate) | | | Class C: a different algorithm, not a rounding of the same one; under the same flag (decided). |

Not in the catalogue, because the exact path is already as fast as
OpenCV's, or the result has no rounding to relax: kernel filters
(Pillow-SIMD is faster), rank filters and morphology, reductions, point
tables, chops, transposes, conversions without arithmetic, drawing and text.
Their remaining gaps are bytes per pixel (structural) or class-A changes.

**Drawing (lines, polygons, text), assessed for phase 1 and left out.**
Pillow-SIMD already beats OpenCV on lines (x86 2.02 vs 3.34 ms for 500
lines 5 px wide; Jetson 3.51 vs 6.02), polygons (25.1 vs 28.4; 56.3 vs
62.0) and rectangles; OpenCV's text (2.1 vs 3.2 ms) is Hershey strokes, not
comparable glyphs. Where the time goes (x86, pillow-simd-perf's `analysis/drawprof.py`):

- polygons: 24.6 ms for the 200 benchmark hexagons, 0.66 ms for the same
  hexagons scaled to 1/40: the cost is the span stores, near memory
  bandwidth, with no arithmetic to relax;
- lines: 0.87 ms for 500 lines 1 px wide, 1.38 ms at 5 px: mostly the Python
  call per line;
- text: 2.6 ms for 100 lines, of which 1.3 ms composing each line's bitmap
  from the cached glyphs and 0.95 ms blending it into the image; the rest is
  Python. A faster variant would have to place or shape glyphs differently,
  which changes the drawing, not its rounding.

None has a rounding to relax, so a fast mode gains nothing a user would
accept as the same drawing. Exact-compatible work remains possible (an
active edge table for polygons with the same intersection formula, fewer
Python calls per text line) and belongs in the default path.

The benchmark now also has unfilled outlines as overlays draw them
(`draw/outline/rectangles`: 500 bounding boxes; `draw/outline/polygons`:
100 ROI polygons of 12 points; both 2 px). Both had default-path problems,
fixed with the same pixels: thin lines went through a function pointer
per pixel since Pillow 13.0 (bounding boxes 4.03 -> 2.53 ms, AVX2;
OpenCV 4.0), and a wide polygon outline allocated and cleared an
image-sized mask per polygon (17.2 -> 2.6 ms; OpenCV 1.7).

## Implementation plan

### Phase 0: the switch, with no fast kernel yet

1. libImaging: the global word and `ImagingSetFast()` / `ImagingGetFast()`
   (new `src/libImaging/Fast.c`, declared in `Imaging.h`); an
   `IMAGING_FAST_ALL` bit.
2. `_imaging.c`: `set_fast`, `get_fast`, `HAVE_FAST`; `features.py`:
   `"fast"`; `Image.py`: `set_fast`, `get_fast`, `fast_mode()`, and the
   variable in `_apply_env_variables()` (Boolean parsing beside the integer
   ones).
3. `src/PIL/_imaging.pyi` stubs; type checks pass (`tox -e mypy`).
4. Tests (`Tests/test_fast_mode.py`): default off; setter and getter;
   environment parsing, including the warning for bad values; the context
   manager restores the previous value, also after an exception; a threaded
   test that toggles the flag while images are processed.
5. Docs: the user page and release note entries (below), marked as having no
   fast kernels yet, or held back until phase 1 lands.

### Phase 1: GaussianBlur, YCbCr, HSV, geometric transforms (items 1 to 4)

The largest gains, as decided; the transforms (item 4) are the biggest
arm64 gap (about 3x). Drawing was assessed and left out (above).

For each kernel:

1. The portable C kernel, which defines the fast result.
2. SSE4, AVX2 and NEON versions that match it bit for bit, tested on
   every build the way the exact kernels are (`checks/simd_hashes.py`
   areas, run per build and compared).
3. Dispatch in the operation's entry point: `if (fast && fast kernel
   applies) ... else exact path`. The exact path is not touched.
4. Tests: the fast result's difference from the exact result and its error
   against the reference stay within the documented bounds, on the test
   images and on the benchmark's photo; exhaustive where the domain is small
   (all 2^24 RGB values for HSV and YCbCr, every alpha for the reciprocal
   tables); the fast path's hash is the same on every build.
5. Benchmarks: a `-fast` variant of each branch build in pillow-simd-perf
   (the same build run with `PILLOW_SIMD_FAST=1`), so the page shows
   exact and fast side by side; `analysis/accuracy.py` gains the Pillow fast
   rows.
6. Docs: the kernel's entry in the user page's table and in
   `SIMD_FAST_KERNELS.md` (below).

The Gaussian needs one measurement before the code: the crossover σ where
the direct kernel stops beating the three box passes, per architecture,
taken as a constant shared by all builds (so results don't depend on the
build).

The transforms need the float reproducibility rules settled first: the
operation order written out, contraction off for those files, the result of
the portable C version on x86 and arm64 compared in CI. If that can't be
pinned on every supported compiler, they use fixed point instead (OpenCV's
5-bit sub-pixel tables).

Done that way, in fixed point (2026-10-31): positions in exact integers
rounded to 1/256 pixel, the perspective's integer sums exact in doubles
with one IEEE division and product, so contraction can't change them;
integer filters. The float32 design in the catalogue row was not needed.

### Phase 2: resize precision, unpremultiply, blend (items 5 to 7)

The resize item starts after the class-A fusion has landed in the default
path, so the fast change is only the intermediate's precision and the
arm64 `SDOT` kernel.

### Phase 3: large downscales (item 8)

Under the same flag, as decided: in fast mode, `resize` and `thumbnail`
without an explicit `reducing_gap` use 2.0 for downscales of 4x and more.

### Parallel track: class-A changes in the default path

Fused blur passes, 16-bit running column sums, fused resize passes with the
premultiply folded in, morphology row-pair sharing, transform row loops.
They keep every pixel, so they need only the existing tests and hashes, and
they make the exact path faster for every user, which narrows what the fast
mode has to offer. Doing them first also means each fast kernel is measured
against the best exact kernel, not against today's.

## Documentation plan

What gets written, where, and when.

| Document | Readers | Content | When |
|---|---|---|---|
| `docs/reference/fast_mode.rst` (new, linked from the reference index) | users | what the flag does and doesn't do; how to enable it (environment, functions, context manager); the guarantees (default off, deterministic and the same on every build, may change between versions); a table per operation: fast kernel, speed-up on x86 and arm64, largest difference from the exact result, error against the float reference; caveats (process-wide, read once per call, pipelines that hash pixels) | phase 0, filled per kernel |
| `docs/reference/Image.rst` | users | `set_fast`, `get_fast`, `fast_mode` with autodoc | phase 0 |
| `docs/reference/block_allocator.rst`, or a new environment-variables section | users | `PILLOW_SIMD_FAST` with the other `PILLOW_*` variables | phase 0 |
| `docs/releasenotes/<version>.rst` | users | "API additions" (the switch), and per release the fast kernels added or changed ("Other changes") | each release |
| `README.md` | users | a short "Exact and fast modes" section: the default, how to opt in, a link to the reference page | phase 1 |
| `SIMD_FAST_KERNELS.md` (new, repo root) | contributors and reviewers | per fast kernel: the algorithm, the arithmetic and where it rounds, the portable reference function, the SIMD versions and how they match it, the error analysis (method, inputs, numbers on both machines), the benchmark numbers, known limits (crossover σ, sizes); the rules every fast kernel follows (the semantics above) | grows with each kernel |
| `SIMD_OPTIMIZATIONS.md` (from [SIMD_EXPLAINER_PLAN.md](SIMD_EXPLAINER_PLAN.md)) | contributors | a chapter on the two modes, linking to `SIMD_FAST_KERNELS.md` | with the explainer |
| `OTHER_LIBRARIES_KERNELS.md`, `PERFORMANCE_GAPS.md` | contributors | updated when a gap closes or a library changes (OpenCV, KleidiCV, libvips versions) | as needed |
| Source comments | contributors | each fast kernel's header comment names its error bounds and its section in `SIMD_FAST_KERNELS.md` | with the code |
| pillow-simd-perf page | users | the `-fast` columns, and an accuracy table from `analysis/accuracy.py` | phase 1 |

## Testing and CI

- The existing suite runs in exact mode only, as now; it already asserts
  exact results in many places, so running it with the flag on would fail
  by design.
- `Tests/test_fast_mode.py` runs on every build in CI (scalar, SSE4,
  AVX2, AVX-VNNI, NEON, and the arm64 job), with tolerance tests per kernel
  and the cross-build hash check of the fast path.
- `checks/simd_hashes.py --fast` hashes the fast path's areas, so the
  remote test scripts compare fast results across builds as they compare
  exact ones.
- A CI job that imports Pillow with `PILLOW_SIMD_FAST=1` and checks
  that the flag is on (environment handling end to end).

## Risks and open questions

- **Placement** of the functions: functions on `PIL.Image` don't exist in upstream
  Pillow; `features.check_feature("fast")` is the portable test.
- **One flag or levels.** One Boolean, as decided, covering the class-C
  substitution too; the stored bit set leaves room for levels later.
- **Maintenance.** Each fast kernel adds a second implementation and its
  tests. The catalogue is kept short on purpose: only operations with a
  measured gain of about 1.5x or more on at least one architecture.
- **Users who compare pixels** (golden images, caches keyed by content)
  must leave the flag off; the reference page says so first.
- **Float reproducibility** for the transforms: settled by fixed point
  (phase 1, above); the only float operations left, the perspective's
  division and product, are correctly rounded everywhere.
- **Free-threaded Python**: the flag is read atomically and once per call;
  the threaded test covers toggling while operations run.

## Decisions (2026-10-31)

1. Names: `PILLOW_SIMD_FAST` and `set_fast()` (with `get_fast()` and the
   `fast_mode()` context manager beside it).
2. Large downscales (item 8) are under the same flag.
3. Phase 1: Gaussian, YCbCr, HSV, geometric transforms; lines, polygons and
   text if useful, which the measurements above say they are not.
