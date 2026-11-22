# Plain C the compiler leaves scalar: an assessment (2026-11-22)

Where Pillow-AllSIMD's own C spends time in loops that have no kernel and
that GCC does not vectorize, on x86-64 and on arm64, and which of them a
rewrite of the C would hand to the compiler on both at once. This is
stage 3 of [OPTIMIZATION_METHOD.md](OPTIMIZATION_METHOD.md) ("help the
compiler") applied to the whole library instead of one loop. Nothing here
is implemented; the last section orders the work.

It started from one loop. In the scalar tier, fast mode's fused resize was
1.2 to 2.2 times slower than the exact resize, because its portable
vertical loop summed the taps of one value at a time. Rewritten for the
compiler (chunks of a constant size, one tap over a whole chunk, 16-bit
sums, the rounded product from its two 16-bit halves, the per-tap loop in
a function of its own) it runs up to 61% fewer instructions and gives
the same pixels (`ResampleFast.c`, `rf_vtap`). The question was how many more
loops are like it.

## Short answer

- With kernels in place, little heavy pixel work is left in plain C on
  either architecture. The AVX2 tier and the NEON build cover nearly the
  same operations (the tables differ in 13 of 99 entries).
- Four places still pay in every build, each fixed by a small rewrite
  that was compiled for both targets here: **ImageMath** on integer
  images, **`Image.point()` with a scale and offset** on I, F and I;16
  images, **the glyph compositing of text**, and **ImageMorph**.
- arm64 alone has two more: **float kernels on single-band images**
  (`ImageFilter.Kernel` with coefficients the integer kernels cannot
  hold) and, in `-O2` builds, **the vertical pass of the box blur**.
- Builds without kernels (the scalar tier of the x86-64 wheel on CPUs
  before SSE4.1, and every architecture other than x86-64 and arm64) have
  the most to gain: 53 loops vectorize once two obstacles are removed,
  and **fast mode's Gaussian blur is 1.5 to 4.3 times slower than the
  exact one there**, for the reason the resize was.
- The obstacles are few and mechanical. The commonest is an `int` loop
  index used in arithmetic such as `in[x - 2]`, which GCC will not
  vectorize under `-fno-strict-overflow`, the flag every common Python
  build passes.

## Method

1. **Where the instructions are.** `checks/simd_callgrind.py` on the 219
   operations of `simd_bench.py --set full` at 640x400, on the dispatch
   build of the working tree at `-O3` (the benchmark image, GCC 14.2), once
   in the AVX2 tier and once in the scalar tier. A parser
   (`bench-results/cfriendly/hot.py`) reads each dump into instruction
   counts per source line of Pillow's own files.
2. **What the compiler did with each loop.** Every compile command of the
   build replayed one translation unit at a time with
   `-fopt-info-vec-optimized -fopt-info-vec-missed`, at `-O3` and `-O2`,
   for x86-64 and, with the cross compiler of the same GCC, for aarch64
   (`static2.sh`). A small scanner (`loops.py`) finds the extent of every
   loop in the sources, so that a hot line maps to its innermost loop and
   the loop to its remarks (`xref.py`).
3. **Why, in bulk.** The same replay with `-fstrict-overflow`, with
   `-fno-trapping-math`, and with both (`flags.sh`). A loop that
   vectorizes only with the flag is blocked by exactly that obstacle, so
   the difference between the runs is the list of loops to rewrite. The
   flags are a probe, not a proposal: see "The obstacles".
4. **Does the rewrite work.** Extracts of the candidate loops compiled
   for both targets at both levels (`exp/`), and the generated loop
   bodies counted.

Limits. The Jetson was unreachable that day, so arm64 has the compiler's
reports and its generated code, and no timings. What arm64 executes in
plain C is inferred from the kernel tables: the AVX2 tier's plain loops,
plus the fallbacks of the kernels only x86 has. One compiler (GCC 14.2);
the macOS wheels are built with Clang and the Windows ones with MSVC,
whose vectorizers differ. Instruction counts are not time: they rank
loops, and every gain below marked "estimated" comes from counting the
instructions of the generated loop, not from a clock.

## The loops, counted

1,276 innermost loops in `src/libImaging` and `src/*.c`; 202 contain
intrinsics, 1,074 are plain C.

| plain loops, by what GCC 14 does at `-O3` / `-O2` | loops | instructions in the AVX2-tier profile (M) |
|---|---:|---:|
| vectorized on both targets at both levels | 193 | 80 |
| vectorized on neither target | 554 | 426 |
| no remark (not a loop any more after inlining, or dead) | 229 | 14 |
| vectorized on arm64, not on baseline x86-64 | 34 | 1 |
| vectorized at `-O3` only, both targets | 15 | 4 |
| vectorized on x86-64, not on arm64 | 7 | 6 |
| other combinations | 42 | 17 |

557 are vectorized on neither target at `-O3`. 235 of those are in the
codecs, the quantizers, drawing and the glue to Python. The other 322, by
the compiler's reason (a loop can have several):

| reason given | loops | what it usually is |
|---|---:|---|
| unsupported control flow | 105 | a branch per pixel; some avoidable |
| number of iterations cannot be computed | 54 | the bound is reloaded from memory every iteration |
| no reason given | 54 | mostly outer loops |
| relevant statement not supported | 42 | often the index obstacle below, misreported |
| no vector type for a statement | 35 | table lookups, pointers loaded in the loop |
| not profitable | 34 | short trip counts |
| two or more inner loops | 13 | outer loops |

The kernel tables: 92 entries in the AVX2 tier, 93 in the NEON build.
Only x86 has `Blur.columns`, `Blur.line32x2`, `Blur.interleave4`,
`Blur.deinterleave4`, `Filter.f3x3_u8` and `Filter.f5x5_u8`. Only NEON has
`Point.lut_8_8`, `Point.lut_nx8`, `Filter.ismall`, `Convert.l2rgb`,
`Convert.rgb2rgba`, `Convert.rgb2cmyk` and `Bands.merge`.

## The obstacles

Each with how it shows, how many loops it holds back, and the rewrite.

### 1. `int` index arithmetic under `-fno-strict-overflow`

```c
for (x = 2; x < xsize - 2; x++) {          /* int x */
    ss += (float)in0[x - 2] * kernel[0] + ...;
```

`x - 2` in `int` may wrap under `-fno-strict-overflow`, so the address is
not a linear function of the loop counter and the loop stays scalar at
every level. GCC's message does not say so: it reports "unsupported
data-type float" on x86-64 and "relevant stmt not supported: _5 = (float)
_4" on arm64. A six-line test separates the cause: the loop vectorizes
with `long x`, or without the flag, and not otherwise
(`exp/g.c`).

Compiling everything with `-fstrict-overflow` as a probe, 58 more loops
vectorize on x86-64 and 43 on arm64. The rewrite is the index type
(`ptrdiff_t`), or pointers that advance; the method note already says so
for `in + x * 4`, and this is the same trap in its `in[x - d]` form.
Turning the flag off for the extension is not proposed: Python's flags
carry it because C code checks for overflow after the fact, and Pillow's
size checks have not been audited for that.

### 2. Float clamps written as ternaries

```c
in = in < 0.0f ? 0.0f : in;
in = in > 255.0f ? 255.0f : in;
return (UINT8)in;
```

A float comparison may raise an exception, so GCC keeps the selects as
branches ("unsupported control flow"). `Filter.c`'s `clip8()` is this, and
its comment says "branchless". With `-fno-trapping-math` as a probe 5 more
loops vectorize, and 13 more once obstacle 1 is gone too: the nine generic
loops of `Filter.c` need both. Two rewrites compile to vectors on both
targets: comparing the float and selecting among integers, or the flag
itself for the extension, which changes no result (it only says nobody
reads the floating-point exception flags) and is what Clang assumes by
default. The first is local. For the second, whether a program that embeds
Pillow enables floating-point traps was not looked into.

### 3. A bound the compiler must reload

```c
for (x = 0; x < out->xsize; x++) {     /* ImageMath, INT32 *p0 */
    *p0 = *p1 + *p2;
```

A store through `INT32 *` may change `out->xsize`, an `int`, so the trip
count is unknown ("number of iterations cannot be computed"). The same
loop on `FLOAT32` vectorizes, because a `float` store cannot alias an
`int`: in `_imagingmath.c` the 14 float operations other than division,
modulo and power are vectorized, and all 23 integer ones are scalar. 54 loops outside the codecs carry this remark:
32 in `Reduce.c` (`x < box[2] / xscale`), 8 in `Point.c`
(`x < imIn->xsize`), the rest scattered. The rewrite: the bound in a
local, and `restrict` on the output row.

### 4. No `restrict`, at `-O2`

Without `restrict` GCC vectorizes behind a run-time overlap check, and at
`-O2` it gives up when a loop would need more than a few of them.
`ImagingBoxBlurColumns` has five pointers and is vectorized at `-O3` and
scalar at `-O2` on both targets ("number of versioning for alias run-time
tests exceeds 6"). 16 plain loops are vectorized at `-O3` only; this is
the one that matters.

### 5. Loops merged by `-O3`

In the resize fix the per-tap loop, inside the loop over the taps, was
reported "vectorized", and a vector copy was in the binary, but the code
that ran was scalar: `-O3` had unrolled the tap loop by two and merged
the two inner loops into one that it then left alone. The remarks do not
show this; the instruction count per line does. The per-tap loop is now a
function that is never inlined (`SIMD_NOINLINE`). Any other loop of the
shape "for each tap, for each value" is a suspect, and the fast
Gaussian's is one.

### 6. Branches that are not needed

`_imagingft.c` skips pixels whose coverage is zero and takes another
branch where the target is zero. The blend formula gives the same value
in both cases (checked for all 65,536 input pairs), so the branches only
keep the loop scalar.

## Candidates

Instruction counts per pixel are from the profile (640x400); "after" is
the generated loop of the rewritten extract, counted, on x86-64 and arm64.

### In every build, x86-64 tiers and arm64

| what | where | now | after (estimated) | rewrite |
|---|---|---|---|---|
| ImageMath on I images: 23 operations | `_imagingmath.c`, the `UNOP` and `BINOP` macros | 7 per pixel and operation | 18 and 20 more loops vectorized of 40 (x86-64, arm64) | bounds in locals, `restrict` output, indexed rows |
| `point()` with scale and offset on I, F, I;16 | `Point.c:325-370` | 10 | about 4 (2 lanes of `double`) | the same |
| text: glyph coverage into the mask | `_imagingft.c:1380-1476` | 7 to 25, by branch | 3 (x86-64), 1.5 (arm64) | branch-free blend |
| ImageMorph `apply` | `_imagingmorph.c:95-115` | 46 | about 15 | the 9-bit index of a row in one loop, the table in a second, the count in a third |

- ImageMath's output is a new image (`ImageMath.py` creates it), so
  `restrict` holds. Division, modulo and power stay scalar. At `-O2`
  neither version is vectorized: `_imagingmath` is a separate extension
  that does not get `_imaging`'s flags, and its trip counts are unknown.
  Constant-size chunks, as in `rf_vtap`, would cover `-O2`.
- The glyph loop is 18% of the instructions of drawing text (the rest is
  FreeType and the paste), so the whole operation gains about a seventh.
- ImageMorph's table lookup stays scalar; the gain is in building the
  index.
- `Point.c`'s table lookups (`ImageEnhance`, `colorize`) carry the same
  reloaded bound and stay table lookups: nothing to vectorize on x86, and
  NEON has kernels for them.

### On arm64 only

| what | where | why only arm64 | rewrite |
|---|---|---|---|
| float kernels on L images, 3x3 and 5x5 | `Filter.c:225-231`, `404-412` | `Filter.f3x3_u8` and `f5x5_u8` exist for x86 only | obstacles 1 and 2 |
| box and Gaussian blur, vertical pass, `-O2` builds | `BoxBlur.c:495-522` | `Blur.columns` exists for x86 only; arm64 relies on the compiler here | `restrict` on the five pointers |
| mode filter, the count of equal values | `ModeFilter.c:59-61` | vectorized on x86-64, not on arm64 | not looked into; it runs where the 3x3 kernel does not |

The 5x5 float row was compiled both ways for arm64: 164 instructions per
pixel as written, about 43 once it vectorizes (16 pixels in 690
instructions). On baseline x86-64 the same vector loop is no shorter than
the scalar one, since SSE2 widens bytes to floats in many steps; there
the kernel does the work.

### In builds without kernels

The scalar tier of the x86-64 wheel and every other architecture run the
portable version of each kernel. Fast mode is the default, so they run
the portable versions of the fast kernels too.

| what | instructions in the scalar-tier profile (M) | obstacle |
|---|---:|---|
| fast Gaussian, `gauss_taps` (`GaussianFast.c:150-156`) | 831 | one value at a time over the taps; pointers loaded in the loop |
| generic 3x3 and 5x5 filters (`Filter.c`, 9 loops) | 764 | 1 and 2 |
| portable resampling kernels (`ResampleHorizontal.c`) | 404 | 1 |
| bicubic and bilinear transforms (`TransformFast.h`) | 411 | 1, and row pointers loaded per tap |
| 1-band resampling (`Resample.c:479`) | 51 | 1 |
| YCbCr conversions (`ConvertYCbCr.c`) | 20 | 1 |
| `Bands.c` (extrema, split, merge) | 15 | 1 |
| `Reduce.c`: 15 loops by the index, 32 by the bound | 3 | 1 and 3 |
| `Pack.c`, `Unpack.c`, `Geometry.c` | under 1 | 1 |

The fast Gaussian in the scalar build, `-O3`, 2560x1600, exact against
fast in one process:

| case | exact, ms | fast, ms |
|---|---:|---:|
| Gaussian RGB, radius 1 | 113 | 187 |
| Gaussian RGB, radius 5 | 173 | 745 |
| Gaussian L, radius 5 | 47 | 166 |
| unsharp mask RGB, radius 2 | 232 | 349 |
| Gaussian RGB, radius 20 (beyond fast mode's limit) | 175 | 178 |

This is the resize's problem again and has the resize's fix. Until it is
fixed, a build without kernels is better off in exact mode for Gaussian
blur and unsharp mask.

### `-O2` builds

Debian's, Ubuntu's and conda's Pythons compile extensions at `-O2`, the
Docker Hub Python images at `-O3` (the level of the wheels' interpreters
was not checked here). The dispatch build
of the working tree at both levels, 219 operations at 2560x1600, three
interleaved rounds, the tier this CPU selects (AVX-VNNI),
`bench-results/cfriendly/o2-vs-o3.txt`: the `-O2` build is 2.5% slower in
the geometric mean, 21 operations by more than 10%, none faster by as
much.

| operations | `-O2` time over `-O3` |
|---|---:|
| box blur, radius 2, RGB and L | 1.45 |
| transposes with a 90-degree turn, RGB and L | 1.25 to 1.52 |
| bilinear transforms (quad, mesh, rotate 30 degrees) | 1.33 to 1.44 |
| 3x3 filters on L, median 5 on L | 1.21 to 1.22 |
| resize of F and I;16 | 1.14 to 1.18 |

The machine was running other jobs during the run (the `-O3` build's own
rounds differ by 2% in the median and 30% at the 90th percentile), so the
groups are to be trusted and single lines are not. These are kernels,
not plain loops: `-O3` unrolls and inlines around the intrinsics (the box
blur's unrolling was measured at 1.46 before, in the method note). The
vectorizer's remarks tell the same story from the other side: 669 loops
vectorized at `-O3` and 521 at `-O2` on x86-64, 522 and 385 on arm64.

One line in `setup.py` would remove the difference for every source
build on both architectures: `-O3` appended to the extension's flags for
GCC and Clang, where the interpreter's `-O2` comes first and the later
flag wins. It needs the test suite run once on an interpreter built
at `-O2` with the flag in place (the Docker images used for tests and
benchmarks are all `-O3`), and a decision that a distribution's choice
of level may be overridden.

### Not candidates

Hot, scalar, and sequential by nature: the median cut and its sort, the
nearest-colour search with its early exit, the hash tables; the
error-diffusion dithers; LZW in GIF and TIFF; the polygon scanner; the
histograms; palette and table lookups. Their gains, where there are any,
are algorithmic and were the subject of
[PROFILE_FOLLOWUP_PLAN.md](PROFILE_FOLLOWUP_PLAN.md) and
[ROUND_2026-11-05_PLAN.md](ROUND_2026-11-05_PLAN.md).

## Order of work

1. **`gauss_taps`**: a regression of the default mode against the exact
   one wherever there is no kernel. The resize fix is the template, and
   the instruction-count gate's scalar tier will show it.
2. **ImageMath macros and `ImagingPointTransform`**: two small edits, in
   every build, with no exactness question (the same operations per
   element).
3. **The glyph blend**: the same values, proven by enumeration; a test
   that compares the two forms over all inputs belongs with it.
4. **`Filter.c`**: index type and clamp for the 1-band float loops
   (arm64), then the other seven loops (kernel-less builds). Float sums
   keep their order per pixel, so the pixels do not change; the
   consistency tests decide.
5. **`restrict` in `ImagingBoxBlurColumns`**, with a check that the
   callers never pass overlapping rows; and the decision on `-O3` in
   `setup.py`, which covers this loop and the kernels that lose at `-O2`.
6. **ImageMorph**.
7. The rest of the kernel-less list, by the table above, if those builds
   matter: `ResampleHorizontal.c`, `TransformFast.h`, `Reduce.c`.

For each: the method's rules. Remarks at `-O2` and `-O3` on both targets
before and after, the instruction count per line to confirm that the
vector copy is the one that runs, output hashes in every tier, timings on
`-O3` builds interleaved, and arm64 timings on the Jetson, which this
assessment does not have.

## Reproducing

In `bench-results/cfriendly/` on the laptop (not tracked):
`static2.sh` and `flags.sh` run in the image `pillow-simd-cross` (the
benchmark image plus `gcc-aarch64-linux-gnu`) and write one log per unit
under `out/`; `hot.py DUMPS OUT.json` after `simd_callgrind.py --set full
--size 640x400`; `xref.py hot.json` prints the hot loops with their status
per target; `exp/` holds the extracts (`g.c` the index test, `f3.c` the
filter row, `h.c` point, glyph and morphology).
