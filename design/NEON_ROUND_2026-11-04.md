# NEON round of 2026-11-04: assessment and improvements

Status: done 2026-11-05 (started 2026-11-04). The Jetson Orin became usable again
(66% idle under other work), after a day of being saturated: first the
arm64 assessment of the branch as it stands after
[PROFILE_FOLLOWUP_PLAN.md](PROFILE_FOLLOWUP_PLAN.md), then the NEON
improvements the assessment points at, in the order of
[OPTIMIZATION_METHOD.md](OPTIMIZATION_METHOD.md).

## Assessment

Two sources: the published benchmark data (the branch before the
kernel-table refactor, `pillow-allsimd-perf` at 0d7d352), compared
between the two machines, and the A/B on the Jetson of HEAD against the
pre-refactor commit (`bench-results/jetson-ab.run.sh b8fb92778 HEAD`:
NEON and scalar builds of both, their output hashes, then perf.py's 111
cases interleaved with OpenCV, five rounds).

**Where NEON gains least, relative to AVX2.** Each case's NEON gain over
the arm64 scalar build against its AVX2 gain over the x86 scalar build
(fast mode, the default, where it differs), from the published data:

| case | NEON over scalar | AVX2 over scalar | NEON ms | note |
|---|---:|---:|---:|---|
| filter RGB 3x3 (SHARPEN, FIND_EDGES) | 3.6 to 5.5x | 12.6 to 33x | 27 | exact only |
| gaussian RGB radius 10 | 1.25x | 3.4x | 91 | fast taps |
| unsharp RGB | 3.5x | 7.4x | 73 | fast taps + row |
| scale RGB 2048x1280 bicubic, lanczos | 1.8 to 1.9x | 6.3 to 6.6x | 29 to 38 | |
| 3D LUT RGB 16 | 2.0x | 7.6x | 52 | exact only on arm64 |
| box blur RGB 10 | 1.3x | 3.7x | 29 | |
| YCbCr to RGB, RGB to YCbCr | 3.0x, 2.4x | 7.2x, 5.0x | 12, 17 | exact 1.2x, 1.1x |
| alpha composite | 1.2x | 6.8x | 18 | known: no fast divide; 13x ahead of OpenCV |

The Gaussian on L shows 1.0x on both scalar and NEON because the scalar
build's portable taps loop already vectorizes there; it is not a NEON
weakness.

**What the code says.** Reading the NEON kernels of the first rows: the
Gaussian taps accumulate all 2r + 1 taps into one register pair, and the
integer kernel filters accumulate all 9 or 25 taps into one set of four
32-bit accumulators. A NEON multiply-accumulate into a register waits for
the previous one into the same register, four cycles on the Cortex-A78,
so one chain of n of them costs 4n cycles whatever the pipes could do:
for a 3x3 filter on 4-byte pixels, 9 chained MLAs per 4 pixels is 9
cycles per pixel, which is what the 14 cycles per pixel measured mostly
is. The resample kernels already use two chains per output pixel; the
LUT does one pixel per iteration through seven dependent interpolations
and eight table loads.

## Items

- **N1. Gaussian taps in two chains** (`GaussianFastSIMD.c`, NEON): the
  odd and the even taps into separate accumulator pairs, added before
  the shift. The partial sums fit 16 bits as the whole does (the
  weights sum to 256) and integer sums reassociate exactly: same bytes.
- **N2. Kernel filters in two chains** (`FilterSIMD_NxNi.c`, NEON):
  the second tap of each pair into a second set of accumulators, the
  odd last tap of a row alternating between the sets by row, the sets
  added before packing. Integer sums: same bytes. The x86 tiers are
  untouched (their pairs go through `pmaddwd`): the x86 hashes and the
  filter tests are unchanged.
- **N3. Kernel filters with small coefficients** (decided from the
  counters below, done): see "The multiplier pipe".
- **N4 (not done)**: the resample kernels
  and the 3D LUT, with `perf` on the Jetson (`bench-results/neon/prof/`),
  before changing anything; the YCbCr exact kernels, whose per-quarter
  arrays may go through memory (an earlier arm64 lesson).

## Measurements

**HEAD against the pre-refactor commit on the Jetson (2026-11-04, load
13 to 15 from other containers, one pinned core).** Hashes: the NEON
builds of the two commits agree on every area but the PNG bytes, which
libdeflate compresses now (the benchmark image has it), and so do the
scalar builds; NEON against scalar differs only in the two known float
kernels (DETAIL, SMOOTH_MORE). So the kernel-table refactor, the
dispatch work and the follow-up round give the same pixels on arm64.
Timing, perf.py's 111 cases, old over new: geometric mean 1.010; the
cases that moved are the ones changed, quantize mediancut 1,396 to 678 ms
(2.06x) and PNG encode 717 to 230 ms (3.12x; the file 7% smaller), and
three suspects to recheck in the next run, draw rectangles 0.78x, draw
lines 0.92x and min 3x3 0.88x, which no change touched (the load peaked
at 23 during the run). Raw: `bench-results/jetson-ab/
ab-head-vs-prerefactor.{json,txt}`, `hash-*.txt`.

**N1 and N2 against HEAD: measured, both dropped.** Same hashes on NEON
and scalar. Kernel filters with two accumulator sets: 0.87x to 0.95x
(RGB SHARPEN 27.2 to 28.6 ms, FIND_EDGES 26.7 to 30.2, L SHARPEN 6.0 to
6.8), so the chain was not the bound. The bound is the multiplier: each
tap is four widening multiply-accumulates per 16 bytes, 36 per four
pixels for a 3x3 kernel, and the Cortex-A78 runs vector multiplies on
one pipe, which is 9 cycles per pixel of the 14 measured; splitting the
sums only added the final additions and register pressure. AVX2 does a
pair of taps for 16 lanes in one `pmaddwd` on either of two ports, so
the 6x difference between the machines here is the hardware's, and the
16-bit coefficients leave no narrower multiply to use. Gaussian taps in
two chains: 1.06x at radius 1 and 0.97x to 0.99x on unsharp, within
noise; and the radius 10 and 30 cases are above the fast kernel's sigma
limit (6), so they run the box passes, which the taps never touch.
Raw: `bench-results/jetson-ab-n12/`. The lesson recorded on 2026-10-31
holds: on this core, measure a layout change, don't reason it out.

**The three suspects are real.** In that second run HEAD measured the
same as in the first: draw rectangles 11.9 ms, draw lines 4.1, min 3x3
3.5, against 9.5, 3.7 and 3.1 before the refactor (and 9.2, 3.5 and 3.0
in the published data). Diagnosis below.

## The drawing regression: a kernel call per span

A third run with a fresh pre-refactor build (`bench-results/
jetson-regress.sh`) confirmed rectangles (9.1 against 12.3 to 13.0 ms)
and lines (3.55 against 4.1 to 4.25) and cleared the min 3x3 filter,
which came out faster at HEAD that time (3.3 against 3.8): build-to-build
variation, not a regression.

The rectangle case alone, split (`rect.py`): creating the canvas costs
the same (0.6 ms); drawing the 500 rectangles is 8.8 ms before and 11.8
after. `perf` puts 79% of HEAD's time in `draw_fill32_simd` and 13% in
`hline32`, and the store loop is instruction for instruction the one
that used to sit inside `hline32`; what changed is that the refactor
(step A10) made the fill a kernel of the table, so each span loads the
table through the GOT, saves and restores registers in `hline32`, spills
the ink and calls through a pointer: 2.3 million more instructions and
7.4 million more cycles per run for 76,000 spans, far more cycles than
the instructions explain, on a loop that waits on its stores (backend
stalls 83% of the cycles at HEAD).

Variants of `Draw.c` built and timed interleaved on the Jetson
(`variants.sh`): the fill as a plain loop over fixed chunks of four
pixels, inline, and of eight.

| variant | rectangles | lines (width 5) |
|---|---:|---:|
| HEAD, kernel call per span | 11.9 to 12.0 ms | 2.70 to 2.89 ms |
| inline, 4 pixels per step | 9.06 to 9.13 | 2.50 to 2.57 |
| inline, 8 pixels per step | 9.08 to 9.18 | 2.62 to 2.64 |

So the fill is plain C in `Draw.c` again, four pixels per step, which
needs no instruction-set guard (the four stores merge into one 16-byte
store at `-O2` on any target), and the `fill32` kernel is gone from the
table. On x86-64 at `-O3` and `-O2`, the perf repository's drawing cases:
lines 1.07x and 1.08x, outline polygons 1.04x and 1.07x, the others
within 2%; outputs identical.

The rule this adds to the dispatch plan's "row or image granularity": a
kernel behind the table must do enough work per call to bury a call that
the compiler cannot see through; a span of a drawn shape does not.

**x86-64, found on the way.** The published x86 data had the wide lines
at 2.01 ms and the baseline of HEAD at 2.45. A bisect over seven commits
as single-tier AVX2 builds (`bench-results/followup/bisect-draw.sh`)
puts 0.2 ms at the same commit (A10) and shows the inline fill taking
it back (2.02 ms). The rest belongs to the dispatch build alone: there
Pillow's own files are compiled for baseline x86-64, where `floor` and
`ceil` are library calls (SSE4.1 has the instruction), and `Draw.c`'s
`ROUND_UP` and `ROUND_DOWN` call them once or twice per span of a
polygon: 28 calls to `floor` in the dispatch build's `_imaging` against
6 in the AVX2 build. They are now the truncation corrected by one
comparison (`draw_floor`, `draw_ceil`), which gives the same integers
for every value in int range on any target: 9,000 random shapes with
fractional and negative coordinates in three modes hash the same, as
does the hash set. Dispatch build, perf repository cases: lines 2.47
to 2.32 (the fill) to 2.21 ms (the rounding), outline polygons 2.77 to
2.53. What is left against the single-tier build (2.03) is inside
`polygon_generic`, which the compiler inlines and specializes
differently when the whole file is compiled for AVX2; no call differs.
`ImagingTransformFast` and `ImagingTransformAffine` still call `floor`
in the dispatch build (22 and 6 call sites, per row or per image, not
per pixel); their cases measure the same as the single-tier build's.

## The arm64 counters, and the multiplier pipe

`bench-results/jetson-profile.sh` ran `checks/simd_counters.py` on the
HEAD NEON build (61 cases, two rounds, 1.98 GHz) with `perf` symbols for
nine cases; the table is `bench-results/jetson-profile/
b-old-neon-counters.md`. Against the x86 AVX2 cycles of the baseline,
the NEON build needs 4.1 to 4.4 times the cycles on the 3x3 kernel
filters and the alpha composite, 3.6 on the 3D LUT, 2.9 to 3.3 on the
horizontal resamples and RGB to YCbCr, 2.5 to 2.7 on the box blur and
bilinear upscale, and 1.0 on the copies. Instructions per cycle are 2.3
on the filters, 2.7 on the LUT and 1.5 on the fast YCbCr kernels with no
cache misses, where the resamples run at 4.1 to 4.6: the low ones are
not waiting on memory or on the instruction count but on the one pipe
the Cortex-A78 has for vector multiplies. A 3x3 integer kernel on RGB is
36 widening multiply-accumulates per four pixels.

**N3.** Fewer multiplies, exactly: `filter_small_kernel()` divides out
the power of two a kernel's fixed-point coefficients and offset share,
and where what is left is small (coefficients up to 255, sums within 16
bits) the NEON kernel adds in 16-bit lanes, eight per operation, with
adds for coefficients of one. A model of the normalization and the
reduction over extreme and random windows had no mismatch for any
built-in or test kernel before the kernel ran. On the Jetson, HEAD
against the working tree, hashes equal for NEON and for scalar:

| case | HEAD | new | |
|---|---:|---:|---:|
| filter RGB SHARPEN | 27.2 ms | 14.7 | 1.84x |
| filter RGB EDGE_ENHANCE | 26.7 | 14.7 | 1.81x |
| filter RGB FIND_EDGES | 26.7 | 14.7 | 1.81x |
| filter L SHARPEN | 6.0 | 4.0 | 1.49x |
| filter RGB SMOOTH, SMOOTH_MORE (not reducible) | 27.1, 65.2 | same | 1.00x |
| draw rectangles | 12.1 | 9.7 | 1.25x |
| draw polygons | 64.0 | 59.4 | 1.08x |

The first version stopped one bit short of the full precision and left
FIND_EDGES with coefficients of -2 and 16 (1.34x); dividing the whole
precision out, where the rounding half can never carry, made it -1 and
8. All reducible 3x3 kernels then cost the same 14.7 ms whatever their
multiplies, so the adds' chain looked like the next bound; two chains of
adds measured 0.79 to 0.88x of one (`variants.sh chain1 chain2`), and
the kernel has one.

**The rounding in `Draw.c`, the other way round.** The integer form of
floor and ceil that removes a library call on baseline x86-64 made the
wide lines 15% slower on arm64 (2.49 against 2.88 ms), where floor is
one instruction: it is compiled only for x86 without SSE4.1. And a
length threshold before the RGBA-ink blend kernel, to spare short spans
the call, made small ellipses slower (2.71 against 2.83 and 3.01 ms):
the blend is worth its call even on a few pixels.

**Left for a later round, with what the counters say.** The horizontal
resamples run at more than four instructions per cycle: they need fewer
instructions, which for exact results means a different formulation,
not tuning. The 3D LUT (one pixel per iteration, seven interpolations)
and the fast YCbCr kernels (1.5 instructions per cycle, multiplies of
14-bit coefficients in 32-bit lanes) are bound by the same multiplier
pipe as the filters were; for YCbCr a 16-bit form would change the fast
kernels' defined result on every build, which fast mode allows but a
release note must say. The alpha composite's division stays.
