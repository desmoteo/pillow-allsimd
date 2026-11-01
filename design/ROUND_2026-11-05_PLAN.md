# Plan: the round of 2026-11-05 (eight items)

Status: planned 2026-11-05, after
[NEON_ROUND_2026-11-04.md](NEON_ROUND_2026-11-04.md); the user chose all
eight of the candidates listed at its end. The order below is the order
of work; each item is measured, verified pixel for pixel and committed on
its own, as [OPTIMIZATION_METHOD.md](OPTIMIZATION_METHOD.md) has it, and
what came out of each is at the end of this page.

## Why this order

Three regressions of the last week were found late and by accident: the
span fill as a table kernel, the library calls to `floor`, and the
octree quantizer's gap. All three are costs of what the x86-64 wheels
ship (Pillow's own files compiled for baseline x86-64, a call through
the table) or of work per call, and neither the benchmark page, which
measures single-instruction-set builds, nor CI, which measures nothing,
could show them. So the measuring comes first (1), then the two items
with the largest gain for the most users (2, 3), then the rest.

## Items

### 1. The wheel build on the benchmark page, and an instruction-count gate in CI

- **Benchmark.** `run.sh` gains a `branch-wheel` variant on x86-64: the
  dispatch build (`-C simd=dispatch`), in fast mode, the default, which
  is what `pip install pillow-allsimd` gives. The page labels it
  "Pillow-AllSIMD 13 wheel" in the fast family. A first run on the laptop
  gives the list of cases where the wheel is slower than the single-tier
  build, which is item 3's input.
- **CI gate.** `checks/simd_ir_gate.py`: a fixed list of operations, one
  or more per kernel group and per-call path (drawing spans, quantizers,
  PNG), each run under callgrind for the tiers of the dispatch build
  (`PILLOW_ALLSIMD_ISA`), counting the instructions executed in Pillow's
  own code only (`checks/simd_callgrind.py` already separates them from
  libc, Python and the codec libraries). The counts are compared with a
  committed baseline, `checks/simd_ir_baseline.json`, and a case more
  than 3% above it fails the job; `--update` rewrites the baseline, and
  the commit that does so says why. Instructions are not time and a
  vector instruction counts as one, so the gate catches what it can:
  a kernel that fell back to scalar, a call per pixel or per span, a
  library call in a loop. A workflow job on x86-64 Linux runs it.
- Risk: instruction counts move with the compiler; the baseline is tied
  to the runner image's GCC, and the job prints the differences so that a
  compiler update is a one-commit refresh.

### 2. PNG decoding with libdeflate

- Evidence: the benchmark's PNG (2560x1600 RGB, 7.3 MB of IDAT) inflates
  in 77 ms with zlib and 28 ms with libdeflate, of a 98 ms decode.
- Change: libdeflate needs the whole compressed stream. The PNG plugin
  learns the total size of the consecutive IDAT chunks by walking the
  chunk headers when the file is seekable, and passes it to the decoder;
  `ZipDecode.c` then gathers the compressed bytes and, once it has them
  all, decompresses the image in one piece and runs its existing loop
  (unfilter, unpack, interlace) over the rows. Everything else stays on
  zlib's stream: a stream that is not seekable, `LOAD_TRUNCATED_IMAGES`
  (which must keep giving the rows that arrived), APNG frames and
  `fdAT`, an image whose data exceeds the cap of item P1 (256 MB), and
  any failure of the one-piece decompression, in which case the gathered
  bytes go through the zlib loop, so that corrupt files give the errors
  and the partial images they give today.
- Exactness: lossless both ways, the same rows; the test suite's PNG
  files, the broken ones included, decode to the same pixels or raise the
  same errors with the path on and off (`PILLOW_LIBDEFLATE_MAX=0`).
- Measurement: perf repository `codec/decode/png`, `simd_bench.py`
  "decode PNG *".

### 3. Hot scalar code per tier in the x86-64 wheels

- Evidence so far: the octree quantizer, 25 ms in the dispatch build
  against 15.6 ms in an AVX2 build, 40% more instructions (shifts by
  amounts read from the cube, which AVX2 vectorizes); the rest from
  item 1's list.
- Change: plain C loops that gain from the tier's flags move into the
  companion of their area and are called through the table, once per
  image or per row, never per pixel or span (the rule of the drawing
  regression). No intrinsics: the same source compiled per tier, so the
  pixels are the same.
- Not done this way: code where a tier's flags change nothing, and code
  called per span (the polygon scanner: item 8 instead).

### 4. The quantizers' per-call arrays

- Evidence: two arrays of four bytes per pixel, zero-allocated on every
  call; with the huge-page advice a fast-octree call is 19.6 ms when
  they are mapped afresh and 15.6 ms when recycled.
- Change: the index array as one byte per pixel (the palette has at
  most 256 entries), written straight into the output image where the
  rows allow; the pixel array without zeroing where every byte is
  written. The merge sorts of the median cut (12% of the adaptive
  conversion) only if an equivalent that keeps the lists' order on ties
  is cheap to show equal.
- Exactness: `bench-results/followup/quantcheck.py`, all three methods.

### 5. arm64 kernels on the multiplier pipe

- **3D LUT** (52 ms, 2.0x over scalar where AVX2 has 7.6x): seven 32-bit
  multiplies per pixel, one pixel per iteration. Where the table's
  values fit 15 bits (Pillow's own range), the interpolation
  `a + (((b - a) * s) >> 15)` is `sqdmulh` in 16-bit lanes, two pixels
  per vector: half the multiplies, the same integers; other tables keep
  the 32-bit kernel.
- **Box blur** (1.3x over scalar against 3.7x on x86): read first.
- **Fast-mode YCbCr** (2.4x and 3.0x against 5.0x and 7.2x): the
  products need 32-bit lanes at the present definition. The form OpenCV
  uses (Y, then Cb and Cr from B - Y and R - Y: five multiplies instead
  of nine) would change the fast kernels' result on every build, which
  fast mode permits and a release note must state. Decided after the
  first two: only if its accuracy against the float reference is no
  worse than the present kernels'.

### 6. arm64 resampling

- Evidence: 17 to 64 ms, the largest NEON times, at 4.1 to 4.6
  instructions per cycle and about 43 instructions per output pixel in
  the horizontal pass, where the taps themselves need about 12.
- Change to try: the horizontal kernel with the coefficients of an
  output pixel loaded once for four rows and fixed tap counts for the
  common kernels, as the x86 kernel is built. An earlier four-row kernel
  measured no gain (2026-10-31), so this starts from a per-instruction
  profile (`hist.sh`), and is dropped if the A/B says so.

### 7. Loop alignment on the Cortex-A78

- Evidence: min 3x3, ellipses and nearest rotation move 10 to 20%
  between builds with no change in their code.
- Change to try: `-falign-loops` and `-falign-functions` for the tier
  unit and the files of those loops on arm64, measured as the spread of
  those cases over builds that differ only in unrelated code.

### 8. Drawing

- Evidence: the polygon scanner is 62% of wide lines; it calls `roundf`,
  a library call on every x86 build (SSE4.1 has no round-half-away),
  two to four times per span, allocates per shape, and the profile
  counts one to two thousand allocations per drawing operation.
- Change: `roundf` as an exact inline form where it is a call, the
  scanner's per-shape arrays on the stack for small shapes, and whatever
  the per-instruction profile shows next.
- Exactness: the random-shape comparison (9,000 shapes, three modes)
  and the hash set.

## Rules

- A/B at `-O3` in the benchmark image, old and new interleaved, with the
  output check; on the Jetson through the variant runner, one run at a
  time; Pillow 13 first in any A/B of the quantize cases.
- A change that does not measure faster is reverted and recorded.
- The benchmark and the documentation tables are regenerated once, at
  the end; publishing them (the perf repository and gh-pages) is asked
  for then.

## Done

### 1 (first half) and 3. The wheel against a single-tier build, and the loops that differed (2026-11-05)

- The 111 cases of the benchmark page, wheel (`-C simd=dispatch`) against
  the AVX-VNNI build, five rounds: geometric mean 1.011; four cases apart
  by more than 5%: mask composite 1.85x, fast octree 1.59x, lines 1.09x,
  polygon outlines 1.06x (`bench-results/followup/x/wheel-vs-single.txt`).
- The 216 operations of `checks/simd_bench.py --set full`: geometric mean
  1.060, 33 operations slower by more than 5%, the worst soft light,
  hard light and overlay (3.3 to 3.4x), the pastes through a mask (1.7
  to 2.1x), I to L, the colour matrix, rank 7, `ImageEnhance.Color` and
  `Sharpness` (1.6 to 1.9x) and the octree (1.63x)
  (`wheel-vs-single-full.txt`). The page's cases alone would have
  missed most of them.
- Pasting through a mask got vector kernels (`paste_mask_L_simd`,
  `paste_mask_RGBA_simd`: the masked fill's blend with a source row),
  which beat the compiler's own vectors too: 6.3 to 3.0 ms in the AVX2
  build, 11.7 to 3.0 ms in the wheel.
- The others are plain C in `PlainLoops.h`, compiled by the owning file
  for the scalar table and by `SIMDKernels.c` for each tier, called once
  per row through the table: the two blend-mode loops, the two float
  blends, the two matrix rows, L to I and I to L, the rank network. The
  wheel is level with the single-tier build on all of them after the
  change (soft light 24.5 to 7.0 ms, rank 7 53.6 to 31.1 ms, and so on),
  and the single-tier build is unchanged.
- Exactness: `bench-results/followup/pastecheck.py` and `plaincheck.py`
  give one hash in the build before, and in the SSE4, AVX2 and AVX-VNNI
  tiers after; the scalar tier gives its own hash before and after (its
  `convert("L")` and smoothing kernel differ by design, and
  `ImageEnhance` uses both). `Tests/test_simd_tiers.py` has every one of
  these operations; `Tests/test_simd_consistency.py` checks the mask
  paste against the formula.
- Left for item 4: the octree's two loops, rewritten there. Left for
  item 8: lines and polygons. Below 1.25x and not touched: `getextrema`,
  `point` on I, flood fill, `ImageMath`, L to I;16.

### 2. PNG decoding with libdeflate (2026-11-05)

Done as planned, with two changes found by measuring.

- `PngImageFile._idat_stream_size()` walks the `IDAT` headers (a seek and
  an 8-byte read per chunk, the position restored) and the length goes to
  the decoder as a second configuration value; `ZipDecode.c` is now a
  row step (`zip_decode_row`) shared by the stream loop
  (`zip_decode_stream`) and the one-piece path (`zip_decode_whole`).
- The first version copied each row out of the decompressed buffer and
  measured 68.7 ms against 97.0 (1.41x), where the inflate times alone
  (67.1 ms by rows with zlib, 28.5 with libdeflate, measured apart with
  the same data) promised about 58. Timing the phases showed where the
  rest was: 5.7 ms gathering, 32.3 ms inflating, 29.2 ms in the rows.
  Unfiltering in place, without the copy, gave 1 ms; the huge-page
  advice on the two buffers, which the allocator maps afresh on every
  decode, gave 6 ms. Final: **62.4 ms, 1.56x** on the benchmark's photo;
  RGB 1.60x, RGBA 1.67x, L 1.40x, P 4.0x on a noisier image, 1.6x at
  512x512, 1.8x at 128x128, 1.02x at 16x16.
- What is left is the unfiltering, 29 ms of scalar Sub, Average and
  Paeth; the vector forms were measured slower in round 2 and are not
  retried.
- Checked: the suite (8,510 passed); `bench-results/followup/pngcheck.py`,
  15,666 cases (every PNG of the test suite whole, cut and with flipped
  bits, from a path and from memory, with and without
  `LOAD_TRUNCATED_IMAGES`): the same pixels or error text with the path
  on, off (`PILLOW_LIBDEFLATE_MAX=0`) and in the build of 2026-10-30;
  Memcheck clean on the 10,518 cases of the files under 3 kB.
- Not covered by a test: that the fast path is the one taken, since it
  is unobservable by design. The instruction-count gate of item 1 has a
  PNG decode case for that.

### 4. The fast octree without its arrays (2026-11-05)

The item as planned (bytes for the index array, no zeroing) would have
gained little: removing both arrays altogether gave 15.6 to 15.3 ms.
Timing the phases showed why, and what to do instead.

- `quantize_octree_rows()` reads an RGB or RGBA image's rows and writes
  the P image's rows; the palette computation between the passes is
  `octree_build()`, shared with the array interface, which other modes
  still use. Phases after this step: counting 11.1 ms, palette 0.4 ms,
  mapping 4.0 ms. The arrays were never the cost when the allocator
  recycled them; they are now never a cost.
- Counting, RGB: four 32-bit sums per bucket (r, g, b, count) in a 64 KB
  table, one vector addition per pixel, moved to the 64-bit buckets
  every 2^24 pixels. Mapping: a 16 KB table of the lookup cube's
  indices. 10.0 ms (counting 6.0, mapping 3.6).
- The RGB cube's four bits per channel as constants: 7.9 ms. The shifts
  by variable counts were the larger part of what was left: `shl %cl` is
  three micro-operations without BMI2, six of them per pixel.
- Tried and dropped: two tables of sums for alternate pixels (the same
  10.0 ms: the pass was not waiting on store forwarding).
- Result on the benchmark's case: 25.0 ms Pillow 13, 15.6 ms the AVX2
  build and 25.1 ms the wheel before, **7.9 ms** in both now; the scalar
  tier 9.5 ms and SSE4 8.4 ms.
- Exactness: `quantcheck.py` (extended with RGBA images with no, one,
  many and only transparent pixels, RGB with other fourth bytes, one-row
  and one-pixel images) gives one hash before and after and in every
  tier; `Tests/test_image_quantize_octree.py` compares the rows path
  with the array path on the same pixels, holds digests that upstream
  Pillow 13.0.0 reproduces, and has an image whose red sum passes 2^32
  in one bucket. Memcheck clean on the octree tests.
- Not done: the median cut's two arrays. They are 2 x 16 MB of a 355 ms
  call, and its interface, shared with libimagequant, wants them.

### 1 (second half). The instruction-count gate (2026-11-05)

- `checks/simd_ir_gate.py`, `checks/simd_ir_baseline.json` and the
  workflow `.github/workflows/simd-instructions.yml`, as planned, with
  these differences. The list is 59 operations, not one per kernel:
  three whose own count is some ten thousand instructions (JPEG and WebP
  decoding, `getbbox`) were dropped, since 3% of that is an argument
  check. The count of Pillow's own code comes from the callgrind dump
  itself, per object: `callgrind_annotate` names no object for code
  inlined from a header, and the first version of the gate counted 0.06
  million instructions for a bicubic rotation that executes 290 million.
  The tiers are scalar, SSE4.1 and AVX2: valgrind has no AVX-VNNI.
- The job runs in `python:3.12-trixie` (GCC 14.2, Valgrind 3.24), where
  the baseline was made; two runs in fresh containers gave 186 of 186
  counts within 3%, and the whole job takes 1 min 40 s on ten cores.
- What it would have caught: the octree in the wheel (the scalar tier's
  count was the SIMD tiers' before item 4), a PNG that left libdeflate's
  path (the whole count of the decode), the span fill through the table.
- The baseline is refreshed by the last commit of this round, after the
  items that change counts on purpose.

### 8. Drawing: the evidence was wrong, and two changes that did not survive (2026-11-05)

Profiled again before changing anything (callgrind by line, then cycles,
instructions and branch misses per run with `perf_event_open`,
`bench-results/followup/drawcyc.py`: 2,000 lines, 400 rectangles, 60
polygons on 2560x1600).

- `roundf` is called where two edges meet at a corner, a few times per
  polygon, not per span; the per-shape `calloc`s are 0.3% of the
  instructions of wide lines. Neither is worth touching. The plan's item
  came from reading the source, not from a profile.
- A wide line costs about 215 instructions per row, spread: the scan over
  the four edges 75, the insertion sort of two values 32, the call to the
  span function with its two roundings 28, the span function with its
  clipping and a fill of five or six pixels 58. At 5.3 instructions per
  cycle it is instruction-bound, with no single place to gain.
- The wheel runs 13% more instructions than the AVX-VNNI build on wide
  lines (255 against 225 million; 48.7 against 42.4 million cycles) and
  none more on one-pixel lines: baseline encoding over the whole scanner,
  not a missing kernel. The 9% that the first comparison showed on
  `draw/lines` is this.
- Tried, measured, reverted:
  - a two-value case in `sort_intersections()`: 6% fewer instructions
    and **twice the time** (85 against 43 million cycles, 2.5
    instructions per cycle). GCC loads the two floats with one 8-byte
    load right after they were stored one by one, and a load wider than
    the stores it overlaps is not forwarded. A comment in the function
    now says so.
  - rounding the intersections in `float` instead of through `double`:
    the same cycles within 1% in both builds.
  - `ImagingFill()` as one row and copies of it: `Image.new()` with a
    colour takes 0.63 ms for 2560x1600 in every build either way; it is
    bound by the memory.
- Not done, a candidate: the scanner compiled per tier and called once
  per polygon, which is enough work per call. It would give the wheel
  the 13% on wide lines and move about 150 lines of `Draw.c` into a
  header; polygons, bound by their fills and branch misses (1.6
  instructions per cycle), would gain less.
- The same pixels throughout: the random-shape hash (9,000 shapes,
  three modes) is the same in all builds measured.

### 5. arm64 kernels on the multiplier pipe (2026-11-05)

The Jetson was at a load average of 15 under other people's containers,
so everything here is user-mode cycles per run, read with
`perf_event_open` (`bench-results/arm-round/armcyc.py`), which repeat to
0.1% there; wall-clock times follow them (2 GHz).

- **This round on arm64 first** (`jetson-ab.run.sh 832065f37 HEAD`):
  NEON and scalar hashes identical before and after; fast octree 42.4 to
  13.4 ms (3.2x), mask composite 11.1 to 6.2 ms (1.8x), PNG decoding
  100.9 to 79.3 ms (1.27x); 111 cases 1.02x.
- **3D LUT:** `neon_lut_pair()`, two pixels in 16-bit lanes with
  `sqdmulh`, where the whole table is within -16384..16383 (one pass
  over the table per call, checked with min and max): 103.4 to 63.5
  million cycles (1.63x), instructions per cycle 2.7 to 3.3; 4-channel
  tables 100.0 to 63.2. The same hash as HEAD and as the scalar build
  over table sizes, channels, ranges on both sides of the limit and odd
  and even widths (`lutcheck.py`). Tried on top and dropped: tables of
  each channel's offset and weight (62.2 against 63.5 million cycles,
  with more instructions).
- **Box blur:** the far weight `fw` is 0 for the radii whose box leaves
  0 or 1 of 2^24, the benchmark's 10 among them, and its product was
  half of the multiplies: skipped in both running sums, 57.1 to 42.3
  million cycles at radius 10 (1.35x), and on x86 7.5 to 5.5 ms (AVX2),
  13.8 to 9.8 ms (SSE4.1). Taking `add - subtract` first, so that the
  sum's chain is one addition, changed nothing by itself (57.17 against
  57.21 at radius 5): the pass waits for the multiplier, not the chain.
  Radii with a far weight are unchanged; a 16-bit product for it, two
  pixels a vector, would save a quarter of their multiplies and was not
  tried.
- **Fast YCbCr: not done, and not evaluated.** The conversion is 16.0
  and 9.3 million cycles (8 and 4.7 ms) at 1.5 instructions per cycle.
  The five-multiply form would change the fast kernels' output on every
  build; its accuracy against the float reference, the condition this
  plan set, was not measured. A candidate still.

### 6. arm64 resampling (2026-11-05)

Started from the disassembly of the horizontal kernel, as planned, and it
was not the four-row structure that cost.

- **Horizontal** (`neon_resample_horiz_px`, 72% of a 1/4 bicubic): with
  an `int` index GCC computed four sign-extended addresses per step and
  loaded each 16-bit coefficient by itself: 17 instructions for four
  taps. With pointers that advance and one load of the four
  coefficients, multiplied by lane (`smlal` by element): 10. Instructions
  of the whole resize 108.6 to 80.0 million, cycles 27.0 to 25.3: the
  loop went from instruction-bound (4.0 a cycle) to the multiplier (3.2),
  one `smlal` per tap and pixel on the A78's one pipe, which is where it
  stays without another definition of the arithmetic.
- **Vertical, fast mode** (`resample_fast_vrow_simd`, a third of an
  enlargement): the loop over the taps loaded a row pointer and a weight
  for every tap of every 16 bytes. For 2 to 16 taps the loop is now
  compiled with the count as a constant, pointers and weights in
  registers: 11 instructions per tap and step to 6. Lanczos x1.5: 116.8
  to 104.3 million cycles.
- Together, cycles on 2560x1600 (HEAD to now): RGB 1/4 bicubic 26.97 to
  24.43 (1.10x), RGBA 34.18 to 31.74 (1.08x), RGB 1/2 bilinear 22.73 to
  21.09 (1.08x), RGB x1.5 Lanczos 120.47 to 105.22 (1.15x). L is another
  kernel and unchanged.
- The same pixels: `resizecheck.py` (nine modes, five filters, sizes,
  boxes) gives one hash for HEAD, the change and the scalar build, in
  fast and in exact mode.
- The earlier four-row kernel that "measured no gain" shared the
  coefficients between rows; the coefficients' loads were a small part
  of what this found.

