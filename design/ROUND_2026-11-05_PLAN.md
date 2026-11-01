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
