# Plan: the baseline's four findings

Status: planned and done 2026-11-04, for the "what this suggests" items
of [BASELINE_2026-11-04.md](BASELINE_2026-11-04.md), in the order of
[OPTIMIZATION_METHOD.md](OPTIMIZATION_METHOD.md). What came out of it is
at the end: P1 to P5 done, P6 measured and left.

The four findings, as the baseline stated them:

1. The conversions no faster than scalar sit at the memory copy limit.
2. The largest costs in Pillow's own code are algorithmic: the quantizer's
   allocations, drawing's per-shape allocations, the masked histogram at
   0.6 instructions per cycle.
3. Core-bound C without a kernel: the mode filter, the dithered
   conversions, `logical_and` on mode 1.
4. zlib's match search in PNG and TIFF encoding is the largest single item.

## Measurements made before deciding (2026-11-04, laptop, CPU 2)

- **Copies.** A 16 MB `memcpy` on the pinned core runs at 31 GB/s
  (read plus write); the row-by-row flip of a 2560x1600 RGB image runs at
  27 GB/s, RGB to RGBA at the same. The kernel's share (page faults) is
  under 10% of the cycles and glibc recycles the blocks, so the copies
  are at 85 to 90% of the core's copy speed. A 32 MB `memcpy`, above
  glibc's non-temporal threshold, runs at 37 GB/s: streaming stores are
  worth at most 20%, only on images larger than the last-level cache,
  and need 32-byte aligned rows. Finding 1 is confirmed and gets no
  kernel change.
- **Quantizer.** The cost is not the allocations: the hash nodes already
  come from blocks, and the 22,300 allocations per call are the
  per-colour pixel-list nodes, which are cheap. The time is in the
  colour hash table: `pixel_cmp` 31%, `hashtable_insert`/`lookup` 17%,
  `pixel_hash` 7% of the median cut. The chains are kept sorted by the
  packed hash value, and every comparison recomputes both pixels' packed
  values with the scale shifts; every lookup takes a `%` by a prime.
- **Masked histogram.** 5 instructions per pixel at 0.6 per cycle: the
  `if (mask[x])` branch on a random mask, and the dependent increments.
- **Mode filter 3x3.** 242 instructions per pixel: the column-updated
  histogram, then the nine-value scan with its data-dependent compares.
- **Dithers.** The Floyd-Steinberg loops are a serial chain through the
  error term, about 10 dependent operations per pixel, which is the 16
  cycles per pixel measured; `logical_and` on mode 1 in the benchmark is
  87% the two dithered conversions that make its inputs.
- **PNG deflate.** The compressor matters more than anything in Pillow's
  own encoder: on the PNG-filtered bytes of a 1280x800 RGBA photo-like
  image (4 MB), zlib 1.3.1 against libdeflate 1.19, best of five:

  | level | zlib | libdeflate | speed | zlib size | libdeflate size |
  |---:|---:|---:|---:|---:|---:|
  | 1 | 63 ms | 16 ms | 4.0x | 1,811 KB | 1,740 KB |
  | 3 | 104 ms | 39 ms | 2.7x | 1,768 KB | 1,720 KB |
  | 6 (default) | 459 ms | 83 ms | 5.5x | 1,725 KB | 1,587 KB |
  | 9 (optimize) | 3,504 ms | 168 ms | 21x | 1,597 KB | 1,576 KB |

  zlib-ng, which the wheels link, measured the same as zlib on this data
  in an earlier round ([PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md)).
  libdeflate is smaller and faster at every level because it compresses
  a whole buffer at once with a better match finder; it has no streaming
  API, which is why Pillow's row-at-a-time encoder cannot use it as is.

## Items

### P1. libdeflate for the PNG encoder (finding 4)

- Change: `ZipEncode.c` gathers the filtered rows into one buffer and,
  after the last row, compresses them with `libdeflate_zlib_compress`
  (a zlib stream, what IDAT holds), then hands the result out in the
  caller's buffer sizes. The row filters, their selection and the
  decoder are untouched. The zlib path stays for: builds without
  libdeflate (`HAVE_LIBDEFLATE` unset), a predefined dictionary, an
  explicit `compress_type` (zlib strategies have no libdeflate
  equivalent), and images whose filtered bytes exceed a cap (256 MB),
  since the whole-buffer path holds the filtered image and the output
  bound in memory where the stream path holds a few rows.
- Levels: `compress_level` 0 to 9 map to libdeflate's 0 to 9 (0 is
  stored, as in zlib); `optimize` uses 9 as it does with zlib (libdeflate
  goes to 12, measured as part of this item).
- Build: an optional `libdeflate` feature in `setup.py` (`libdeflate.h`
  and `-ldeflate`; `-C libdeflate=disable`), `features.check("libdeflate")`
  and its version in `pilinfo`, the external-libraries list of the
  installation docs. The wheels: `wheels-dependencies.sh` builds
  libdeflate with cmake as it builds brotli, on every platform; the
  wheel check expects the feature.
- Exactness: the pixels are unchanged by construction (the filters are
  the same bytes, the compressor is lossless); the PNG bytes differ from
  zlib's, as they already differ between zlib and zlib-ng builds. The
  test suite's PNG tests round-trip; a test checks libdeflate and zlib
  outputs decode to the same pixels at every level, and that the
  fallbacks engage (dictionary, strategy, the cap through a test hook).
- Measurement: `simd_bench.py` "encode PNG RGBA/L/P" and the perf
  repository's `codec/encode/png` case, against the zlib build; the
  file sizes alongside.

### P2. Masked histogram (finding 2)

- Change: `Histo.c`'s masked loops count `t[in[x]] += mask[x] != 0` into
  the four tables the unmasked path uses, so there is no branch and no
  dependent chain. Same counts.
- Measurement: `simd_bench.py` "histogram L masked".

### P3. The quantizer's colour hash (finding 2)

- Change, keeping the table bit-identical in structure and iteration
  order (the palette depends on the pixel lists' order): the pixel hash
  table stores each node's packed key (the value `pixel_cmp` orders by)
  so chain walks compare integers instead of calling `pixel_cmp`, which
  recomputed both packed values; the bucket index uses a precomputed
  fast modulus for the prime; `create_pixel_hash` keeps the node of the
  last pixel and increments its count directly while the next pixels
  repeat it (reset at every rehash). Nothing changes the chains' order
  or the rehash points, so the palettes and the indices are the same.
- Exactness: palette and index bytes of `quantize()` and of
  `convert("P", palette=ADAPTIVE)` compared old against new over the
  hash set's images and sizes, with `kmeans` on and off.
- Measurement: `simd_bench.py` "quantize mediancut" and "convert RGB->P
  adaptive", and the perf repository's quantize cases.

### P4. The mode filter's 3x3 window as a kernel (finding 3)

- Change: a kernel for size 3 (the default and the common case) in a
  companion `ModeFilterSIMD.c` through the table: for each run of
  pixels, the nine window bytes, the count of each against the other
  eight (81 byte compares per vector), and the choice of the highest
  count with the lowest value on a tie by packing count and inverted
  value into 16-bit lanes and taking the maximum; the original pixel
  where the count is 2 or less. The scalar loop keeps the borders and
  every other size.
- Exactness: the scalar scan's result is exactly "highest count, lowest
  value on a tie", so the kernel's pixels equal the scalar's; the hash
  set gets a mode-filter area.
- Measurement: `simd_bench.py` "mode 3 L", counters for the kernel.

### P5. The dithers (finding 3)

- Change: the Floyd-Steinberg loops of `Convert.c` (L and RGB to mode
  1, RGB to P with the web palette) rewritten for a shorter dependent
  chain with the same integer arithmetic: the division of a possibly
  negative error by 16 as the same truncating division, the clamp, and
  the propagation unchanged in value. The serial dependency sets the
  floor, so the target is about 1.5x, not more.
- Exactness: the output must be bit-identical (the hash set covers the
  dithered conversions).
- Measurement: `simd_bench.py` "convert L->1 dither", "convert RGB->1
  dither", "convert RGB->P web dither", "chops logical_and 1".

### P6. The copy-limited conversions (finding 1)

- No change: measured at 85 to 90% of `memcpy`. Recorded for the next
  reader: streaming stores are the only lever (at most 20%, above the
  last-level cache, aligned rows), and fusing a conversion into the
  operation that consumes it is an API question, not a kernel one.

Drawing's allocations (one or more per shape) are left for a later
round: they are a Python-to-C call-pattern question
(`simd/python-to-c-assessment`), not a kernel one.

## Order and rules

P2, P1, P3, P4, P5: the smallest first to validate the measurement path,
then the largest. Each item: the change in its own commit, the
measurement in `bench-results/` and in this page's "Done" section, the
kernels page updated where a kernel is added (P4), `CHANGES.SIMD.rst`,
and the hash set extended before the change where it does not cover the
operation. Builds for A/B at `-O3` through `checks/simd_dbench.sh`;
exactness through `checks/simd_hashes.py` and the per-item comparisons
above.

## Done

Measured on the laptop at `-O3` (`checks/simd_dbench.sh`, CPU 2, best of
rounds, old and new builds interleaved), `simd_bench.py --check` confirming
the outputs; raw results in `bench-results/baseline-2026-11-04/` and
`bench-results/followup/`.

- **P2, masked histogram (2026-11-04).** `histogram L masked` 12.51 to
  2.07 ms (6.0x) at 2560x1600; the counts identical over eight modes,
  fourteen widths and random L and mode-1 masks (`histcheck.py`); the
  unmasked histogram unchanged (1.00x).
- **P1, libdeflate for the PNG encoder (2026-11-04).** `simd_bench.py`
  at 2560x1600, zlib build against libdeflate build: encode PNG RGB
  1,177 to 232 ms (5.07x), RGBA 2,523 to 396 ms (6.37x), L 714 to 133 ms
  (5.37x), P 116 to 60 ms (1.94x), level 1 RGB 162 to 110 ms (1.48x);
  decoding unchanged (0.91 to 1.02x, noise). Sizes on the 1280x800
  photo-like image at the default level: RGB 1,532 against zlib's 1,498
  KB (2% larger), RGBA 1,588 against 1,678 (5% smaller), L 600 against
  597, P 596 against 596. For `optimize`, libdeflate's levels 9 to 12 on
  the filtered rows: level 10 gives level 12's size at 60% of its time
  (RGB 249 ms and 1,448 KB against zlib 9's 537 ms and 1,498 KB; RGBA 404
  against 3,691 ms, 1,492 against 1,564 KB; L 41 against 37 ms, 573
  against 632 KB; P 89 against 50 ms, 554 against 593 KB), so `optimize`
  is level 10. The test suite passes on the libdeflate build (8,365
  passed, with the new `test_file_png_libdeflate.py`); the wheels build
  libdeflate 1.26 (`wheels-dependencies.sh`, before libtiff, which picks
  it up on Linux), the SBOM lists it, the wheel check expects the
  feature. The benchmark's output check reports the encode cases as
  different, which is the bytes, not the pixels. The wheel build checked
  end to end with cibuildwheel from a clean clone (cp312 manylinux
  x86_64): libdeflate 1.26 built and bundled (`pillow_allsimd.libs/`),
  `HAVE_LIBDEFLATE` in the compile, the wheel checks (features and tiers)
  and the suite inside the wheel passed (8,224 passed, 106 skipped).
- **P3, the quantizer's colour hash (2026-11-04).** Three changes, each
  keeping the table's chains, merge points and iteration order as they
  were, so the palettes and indices are bit-identical (one hash over
  five images, four colour counts, three kmeans values, both median
  methods, adaptive and web conversions with and without dither, old
  against new: equal; `simd_bench --check` identical): the nodes keep
  their hash and the pixel table, whose comparison orders by that hash,
  walks chains on integers (`hashtable_order_by_hash`); the bucket is a
  fast modulus (Lemire's multiply-shift, the `%` where no 128-bit
  product exists); a run of equal pixels is one insertion counting them
  all. Then the per-pixel passes that looked each pixel up again: with
  an unscaled table (up to 65,536 colours) the palette's per-box sums
  come from the leaves' pixel lists, which hold each colour's count (the
  hash is injective at every scale, checked over all 16.7 million
  colours); with a scaled table, a direct map from scaled cell to box
  (8 MB at scale 1) replaces the chain walk per pixel in both the sums
  and the mapping pass. `simd_bench.py` at 2560x1600: quantize mediancut
  362 to 171 ms (2.11x), convert RGB to P adaptive 689 to 400 ms (1.72x);
  the other quantizers unchanged. What remains in the adaptive case is
  the three linked-list merge sorts (12%) and the first pass's inserts.
- **P4, the 3x3 mode filter as a kernel (2026-11-04).**
  `ModeFilterSIMD.c` (the `mode` group, SSE4.1, AVX2 and NEON from one
  source): nine loads, 36 pair compares subtracted into the nine counts,
  nine packed keys (count above, inverted value below) and their maximum,
  the centre pixel where the count is two or less. `mode 3 L` at
  2560x1600: 62.7 to 4.5 ms (13.9x) on AVX2, outputs identical. The
  scalar file keeps the borders and the other sizes through a per-pixel
  count (`mode_window_pixel`). `Tests/test_simd_consistency.py` compares
  widths 1 to 301 and heights 1 to 7 against a per-pixel definition
  under every tier; the hash set's rank area now holds the mode filter
  (sizes 3 and 5, noise and a flat-region image) and agrees across the
  old build, the new build and the scalar tier; the NEON tier unit
  compiles with zig for aarch64 (not run: the Jetson is loaded).
- **P5, the dithers (2026-11-04).** Not the arithmetic, which is already
  about ten dependent operations per pixel: the rows. `tobilevel()` and
  `topalette()`'s Floyd-Steinberg loops do two rows at a time, the
  second two pixels behind the first, so that two chains of error terms
  overlap; a pixel needs the errors of the row above at x + 1, which the
  first row has written by then, and the rows alternate between two
  error buffers. The per-pixel body is Pillow's, as a macro, so the
  pixels are the same (one hash over L and RGB noise and gradients at
  fourteen widths and eight heights, to mode 1 and to the web palette
  with and without dither, equal old against new; the palette dither's
  closing write of the blue terms into the three slots past the row's
  end is kept as it was). At -O3 on 2560x1600: convert L to 1 dither 21.3
  to 12.2 ms (1.74x), RGB to 1 dither 1.60x, RGB to P web dither 58.5 to
  36.0 ms (1.62x), the benchmark's `logical_and` on mode 1 (two
  dithered conversions and the and) 1.47x. Four rows would overlap more,
  at the cost of readability; left here.
- **arm64 (2026-11-04, later the same day).** The Jetson A/B of HEAD
  against the pre-refactor commit: the NEON and scalar builds hash every
  area identically except the PNG bytes (libdeflate), and the 111 cases
  average 1.01x with quantize mediancut 2.06x and PNG encode 3.12x
  ([NEON_ROUND_2026-11-04.md](NEON_ROUND_2026-11-04.md)).
