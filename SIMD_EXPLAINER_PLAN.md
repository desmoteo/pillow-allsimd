# Plan: a detailed explanation of the Pillow-SIMD optimizations (x86-64 and arm64)

Status: proposal. The reader and the location below are proposed, to be
confirmed before writing.

## What it is for

A long technical document that explains, operation by operation, how each
SIMD optimization works on x86-64 (SSE4, AVX2, AVX-VNNI) and on arm64 (NEON),
why the two architectures sometimes take different routes, and how each result
is kept bit-exact with the portable C code. It should let a reviewer of the
upstream PRs, or a new contributor, understand a kernel without reverse
engineering the intrinsics, and let a reader of the benchmark page see where
each speed-up comes from.

Proposed reader: a C programmer who knows what a SIMD register is but not the
Pillow internals. Code excerpts are short (5 to 20 lines) and always point to
the file and function.

Proposed location: `SIMD_OPTIMIZATIONS.md` at the repo root, next to
`X86_SIMD_ASSESSMENT.md` and `ARM64_SIMD_ASSESSMENT.md` (so it is versioned with
the code it describes), plus a "How it works" page on the perf site that links
to each section from the corresponding benchmark suite.

## Template for every section

1. **The operation and the scalar baseline**: what Pillow computes, the exact
   arithmetic (fixed-point format, rounding), and where the scalar time goes.
2. **The idea**: the one or two tricks that make it fast, independent of ISA.
3. **x86-64**: SSE4 kernel, what AVX2 adds (wider lanes, cross-lane fixes),
   AVX-VNNI where used; key intrinsics and data layout in registers.
4. **arm64**: the NEON kernel and how it differs, or why the portable C code is
   used instead (measured on the Cortex-A78AE with GCC 12/14).
5. **Exactness**: why the output equals the scalar build bit for bit, and how
   that is checked (`checks/simd_hashes.py` area, exhaustive checks, tests).
6. **Numbers**: before/after on both machines from the perf page (2560x1600),
   against Pillow 13 and against OpenCV/libvips where they compete.
7. **Code map and pitfalls**: files, functions, and the traps found on the way.

## Sections

### 0. Ground rules that shaped every kernel

- The exactness contract: every SIMD build produces the same pixels as the
  scalar build. Tooling: `checks/simd_hashes.py` (per-area MD5 of 67x23 and
  full-size outputs), `hsvcheck.py` (all 2^24 RGB values), `filtcheck.py`,
  the test suite on scalar/SSE4/AVX2/AVX-VNNI and on NEON/scalar on the Jetson.
  The documented exceptions and their causes: alpha composite (x86), the NEON
  float filters DETAIL and SMOOTH_MORE (FMA contraction), RGB→L ±1 on NEON.
- Build model: compile-time `-C simd=sse4|avx2|avxvnni|neon|native|disable`,
  no runtime dispatch, and why (README "Why not contribute to Pillow").
  `ImagingSIMD.h` as the shared helper layer.
- Compiler realities that decided many designs:
  - Python's `-fno-strict-overflow` stops GCC vectorizing `in + x*4` with int
    indices (the `CONVERT_PIXELS` chunked-loop pattern).
  - The -O2 "very cheap" cost model vs -O3: Debian/Ubuntu/conda Pythons are -O2,
    Docker python images -O3; explicit SIMD or fixed-trip chunks for -O2.
  - FMA contraction on arm64 changes float results; `vfmaq_f64` where scalar
    double sums must be matched.
  - When portable C beats hand-written NEON on the A78AE, and the reverse.
- Pillow's memory layout: RGB stored as 4 bytes per pixel, 16 MB blocks,
  transparent huge pages via `madvise` (and why only in madvise mode).
- Cost differences between the two ISAs that recur: pmaddwd vs vmlal, cheap
  32-bit `MUL.4S` and `vdivq_f32` on arm64, pshufb vs tbl, AVX2 lane crossing.

### 1. Resampling (resize): the original Pillow-SIMD kernel

- Separable convolution, coefficients in fixed point (`PRECISION_BITS`), the
  horizontal and vertical passes and their intermediate image.
- x86 horizontal kernel: pmaddwd over coefficient pairs, shuffles that
  interleave the channels of neighbouring pixels, AVX2 two pixels per lane;
  vertical kernel: madd over row pairs, 8 then 2 then 1 pixels per step.
- AVX-VNNI `vpdpwssd`, and why long single-accumulator chains made it 1.55x
  slower than AVX2 on extreme downscales (fixed in b87e97786).
- 1-byte (L) images: four rows at a time; NEON versions (vmlal_s16, vqshrun).
- F, I and I;16 images in vectors of doubles (exact with scalar double sums).
- Reduce (integer box downscale): per-factor kernels, NEON up to factor 512.
- Pass order: why vertical-first was measured and not adopted (it would make
  the result build-dependent).

### 2. Box blur, Gaussian blur and unsharp mask

- The extended box blur: running sums, fractional fringe weights, the exact
  24-bit form `(c*ww + e*fw + 2^23) >> 24`, Gaussian = 3 box passes.
- x86: 12-bit weight split so pmaddwd computes the 24-bit products exactly;
  NEON: 32-bit `vmulq/vmlaq` plus `vaddhn` (and why the split was slower there).
- Direct passes for small radii (no running sum, vectorized across pixels);
  integer radii 1 to 4 in 16 bits with `mulhi` and per-radius magic numbers
  verified at start-up over every possible sum.
- The vertical cascade: all passes in one sweep with rings of 2r+3 rows.
- What is still two sweeps (horizontal, then vertical) and what fusing them
  would save (see section 11).
- Unsharp mask post-pass (SSE4, NEON).

### 3. Kernel filters (ImageFilter.Kernel, the built-in 3x3 and 5x5 filters)

- Integer kernels at Q13, and the fall-back to the highest exact lower
  precision when a coefficient does not fit (why FIND_EDGES/EDGE_ENHANCE now
  take the integer path and match scalar).
- Float kernels, FMA and the known NEON differences.
- 4-byte vs 1-byte images, AVX2 and AVX-VNNI; NEON constant-shift dispatch
  (`FILTER_NxN`) so the narrowing shift fuses into `sqshrn`.

### 4. Rank filters (median, min, max, mode)

- Sorting networks up to 11x11 for 1-byte images; median 3x3 and 5x5 in
  registers; all bands in one pass for 4-byte images; why it is not at -O2
  without explicit loads (RANK_LOAD macros).
- Min/max: separable with a ring of row results, any size.
- Mode filter: histogram updated by columns.

### 5. Colour conversions

- Whole-pixel conversions (L→RGB, RGB↔RGBA, CMYK) and the chunked pattern;
  NEON `vst4q`/`vld3q` loops and when they beat whole-pixel ORs.
- RGB→L/LA/I (fixed-point weights), premultiply and unpremultiply (exact
  division: reciprocal plus correction on x86, `vdivq` on NEON).
- RGB→HSV: float semantics kept exactly, the exact `fmod` replacement,
  SSE4/AVX2/NEON, exhaustive 2^24 check; why OpenCV's 12-bit reciprocal
  tables are not an option.
- RGB↔YCbCr: Pillow's INT16 tables reproduced exactly without lookups
  (per-term rounding), AVX2 and 16 pixels per step on NEON.

### 6. Compositing and channel operations

- Alpha composite (exact division by 255), blend, paste with masks.
- ImageChops multiply/screen without a divide; saturating add/subtract.
- `point()` tables recognised as byte operations (invert, posterize,
  solarize, XOR/AND); ImageEnhance as tables.
- 3D colour lookup tables (trilinear in fixed point); split, merge, get_channel.

### 7. Geometry

- Transposes and 90/270 rotations in register blocks (byte and dword
  transposes with unpack/zip); flips and 180° with vector row reversal.
- Affine, perspective and quad transforms: tiles for cache locality,
  bilinear/bicubic per pixel in doubles (SSE4/AVX2), perspective coordinates
  for a tile row in one vector loop; why NEON transform filters were removed.

### 8. Drawing and text

- Span fill and RGBA ink blend; masked fills for text and `bitmap()`.
- Polygon scan lines: intersections sorted by insertion sort.
- Text: glyph cache and glyph-advance cache (3.3x x86, 4.2x arm64) — not SIMD
  but where the time was.

### 9. Codecs and packing

- PNG row filters (Sub/Up/Avg/Paeth) in SIMD; RGB pack/unpack and raw codec
  byte-order shuffles; `tobytes` in one chunk.
- Quantize: vector nearest-palette search and a colour cache; hash-table nodes
  from blocks.

### 10. What was tried and dropped, per architecture

A table: idea, why it looked promising, what was measured, why it was dropped
(HSV reciprocal variants, 8-byte-store L table, NEON split-weight blur, NEON
integer unpremultiply, NEON transform filters, vpdpwssd chains, resize pass
order, THP always). Reviewers ask about these first.

### 11. Where OpenCV and libvips are still faster, and why

From the source study of OpenCV 5.0.0 (with IPP on x86 and KleidiCV/Carotene
on arm64) and libvips 8.16, summarised from
[PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md) and
[OTHER_LIBRARIES_KERNELS.md](OTHER_LIBRARIES_KERNELS.md): layout (3 vs 4
bytes), fused pipelines, lower precision, different algorithms, vendor
kernels. For each gap: structural (cannot close without changing Pillow's
results) or closable (and the plan).

### 12. Exact and fast modes

If the opt-in fast mode of
[APPROXIMATE_KERNELS_PLAN.md](APPROXIMATE_KERNELS_PLAN.md) is built: the
switch and its guarantees, and for each fast kernel a short summary that
links to its full description in `SIMD_FAST_KERNELS.md` (algorithm, where it
rounds, error against the exact result and the float reference).

### Appendices

- A. Per-operation speed-up tables for both machines (from the perf page).
- B. File map: which kernel lives where, ISA by ISA.
- C. Verification and benchmarking tools and how to run them.

## How it would be written

- Numbers from the published runs on both machines (28b55224b, 2026-10-31,
  on the pillow-simd-perf page), refreshed when a section's code changes.
- Sections 1, 2 and 5 first (largest, most asked about), then 3, 4, 6, 7,
  then the rest. Each section is checked against the code it describes.
- Diagrams only where they explain data movement: register layout of the
  resize horizontal kernel, the blur ring cascade, the transpose blocks
  (Mermaid or small ASCII).
- Estimated length: 25 to 35 pages of Markdown.
