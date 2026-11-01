# High-performance kernels in OpenCV, KleidiCV, Carotene and libvips

How the fastest image libraries implement the operations where they beat
Pillow-AllSIMD, including the approximate kernels, measured for speed and for
accuracy on the same inputs. It is the groundwork for
[APPROXIMATE_KERNELS_PLAN.md](APPROXIMATE_KERNELS_PLAN.md) (an opt-in fast
mode beside the bit-exact kernels) and extends
[PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md), which explains where the time goes
in the benchmark cases.

## Sources and method

- **OpenCV 5.0.0** (tag 40738fb), the `opencv-python-headless` wheel in the
  benchmark image. The x86-64 wheel links Intel IPP 2026.0 (ICV) as a HAL;
  the arm64 wheel links two HALs, **Carotene** (NVIDIA's Tegra NEON library,
  in OpenCV's tree) and **KleidiCV 26.03** (Arm, the archive OpenCV 5.0.0
  pins, MD5 b85a745b...).
- **libvips 8.16.1**, the Debian trixie package in the image, with its
  Highway (SIMD) paths.
- Measurements on the two benchmark machines: Intel Core Ultra 9 285H
  (performance profile, turbo off, one core) and NVIDIA Jetson Orin
  (Cortex-A78AE, one core), with the scripts in pillow-simd-perf's
  `analysis/` (raw outputs in `analysis/results-2026-10-31/`). Pillow-AllSIMD is
  the AVX2 and NEON build of `simd/pillow-13` at 28b55224b.
- Accuracy is measured against float64 references of the same mathematical
  operation, in 8-bit levels against the reference rounded to the nearest
  integer, over the image interior: the maximum, the mean absolute error and
  the share of values that differ. A "reference" is the textbook operation
  (a true Gaussian of radius 4σ, the exact box mean, JPEG's BT.601 YCbCr,
  colorsys HSV, float bilinear sampling), not any library's output.

## What runs where

Which code a call reaches depends on the build, the HALs and, in OpenCV 5,
an *algorithm hint*:

- **OpenCV `AlgorithmHint`** (new in 4.10): `ALGO_HINT_ACCURATE` uses the
  portable, bit-exact implementation; `ALGO_HINT_APPROX` "allows alternative
  approximations to get faster implementation; behaviour and result depend
  on the platform". Functions take it as a last argument
  (`GaussianBlur`, `warpAffine`, `warpPerspective`, `remap`, `cvtColor`); the
  default comes from a build option and is **ACCURATE in both wheels**
  (`cv2.getDefaultAlgorithmHint()` returns 1). The benchmark therefore runs
  OpenCV's accurate paths.
- **IPP on x86**: in 5.0.0 OpenCV no longer calls IPP for box filter,
  median, morphology or Gaussian (the calls are commented out or behind a
  build option), nor for HSV, YCrCb or gray conversions (`IPP_DISABLE_*`
  for accuracy). Measured with `cv2.ipp.setUseIPP(False)`, IPP only changes
  flip left-right and rotate 180 (1.0 ms with IPP, 4.4 and 5.3 ms without)
  and the RGBA bicubic upscale (32.9 ms with, 100.3 ms without).
- **HALs on arm64**: a KleidiCV hook, when it accepts the arguments, runs
  before Carotene's, which runs before OpenCV's own NEON code. In the
  benchmark cases: KleidiCV runs median, add/absdiff/multiply and gray→RGB;
  Carotene runs erode/dilate 3x3 (KleidiCV declines kernels
  under 5x5 by default), max, bitwise not, addWeighted, RGB↔RGBA, RGB→gray,
  RGB→HSV, RGB→YCrCb, the RGBA bilinear resize and the area resize by 2 and 4.
  KleidiCV's Gaussian is reached only with `ALGO_HINT_APPROX`, and only for
  3 to 9, 15 and 21 taps unless the border is replicated. Everything else
  (box filter, filter2D, warps, RGB bilinear resize, YCrCb→RGB, LUT) is
  OpenCV's own code.

## 1. Separable filters and the row pipeline

### The pipeline

The largest structural difference is how the passes of a separable filter
meet. Pillow-AllSIMD writes the horizontal result for the whole image, then
reads it back for the vertical passes (one temporary image of 16 MB at
2560x1600 RGB, and 800 to 1300 page faults per call for it). The other three
never materialise it:

- **OpenCV FilterEngine** (box, morphology, filter2D, sepFilter2D) and its
  Gaussian (`fixedSmoothInvoker`) run the row filter on each source row into
  a ring of `ksize` rows, then the column filter over the ring. For a 7-tap
  Gaussian on 2560x1600 RGB the ring is 7 x 7680 x 2 bytes, about 107 KB:
  L2-resident.
- **KleidiCV `SeparableFilterWorkspace`** goes the other way: the vertical
  taps over `ksize` source rows into **one** row buffer, then the horizontal
  taps from that buffer to the output. One row of intermediate, in L1.
- **libvips** is demand-driven: an operation asks its input for a region
  (a strip or tile), so the vertical pass pulls a few rows of the horizontal
  pass's output. The intermediate exists only for the rows in flight.

Measured cost of the temporary in Pillow-AllSIMD (ms, 2560x1600 RGB):

| | horizontal only | vertical only | both | the temporary | OpenCV |
|---|---|---|---|---|---|
| box r2, x86 | 2.94 | 4.48 | 9.00 | 1.6 | 3.19 |
| Gaussian r1, x86 | 7.39 | 7.89 | 18.68 | 3.4 | 7.57 |
| box r2, Jetson | 7.88 | 9.90 | 21.29 | 3.5 | 10.48 |
| Gaussian r1, Jetson | 23.34 | 23.41 | 52.36 | 5.6 | 43.93 |

Fusing the passes keeps every intermediate value and the order of
operations, so it is an **exact-compatible** change: it can go into the
default path.

### Gaussian blur

| | kernel | arithmetic | SIMD |
|---|---|---|---|
| Pillow | three extended box passes approximating σ (fractional edge weights) | 24-bit fixed point per pass, 8-bit result after each pass | 32-bit lanes; pmaddwd on 12-bit weight halves (x86), 32-bit `vmulq`/`vmlaq` (NEON) |
| OpenCV accurate | true Gaussian, ksize = round(6σ+1) | 8.8 fixed point (`ufixedpoint16`) weights, 16-bit horizontal result in the ring, rounded once | `v_dotprod` (pmaddwd) on symmetric row pairs, 16 lanes in AVX2 |
| KleidiCV (approx hint) | true Gaussian, 3..9, 15, 21 taps (other sizes with replicated borders only) | 8-bit weights summing to 256, **8-bit intermediate** (rounded after the vertical pass) | `vaddl` of symmetric pairs, `vmlaq_u16` by the weight, `vuzp2q` to keep the high bytes: 16 pixels per step |
| libvips | integer mask `rint(20·g)` truncated where g < `min_ampl` (0.2 for gaussblur) | vector path: masks as 8-bit mantissas sharing one exponent, then 32-bit sums | Highway `ReorderWidenMulAccumulate` |

Measured (σ = 1 unless noted; error against a true Gaussian of radius 4σ):

| | x86 ms | Jetson ms | max | mean | differ |
|---|---|---|---|---|---|
| Pillow-AllSIMD | 17.94 | 51.52 | 2 | 0.501 | 49.0% |
| OpenCV accurate | 7.86 | 44.08 | 1 | 0.004 | 0.4% |
| OpenCV approx (KleidiCV on arm64) | 7.82 | **11.15** | 1 | 0.125 | 12.5% |
| libvips default (vector) | 7.38 | | 3 | 0.393 | 38.6% |
| libvips default, `VIPS_NOVECTOR` | 76.05 | | 3 | 0.393 | 38.6% |
| libvips `min_ampl` 0.01 (vector) | 8.29 | | 5 | 1.982 | 93.2% |
| libvips `min_ampl` 0.01, no vector | 95.96 | | 1 | 0.130 | 13.0% |
| σ 2: Pillow-AllSIMD / OpenCV accurate | 21.02 / 14.58 | 57.21 / 82.12 | 2 / 1 | 0.196 / 0.011 | 19.6% / 1.1% |

What it shows:

- Pillow's result is exact only relative to its own three-box algorithm: as
  a Gaussian it is the least accurate of all (half the values off by one,
  some by two). A true separable Gaussian in 8.8 fixed point is both 2.3x
  faster (x86) and 100x closer.
- KleidiCV's 8-bit weights with an 8-bit intermediate are 4x faster than
  OpenCV's accurate NEON path for a 12.5% share of off-by-one values; still
  more accurate than Pillow's exact path.
- libvips' vector path is 10x faster than its C path; its 8-bit mantissa
  masks cost accuracy as the kernel grows (max 5 with a long kernel), which
  libvips documents and lets users turn off (`VIPS_NOVECTOR`).
- OpenCV's accurate NEON Gaussian is slower than Pillow-AllSIMD's for σ 2
  (82 vs 57 ms): the three box passes cost the same for any σ.

### Box blur

- **OpenCV** (`RowSum` + `ColumnSum<ushort, uchar>`): the row sum for 3 and
  5 taps is a direct sum of shifted byte loads, which compilers vectorize for
  any channel count (`D[i] = S[i] + S[i+cn] + ...`); larger kernels use a
  running sum per channel. Sums are **16-bit** when ksize² · 255 fits. The
  column pass keeps one row of running sums: `SUM += new row`, output, then
  `SUM -= old row`, O(1) per pixel. The division by the area is
  `(SUM + d/2) · m >> 23` with `m` rounded to the nearest, which is exact for
  the 25 of a 5x5 box (0 values differ from the true mean).
- **Pillow-AllSIMD**: 24-bit weights `ww = 2^24 / (2r+1)` with rounding at
  bit 23, so 9.1% of values are one off the true mean for r = 2; the integer
  radii 1 to 4 use a 16-bit `mulhi` form proven equal to that 24-bit result.
  The vertical pass sums 2r+1 rows for small radii instead of a running sum.

| box 5x5 | x86 ms | Jetson ms | max | mean | differ |
|---|---|---|---|---|---|
| Pillow-AllSIMD | 9.52 | 20.31 | 1 | 0.091 | 9.1% |
| OpenCV | 2.87 | 10.56 | 0 | 0 | 0% |

A 16-bit running column sum with the existing exact division is
exact-compatible; OpenCV's exact mean is a different result (it would be
more accurate, but not Pillow's).

### 2D kernels (filter2D, ImageFilter.Kernel)

- **OpenCV** `FilterVec_8u`: non-separable 8-bit kernels accumulate in
  **float32** (`v_muladd` on converted pixels), rounded once at the end.
- **libvips** `conv` with integer precision: on the vector path the mask is
  turned into 8-bit mantissas with a shared exponent ("can produce slightly
  different results"), Highway multiply-accumulate in 32 bits.
- **Pillow-AllSIMD**: 16-bit coefficients at Q13, or at the highest exact lower
  precision, 32-bit sums; the fastest of the three now (FIND_EDGES 4.4 ms
  against OpenCV 6.6 and libvips 6.2 on x86; 26.6 against 39.8 on the Jetson).

## 2. Resampling

| | method | arithmetic | structure |
|---|---|---|---|
| Pillow-AllSIMD | convolution with the filter scaled by the downscale factor (antialiased) | 16-bit coefficients (precision chosen per call), 32-bit sums; **8-bit intermediate** between the passes | horizontal pass for the whole image, then vertical; RGBA premultiplied before and divided after |
| OpenCV bilinear (8U) | bilinear, no antialiasing when shrinking | `ufixedpoint16` (8.8) horizontal results, vertical blend, rounded once | per output row: horizontal resize of the needed source rows into a small cache (reused across output rows), then vertical |
| OpenCV INTER_AREA | box average (the benchmark's choice for downscales) | integer sums for integer factors (`resizeAreaFast_`), float weights otherwise | one pass |
| OpenCV cubic, Lanczos | 4 / 8 taps, no antialiasing | 11-bit (`INTER_RESIZE_COEF_BITS`) integer coefficients for 8U | fused like bilinear |
| Carotene bilinear (arm64, 1 and 4 channels) | bilinear | **7-bit** weights (`vmull_u8`, `vrshrn_n_u16(..., 7)`) | vertical blend of up to 8 rows into a buffer, then horizontal with gathered columns |
| KleidiCV bilinear | exact 2x/4x upscales of 1 channel; downscales 1/3 to 1 for up to 3 channels | 16.16 fixed-point coordinates, the top 8 fraction bits as weights | fused |
| libvips `resize` | box `shrink` by an integer factor first (`gap` 2.0: the remaining factor is at most about 2), then `reduce` with Lanczos3 | 12-bit masks precomputed for 64 sub-pixel positions (coordinates rounded to 1/64 px) | demand-driven; Highway `ReorderWidenMulAccumulate` per output pixel |

Measured, bilinear upscale 2560x1600 → 5478x3424 (error against float
bilinear sampling with the same pixel-centre convention):

| | x86 ms | Jetson ms | max | mean | differ |
|---|---|---|---|---|---|
| Pillow-AllSIMD RGB | 25.65 | 93.65 | 1 | 0.179 | 17.9% |
| OpenCV RGB (own code) | 26.74 | 69.81 | 1 | 0.127 | 12.7% |
| Pillow-AllSIMD RGBA (opaque) | 39.37 | 149.79 | 1 | 0.134 | 13.4% |
| OpenCV RGBA (Carotene on arm64) | 31.91 | 43.87 | 1 | 0.137 | 13.7% |

- Pillow's 8-bit intermediate costs accuracy (17.9% against OpenCV's 12.7%
  with an 8.8 intermediate) as well as the temporary image.
- Carotene's 7-bit weights in a fused pass are 3.4x faster than Pillow-AllSIMD
  on the RGBA upscale for the same accuracy. Pillow-AllSIMD's RGBA time is the
  RGB resize plus the premultiply of the input (3.0 ms) and the unpremultiply
  of the 75 MB output (about 50 ms on the Jetson, where it divides).
- On the Jetson the NEON resample kernel is relatively weak: an 8x Lanczos
  downscale takes 16.0 ms (horizontal pass) and 18.8 ms (vertical) against
  2.9 and 4.2 on x86, a much wider ratio than elsewhere. `vmlal_s16` gives
  4 lanes per instruction; the A78's `SDOT` multiplies 16 bytes into 4
  32-bit sums but needs 8-bit weights, which is approximate.
- libvips' default `gap` is the same idea as Pillow's opt-in
  `reducing_gap`: shrink with a box first, then convolve the rest.

## 3. Geometric transforms

| | coordinates | interpolation | vectorization |
|---|---|---|---|
| Pillow-AllSIMD | double, per pixel, through a filter function pointer | double weights; the result **truncated** (`(UINT8)v`) | within one pixel: 4 channels in doubles (`__m256d`, or 2 x `float64x2_t`) |
| OpenCV 5 accurate (`warp_kernels.simd.hpp`) | float32, 8 (NEON) or 16 output pixels per step, incremental | float32 bilinear, rounded | across pixels; an "all inside" fast path; NEON byte shuffles (`vtbl`) to gather the 4 corners of 8 pixels |
| OpenCV 5 approx | as above | FP16 arithmetic where available (arm64) | 8 lanes of FP16 |
| OpenCV `remap` with 16-bit maps | 5-bit sub-pixel fractions (`INTER_TAB_SIZE` 32) | 15-bit integer weights from a 32x32 table | |
| KleidiCV | warpPerspective for 1 channel only; remap with s16 maps and 5-bit fractions | integer | |
| libvips `affine` | double per pixel | bilinear 12-bit weights; bicubic from tables at 1/64 px | per pixel, no SIMD |

Measured, rotation by 30° with bilinear filtering (error against float
bilinear sampling of the same map, inside the source only):

| | x86 ms | Jetson ms | max | mean | differ |
|---|---|---|---|---|---|
| Pillow-AllSIMD | 36.12 | 109.83 | 1 | 0.414 | 41.4% |
| OpenCV accurate | 23.53 | 39.57 | 1 | 0.000 | 0.0% |
| OpenCV approx (FP16 on arm64) | 23.76 | 36.34 | 1 | 0.036 | 3.6% |

Pillow's truncation makes 41% of its values one below the true sample. The
3x gap on the Jetson comes from vectorizing within a pixel in doubles (two
lanes) against across pixels in float32 (four lanes, and no per-pixel call).
Vectorizing Pillow's double arithmetic across pixels is exact-compatible but
gains little on NEON; float32 coordinates and weights are not.

## 4. Rank filters and morphology

- **OpenCV morphology**: separable (row min/max into the ring, then the
  column), and the column pass makes **two output rows per step**, sharing
  the min/max of the rows they have in common (1.5 operations per pixel for
  3x3 instead of 2).
- **OpenCV median**: sorting networks for 3x3 and 5x5 (`medianBlur_SortNet`),
  a multi-level histogram for medium sizes and the O(1) histogram method
  (Perreault) for large ones.
- **KleidiCV median**: sorting networks for 3x3, 5x5 and 7x7, histograms
  above.
- **Pillow-AllSIMD**: sorting networks and register medians for 3x3 and 5x5,
  separable min/max with a ring, all exact (a rank filter has no rounding).
  Remaining gap: the 4 bytes per RGB pixel, the row-pair sharing, and a
  per-row edge-padding copy (min3 RGB 2.06 vs 1.40 ms x86, 5.12 vs 3.13
  Jetson). All of it is exact-compatible.

## 5. Colour conversions

| | Pillow-AllSIMD | OpenCV | KleidiCV / Carotene |
|---|---|---|---|
| RGB→YCbCr | per-term INT16 tables (6-bit fraction), summed then shifted (**floor**) | 14-bit coefficients, one rounding (`yuv_shift` 14), `v_dotprod` | KleidiCV 14-bit (`kRYWeight` 4899 = 0.299·2^14), declines the CbCr order |
| YCbCr→RGB | tables, floor; reproduced exactly in float32 lanes on NEON | 14-bit, `v_mul_expand`, one rounding | not hooked |
| RGB→HSV | float division, `(int)(h * 255)` (**truncation**) | integer division by reciprocal tables at 12 bits (`sdiv_table`, `hdiv_table`), gathered (`vx_lut`, vpgatherdd on AVX2) | Carotene: float reciprocals rounded to the same 12-bit scheme |

| | x86 ms | Jetson ms | max | mean | differ |
|---|---|---|---|---|---|
| RGB→YCbCr Pillow-AllSIMD | 3.15 | 17.39 | 1 | 0.472 | 47.2% |
| RGB→YCbCr OpenCV | 2.56 | 10.38 | 1 | 0.107 | 10.7% |
| YCbCr→RGB Pillow-AllSIMD | 1.99 | 12.21 | 1 | 0.475 | 47.5% |
| YCbCr→RGB OpenCV | 2.46 | 5.70 | 1 | 0.031 | 3.1% |
| RGB→HSV Pillow-AllSIMD (hue / sat, val) | 8.96 | 35.56 | 1 / 1 | 0.500 / 0.242 | 50.0% / 24.2% |
| RGB→HSV OpenCV (hue / sat, val) | 3.56 | 13.60 | 1 / 1 | 0.491 / 0.004 | 49.1% / 0.4% |

Pillow's tables and truncations put about half of all values one level
below the rounded result: a fast path with a single rounding would be both
faster and more accurate, but not bit-compatible. (OpenCV's hue is no more
accurate than Pillow's: its 12-bit tables round, Pillow truncates, and
both land on about half the values off by one.)

## 6. Arithmetic, tables and memory

- **Bandwidth-bound operations** (add, difference, lighter, invert, flips,
  RGBA↔RGB, L→RGB) run at the same bytes per second in Pillow-AllSIMD and
  OpenCV (x86: 25 to 27 GB/s for both; Jetson: 15.4 vs 15.8 GB/s). The time
  ratio is Pillow's 4 bytes per RGB pixel against 3. Structural.
- **LUT**: OpenCV's `LUT` for 8U is a plain scalar loop over the image as
  one row; it still beats Pillow's per-row `point()` loop by 1.4x (1.04 vs
  1.52 ms for L, x86).
- **Page faults**: zero per call for most operations in both libraries
  (glibc reuses freed blocks); Pillow's blur and resize fault in their
  temporaries; numpy places large OpenCV outputs on huge pages (about 450
  faults for 56 MB against 3400 to 5700 for Pillow).
- **blend / addWeighted**: OpenCV computes in float32 and rounds once;
  Pillow-AllSIMD in its float semantics (3.12 vs 1.74 ms x86).

## 7. Drawing and text

OpenCV fills ellipses as polygons of at most 72 vertices (one every 5° once
the axes reach 15 px) with a fixed-point (16.16) convex fill and spans
copied by doubling memcpys; its text is Hershey vector strokes. Pillow
rasterizes exact ellipses and FreeType glyphs. Neither is a candidate for an
approximate mode: the output is a different shape, not a rounding of the same
one.

## 8. Precedents for a user-selectable approximation

| Library | Switch | Scope | Default |
|---|---|---|---|
| OpenCV ≥ 4.10 | `AlgorithmHint` argument per call; build-time default (`OPENCV_ALGO_HINT_DEFAULT`) | Gaussian, warps, remap, YUV conversions | accurate |
| libvips | `VIPS_NOVECTOR` env var, `--vips-novector`, `vips_vector_set_enabled()` | integer convolution, morphology, reduce (the Highway paths) | vector (approximate) on |
| KleidiCV | none; documents each function that is "not bit exact with OpenCV" (Gaussian, resize, remap, warpPerspective, convertTo) | | |
| Pillow | `reducing_gap` argument of `resize` and `thumbnail`, `draft()` for JPEG | resize | off (thumbnail: 2.0) |

## 9. What this means for Pillow-AllSIMD

The techniques fall into three classes.

**A. Exact-compatible restructurings** (the same pixels as now; for the
default path):

1. Fused blur passes: the horizontal passes into the ring the vertical
   cascade keeps. 1.6 to 5.6 ms per blur and the page faults.
2. 16-bit running column sums for box blurs with integer radii.
3. Fused resize passes with a ring of horizontally resampled rows (the 8-bit
   intermediate kept), and the RGBA premultiply/unpremultiply folded into
   the passes' loads and stores. 8 ms of 25 (x86 RGB upscale); about 50 ms
   of 150 on the Jetson RGBA upscale.
4. Morphology: two output rows per column step; edges read in place.
5. Warps: row loops with an all-inside fast path and the filter inlined,
   still in doubles.

**B. Approximate arithmetic** (different pixels, bounded error; for an
opt-in fast mode):

1. Gaussian as a true separable kernel in 8.8 fixed point, fused (x86 ~2.3x;
   more accurate than the current result) or with 8-bit weights and an 8-bit
   intermediate (Jetson ~4.6x, KleidiCV's error).
2. Resize with an 8.8 intermediate in one fused pass (more accurate), 8-bit
   weights through `SDOT` on arm64, and an RGBA path without the division.
3. YCbCr and HSV with one rounding (14-bit coefficients; 12-bit reciprocal
   tables for HSV): 1.2 to 2.6x, more accurate than the tables.
4. Warps in float32 across pixels (Jetson ~3x), FP16 where available.
5. Unpremultiply by a reciprocal table instead of a division.

**C. Algorithmic substitutions** (a different operation; opt-in, and only
where the result is a better or accepted approximation of the same intent):
shrink-then-convolve for large downscales (libvips' `gap`, Pillow's
`reducing_gap`), area averaging for integer factors.

Classes B and C are what [APPROXIMATE_KERNELS_PLAN.md](APPROXIMATE_KERNELS_PLAN.md)
plans; class A belongs in the default path and is listed in
[PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md).
