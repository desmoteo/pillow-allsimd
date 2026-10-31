# Fast-mode kernels

The kernels that run when fast mode is on (`PILLOW_SIMD_FAST=1`,
`Image.set_fast()`; user documentation in `docs/reference/fast_mode.rst`).
The plan, the catalogue of candidates and the evidence behind them are in
[APPROXIMATE_KERNELS_PLAN.md](APPROXIMATE_KERNELS_PLAN.md) and
[OTHER_LIBRARIES_KERNELS.md](OTHER_LIBRARIES_KERNELS.md).

## Rules every fast kernel follows

- **Off by default.** The exact path is untouched; the fast kernel is a
  separate function the operation's entry point calls when
  `ImagingGetFast() & IMAGING_FAST_ALL` is set, read once per call.
- **Deterministic.** A portable C function defines the result. The SSE4,
  AVX2 and NEON versions compute the same integer arithmetic, so every build
  on every architecture gives the same pixels; `checks/simd_hashes.py` has a
  `fast:` area per kernel to compare builds, and its hash is the same on x86
  and arm64.
- **Bounded.** The largest difference from the exact result and the error
  against a float reference of the operation are measured on the benchmark
  photo, `Tests/images/hopper.ppm` and noise, written below, and asserted by
  `Tests/test_fast_mode.py`. The error against the reference must not be
  worse than the exact path's.
- **Falls back.** Arguments, modes or sizes the fast kernel doesn't take go
  to the exact path, which also reports the errors, so fast mode never
  changes which calls fail.
- **Same shape.** Output mode, size and edge handling are the exact path's.

## Gaussian blur (`GaussianFast.c`)

**Operation.** `ImageFilter.GaussianBlur` and the blur inside
`ImageFilter.UnsharpMask`, through `ImagingGaussianBlur()` with three passes,
for 8-bit modes L, LA, La, RGB, RGBA, RGBa, RGBX and CMYK and radii (σ) up to
6 on each axis.

**Exact path.** Three extended box passes per axis (BoxBlur.c), each with
24-bit weights (`ww` inside, `fw` on the fractional edge) and rounded to 8
bits, horizontally on each row, then vertically in one sweep; each pass
replicates its input's edge pixels.

**Fast kernel.**

- The three boxes, built with the exact path's own radius
  (`ImagingGaussianBlurBoxRadius`) and weights, are convolved in double into
  one kernel of radius 3(r+1), whose weights are rounded to multiples of
  1/256 with the remainder in the centre. It is Pillow's blur, not a sampled
  Gaussian: a sampled Gaussian has less variance than σ² at small σ (at
  σ 0.3 it is nearly the identity, where Pillow blurs), and its shape differs
  from three boxes by up to 25 levels at sharp edges.
- One sweep down the image: for each output row, the vertical taps of its
  source rows into one row, rounded to 8 bits, then the horizontal taps of
  that row (padded with its edge pixels) into the output, KleidiCV's order.
  Every sum is at most 128 + 255 · 256, so it fits 16 bits: 8 lanes per SSE4
  or NEON register, 16 per AVX2 register (`mullo_epi16`, `vmlaq_n_u16`);
  symmetric taps are added in pairs first.
- **Edges.** One kernel can't clamp after each pass as the exact path does,
  which near a corner changed up to three quarters of the kernel's weight
  (30 levels on noise). The bands within the kernel's reach of the edges are
  therefore taken from the exact blur of strips twice as wide (top and
  bottom; left and right side by side in one image): a pass clamping at a
  strip's inner edge, or mixing the two strips at the seam, changes the
  strip by at most the reach from there, so the outer halves are exactly the
  exact blur of the whole image. Images no wider or taller than four reaches
  are blurred exactly altogether.
- σ above 6 keeps the exact path, whose cost doesn't grow with σ: x86 and
  arm64 both cross over between 6 and 8, and the limit must be the same
  constant everywhere, or the same call would give different pixels on
  different machines.

**Accuracy.** Largest difference from the exact result: 3 levels (benchmark
photo, hopper, noise; all σ from 0.3 to 6); mean 0.1 to 0.7. Against a true
Gaussian of radius 4σ, on the photo (share of values that differ, maximum):

| σ | exact | fast |
|---|---|---|
| 0.5 | 59.1%, 3 | 20.0%, 1 |
| 1 | 49.0%, 2 | 13.7%, 1 |
| 2 | 19.6%, 2 | 9.4%, 1 |
| 4 | 13.6%, 1 | 9.9%, 1 |

The fast result is closer: one rounding instead of three.

**Speed** (2560x1600 RGB, one core, ms, exact → fast):

| σ | x86 AVX2 (Core Ultra 9 285H) | arm64 NEON (Jetson Orin) |
|---|---|---|
| 0.5 | 18.7 → 4.7 (4.0x) | 52.2 → 12.0 (4.4x) |
| 1 | 18.8 → 5.8 (3.3x) | 52.2 → 16.5 (3.2x) |
| 2 | 21.5 → 8.5 (2.5x) | 58.1 → 22.3 (2.6x) |
| 3 | 25.7 → 13.1 (2.0x) | 91.1 → 33.0 (2.8x) |
| 4 | 25.9 → 16.7 (1.5x) | 90.9 → 40.0 (2.3x) |
| 6 | 26.9 → 26.4 (1.0x) | 91.3 → 72.5 (1.3x) |

**Code map.** `gauss_kernel()` builds the kernel; `gauss_taps()` is the
portable reference with its SIMD loops; `gauss_exact_edges()` the edge
bands; `ImagingGaussianBlurFastApplies()` decides, `ImagingGaussianBlurFast()`
runs. Tests: `Tests/test_fast_mode.py` (difference from exact per mode and
radius, exact edge bands, closer to a true Gaussian, uniform images stay
uniform, fall-backs). Hashes: `fast: gaussian` in `checks/simd_hashes.py`.
Measurements: pillow-simd-perf `analysis/` (accuracy) and the σ sweep.

## RGB ↔ YCbCr (`ConvertYCbCr.c`)

**Operation.** `Image.convert("YCbCr")` from RGB, RGBA and RGBX, and
`convert("RGB")` from YCbCr: the conversion table's row functions, switched
in `ImagingConvert()` (`fast_converter()`). Palette images and the JPEG 2000
decoder keep the exact functions.

**Exact path.** Per-term INT16 tables with 6 fractional bits, summed and
shifted right: each sum is floored, so about half the values are one below
the rounded result of JPEG's formulas.

**Fast kernel.** JPEG's BT.601 full-range formulas with 14-bit coefficients
(`Y = 4899 R + 9617 G + 1868 B`, ..., `R = Y + 22970 (Cr - 128)`, ...) and
one rounding, as OpenCV and KleidiCV compute them. Every value is an integer
sum of products in 32 bits:

- x86: RGB→YCbCr with `pmaddwd` on the pixels as 16-bit words and `phaddd`
  (4 pixels per SSE4 step, 8 per AVX2 step with a permute to undo the
  per-lane order); YCbCr→RGB with `pshufb` making (Cb, Cr) word pairs and one
  `pmaddwd` per channel.
- NEON: 16 pixels per step through `vld4q`/`vst4q`, `vmull_n_s16` and
  `vmlal_n_s16` per quarter, `vshrn` and saturating narrows.
- Cb and Cr reach 256 at most (pure blue, pure red), clipped to 255;
  YCbCr→RGB clips to 0..255; the fourth byte is copied, as before.

**Accuracy** (all 2^24 inputs in each direction): at most 1 level from the
exact result. Against the float formulas, rounded: RGB→YCbCr 0.29% of values
off by one (exact: 47.2%), YCbCr→RGB 0.06% (exact: 32.1%). The hashes of the
whole domain are the same on SSE4, AVX2, NEON and scalar builds.

**Speed** (2560x1600 RGB, one core, ms, exact → fast; OpenCV in brackets):

| | x86 AVX2 | arm64 NEON |
|---|---|---|
| RGB→YCbCr | 3.14 → 2.04 (1.5x) [2.57] | 17.41 → 7.81 (2.2x) [10.37] |
| YCbCr→RGB | 2.00 → 1.58 (1.3x) [2.47] | 12.22 → 7.19 (1.7x) [5.69] |

**Code map.** `ImagingConvertRGB2YCbCrFast()`, `ImagingConvertYCbCr2RGBFast()`
(the scalar loop at the end of each defines the result). Tests:
`Tests/test_fast_mode.py` (difference from exact over a lattice of every
third value, accuracy against the float formulas, palette images
unchanged). Hashes: `fast: ycbcr`.

## RGB → HSV (`Convert.c`, `rgb2hsv_fast`)

**Operation.** `Image.convert("HSV")` from RGB, RGBA and RGBX, switched in
`fast_converter()`.

**Exact path.** colorsys's formulas in float32 for the ratios and double for
`h / 6 + 1` modulo 1, then truncated: `(int)(h * 255.0)`. The float steps
sometimes land just below an integer the exact ratio reaches, and the
truncation then gives 1 less.

**Fast kernel.** The same formulas, truncated exactly: with `d = max - min`
and the hue's numerator `n` (`g - b`, `2d + b - r` or `4d + r - g` by the
largest channel, Pillow's tie order, plus `6d` if negative),
`h = floor(255 n / 6d)`, `s = floor(255 d / max)`, `v = max`. One float32
division gives each floor exactly: the numerator and denominator are
integers below 2^24, and a quotient that isn't an integer is at least
1/1530 from one, far more than the division's rounding error, so the result
doesn't depend on the floating-point unit. Integer channels, selects and the
division in 8 (AVX2), 4 (SSE4) or 4 × 4 (NEON, after `vld4q`) lanes; pixels
with `max == min` are masked to 0, as before.

**Accuracy** (all 2^24 inputs): equal to the exact integer floors;
saturation and value equal to the exact path's everywhere, hue 1 above it
for 0.28% of inputs and equal elsewhere. The same hash on SSE4, AVX2, NEON
and scalar builds.

**Speed** (2560x1600 RGB, one core, ms, exact → fast; OpenCV in brackets):
x86 AVX2 8.96 → 3.06 (2.9x) [3.66]; arm64 NEON 35.57 → 17.74 (2.0x)
[13.61]. On arm64 the two divisions per four lanes dominate; a reciprocal
estimate with an exact integer correction step would remove them.

**Code map.** `rgb2hsv_fast()`, with `rgb2hsv_fast_pixel()` defining the
result. Tests: `Tests/test_fast_mode.py` (the floors over a lattice of every
third value, saturation and value equal to exact). Hashes: `fast: hsv`.

## Affine, perspective and quad transforms (`TransformFast.c`)

**Operation.** `Image.transform()` with `AFFINE` (and `EXTENT`, which
becomes affine), `PERSPECTIVE`, `QUAD` and `MESH` (a quad per tile), and
`Image.rotate()` at angles other than multiples of 90 degrees, with the
bilinear or bicubic filter, for L and the 4-byte modes RGB, RGBA (which
`transform()` premultiplies to RGBa first), RGBa, RGBX and CMYK: in
`ImagingGenericTransform()`, after the filter lookup.

**Exact path.** Each output pixel's source position in doubles, then the
filter in doubles, truncated to 8 bits; the vectorized perspective
coordinates divide twice per pixel. Truncation puts 41% of bilinear values
one below the rounded interpolation, and the compilers fuse the
multiply-adds differently on x86 and arm64.

**Fast kernel.**

- **Positions** in exact arithmetic, a 64x64 tile at a time, one row of the
  tile per call. Affine and quad: the coefficients times 2^31 or 2^32,
  rounded once, then integer sums along the row (2^-32 pixel). Perspective:
  the numerators and the denominator as exact integers (coefficients times
  2^36, below 2^53, so in doubles a fused or separate multiply-add gives
  the same sums), then `p = n * (512 / d)`, one IEEE division and one
  product, which every unit computes alike, truncated to 1/512 pixel: two
  lanes per step on SSE4 (`divpd`), four on AVX2 (`vdivpd`), two on NEON
  (`fdiv`), truncated by `cvttpd2dq` or `fcvtzs`.
- Each position, minus half a pixel, is rounded to 1/256 pixel; the integer
  part picks the pixels, the fraction the weights. A pixel is outside if
  its position is (the exact path's test, at the fixed point's
  precision), and near an edge if a neighbour is out of the image, where
  the scalar function clamps it as the exact path does.
- **Bilinear**, vertical first: `v = p0 (256 - fy) + p1 fy` for the two
  columns, at most 65280, so 16-bit lanes hold it, then
  `((v0 >> 1) (256 - fx) + (v1 >> 1) fx + 2^14) >> 15`, one rounding (the
  halving, 1/512 of a level, lets the horizontal step use a signed 16-bit
  multiply-add). AVX2: four pixels per step, two per register, `pshufb`
  making (p0, p1) byte pairs per channel, `pmullw` for the vertical step
  and `pmaddwd` with the (256 - fx, fx) pairs; SSE4 two pixels. NEON: one
  pixel per half register, `p0 (256 - fy)` as `(p0 << 8) - p0 fy` so the
  weights fit 8 bits (`vshll`, `vmlsl_u8`, `vmlal_u8`), `vmull_n_u16`,
  `vrshrn` by 15. L: eight pixels per step from one 16-bit load of (p0, p1)
  per row and pixel, in 16-bit lanes (SSE4, NEON).
- **Positions and kinds** for a row: AVX2 compares the affine and quad
  positions in four 64-bit lanes (`vpcmpgtq`) and takes the 1/256 pixel
  positions from the low halves of a logical shift, which equal the
  arithmetic shift's there, and computes the kinds eight at a time.
- **Bicubic:** Pillow's cubic (a = -1) as a table of 256 weight sets, 12
  fractional bits each, rounded with the remainder in the largest so they
  sum to 4096; each row's sum rounded to 8 bits, then the column sum,
  rounded and clipped. One pixel per step: `pmaddwd` on two pixel pairs
  per row (SSE4), `vmlal_n_s16` per pixel (NEON).
- Positions the fixed point can't hold (a corner of the region beyond 2^30
  pixels, perspective sums beyond 2^52, images of 2^22 pixels or more on a
  side) keep the exact path, as do the nearest filter, other modes (LA and
  La included: their exact filters copy the first channel into the second
  and third bytes) and the other transforms.

**Accuracy.** At most 2 levels from the exact result (noise, hopper; every
transform, mode and filter of the tests); 1 on the benchmark photo, mean
0.4, about 40% of values differ, nearly all by the exact path's truncation.
Against bilinear interpolation in doubles at the exact positions (photo,
rotation by 30 degrees): exact 41.4% of values off by one, fast 0.6%. RGBA
differs more where alpha is low (up to 37 levels on the photo's alpha):
`transform()` premultiplies, and a premultiplied value 1 apart is up to
255 / alpha apart once divided by alpha again. The same hash on SSE4, AVX2,
NEON and scalar builds, x86 and arm64.

**Speed** (2560x1600, one core, ms, exact → fast; OpenCV `warpAffine` in
brackets):

| | x86 AVX2 | arm64 NEON |
|---|---|---|
| rotate 30° RGB bilinear | 39.6 → 14.8 (2.7x) [29.9] | 108.3 → 41.1 (2.6x) [35.4] |
| rotate 30° RGB bicubic | 98.3 → 33.6 (2.9x) [62.7] | 263.7 → 78.6 (3.4x) [79.2] |
| rotate 30° RGBA bicubic | 109.0 → 40.7 (2.7x) | 340.9 → 97.3 (3.5x) |
| perspective RGB bilinear | 32.3 → 15.2 (2.1x) | 117.5 → 61.5 (1.9x) |
| rotate 30° L bilinear | 35.1 → 12.5 (2.8x) | 60.8 → 31.7 (1.9x) |
| perspective L bilinear | 27.1 → 14.1 (1.9x) | 67.5 → 48.7 (1.4x) |
| rotate 30° L bicubic | 84.4 → 43.5 (1.9x) | 126.6 → 69.5 (1.8x) |

The x86 machine had a browser running for this table; OpenCV's numbers
are from a quiet run. On arm64 the perspective's divisions dominate (fast
affine positions 42 ms, perspective 55 ms for the same shift).

**Code map.** `tf_map_init()` and `tf_region_ok()` set up and check the
fixed point; `tf_row()` gives the positions and kinds of a row;
`tf_bilinear_pixel()` and `tf_bicubic_pixel()` define the result, with
`tf_bilinear_l()`, `tf_bicubic_l()`, `tf_bilinear8_l_sse()`,
`tf_bilinear8_l_neon()`, `tf_bilinear2_sse()`, `tf_bilinear4_avx2()`,
`tf_bicubic1_sse()`, `tf_bilinear2_neon()` and `tf_bicubic1_neon()` for
pixels away from the edges;
`ImagingTransformFast()` decides and runs. Tests: `Tests/test_fast_mode.py`
(difference from exact per mode, filter and transform, accuracy against
float bilinear, fill colour, small images, mesh, out-of-range coefficients
and other modes on the exact path). Hashes: `fast: transform`.

## Blend (`Blend.c`)

**Operation.** `Image.blend(im1, im2, alpha)` for 0 < alpha < 1, on every
mode it takes (8-bit bands); alpha 0 and 1 copy an image, and alphas
outside 0..1, which extrapolate and clip, keep the exact path.

**Exact path.** `in1 + alpha (in2 - in1)` in floats, truncated, so half of
the values are one below the rounded result; the compiler vectorizes it.

**Fast kernel.** alpha in units of 2^-15, `a = round(alpha 2^15)` (at most
32767), then `in1 + ((in2 - in1) a + 2^14) >> 15`: the rounding of
`pmulhrsw` (SSE4, AVX2) and of `vqrdmulh` (NEON), which compute the same
integer, 16 or 32 bytes per step. The result lies between the two inputs,
so nothing needs clipping.

**Accuracy.** At most 1 level from the exact result. It is the float result
rounded to the nearest integer, except where that is within 0.004 of a
half (alpha's rounding moves the product by at most 255 · 2^-16). The same
hash on SSE4, AVX2, NEON and scalar builds.

**Speed** (2560x1600 RGB, alpha 0.3, one core, ms): x86 AVX2 3.32 → 1.93
(1.7x); arm64 NEON 12.97 → 3.13 (4.1x). OpenCV's `addWeighted` (float):
1.8 and 11.9.

**Code map.** `blend_fast_row()`. Tests: `Tests/test_fast_mode.py`
(difference from exact per mode, distance from the float result, alphas
outside the open interval exact). Hashes: `fast: blend`.

## Large downscales (`Image.py`, `resize()`)

**Operation.** `Image.resize()` with the bicubic, Lanczos, MKS2013 or
MKS2021 filter and no `reducing_gap`, on every mode but I;16 (which
`reduce()` doesn't take). Thumbnails already use `reducing_gap=2.0`.

**Exact path.** One convolution with the filter stretched over the whole
scale: 1/8 Lanczos reads 48 source pixels per output pixel and axis.

**Fast kernel.** `reducing_gap=3.0`: for a downscale of 6x or more the
image is first shrunk by an integer factor with `reduce()` (box averages,
exact integer arithmetic) so that the convolution does the last 3x to 6x.
The libvips-style step is a different algorithm, not a rounding, so it is
bounded by measurement:

- 3.0 rather than thumbnail's 2.0: at 2.0 the 16x results were up to 6
  levels from the exact result on hopper scaled to 2048x1536, and 4x
  downscales changed too; at 3.0 they stay within 2 levels on photos,
  which Pillow's documentation calls indistinguishable from fair
  resampling in most cases.
- Only the filters it makes faster: with box, bilinear and Hamming the
  exact kernels are already as fast as `reduce()` plus a shorter filter (at
  1/8: 0.75x, 0.96x, 1.0x on x86).

Both steps are exact kernels, so the hash is the same on every build.

**Accuracy** (2560x1600 photo to 426x266, 320x200 and 160x100): at most 2
levels from the exact result, 0.2 to 0.4 on average; against the same
filter in floats (Pillow's F resize per channel) PSNR 52 to 56 dB, the
exact result 59 to 61 dB. Tiny results differ more at their edges, where
`reduce()` averages partial boxes (512x384 to 32x24: up to 9 levels on the
border, 4 inside).

**Speed** (2560x1600 RGB, one core, ms, exact → fast):

| | x86 AVX2 | arm64 NEON |
|---|---|---|
| 1/6 bicubic | 2.64 → 1.58 (1.7x) | 13.23 → 5.83 (2.3x) |
| 1/8 Lanczos | 3.52 → 1.88 (1.9x) | 18.21 → 7.39 (2.5x) |
| 1/16 Lanczos | 3.06 → 1.56 (2.0x) | 16.48 → 3.67 (4.5x) |
| 1/8 MKS2021 | 4.44 → 2.24 (2.0x) | 25.90 → 9.15 (2.8x) |

**Code map.** `Image.resize()`, the `reducing_gap` default. Tests:
`Tests/test_fast_mode.py` (equal to `reducing_gap=3.0`, close to exact at
photo sizes, below 6x and with other filters, an explicit gap or I;16
exact). Hashes: `fast: downscale`.
