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
