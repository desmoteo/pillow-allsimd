# Pillow 12.3 and 13.0 as a source of optimizations for pillow-simd

Companion to [X86_SIMD_ASSESSMENT.md](X86_SIMD_ASSESSMENT.md) and [ARM64_SIMD_ASSESSMENT.md](ARM64_SIMD_ASSESSMENT.md). Branch `simd/x86-fixes-arm64-neon`; review written at `2e6edeec2`, step 1 of section 5 done in `311d9e4e5`.

## Sources and method

- Pillow 12.3.0 and 13.0.0 sdists from PyPI, compared file by file.
  - The 12.3.0 sdist's `src/libImaging` is identical to this fork's base (`bb1d8e8ab`). Everything 12.3 added is already in pillow-simd.
- The 13.0.0 release notes (`docs/releasenotes/13.0.0.rst`) and the C diffs for every file 13.0 changed that does pixel work: Resample, Filter, Convert, Chops, Bands, BoxBlur, ColorLUT, Geometry, GetBBox, Histo, Offset.
- Benchmarks of this branch on a Jetson Orin (Cortex-A78AE, Ubuntu 22.04, GCC 11.4, Python 3.10), run with `checks/simd_bench_remote.sh` at commit `b1730a851`. They guided which ideas matter on arm64.
- Points, Fill, Copy, Pack, Matrix and Negative changed only a little in 13.0 and were not reviewed line by line.

## Summary

1. **Rebase pillow-simd onto 13.0 rather than cherry-pick.** 13.0 rewrote much of the scalar C so compilers vectorise it, added free-threading locks, a new `IMAGING_TYPE_I16` and two resampling filters, and changed several internal return conventions. Cherry-picking would leave the fork diverging on every file we touch.
2. **Four upstream bug fixes affect code this branch changed.** All four are now ported (`311d9e4e5`):
   - The `I;16` filter computation.
   - Box-blur radius validation (our new vertical pass loops over the radius, so a huge radius hangs).
   - Masked LA histograms.
   - Filter expansion overflow checks.
3. **13.0 has a few scalar ideas worth applying inside our SIMD code too:**
   - vertical-first resampling when the height shrinks much more than the width,
   - 32-bit palette copies,
   - an OR-reduce in `getbbox`,
   - row `memcpy` in `offset`.
4. **The Jetson run showed that compiler auto-vectorisation can beat hand-written NEON.** On GCC 11 it beat our NEON kernels for blur, unsharp mask and alpha composite. Those three now use the portable code on arm64 (`2e6edeec2`).
5. **New SIMD targets** that upstream does not cover: `ImageChops.multiply`/`screen` (exact divide by 255 is possible), NEON ColorLUT, NEON `reduce` 3 and 4, and the horizontal box-blur pass.

## 1. What the Jetson run says (commit `b1730a851`, ms, best of 5)

Upstream is the Pillow 12.3.0 wheel from PyPI. "scalar" is this branch built with `-C simd=disable`.

| Operation | upstream | scalar | NEON | Reading |
|---|---:|---:|---:|---|
| filter L 3×3 | 48.7 | 46.7 | 8.9 | NEON 5.5x |
| filter RGB 3×3 | 126.4 | 134.2 | 35.0 | NEON 3.6x |
| resize RGBA ¼ bicubic | 51.8 | 52.6 | 20.3 | NEON 2.6x |
| reduce RGB (1, 2) | 7.0 | 7.0 | 1.6 | NEON 4.4x |
| convert RGBa→RGBA | 38.7 | 38.6 | 14.8 | NEON 2.6x |
| gaussian RGB r=5 | 160.8 | **111.7** | 144.9 | scalar wins: vertical pass without transposes plus compiler vectorisation |
| gaussian L r=5 | 54.0 | **29.6** | 36.6 | scalar wins |
| unsharp RGB | 236.8 | **144.8** | 166.8 | scalar wins: GCC turns `/ 100` into a multiply-high; our NEON used `vdivq_f32` |
| alpha_composite | 22.0 | 22.2 | 28.6 | NEON loses |
| convert RGBA→RGBa | 3.4 | 10.6 | 3.6 | scalar 3x slower than upstream (see 3.3) |
| chops multiply | 14.4 | 21.3 | 21.3 | 0.68x, no NEON (see 4.1) |
| colorlut RGB 17 | 104.1 | 101.3 | 100.7 | no NEON yet |

Action taken: BoxBlur (both passes), UnsharpMask and AlphaComposite now use the portable code on arm64. All three were exact, so results do not change. x86 keeps its SSE4 paths, which measured faster there.

Compiling those files with the same GCC 11.4 for aarch64 showed no vector spills in UnsharpMask or AlphaComposite, and 6 in the horizontal box-blur pass. Register spills are therefore not the main cause. Treat "compiler beats intrinsics" as something to re-measure per core and compiler, not as a general rule. Measure again on Apple M-series or Graviton before reintroducing these NEON kernels.

## 2. Bugs fixed in 13.0 that affect this branch

| Area | 12.3 behaviour (and this branch) | 13.0 | Action |
|---|---|---|---|
| `I;16` 3×3/5×5 filters (`kernel_i16`) | High byte shifted right (`>> 8`) instead of left, and each byte clipped separately: a flat `I;16` image of 1000 filters to 233 | `<< 8`, clamped to 0..65535 with `clip16`; `IMAGING_TYPE_I16` loop separate from the 8-bit loop | Ported. `test_i16_uses_generic_kernel` (commit `cf334ba71`) pinned 12.3's wrong values; it now compares against an `I` image |
| BoxBlur radius | NaN, infinite and ≥ 2^31 radii accepted; converting them to `int` is undefined (NaN segfaults GaussianBlur) | Rejected with `ValueError` | Ported. Matters more here: `ImagingVerticalBoxBlur` loops from `-radius-1` to `radius`, so a huge radius hangs |
| `ImagingExpand` (filter margin) | Size check can overflow, and divides by zero for margin 0 (SIGFPE) | 64-bit check; empty images handled | Ported |
| Masked histogram of LA images | Second band counted from byte 1 (padding), not byte 3 (alpha); the unmasked path was already right | Byte 3 for two-band images | Ported (scalar) |

## 3. 13.0 ideas to reuse, mapped to pillow-simd

### 3.1 Resample: vertical pass first when the height shrinks most
- 13.0 runs the vertical pass first when `ysize_in - ysize_out > 2 * (xsize_in - xsize_out)`. The horizontal pass, the more expensive one per output pixel, then works on fewer rows.
- It slots in unchanged with our kernels, since both passes have SSE4/AVX2/NEON versions.
- Results change slightly versus horizontal-first, because the intermediate image is rounded to 8 bits in a different order. Upstream accepted that, and following it keeps pillow-simd close to upstream.
- Effort: small (`ImagingResampleInner` only).

### 3.2 Filter: split the 16-bit loop from the 8-bit loop
- 13.0's "filter up to 3.5x" comes from moving the `I;16` branch out of the 8-bit inner loop, so the 8-bit loop has no per-pixel test and vectorises.
- Our SIMD path already handles 8-bit images. The change helps the portable fallback (`-C simd=disable`, other architectures) and float kernels with large coefficients.
- Comes with the rebase.

### 3.3 Convert
- P, PA → RGB, RGBA: copy the 4-byte palette entry as one `UINT32` and set alpha with a mask (`SET_ALPHA_32`); up to 2.5x. A SIMD gather (AVX2 `vpgatherdd`, 8 pixels per instruction) could go further on x86. NEON has no efficient 1 KB table lookup, so arm64 keeps 13.0's scalar version.
- RGBA → RGB and RGBX → RGB reuse `rgb2rgba` (copy, alpha = 255); `i2l` clamps without branches. Scalar, comes with the rebase.
- RGBA → RGBa premultiply: the portable build is 3x slower than the upstream wheel on the Jetson (10.6 vs 3.4 ms) even with `-ftree-vectorize`. 13.0 did not change `rgbA2rgba`, so the gap is compiler version (GCC 11 vs the wheel's newer GCC). NEON already matches the wheel (3.6 ms), so nothing to do for arm64; worth rechecking for other architectures.

### 3.4 Chops
- 13.0 writes `difference` as a branch-free ternary without the clip (up to 3x). Our SIMD branch only touched Chops for the GIL. Comes with the rebase.
- `multiply` and `screen` still divide by 255 per byte. See 4.1.

### 3.5 Geometry, GetBBox, Offset, Histo
- Flip left-right and rotate 180 copy whole typed elements (16-bit for `I;16`, 32-bit for 4-byte modes) so the reversed copy vectorises (up to 10x).
- `getbbox` first OR-reduces each row and only scans a row element by element when it has data (up to 24x).
- `ImageChops.offset` copies each row as two `memcpy`s instead of a modulo per pixel (up to 46x).
- Histogram loops use `restrict` row pointers.
- All scalar, all from the rebase, and all leave nothing for hand-written SIMD to add, except possibly an explicit NEON/SSE OR-reduce for `getbbox` if the compiler doesn't vectorise it.

### 3.6 Features that interact with the SIMD code
- **New resampling filters `MKS2013` (support 2.5) and `MKS2021` (support 4.5).** They use the same convolution path, so our resample kernels handle them. The 16-bit coefficient normalisation (`normalize_coeffs_8bpc`) should be tested with their negative lobes. Needs the Python-side enum from 13.0.
- **Free-threading.** 13.0 locks `imIn->mutex` around pixel loops (`MUTEX_LOCK`). Every function this branch added or restructured needs the same locks after the rebase:
  - `ImagingFilterSIMD`
  - `ImagingVerticalBoxBlur`
  - the `ReduceSIMD.c` and `ResampleSIMD*.c` callers
  - `ImagingUnsharpMaskRow`
  - Bands
- **`IMAGING_TYPE_I16`.** 13.0 gives 16-bit images their own type instead of `IMAGING_TYPE_SPECIAL`. Our `I;16` dispatch fix in Filter.c and the Reduce/Resample type checks use `IMAGING_TYPE_SPECIAL` and must switch.
- **Changed internal return conventions:**
  - `ImagingSplit` returns 0 or −1, not the band count.
  - `ImagingTransformAffine` returns `int`.

  Our Bands.c rewrite touches `ImagingSplit`, so expect a conflict there.

## 4. SIMD opportunities not covered upstream

### 4.1 `ImageChops.multiply` and `screen`
- They compute `a * b / 255`, which the Jetson shows at 0.68x of the upstream wheel.
- In 16-bit lanes, `floor(x / 255) == (x + 1 + (x >> 8)) >> 8` for every product `x = a * b` of two bytes. Checked exhaustively; it fails only at x = 65535, which a byte product never reaches.
- That gives an exact SSE4/NEON kernel with one widening multiply, two adds and two shifts per 8 bytes. `screen` is `255 - multiply(255 - a, 255 - b)`.
- Expected 2–4x. Effort: small.

### 4.2 NEON ColorLUT
- 101 ms on the Jetson for a 2560×1600 RGB image, the slowest remaining operation without NEON.
- The x86 kernel's structure (16-bit fixed point, per-pixel table gathers) maps to NEON, with the gathers done as scalar loads.
- Effort: medium–high, as estimated in the ARM64 assessment.

### 4.3 NEON reduce 3, 4 and N×N; 1-byte reduce
- `reduce(3)`, `reduce(4)` and all L reductions run the portable code on arm64 (1.0x).
- The 2×2 approach (`vld4q_u8` + pairwise widening adds) extends to 4×4 with a second pairwise add, and to 3×3 with three loads.
- Effort: medium.

### 4.4 Horizontal box-blur pass
- Gaussian RGB is now dominated by the horizontal pass, which processes one pixel at a time on both x86 and arm64.
- Option: blur several rows at once so each vector holds the same column of 4 rows, transposing small blocks in registers.
- Effort: medium–high. Expected up to 2x on Gaussian RGB.

## 5. Recommended order

1. ~~**Port the 13.0 bug fixes in section 2 now**, on this branch: `I;16` filter, box-blur radius validation, LA histogram, expansion check. Update `test_i16_uses_generic_kernel`.~~ Done; see "Status of step 1" below.
2. ~~**SIMD `ImageChops.multiply`/`screen` (4.1).**~~ Done; see "Status of step 2" below.
3. **Rebase onto Pillow 13.0.0** as the fork usually does: version bump, then the SIMD commits on top.
   - Expect conflicts in Bands.c (`ImagingSplit`), Filter.c (`IMAGING_TYPE_I16`, I;16 loop), Resample.c (pass order, `MUTEX_LOCK`), BoxBlur.c and Convert.c.
   - Re-run the diff harness and both full test suites afterwards.
   - About 2–3 days.
4. **After the rebase:**
   - take vertical-first resampling (3.1) through the SIMD kernels;
   - add mutexes to the SIMD entry points (3.6);
   - check the 16-bit coefficients with MKS2021.
5. **NEON ColorLUT (4.2) and NEON reduce 3/4 (4.3)**, then the horizontal box blur (4.4).
6. ~~**Items from [CALLGRIND_PROFILE.md](CALLGRIND_PROFILE.md)**, which are not in this review.~~ Done:
   - Kept: SIMD horizontal resample for L, SIMD reduce for L, and L blur four rows at a time.
   - Measured and dropped: column strips, the SSE4 blend and merge kernels, and two-row AVX2 blur. See the "Outcome" section of that document.
7. **Re-run `checks/simd_bench_remote.sh`** on the Jetson after each step, and on a second arm64 machine (Apple M-series or Graviton) before deciding the final NEON/portable split.

## Status of step 1

All four fixes from section 2 are ported, with 13.0's tests:
- `Filter.c`: the `I;16` loop is separate from the 8-bit loop, as in 13.0, but still keyed on `IMAGING_TYPE_SPECIAL` until the rebase. `test_i16_uses_generic_kernel` now compares against the same values in an `I` image, not 12.3's output.
- `BoxBlur.c` and `ImageFilter.BoxBlur`: NaN, infinite and ≥ 2^31 radii raise `ValueError`, for `GaussianBlur` and `UnsharpMask` too since they go through `ImagingBoxBlur`.
- `ImagingExpand`: 64-bit size check. Stock 12.3 crashed with SIGFPE both on huge margins and on `MinFilter(1)` applied to a zero-size image.
- `Histo.c`: the masked path counts byte 3 for LA. The unmasked path was already correct in 12.3.

Also found: the `I;16N → I` conversion clips to 255 in 12.3. That is a separate upstream bug and not touched here.

## Status of step 2

`Chops.c` computes multiply and screen with `(x + 1 + (x >> 8)) >> 8` for `x = a * b`. Screen is multiply with every byte inverted on the way in and out (XOR 255). There are AVX2, SSE4 and NEON loops, and the scalar loop for the row tail and `-C simd=disable` uses the same formula, so every build gives the same bytes as the division.

Correctness: `test_multiply_screen_exact` checks all 65,536 byte pairs in L, RGB and RGBA at widths from 1 to 256. It passes on the scalar, SSE4 and AVX2 builds on x86, and the same check passes on the NEON and scalar builds under arm64 emulation. Output matches upstream 12.3 byte for byte.

x86 (this machine, 2560×1600, best of 7, ms):

| Operation | upstream 12.3 | scalar before | scalar | SSE4 before | SSE4 | AVX2 |
|---|---:|---:|---:|---:|---:|---:|
| multiply RGB | 18.65 | 18.70 | 2.90 | 11.00 | 2.68 | 2.41 |
| screen RGB | 23.43 | 23.31 | 2.92 | 12.57 | 2.68 | 2.41 |
| multiply L | 4.64 | 4.65 | 0.64 | 2.71 | 0.58 | 0.38 |

Most of the gain comes from the formula itself: without the division, GCC vectorises the scalar loop. The intrinsics add 8–70% on top, and matter more for compilers that vectorise less, such as the Jetson's GCC 11. The NEON kernel has not been timed on real arm64 hardware yet. `checks/simd_bench.py` now also times `chops screen` and `chops multiply L`.

## Caveats

- Upstream speed-up figures are quoted from the 13.0 release notes, not measured here.
- Only one arm64 machine was measured, with one compiler (GCC 11.4). Newer compilers vectorise more, which can move the NEON/portable balance in either direction.
- The removal of three NEON kernels has since been re-timed on the Jetson; see "Jetson re-run" below.

## Jetson re-run at `b913b10e8`

Same machine and script as section 1. The upstream column matched the earlier run to within 1%, so differences of a few percent are real here.

| Operation | NEON at `b1730a851` | NEON at `b913b10e8` | Change | Cause |
|---|---:|---:|---:|---|
| reduce L 2 | 2.33 | 0.23 | 10.1x | NEON 2×2 for 1-byte images |
| reduce L 4 | 1.81 | 0.23 | 7.9x | NEON 4×4 for 1-byte images |
| chops multiply | 21.25 | 4.15 | 5.1x | divide-free multiply (step 2) |
| resize L 1/4 bicubic | 12.03 | 5.39 | 2.2x | NEON horizontal pass for 1-byte images |
| gaussian RGB r=5 | 144.86 | 111.21 | 1.30x | portable code instead of NEON (`2e6edeec2`) |
| box RGB r=2 | 47.75 | 36.62 | 1.30x | same |
| alpha_composite | 28.62 | 22.22 | 1.29x | same |
| gaussian L r=5 | 36.57 | 29.54 | 1.24x | same |
| unsharp RGB | 166.81 | 144.48 | 1.15x | same |
| **resize L x1.5 lanczos** | 45.11 | 47.35 | **0.95x** | NEON horizontal pass slower than the compiler's code for short kernels (about 7 taps) |

- This confirms the decision in `2e6edeec2`: alpha_composite with the portable code now runs at the upstream wheel's speed (22.2 vs 22.0 ms), against 28.6 ms with the old NEON kernel.
- The outputs differ from upstream in exactly the same 7 operations as before. The new kernels add no differences.

Open on arm64:
- The L upscale regression above. The NEON 1-byte kernel handled one row at a time and paid for a horizontal `vaddvq` on every output pixel. Fixed in `4c45bd949`: the kernel now does four rows at a time, as on SSE4, with one `vpaddq` reduction per output pixel and the last 1–3 taps done across the four rows in one vector. Output is unchanged. On the Jetson (upstream column again within 0.3%):
  - `resize L x1.5 lanczos`: 47.35 → 34.03 ms, 2.48x upstream and 1.43x the scalar build;
  - `resize L 1/4 bicubic`: 5.39 → 4.37 ms, 3.34x upstream.
- No NEON code yet for `reduce` 3 and 4 or ColorLUT, which stay at 1.0x upstream.
- The portable build is slower than the upstream wheel for `convert RGBA->RGBa` (0.32x), `merge RGB` (0.70x) and `convert RGB->L` (0.80x). The NEON build is unaffected, but builds without NEON pay this.
