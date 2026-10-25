# Pillow 12.3 and 13.0 as a source of optimizations for pillow-simd

Companion to [X86_SIMD_ASSESSMENT.md](X86_SIMD_ASSESSMENT.md) and [ARM64_SIMD_ASSESSMENT.md](ARM64_SIMD_ASSESSMENT.md). Branch `simd/x86-fixes-arm64-neon`; review written at `2e6edeec2`, step 1 of section 5 done in `311d9e4e5`.

## Where things stand (2026-10-29)

- Done: the 13.0 bug fixes (step 1), every SIMD target of section 4 except the RGB horizontal box-blur pass, the callgrind items, x86 work ported to arm64, and the new x86 build options (sections at the end).
- Done on branch `simd/pillow-13`: **the merge of Pillow 13.0.0** (steps 3 and 4 of section 5). See "Status of steps 3 and 4" below.
- Open: timing on arm64 hardware after the merge, a second arm64 machine for the NEON/portable split, and runtime dispatch (not planned on this branch).

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
   - row `memcpy` in `offset` (done, `bc77919ef`).
4. **The Jetson run showed that compiler auto-vectorisation can beat hand-written NEON.** On GCC 11 it beat our NEON kernels for blur, unsharp mask and alpha composite. Those three now use the portable code on arm64 (`2e6edeec2`).
5. **New SIMD targets** that upstream does not cover: `ImageChops.multiply`/`screen` (exact divide by 255 is possible), NEON ColorLUT, NEON `reduce` 3 and 4, and the horizontal box-blur pass. All done except the RGB horizontal box-blur pass; for 1-byte images it is done on x86.

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
5. ~~**NEON ColorLUT (4.2) and NEON reduce 3/4 (4.3)**~~ Done (`9fd5b665a`). Horizontal box blur (4.4): done for 1-byte images on x86; two RGB rows per AVX2 vector and the arm64 port were measured and dropped.
6. ~~**Items from [CALLGRIND_PROFILE.md](CALLGRIND_PROFILE.md)**, which are not in this review.~~ Done:
   - Kept: SIMD horizontal resample for L, SIMD reduce for L, and L blur four rows at a time.
   - Measured and dropped: column strips, the SSE4 blend and merge kernels, and two-row AVX2 blur. See the "Outcome" section of that document.
7. **Re-run `checks/simd_bench_remote.sh`** on the Jetson after each step (done throughout), and on a second arm64 machine (Apple M-series or Graviton) before deciding the final NEON/portable split (not done).

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

## Status of steps 3 and 4 (branch `simd/pillow-13`)

The branch merges the `13.0.0` tag of python-pillow/Pillow into `simd/x86-fixes-arm64-neon` instead of rebasing, so the commits of PRs #160–#164 stay unchanged underneath. The version is `13.0.0.post0`.

- **Conflicts:** 24 files, as expected in CI, `setup.py`, `Makefile`, the tests and the C files Bands, BoxBlur, Chops, ColorLUT, Filter, Geometry, Histo, Offset and AlphaComposite. Resample, Reduce and Convert merged without conflicts.
- **Two semantic conflicts** that only the compiler found:
  - 13.0's `XCLIP`/`YCLIP` take one argument (the image sizes come from locals), so the SIMD bilinear and bicubic transform filters were adapted.
  - `MIN`/`MAX` now come from `ImagingUtils.h`.
- **Every 13.0 security fix is in place:**
  - the `ImagingMerge` band check;
  - the affine-transform double free;
  - the rank filter margin overflow;
  - the reduce size overflow.
- **Every `MUTEX_LOCK` of 13.0 is kept.** The SIMD filter dispatch and the SIMD transform tiles run inside 13.0's locks, and so does the four-table histogram. Reduce, BoxBlur, Bands and UnsharpMask have no locks in 13.0 either.
- **3.1 vertical-first resampling** comes with the merge. Each pass keeps its SIMD kernel, so either order uses them.
- **`IMAGING_TYPE_I16`** replaces `IMAGING_TYPE_SPECIAL` everywhere; the SIMD dispatch only takes `IMAGING_TYPE_UINT8`.
- **MKS2013 and MKS2021** run through the same SIMD kernels.
  - Against the scalar build (`-C simd=disable`), resized L, LA, RGB, I;16, I and F images are identical for MKS2013, MKS2021, Lanczos and Bicubic, at 15 sizes up and down.
  - RGBA differs for all four filters. It did already before the merge for Lanczos and Bicubic, so this is not new.
  - Against stock 13.0, I;16, I and F are identical. The 8-bit modes differ, as they always have in pillow-simd, which uses 16-bit fixed-point coefficients.
- **One 13.0 test needed a change.**
  - `test_consistency_i16` compares `hopper("I")` with `hopper("I;16")` after filtering. In pillow-simd the SIMD RGB→L/I conversion has 15-bit coefficients (from upstream pillow-simd), which round one hopper pixel differently from RGB→I;16.
  - The test now makes its `I` reference from the `I;16` image. That became possible with 13.0, which fixed the `I;16N → I` clipping noted in step 1.

Measured on this machine (2560×1600, ms, best of 25, pinned to one CPU):

| Operation | upstream 13.0 wheel | scalar | AVX2 |
|---|---:|---:|---:|
| resize RGB 1/4 MKS2021 | 70.5 | 76.6 | 6.5 |
| resize RGB to 1/8 height, Lanczos (vertical first) | 49.8 | 15.3 | 5.7 |
| P→RGB / P→RGBA | 2.1 / 1.8 | 1.6 / 1.5 | 1.6 / 1.5 |
| getbbox, flip, rotate 180, difference, offset | within ±10% of each other | | |

The 13.0 scalar speedups (palette copies, getbbox, flips, difference, offset) are memory-bound at this size: 16 MB in and out in about 2.5 ms. Hand-written SIMD has nothing to add there, so none was written.

Tests: 6095 passed with `-C simd=avx2`, `sse4` and `disable` on the host (Python 3.14). 6129 passed for scalar, SSE4, AVX2 and AVX-VNNI in a Python 3.13 container via `checks/simd_test_remote.sh local`, which has more optional test dependencies.
On arm64 under qemu (`--platform linux/arm64`), the filter, resample, resize, reduce, convert, chops, ColorLUT, box blur, transform, split, histogram and point tests (1375 tests) passed for the NEON and scalar builds. The output comparison shows only the known ±1 differences: the 5×5 float filter and the 15-bit RGB→L conversion.

Python 3.10, which the Jetson runs, is no longer supported. `checks/simd_test_remote.sh` runs the test suite there in a container with a newer Python. `checks/simd_bench_remote.sh --docker` does the same for benchmarks.

## Performance after the merge (2026-10-29)

Baseline: the Pillow 13.0.0 wheel from PyPI. 2560×1600 images, best of 5 rounds, `checks/simd_bench.py --set all`, pinned to one CPU. Reports are in the git-ignored `bench-results/`.

**Geometric mean over all operations, speed-up against the 13.0 wheel:**

| Machine | scalar | SSE4 | AVX2 | AVX-VNNI | NEON | pre-merge build |
|---|---:|---:|---:|---:|---:|---:|
| x86-64 laptop, Python 3.14, GCC 13 (82 ops) | 1.37x | 2.75x | 3.12x | 3.17x | | 3.05x (AVX2) |
| Jetson Orin, Python 3.12 container, GCC 14 (86 ops, `ff08470e7`) | 1.25x | | | | 1.85x | 1.81x (NEON, same container) |

On the same compiler, the merge cost nothing on average: AVX2 is 1.02x and NEON 1.02x the pre-merge build. The only operations more than 8% slower are flip left-right on x86 (0.91x), and `frombytes RGB` (0.85x) and `ROTATE_90` (0.88x) on arm64; see "Not faster than the wheel" below. The merge gained 13.0's scalar rewrites:

| Operation (ms) | x86 AVX2 pre → now | arm64 NEON pre → now |
|---|---:|---:|
| convert P→RGB | 4.41 → 1.65 | 8.04 → 3.61 |
| chops difference | 4.32 → 2.43 | 6.26 → 3.10 |

**Selected operations (ms):**

| Operation | x86 13.0 wheel | x86 AVX2 | arm64 13.0 wheel | arm64 NEON |
|---|---:|---:|---:|---:|
| resize RGB 1/4 bicubic | 33.4 | 3.40 | 34.5 | 14.8 |
| resize L 1/4 bicubic | 15.4 | 1.80 | 14.5 | 4.24 |
| resize RGB 1/4 MKS2021 | 70.4 | 6.67 | 76.7 | 29.7 |
| resize L 1/4 MKS2021 | 34.3 | 3.11 | 30.1 | 8.16 |
| resize RGB to 1/8 height, Lanczos (vertical first) | 49.7 | 5.53 | 73.2 | 18.7 |
| reduce L 2 | 1.63 | 0.21 | 2.32 | 0.21 |
| gaussian RGB r=5 | 139.9 | 53.4 | 158.2 | 84.7 * |
| unsharp RGB | 229.7 | 63.0 | 234.6 | 102.6 * |
| filter L 3×3 SHARPEN | 48.9 | 2.08 | 47.4 | 6.09 |
| filter RGB 5×5 SMOOTH_MORE | 168.6 | 18.8 | 227.2 | 65.9 |
| convert RGBa→RGBA | 33.7 | 1.75 | 38.6 | 11.1 |
| colorlut RGB 17 | 101.1 | 13.0 | 104.2 | 52.4 |
| chops multiply | 18.7 | 2.47 | 14.0 | 3.65 |
| median 3 L | 546 | 9.6 | 453 | 13.3 |
| mode 3 L | 1039 | 77.7 | 1251 | 99.1 |

\* after `a0fe1ba89`, see below.

**Not faster than the wheel:**
- Resize of F and I;16, YCbCr→RGB, L→1 dither, transposes, `chops offset`, `point` and `histogram` on x86 are within ±5%; none of them has a SIMD path.
- On arm64 `getextrema` is 4% slower with identical code.
- `ROTATE_90` on arm64 is 13.0's code now and 12% slower than 12.3's loop under GCC 14, though within 1% of the wheel.
- `frombytes RGB` lost 15% with 13.0 on both the wheel and this build; it is still 1.6x the wheel.

**arm64 kernels re-timed with newer compilers (`a0fe1ba89`).**

`2e6edeec2` replaced three NEON kernels with the portable code, based on the host's GCC 11. In containers with GCC 12 (Debian 12) and GCC 14 (Debian 13), with the upstream column stable to 0.3%:

| Operation (ms, GCC 14 / GCC 12) | portable | all three NEON kernels back | kept: 4-byte horizontal pass and unsharp row |
|---|---:|---:|---:|
| gaussian RGB r=5 | 107.3 / 112.2 | 91.1 / 93.4 | 84.7 / 87.6 |
| box RGB r=2 | 35.6 / 37.1 | 30.3 / 31.0 | 28.2 / 29.0 |
| unsharp RGB | 134.7 / 142.8 | 108.9 / 111.1 | 102.6 / 105.3 |
| gaussian L r=5 | 28.5 / 28.9 | 29.9 / 30.3 | 28.4 / 28.8 |
| alpha_composite | 22.0 / 21.3 | 25.1 / 26.4 | 21.9 / 21.3 |
| histogram RGB (13.0's loops on arm64) | 10.3 / 10.0 | | 9.28 / 8.86 |

- The NEON vertical pass (shared by L and RGB) and NEON alpha composite stay removed.
- GCC 11 was not re-measured: a Python 3.11+ build environment with GCC 11 is needed, and the Debian 11 image no longer installs packages.
- Outputs are identical to the scalar build and to the 13.0 wheel.

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
- ~~No NEON code yet for `reduce` 3 and 4 or ColorLUT, which stay at 1.0x upstream.~~ Done in `9fd5b665a`. On the Jetson, with the upstream column again within 1%:
  - `reduce RGB 3`: 5.06 → 1.84 ms (2.82x upstream);
  - `reduce RGB 4`: 4.47 → 1.64 ms (2.71x);
  - `colorlut RGB 17`: 100.98 → 68.31 ms (1.53x).

  ColorLUT on NEON keeps the portable code's arithmetic, so its output is identical to upstream. The SSE4 kernel uses its own precision.
- Left as is: `convert RGB->LA` is bound by memory bandwidth (0.99x), and `convert RGBA->RGBa` is 6% behind the upstream wheel.
- The portable build is slower than the upstream wheel for `convert RGBA->RGBa` (0.32x), `merge RGB` (0.70x) and `convert RGB->L` (0.80x). The NEON build is unaffected, but builds without NEON pay this.

## Rank filters, histogram, offset and RGB pack/unpack (`bc77919ef`)

From a callgrind profile of 45 more common operations. These changes are plain C, or SSE4 and NEON for pack/unpack, so they help both architectures. Output is identical to the previous code, and the I;16 offset is now correct (it read and wrote past every row).

| Operation (ms) | x86 before | x86 SSE4 | Jetson upstream | Jetson NEON |
|---|---:|---:|---:|---:|
| median 3 L | 563.1 | 10.5 (54x) | 442.7 | 16.9 (26x) |
| max 3 RGB | 1180.6 | 36.7 (32x) | 964.0 | 67.7 (14x) |
| histogram RGB | 13.8 | 5.8 (2.4x) | 12.5 | 10.2 (1.2x) |
| chops offset RGB | 22.1 | 1.7 (13x) | 26.8 | 1.9 (14x) |
| frombytes RGB | 6.8 | 5.5 (1.2x) | 3.5 | 2.4 (1.5x) |
| tobytes RGB | 4.0 | 3.2 (1.3x) | 3.6 | 3.1 (1.1x) |

## x86 work ported to arm64 (`a85f87dfb`, `07d3ea375`)

Jetson run at `07d3ea375`, with the upstream column again within 1% of earlier runs and the same 7 operations differing from upstream:
- **Kept: NEON reduce for large factors** (xscale 5..512 or yscale above 16). `reduce RGB 5`: 4.12 → 2.70 ms (1.53x the scalar build); `reduce RGB 8`: 3.92 → 1.93 ms (2.03x).
- **Dropped: L blur four rows at a time** through the 4-byte line code, which gives 2.4–3x on x86. On the Jetson it was 1–5% slower than GCC's 1-byte code (gaussian L 29.95 vs 29.55 ms, box L 10.21 vs 9.77 ms). After the revert both match the scalar build (29.58 and 9.76 ms).

No x86 SIMD code is now missing on arm64.

### NEON float 3×3/5×5 filters

- Used for kernels whose coefficients don't fit 16-bit fixed point, including the built-in `FIND_EDGES`, `CONTOUR` and `EDGE_ENHANCE_MORE`. The kernel (`FilterSIMD_NxNf.c`) works 16 bytes at a time and could also do 1-byte pixels.
- Jetson Orin at `a6d6e1645`:
  - `FIND_EDGES` on RGB: 151.6 → 90.5 ms (1.68x the scalar build).
  - `CONTOUR` on L: 15.5 → 22.9 ms, slower, because GCC already vectorises the generic 1-byte code well.

  Since `a6d6e1645`, NEON is used only for 4-byte images, and L stays on the generic code, so its output equals the scalar build's.
- Each kernel row is a multiply followed by fused multiply-adds, then added to the sum, as GCC compiles the generic code for arm64.
- Results: under emulation, 768 battery outputs were checked against the scalar arm64 build (GCC 12).
  - Kernels with exact coefficients match bit for bit.
  - One inexact 3×3 kernel differs by 1 in 16–40 bytes of a 333×211 image, about 0.05%, because GCC fuses that loop slightly differently.
  - For comparison, the x86 SSE4/AVX2 float kernels differ from x86 scalar by 1 in up to 1.4% of bytes for inexact kernels.
- Exactness here depends on the compiler's fusion choices, which can change between GCC versions. Making it exact would mean compiling the generic code without fused multiply-adds, which would also change arm64 scalar output.

## Newer x86 instruction sets: Xeon Gold 5512U (Emerald Rapids)

Both runs used `checks/simd_bench_remote.sh --docker` (ms, 2560×1600, best of 5). All builds differ from upstream in the same 9 operations, with the same values: the 3 resize rows that also differ in the scalar build, and the known x86 rounding of RGB→L/LA, RGBa→RGBA, alpha_composite and the 5×5 integer filters.

**AVX-VNNI vs AVX2** (`-C simd=avxvnni`: one `vpdpwssd` instead of `pmaddwd` + `paddd`):

| Operation | AVX2 | AVX-VNNI | Gain |
|---|---:|---:|---:|
| filter RGB 5×5 SMOOTH_MORE | 13.57 | 8.41 | 1.61x |
| filter RGB 3×3 SHARPEN | 5.20 | 4.02 | 1.29x |
| filter L 5×5 / 3×3 | 8.77 / 3.51 | 6.88 / 2.85 | 1.27x / 1.23x |
| resize L / RGB ×1.5 lanczos | 13.95 / 27.16 | 12.38 / 25.02 | 1.13x / 1.09x |
| resize RGB 1/4 bicubic | 3.14 | 2.91 | 1.08x |

Unchanged operations are within ±10% from run to run.

**AVX-512 through the compiler** (`-C simd=native`: `-march=native -ffp-contract=off`; `native512`: also `-mprefer-vector-width=512`), against AVX-VNNI:
- `blend RGB`: 4.86 → 3.21 ms with native512 (1.51x); native alone gains nothing.
- `median 3 L`: 5.78 → 5.10 ms with native512 (1.13x).
- `filter RGB 5×5` and ColorLUT: 1.08x with either.
- Everything else within ±3%.

Conclusion: hand-written AVX-512 kernels are not worth their cost. The heavy operations are 6–25x upstream already and near memory speed, and the compiler's AVX-512 found little more. For a build that only has to run on such a Xeon, `-C simd=native512` is the fastest; for portable builds, `avxvnni` (Intel Alder Lake / Sapphire Rapids and newer, AMD Zen 5) or `avx2`.
