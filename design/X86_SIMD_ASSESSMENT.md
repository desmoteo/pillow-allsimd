# x86 SIMD in pillow-simd: defects, improvements and missing coverage

> Written before this fork was named Pillow-AllSIMD: "Pillow-SIMD" here
> means this code base, the fork of uploadcare/pillow-simd.

Companion to [ARM64_SIMD_ASSESSMENT.md](ARM64_SIMD_ASSESSMENT.md) and [ACCELERATION_OPTIONS_ASSESSMENT.md](ACCELERATION_OPTIONS_ASSESSMENT.md). Branch `simd/master` (HEAD `788bf910f`).

Scope: the existing SSE4/AVX2 kernels (correctness, efficiency, AVX2 use) and operations that have no SIMD path yet. Runtime dispatch, AVX-512, threading and codecs are covered in the acceleration document and only referenced here.

## How this was assessed

Unlike the two companion documents, findings here were **measured, not only read**.

- Four builds, all made from `git archive` copies in a scratch directory; the repo was not modified:
  - `sse4`: this branch, default flags (`-O2 -msse4`).
  - `avx2`: this branch, `-O2 -msse4 -mavx2 -mfma`.
  - `base`: parent commit `bb1d8e8ab` (upstream 12.3.0 before the SIMD commits), same toolchain.
  - `upstream`: the installed Pillow 12.3.0 wheel from PyPI (different toolchain).
  - All built without jpeg/zlib, which does not affect the kernels.
- Output comparison: about 100 operations on odd-sized random images (67×23) across modes, diffed byte-by-byte between builds. Plus exhaustive tests where useful (all 65,536 `(value, alpha)` pairs for unpremultiply).
- Benchmarks: 2560×1600 random images on one P-core (Core Ultra 9 285H, `taskset -c 2`).
  - Each figure is the best of 5 interleaved rounds.
  - **The machine is noisy**: the CPU governor is `powersave`, and rounds varied up to 2x. Treat differences under ~15% as noise.
  - Ratios above 1.5x were stable across all runs.
- Existing tests (`test_image_filter`, `test_box_blur`, `test_image_resample`, `test_image_convert`, `test_color_lut`) pass on the `sse4` build. The only failures were 11 tests that need the disabled codecs.

## Verdict

- **The SIMD work delivers where it exists.** Measured against the same-toolchain `base` build:
  - 6–16x on 4-channel resize, 3×3/5×5 filters and RGBa→RGBA.
  - 2–6x on ColorLUT, Gaussian blur, alpha composite, reduce and RGB→L.
- **There are two real correctness bugs and one behavioural regression.** None is caught by the test suite:
  1. BoxBlur/GaussianBlur along a single axis reverses the channel order of every 4-byte mode.
  2. Filters on `I;16` images go to the 8-bit kernel and produce garbage.
  3. SIMD and scalar code in the same function round differently, so **identical pixels give different results depending on their column**. This affects up to 21% of flat rows.
- **The AVX2 tier is worth 1.2–1.5x on the kernels that have it, and is never shipped.** Building it the obvious way (`CFLAGS=-mavx2`) silently produces an `-O0` build that is 3–10x slower.
- **Grayscale (L) is the largest gap.** Resize, reduce and box blur for L have no SIMD at all. `reduce()` on L is **1.8x slower than upstream**, because the SIMD rewrite dropped upstream's 4×4/5×5 specialisations.
- The largest absolute times left are in operations with no SIMD: rotate/affine transforms, median filter, quantize, and the second half of UnsharpMask.

## 1. Correctness defects (all reproduced)

### 1.1 Single-axis BoxBlur / GaussianBlur swaps channels: **bug**
- Where:
  - [BoxBlur.c:22](../src/libImaging/BoxBlur.c#L22) and [BoxBlur.c:133](../src/libImaging/BoxBlur.c#L133) (`ImagingLineBoxBlurZero32`): `shiftmask = _mm_set_epi8(0,…,0, 3, 7, 11, 15)`.
  - `_mm_set_epi8` takes its arguments high byte first, so output byte 0 takes lane 3. Channel order is reversed on every horizontal pass.
- Why tests miss it:
  - A 2-D blur runs `n` passes in x and `n` passes in y, which is always an even number of reversals, so the error cancels.
  - A 1-D blur runs `n` reversals:
    - `BoxBlur((r, 0))` with n=1 is reversed.
    - `GaussianBlur((r, 0))` with 3 passes is reversed.
    - `box_blur((r, 0), n=2)` comes out correct by accident.
- Reproduced: an RGBA pixel `(200, 50, 10, 255)` blurred with `BoxBlur((2, 0))` or `GaussianBlur((0, 2))` comes back as `(255, 10, 50, 200)`. Upstream returns the input unchanged. This affects all 4-byte modes: RGB, RGBA, RGBa, RGBX, CMYK, LA and La.
- Fix: `_mm_set_epi8(-1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 15, 11, 7, 3)` in both functions. Add a test with `(r, 0)` and `(0, r)` radii on a non-grey RGBA colour.

### 1.2 `I;16` filters go to the 8-bit kernel: **bug**
- Where:
  - [Filter.c:504](../src/libImaging/Filter.c#L504) only sends `IMAGING_TYPE_INT32` to the generic `ImagingFilter3x3/5x5`.
  - `I;16*` images have `pixelsize == 2`, so `image8` is set ([Storage.c:224-229](../src/libImaging/Storage.c#L224-L229)). They fall into `ImagingFilter3x3f_u8` / `5x5f_u8`.
  - Those kernels filter the raw bytes of the 16-bit samples over the first half of each row.
- Reproduced: a constant `I;16` image of value 1000 filtered with `SMOOTH` returns 28030 and 11455 at two positions, and other values for `SMOOTH_MORE`, `I;16B` and `I;16L`. There is no out-of-bounds access: an ASan build is clean. The output is simply wrong.
- `test_image_filter.py` already parametrises `I;16*`, but it only checks mode and size, so it passes.
- Fix: `if (im->type == IMAGING_TYPE_INT32 || im->type == IMAGING_TYPE_SPECIAL)` → generic path, matching upstream.
- Note: upstream returns 233 for the same constant-1000 input, which also looks wrong. That is an upstream issue; parity with upstream is the target here.

### 1.3 RGBa → RGBA differs from upstream and from its own scalar tail
- Where: [Convert.c:572](../src/libImaging/Convert.c#L572) `rgba2rgbA`. The SIMD path computes `v * (255.5*256 * rcp_ps(a)) >> 16`; the scalar tail computes `255*v/a`.
- Exhaustive test over all `(v, a)` with `v ≤ a` (valid premultiplied input):
  - The SIMD body differs by ±1 on **6,778 of 32,896 pairs (21%)**.
  - The scalar tail matches upstream exactly.
  - For `a == 0` the SIMD body outputs colour 0; upstream and the tail keep the stored colour.
- Why it matters:
  - `resize`, `reduce` and `transform` on RGBA/LA convert to RGBa and back.
  - Low-alpha pixels amplify a ±1 premultiplied error by up to 255/alpha. On a random-alpha image, bicubic upscale differs from upstream by up to 128 on 64 channels, all at alpha ≤ ~60.
  - Visually negligible, but it is a regression against upstream and depends on column position (1.4).
- Fix:
  - Exact division via a 256-entry table of `ceil(255·2^16 / a)` reciprocals (gather or `pshufb` cannot index 256 entries, but a per-pixel scalar table load of 4 values per vector is cheap).
  - Or use one Newton–Raphson step after `rcp_ps` followed by a correction compare.
  - Handle `a == 0` with a blend.

### 1.4 Column-position-dependent output
Several kernels use different arithmetic in the SIMD body and in the scalar tail. Identical input pixels in one row then produce different outputs. Measured on flat 7-pixel rows, where x 0–3 go through SIMD and x 4–6 through the tail:

| Operation | Rows with non-uniform output | Cause |
|---|---:|---|
| `alpha_composite` | 845 / 4000 (21%) | SIMD: 8-bit coefficient from `rcp_ps`; tail: 7 extra precision bits ([AlphaComposite.c:201](../src/libImaging/AlphaComposite.c#L201)) |
| RGBa→RGBA | 788 / 4000 (20%) | see 1.3 |
| RGB→L / LA / I | 3 / 4000 | SIMD: 15-bit coefficients (9798, 19235, 3735); tail: `L24` 16-bit coefficients ([Convert.c:212](../src/libImaging/Convert.c#L212)) |

The fix is the same each time:
- Make the tail compute exactly what the SIMD body computes, e.g. by reusing the vector code on a 1-pixel load.
- Or, when `xsize ≥ vector width`, finish with one **overlapping** vector over the last N pixels instead of a scalar tail. That is also faster.

Add a test that compares each pixel of a flat image of width 7, 9 and 17 against pixel 0.

### 1.5 Build issues
- **`CFLAGS` replaces, not extends, Python's flags.** setuptools' `customize_compiler` uses `os.environ.get('CFLAGS', cflags)`.
  - A packager who sets `CFLAGS=-mavx2` gets an `-O0` build.
  - Measured: 3–10x slower than the SSE4 build on every operation, including non-SIMD ones like `transpose`.
  - This is the most likely way someone tries to enable AVX2 today. Provide a supported switch instead, e.g. `-C simd=avx2` mapped to `extra_compile_args`, or runtime dispatch (acceleration doc §1.1).
- **macOS/iOS x86_64 probably lose `-msse4`.**
  - [setup.py:641](../setup.py#L641) and [setup.py:652](../setup.py#L652) *assign* `extension.extra_compile_args = ["-Wno-nullability-completeness"]` for every extension when an SDK path is found. That overwrites `-msse4` from [setup.py:1082](../setup.py#L1082).
  - Unless clang's default target already defines `__SSE4_2__`, `ImagingSIMD.h` then includes no intrinsic headers and the kernels fail to compile.
  - Not verified (no Mac here). The fix is `+=`.
- **32-bit Windows.** [ImagingSIMD.h:6](../src/libImaging/ImagingSIMD.h#L6) defines `__SSE4_2__` when `_M_IX86_FP == 2`, which only guarantees SSE2. This builds SSE4.1 code for an SSE2 target.
- **MSVC never gets AVX2.** `-msse4` is ignored by MSVC (D9002 warning), and nothing passes `/arch:AVX2`.

### 1.6 Test gaps
- Both bugs above pass the current suite.
- SIMD/AVX2 equivalence is not tested anywhere.
- The SSE4 and AVX2 builds produced byte-identical output on every operation tested. Keep it that way with a CI job that builds with AVX2 and runs the same tests.

## 2. Measured performance

Times in ms, 2560×1600. Columns:
- **base**: same toolchain, pre-SIMD commit; this isolates the SIMD effect.
- **upstream**: the PyPI wheel, shown for reference. Its toolchain differs, which by itself explains ±30% on untouched code such as Chops and RGB→F.

| Operation | upstream | base | sse4 | avx2 | sse4 vs base | avx2 vs sse4 |
|---|---:|---:|---:|---:|---:|---:|
| resize RGB ¼ bicubic | 15.1 | 15.6 | 2.3 | 1.7 | **6.7x** | 1.35x |
| resize RGBA ¼ bicubic | 25.6 | – | 3.8 | 2.9 | 6.8x (vs upstream) | 1.29x |
| resize RGB ×1.5 lanczos | 101.5 | – | 38.2 | 32.1 | 2.7x (vs upstream) | 1.19x |
| resize **L** ¼ bicubic | 6.9 | 7.3 | 7.6 | 7.3 | **1.0x** | 1.0x |
| resize **L** ×1.5 lanczos | 34.4 | 35.3 | 42.7 | 36.8 | **0.8–0.9x** | 1.16x |
| reduce RGB 4 | 1.69 | 3.73 | 0.82 | 0.92 | 4.5x | 0.9x |
| reduce **L** 4 | 0.52 | 0.52 | 0.94 | 0.93 | **0.55x (regression)** | 1.0x |
| GaussianBlur RGB r=5 | 65.9 | 80.6 | 30.0 | 31.3 | 2.1–2.7x | 1.0x |
| GaussianBlur L r=5 | 21.1 | 46.8 | 21.1 | 21.1 | 2.2x | 1.0x |
| UnsharpMask RGB | 107.8 | 132.5 | 70.2 | 76.1 | 1.6–1.9x | 1.0x |
| filter RGB 3×3 | 59.8 | – | 3.8 | 2.6 | 15.7x (vs upstream) | 1.46x |
| filter L 3×3 | 15.2 | 20.6 | 2.6 | 1.9 | 7.8x | 1.39x |
| RGB→L | 1.39 | 2.18 | 0.82 | 0.97 | 2.3x | 0.85x |
| RGBa→RGBA | 15.1 | 15.8 | 1.24 | 1.14 | 12x | 1.09x |
| alpha_composite | 12.9 | 12.5 | 3.1 | 2.1 | 4x | 1.47x |
| ColorLUT 17³ | 53.5 | 53.1 | 9.3 | 6.0 | 5.7x | 1.54x |
| median 3×3 RGB | 757 | 973 | 909 | 851 | 1.0x | – |
| rotate 17° bicubic | 157 | – | 157 | 149 | no SIMD | – |
| affine bilinear | 41.4 | – | 39.2 | 40.0 | no SIMD | – |
| convert RGB→P adaptive | 4359 | – | 4249 | 3740 | no SIMD | – |

Readings:
- The AVX2 tier helps only where 256-bit code exists: resample, filter, alpha composite, ColorLUT, at 1.2–1.5x.
- Reduce, BoxBlur, Convert RGB→L, Bands and the rest are SSE4-only, so AVX2 changes nothing there.
- L-mode resize and reduce got no SIMD and in places regressed.

## 3. Improvements to existing kernels

Ordered by expected value per effort.

### 3.1 Restore Reduce 4×4 / 5×5 (easy, fixes a regression)
- Upstream `bb1d8e8ab` dispatches `xscale == yscale ∈ {4, 5}` to `ImagingReduce4x4` / `ImagingReduce5x5`. The rewritten [Reduce.c:967](../src/libImaging/Reduce.c#L967) only special-cases ≤3, so L goes through the generic NxN loop and is 1.8x slower.
- Restore both functions. Give the 4-channel versions SIMD bodies too: `reduce(4)` and `reduce(5)` are common thumbnail factors and are used by `thumbnail()` / `draft()`-style pipelines through `reducing_gap`.

### 3.2 Reduce: use the full register width (moderate)
- All 4-channel paths compute **one output pixel per iteration from 64-bit loads**, e.g. `ImagingReduce2x2` at [Reduce.c:427](../src/libImaging/Reduce.c#L427). Half of each 128-bit register is idle and there is no AVX2.
- For 2×2, load 16 or 32 bytes per row and add the rows in 16-bit. Then pair-add adjacent pixels with a `pshufb` + `add`, or `pmaddubsw` with ones on a channel-interleaved shuffle. That gives 4 (SSE) or 8 (AVX2) outputs per iteration.
- Apply the same approach to 1×2, 2×1, 3×3 and NxN rows.
- Expected: 2–3x on the 4-channel reduce. It is memory-bound beyond that.

### 3.3 Grayscale (L) SIMD: resize, box blur, reduce, filter (moderate, high value)
L images (documents, scans, ML preprocessing) get almost no benefit today.
- **Vertical resample is channel-agnostic.**
  - `ImagingResampleVerticalConvolution8u` operates on bytes, so the L vertical pass can reuse it unchanged on `xsize/4` 32-bit words plus a byte tail.
  - The current L vertical loop ([Resample.c:396-409](../src/libImaging/Resample.c#L396-L409)) iterates columns in the outer loop and rows in the inner loop, a strided pattern that is also cache-hostile.
- **Horizontal L resample**: `madd_epi16` over 8 taps at a time against `pmovzxbw` pixels, with one horizontal reduction per output. Alternatively, process 4 rows at once as the 4x variant already does for RGBA.
- **BoxBlur L**: see 3.4. Vectorising across rows works identically for L.
- **Reduce L**: vector `add`s over 16/32-byte rows, then horizontal pair sums with `pmaddubsw(x, ones)`.
- **Filter L**: [FilterSIMD_3x3f_u8.c](../src/libImaging/FilterSIMD_3x3f_u8.c) computes each output with float `hadd` dot products, which costs about 2.7x more per byte than the RGB integer kernel. An integer kernel would be faster:
  - Load 16 pixels per row, convert with `pmovzxbw`, multiply-accumulate each of the 3 or 5 shifted versions by a broadcast 16-bit weight in 32-bit.
  - It follows the existing `fast_path` normalisation in [Filter.c:512](../src/libImaging/Filter.c#L512).

### 3.4 BoxBlur: remove the transposes and vectorise across rows (moderate)
- [BoxBlur.c](../src/libImaging/BoxBlur.c) processes one pixel per iteration. The running sum `acc` is a serial dependency, and `ADD_FAR` does two `mullo_epi32` per pixel, each with ~10-cycle latency on Intel.
- The vertical direction is done by transposing the image twice ([BoxBlur.c:432-439](../src/libImaging/BoxBlur.c#L432-L439)).
- Better:
  - Do the vertical pass directly on rows. The running sum then becomes a vector over x: 16 L pixels or 4 RGBA pixels per `__m128i`, with no transpose, unit-stride access, and naturally wide.
  - Keep the horizontal pass but interleave 2–4 rows to hide the multiply latency.
  - Replace `mullo_epi32` with `madd_epi16` on 16-bit-split weights, or use float.
- Expected: GaussianBlur 2–3x on top of today's, and the same for L, which currently has no SIMD.

### 3.5 Filter AVX2 path is not really 256-bit (easy–moderate)
- [FilterSIMD_3x3i_4u8.c:20-27](../src/libImaging/FilterSIMD_3x3i_4u8.c#L20-L27) loads 128 bits and **duplicates the same data into both lanes**, then uses a different kernel alignment per lane. That yields 2 outputs per step, not 4–8.
- Loading 8 consecutive pixels into a full `ymm`, as the resample kernels do, should give close to 2x over SSE4 instead of the measured 1.4x. The same applies to the 5×5 integer kernel.

### 3.6 Resample (small refinements)
- Vertical AVX2 handles 8 pixels per iteration. 16 pixels per iteration (two loads per row) amortises the `set1` of coefficients and loop overhead, matching what the SSE4 path does with 2×4.
- Horizontal 1-row kernel: the AVX2 `xmax < 8` fallback sets up an SSE accumulator but skips the 4-tap SSE loop, so 4–7-tap filters (bilinear/bicubic upscales) go through the 2-tap loop. Add the 4-tap SSE step.
- AVX-VNNI (`_mm256_dpwssd_avx_epi32`) fuses `madd` + `add` in every resample, filter-int and ColorLUT inner loop. Done as a compile-time option, `-C simd=avxvnni` (`39f7971c4`), without dispatch: integer filters 1.3–2.1x and resize up to 1.13x faster than AVX2, output identical. See [UPSTREAM_13_REVIEW.md](UPSTREAM_13_REVIEW.md).

### 3.7 Convert
- RGB→L/LA/I use `hadd_epi32` (3 uops on Intel) and have no AVX2.
  - Alternative: one `pmaddwd` with the coefficients laid out as `(R,G)(B,0)`, then `pshufb`-pack the two 32-bit partials and add. Or use `pmaddubsw` with 7-bit coefficients in two steps.
  - Also needed for the 1.4 consistency fix: the tail must use the same coefficients.
- Unpremultiply: fix accuracy (1.3) before tuning.
- RGB→YCbCr, RGB→HSV, L→RGB and RGB→F are scalar.
  - YCbCr is the one worth doing (matrix multiply, same pattern as RGB→L). It is used by `Image.convert("YCbCr")` and some encoders.
  - HSV has divisions and branches, so it is only worth SIMD with `rcp`-free float division.

### 3.8 UnsharpMask post-pass (easy)
- After the (SIMD) Gaussian blur, [UnsharpMask.c](../src/libImaging/UnsharpMask.c) runs a scalar per-channel loop with `diff * percent / 100`. That is about half of `UnsharpMask` time (70 ms vs 30 ms for the blur alone).
- Vectorise it as follows:
  - Compute `diff` in 16-bit.
  - Compute `|diff| > threshold` as a mask.
  - Do the multiply-by-percent and exact division by 100 with `mulhi` by the magic constant.
  - Saturate, then `blendv`.
- Expected: about 1.7x on the whole operation.

### 3.9 GIL release
- `AlphaComposite`, `Bands` (split/merge/getchannel), `Blend` and `Chops` never call `ImagingSectionEnter`. Threads therefore cannot run them concurrently.
- Adding the section is a two-line change per function. It matters once the operations are fast enough that Python-side parallelism is the next lever.

## 4. Operations with no SIMD path

Ranked by measured cost and likely usage.

| Operation | Cost (2560×1600) | Approach | Expected |
|---|---:|---|---|
| `rotate` / `transform` / `affine` (bilinear, bicubic) | 40–157 ms | Per output row, source coordinates are linear in x. Compute 4–8 coordinates with vector add, gather the 2×2 or 4×4 neighbourhoods (AVX2 `vpgatherdd` on 32-bit pixels, or scalar loads into vectors), and blend with `pmaddwd` as in resample. Nearest-neighbour is a pure gather. | 2–4x |
| `MedianFilter(3)`, `MinFilter`, `MaxFilter` (RankFilter.c, scalar sort) | 900 ms | Branch-free sorting network with `_mm_min_epu8` / `_mm_max_epu8` over 16/32 pixels at once. The 3×3 median needs 19 min/max pairs, and column sorts can be reused across neighbours. Min/Max is trivially vertical-then-horizontal. | 10x or more for 3×3; standard technique |
| `convert("P", ADAPTIVE)` / `quantize` | 4.2 s | Median cut is not SIMD-shaped. The SIMD-able part is the final nearest-palette mapping (distance to 256 colours). Recommending `Quantize.FASTOCTREE` or libimagequant is likely the bigger win. | Algorithmic, not SIMD |
| `paste` / `composite` with mask (Paste.c) | 7 ms | Same `MULDIV255` blend as alpha composite. | 2x |
| `ImageChops` (add, multiply, screen, difference, …) | 3–13 ms | Each is one or two vector ops (`adds_epu8`, `mulhi`, `subs`/`abs`). The compiler may already auto-vectorise some; check with `-fopt-info-vec`. | 1–3x |
| `resize` / `reduce` / `filter` on `I;16`, `I`, `F` | 6–13 ms | 32-bit float/int convolution with `mul_ps` / `fmadd_ps`. Only worth it if 16-bit or float workflows (scientific, medical) are a target. | 2–4x |
| `point` (LUT), `histogram`, `Bands.merge`, `getbbox`, `transpose` | 1–7 ms | Already near memory speed, or auto-vectorised. Low priority. | small |

## 5. Recommended order

1. **Fix the bugs and add the tests** (1.1, 1.2, 1.6): about 1 day.
2. **Make SIMD and scalar paths agree** (1.3, 1.4): overlapping last vector, exact unpremultiply, same coefficients in the tails. About 2–3 days. This also removes most remaining ±1 differences from upstream other than resample's 16-bit coefficients.
3. **Fix the build pitfalls** (1.5): `+=` on macOS, supported AVX2 switch, MSVC 32-bit guard. About 1 day. Then do runtime dispatch per the acceleration doc.
4. **Restore Reduce 4×4/5×5, then L-mode SIMD** for resize (reuse the vertical kernel), reduce and filter (3.1, 3.3): about 1 week.
5. **BoxBlur restructure, UnsharpMask post-pass, real 256-bit filter path** (3.4, 3.5, 3.8): about 1 week.
6. **New kernels**: median/rank filter, then affine/rotate (section 4). About 1–2 weeks each.
7. **AVX-VNNI and AVX-512VL** tiers once dispatch exists (acceleration doc §1.2). AVX-VNNI now exists as `-C simd=avxvnni`; AVX-512 through the compiler (`-C simd=native512`) was measured on a Xeon and gains little except `blend` (1.5x).

## Caveats

- Benchmarks are from one laptop CPU with a noisy power governor. Use them for direction and order of magnitude; confirm on target servers with `benchmark.yml`.
- "Expected" speed-ups in sections 3 and 4 are estimates and were not prototyped.
- The macOS `-msse4` loss (1.5) is from reading `setup.py`; it was not built on a Mac.
- AVX-512 could not be tested on this CPU. It was later measured on a Xeon Gold 5512U through `-C simd=native`/`native512` (see [UPSTREAM_13_REVIEW.md](UPSTREAM_13_REVIEW.md)).
- Scripts used (output diff, exhaustive unpremultiply test, benchmarks) were kept outside the repo and are not part of this change.

## Implementation checklist

Branch `simd/x86-fixes-arm64-neon`. Decisions (2026-10-22):
- Keep the existing fast approximations (`rcp_ps`, 15-bit RGB→L coefficients).
- Remove **column dependence**: leftover pixels must go through exactly the same arithmetic as the vector body. ±1 differences from upstream Pillow stay.

Every item is done when:
- the change builds with `-C simd=sse4`, `-C simd=avx2` and `-C simd=disable`;
- the SSE4 and AVX2 builds produce byte-identical output;
- a regression test exists for every bug fix.

### Phase 1: bugs
- [x] **1.1 BoxBlur channel order.** Fix `shiftmask` in `ImagingLineBoxBlur32` and `ImagingLineBoxBlurZero32` ([BoxBlur.c](../src/libImaging/BoxBlur.c)).
  - Test: `BoxBlur((r, 0))`, `BoxBlur((0, r))`, `GaussianBlur((r, 0))` on a flat non-grey RGBA image return the input colour, for n = 1, 2, 3 passes.
  - Done in `c4fd9f480`; `test_channel_order` failed 18 times before the fix.
- [x] **1.2 `I;16` filter dispatch.** Send `IMAGING_TYPE_SPECIAL` to the generic `ImagingFilter3x3/5x5` ([Filter.c](../src/libImaging/Filter.c)).
  - Test: `I;16`, `I;16L`, `I;16B` output equals the generic path, i.e. the same result as upstream for a kernel with a known answer.
  - Done in `cf334ba71`.
- [x] **Found during implementation: SSE4 Reduce 1×3 added the rounding term twice** (some outputs 1 too high). Fixed in `bf56b0bfd`, with `test_multiband_matches_single_band`, which checks every 4-byte kernel against the 1-byte code.
- [x] **Found during implementation: the radius-below-1 BoxBlur line functions read `lineIn[-1]` and `lineIn[1]` on 1-pixel lines.** Fixed in `e470d62b6` with `test_one_pixel_line`. All blur results now match upstream.

### Phase 2: build
- [x] **2.1 Feature macros.** `ImagingSIMD.h` defines `IMAGING_SIMD_SSE4`, `IMAGING_SIMD_AVX2` and `IMAGING_SIMD_NEON`; kernels test these instead of compiler macros.
  - MSVC: SSE4 only for `_M_X64`, not for `_M_IX86_FP == 2`.
- [x] **2.2 Unaligned loads.** Replace the type-punned load in `mm_cvtepu8_epi32` with `memcpy` (plus `mm_load_si32` for the vertical resample kernel).
- [x] **2.3 Build options.**
  - `-C simd=auto|sse4|avx2|neon|disable` in `setup.py` adds the matching flags to `extra_compile_args` of `PIL._imaging` only, so the optimisation flags are kept.
  - `auto` means SSE4.1 (`-msse4.1`, was `-msse4`) on x86, NEON on arm64 and nothing elsewhere. The build summary prints the choice.
  - Documented in `README.md`.
- [x] **2.4 macOS/iOS.** `extra_compile_args +=` instead of `=` ([setup.py](../setup.py), SDK branches). Not built on a Mac here.
- [x] **2.5 Scalar fallback.** Every SIMD site has a scalar path (shared with the arm64 work, see ARM64 checklist phase 1).
  - `-C simd=disable` on x86 passes the full test suite.

### Phase 3: column-independent results
- [x] **3.1 Shared tail helper.** Leftover pixels (`xsize % N`) are copied into a zero-padded stack buffer, run through the same vector body, and copied back (`CONVERT_ROW` in Convert.c, inline in AlphaComposite.c).
- [x] **3.2 Convert.** Applied to `rgb2l`, `rgb2la`, `rgb2i`, `rgba2la` and `rgba2rgbA`. `rgbA2rgba` is exact (`MULDIV255`), so its scalar tail already matched.
- [x] **3.3 AlphaComposite.** Applied 3.1.
- [x] **3.4 Tests.** `Tests/test_simd_consistency.py`: flat images of widths 1–17 give flat output for these conversions, alpha composite, and the interior of 3×3/5×5 filters. 12 cases failed on the original branch.

### Phase 4: performance
- [x] **4.1 Reduce.** Reduce.c is upstream's again, which restores `ImagingReduce4x4` / `ImagingReduce5x5`. L `reduce(4)` is 1.8x faster than before (same as `base`). The SSE4 4-byte kernels moved unchanged to `ReduceSIMD.c`.
- [x] **4.2 L vertical resample.**
  - `ImagingResampleVerticalConvolution8u` takes the width in 32-bit words and reads rows through `image[]`.
  - Used for 1-byte images, with a byte tail; output unchanged.
- [x] **4.3 GIL.** `ImagingSectionEnter/Leave` around the pixel loops in AlphaComposite, Bands (split, getband, merge, putband, fillband), Blend and Chops.
- [x] **4.4 UnsharpMask post-pass in SIMD.**
  - SSE4 and NEON, 16 bytes per step, float division by 100 while `|percent| ≤ 65535`; scalar otherwise.
  - Output identical to the scalar loop for 40 percent/threshold combinations; 82 → 47 ms on a 2560×1600 RGB image.
- [x] 4.5 Reduce 4-channel paths: four outputs per iteration for 1×2 (`pavgb`), 2×1 and 2×2 (`pshufb` + `pmaddubsw`), about 2x faster. Other factors keep one output per iteration.
- [x] 4.6 Filter AVX2 path: measured and dropped for 4-byte images. The duplicated lanes carry different kernel alignments, so each 256-bit multiply-add already makes two output pixels; 8 consecutive pixels per step (each lane running the SSE4 scheme) saves shuffles but not multiply-adds, and was 0.93x (3×3) and 1.02x (5×5) with AVX2, 0.89x and 0.46x with AVX-VNNI. The premise in 3.5 was wrong. L images got a 256-bit path instead (32 pixels per step, `FilterSIMD_NxNi.c`): 2x faster than SSE4, see [CALLGRIND_PROFILE.md](CALLGRIND_PROFILE.md).
- [x] 4.7 BoxBlur: the vertical pass works on rows (no transposes), 16 bytes per step with SSE4/NEON. GaussianBlur on L 1.8x faster, BoxBlur on RGB 1.5x, GaussianBlur on RGB 1.1x: the per-pixel horizontal passes now dominate. Later, 1-byte images were blurred four rows at a time through the 4-byte line code (2.4–3x, `b913b10e8`); two RGB rows per AVX2 vector was measured and dropped ([CALLGRIND_PROFILE.md](CALLGRIND_PROFILE.md)).
- [x] 4.8 L-mode integer 3×3/5×5 filter kernel (`FilterSIMD_NxNi.c`, shared with NEON). 1.4x (3×3) and 2x (5×5) faster than the float kernels; x86 and arm64 L filter output is identical.
- [x] 4.9 New kernels: 3×3 rank filters (median and any rank) via a sorting network, and min/max of any size by rows then columns, in portable C that compilers vectorise (`bc77919ef`), 30–130x faster. **Affine/rotate bilinear is not done.**

### Phase 5: verification
- [x] Output diff harness (SSE4 vs AVX2 vs scalar vs upstream vs arm64) run after each change. Final state:
  - SSE4 and AVX2: byte-identical on every operation tested.
  - Scalar build vs upstream Pillow: identical except resize, which keeps pillow-simd's 16-bit coefficients.
  - SSE4 vs scalar: ±1 from the kept approximations (`rcp_ps` in unpremultiply and alpha composite, 15-bit RGB→L coefficients, float 4-byte filters with large coefficients, ColorLUT).
- [x] Full test suite passes on amd64 SSE4, AVX2 and scalar builds (Docker image with all codecs).
- [x] Benchmark table re-run for changed kernels (figures above). The machine had heavy unrelated load in the later runs, so only interleaved before/after ratios are quoted.
- [x] CI: `simd` matrix dimension with AVX2 and scalar jobs on `ubuntu-latest` (prepared, not run).
