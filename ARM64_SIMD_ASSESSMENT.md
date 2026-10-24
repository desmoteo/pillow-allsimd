# Extending pillow-simd to arm64: feasibility assessment

Branch assessed: `simd/master` (HEAD `788bf910f`). Assessment is from static code review on x86_64; nothing was built or run on arm64.

## Verdict

**Feasible, and NEON is a good fit for nearly every kernel. It is a port, not a recompile.**

- Today the SIMD code is x86-only and **unconditionally so**. Building on arm64 fails outright.
- Roughly 7,000 lines across 14 files use SSE4.1/AVX2 intrinsics directly.
- Most of them have no scalar fallback, because the SIMD commits replaced the original C.
- The mechanical porting work is moderate (about 4–6 weeks for one engineer including testing; see the plan).
- Two blockers need a decision first, in the "Blockers" section.

## Current state

### Build and dispatch
- [setup.py:1082](setup.py#L1082) hard-codes `extra_compile_args=["-msse4"]` for `PIL._imaging`. GCC/Clang on arm64 reject this flag (`unrecognized command-line option`), so the build fails before any code is compiled.
- [ImagingSIMD.h](src/libImaging/ImagingSIMD.h) is the only intrinsics gate.
  - It defines `__SSE4_2__` for MSVC (`_M_X64`) and includes `emmintrin.h`, `smmintrin.h` and, if `__AVX2__`, `immintrin.h`.
  - It has no arm branch, and no `arm_neon.h`.
- There is **no runtime CPU dispatch**.
  - SSE4 is baked in at compile time.
  - The AVX2 paths only compile if the builder passes `-mavx2` or `-march=native`; `setup.py` does not.
  - A single wheel therefore never contains AVX2 code unless the packager adds it.

### Guarding and fallbacks per file

| File | Lines | Intrinsic lines | `#if` guards | Scalar fallback | Notes |
|---|---:|---:|---|---|---|
| Convert.c | 1969 | 113 | SSE4 and AVX2 | partial (5 `#else`) | RGB→L, RGBa↔RGBA, etc. |
| Reduce.c | 991 | 179 | none | **none** (~1000 scalar lines deleted) | SSE4 only |
| ColorLUT.c | 559 | 149 | AVX2 only | none | SSE4 path unguarded |
| BoxBlur.c | 480 | 22 | none | none | |
| ResampleSIMDHorizontalConv.c | 368 | 157 | AVX2 | yes | included from Resample.c |
| ResampleSIMDVerticalConv.c | 255 | 130 | AVX2 | yes | |
| Bands.c | 344 | 25 | none | none | byte shuffles |
| AlphaComposite.c | 225 | 72 | AVX2 only | none | |
| FilterSIMD_3x3f_u8 / 5x5f_u8 | 159 / 100 | 69 / 42 | none | none | float |
| FilterSIMD_3x3f_4u8 / 5x5f_4u8 | 237 / 126 | 71 / 32 | partial | partial | float |
| FilterSIMD_3x3i_4u8 / 5x5i_4u8 | 222 / 250 | 103 / 125 | AVX2 | yes | integer, `madd` heavy |

Consequences:
- Reduce, Bands, BoxBlur, AlphaComposite, ColorLUT and several Filter kernels would need a **new scalar or NEON implementation** on arm64.
- Resample and some Filter variants keep their original scalar code under `#else`, so they are the easiest first step.
- The original upstream scalar code for the removed files is still in git history before commits `cb354cc45`..`04a1bb587`. It can be restored as the fallback.

## Intrinsic mapping

The intrinsics used are a small set, and nearly all have direct NEON equivalents.

| x86 (used here) | NEON / AArch64 | Difficulty |
|---|---|---|
| `add/sub/mullo_epi16/32`, `and`, `or`, `set1`, `setzero`, loads and stores | `vaddq`, `vsubq`, `vmulq`, `vandq`, `vorrq`, `vdupq_n`, `vld1q`, `vst1q` | trivial |
| `unpacklo/hi_epi8/16/32` | `vzip1q/vzip2q` | easy |
| `packus_epi16`, `packs_epi32` | `vqmovun_s16`, `vqmovn_s32` (the two-input x86 form becomes `vcombine` of two narrows) | easy |
| `srli/slli/srai` | `vshrq_n`, `vshlq_n` | easy |
| `cvtepu8_epi32/16` | `vmovl_u8` / `vmovl_u16` | easy |
| `cvtepi32_ps`, `mul_ps`, `add_ps` | `vcvtq_f32_s32`, `vmulq_f32`, `vaddq_f32` | easy |
| `cvtps_epi32` | `vcvtnq_s32_f32` (round to nearest even, matches x86 default) | easy, must pick the rounding variant |
| `shuffle_epi8` (pshufb) | `vqtbl1q_u8` | easy, but **index semantics differ**: x86 zeroes on bit 7, NEON zeroes on out of range (>15) index. Table constants using `0x80` for "zero" must be remapped (e.g. to `0xFF`). Heavy use in Bands, BoxBlur, Convert, Resample, ColorLUT. |
| `blendv_epi8`, `blend_epi16` | `vbslq` | easy |
| `shuffle_epi32` | `vgetq_lane` or `vextq` / `vrev64q`, or `vqtbl1q_u8` | easy-moderate |
| `madd_epi16` | `vmull_s16` + `vpaddq_s32` (or `vmlal_s16`) | moderate. Hot in Resample, Filter-int, ColorLUT. |
| `mulhi_epu16` | `vmull_u16` + `vshrn_n_u32(...,16)` | moderate |
| `hadd_epi32`, `hadd_ps` | `vpaddq_s32`, `vpaddq_f32` | easy (note different lane order than x86, which is per-128-bit-lane) |
| `srli_si128` / `slli_si128` | `vextq_u8` with a zero vector | easy |
| `_mm_rcp_ps` | `vrecpeq_f32` (+ one `vrecpsq_f32` Newton step) | **semantic risk**, see below |
| `_mm256_*` (AVX2) | no direct equivalent (NEON is 128-bit) | see below |

### AVX2 paths
- NEON is 128-bit, so the AVX2 variants should not be ported.
- On arm64 the SSE4-equivalent (128-bit) path is the target, written as NEON.
- If the AVX2 sections are guarded behind `__AVX2__`, they are naturally excluded on arm64. This only holds if the 128-bit path exists for every kernel, which today it does not (AlphaComposite and ColorLUT guard the AVX2 path but rely on the SSE4 path being unguarded).
- SVE/SVE2 is possible on newer cores but not worth the cost. Apple Silicon has no SVE, and cloud arm64 (Graviton3+/Neoverse V1+) has 256-bit SVE only on some SKUs.

## Blockers and risks

1. **Build system must stop assuming x86.**
   - Make the compile flag conditional on architecture (`-msse4.1`/`-msse4` on x86, none on arm64 since NEON is baseline on AArch64).
   - Prefer `-msse4.1` over `-msse4`; the code uses nothing newer than SSE4.1 (`_mm_cvtepu8_epi32`, `blendv`, `mullo_epi32`, `packus_epi32`).
   - Check `platform.machine()` and `sys.platform`, also for universal2 macOS builds (clang `-arch x86_64 -arch arm64`): flags must be per-arch, not global.
   - Needs a build-system decision (setuptools compile flags per architecture).

2. **Numerical and test parity.**
   - `_mm_rcp_ps` (AlphaComposite, Convert RGBa→RGBA) is an approximate reciprocal with ~12 bit precision. `vrecpeq_f32` is similar but not bit-identical, and even Intel and AMD differ.
   - Outputs may differ by ±1 level between x86 and arm64.
   - Commit `cb354cc45` already relaxed tests (`assert_image_equal` → `assert_image_similar`) for SIMD differences, so a tolerance precedent exists.
   - Recommended: use a Newton-refined reciprocal or exact division on arm64 (NEON division is fast enough on modern cores), and decide whether cross-arch bit-exactness is a requirement. If it is, the float paths are the hardest to guarantee.
   - Rounding: `cvtps_epi32` rounds to nearest even. Use `vcvtnq`, not `vcvtq` (truncate).

3. **Alignment / aliasing in helpers.**
   - `mm_cvtepu8_epi32(void *ptr)` in `ImagingSIMD.h` does `*(INT32 *) ptr`, which is unaligned type-punning.
   - It works on x86 and mostly on arm64 (unaligned loads are allowed on AArch64 normal memory) but it is undefined behaviour and can miscompile under strict aliasing. Replace with `memcpy` or `vld1q_lane_u32`.

4. **Endianness / lane order.**
   - Several kernels use `_mm_set_epi8` / `_mm_set_epi16` constants written high-lane-first.
   - These must be re-expressed carefully for NEON (`vld1q` from a little-endian array). This is the most common source of porting bugs, so each kernel needs a bit-exact comparison test against the scalar reference.

5. **Over-reads.**
   - Some x86 kernels load 16 bytes where only 12 or 8 are valid (e.g. `_mm_loadu_si128` on 3-byte pixels), relying on the image buffer padding.
   - NEON has no behavioural difference here, but any port must keep the same invariants. Valgrind/ASan runs on arm64 are advisable. The repo already has `test-valgrind*.yml`.

6. **Maintainability.**
   - The code uses raw intrinsics inline. A second full implementation per kernel roughly doubles the surface area.
   - Mitigation: a thin portability header (see plan) so kernels are written once against a small `v128` API, with x86 and NEON backends.

## Per-kernel assessment

Ordered by expected benefit and ease. Speed-up factors are the x86 gains reported in the commit history; NEON typically reaches similar relative gains over scalar since both are 128-bit.

| Kernel | NEON effort | Notes |
|---|---|---|
| Resample (H and V, 8-bit) | Low–medium | Scalar fallback exists. Core is `madd_epi16`, `packs`, `pshufb`. Biggest user-visible win (resize). Do first. |
| Reduce | Medium | Largest file; simple adds, unpack and shifts; no fallback so port or restore scalar. |
| Filter 3x3/5x5 (int and float) | Medium | Fallbacks partially exist. `madd` and `hadd` patterns. |
| BoxBlur | Low | Only 7 distinct intrinsics, mostly `pshufb` plus 32-bit multiply. |
| Bands (split/merge) | Low | Pure byte shuffles: `vqtbl`, or better `vld3q_u8` / `vst4q_u8` for de/interleave, which is *faster* than the x86 approach. |
| Convert | Medium | RGB→L via 16-bit multiplies; RGBa→RGBA uses `rcp`, see risk 2. `vld3q_u8` helps. |
| AlphaComposite | Medium | `rcp_ps`, `mullo_epi32`, shuffles. |
| ColorLUT | High | 149 intrinsic lines, gather-like index logic, 16-bit fixed point, `madd`/`mulhi`. Last. |

## Recommended plan

1. **Make the build arch-aware** (setup.py flags; CI job on `ubuntu-24.04-arm` and `macos` arm64 already exist in `wheels.yml`). Goal: arm64 builds and passes the existing test suite using scalar code only.
2. **Restore scalar fallbacks** for the files that have none (Reduce, BoxBlur, Bands, AlphaComposite, ColorLUT, the f_u8/f_4u8 filters). Take them from git history before the SIMD commits and put them under `#else` branches. After this step the package is correct on any architecture.
3. **Fix `ImagingSIMD.h`.**
   - Add `#elif defined(__ARM_NEON) || defined(__aarch64__)` → `#include <arm_neon.h>` and define a feature macro (for example `IMAGING_SIMD_NEON`).
   - Replace the type-punned loads.
   - Keep `__AVX2__` strictly x86.
4. **Port kernels in the order above**, one per commit. For each one:
   - Add a bit-exactness test against the scalar path, over random data including odd widths, tails and all modes.
   - Do not change tolerances in existing tests.
5. **Optional:** introduce a small portability layer for shared primitives (`pshufb` remap, `madd`, `hadd`, `rcp`) to avoid duplicating logic.
6. **Benchmark** on Apple M-series and Graviton or Neoverse using the repo's `benchmark.yml` against upstream Pillow (which already auto-vectorises some paths), and drop any kernel whose NEON version isn't clearly faster. Compilers on arm64 often auto-vectorise simple loops well, so the benefit for trivial kernels (Bands, BoxBlur) may be small.

## Effort estimate

| Phase | Estimate |
|---|---|
| 1–3 (build, fallbacks, header) | ~3–5 days |
| Resample, BoxBlur, Bands | ~1 week |
| Reduce, Filter, Convert, AlphaComposite | ~2 weeks |
| ColorLUT | ~1 week |
| Testing, benchmarks, CI, wheels | ~1 week, in parallel with porting |

## Open decisions

1. Is cross-architecture **bit-identical** output required, or is ±1 tolerance acceptable (affects `rcp` handling)?
2. Should wheels ship **runtime dispatch** (SSE4/AVX2, NEON) or stay compile-time only as today? Runtime dispatch would also fix the fact that AVX2 is never in the shipped wheels.
3. Is a portability layer acceptable, or should kernels stay as separate `#if` blocks with raw intrinsics (matches the existing style, easier diffs against upstream)?

## Caveats

- No arm64 hardware was used. Instruction mappings and effort figures are from reading the code and from NEON documentation, not from measurement.
- The line counts and intrinsic inventory were obtained with grep over `src/`, and are reliable. The effort and speed-up estimates are judgement calls.

## Implementation checklist

Branch `simd/x86-fixes-arm64-neon`. Resolving "Open decisions" above (2026-10-22). Item 1 and the choice of native NEON were decided by the maintainer; items 2 and 3 are this branch's defaults and can be revisited:
1. **Exactness:** cross-architecture bit-exactness is *not* required. Within one build, results must not depend on pixel position. NEON may use `vrecpeq_f32` where x86 uses `rcp_ps`.
2. **Runtime dispatch (default):** not in this branch. Compile-time selection, with `-C simd=…` (see the x86 checklist, 2.3).
3. **No portability layer (default).** Kernels stay as separate `#if IMAGING_SIMD_SSE4 / #elif IMAGING_SIMD_NEON / #else` blocks with raw intrinsics, matching the existing style.

Test environment: the `python:3.12` arm64 container under qemu-user (binfmt). It runs the real extension and the test suite. Speed under emulation means nothing, so performance needs real hardware (CI `ubuntu-24.04-arm`, Apple M-series).

Every item is done when:
- the arm64 build passes the SIMD-related tests;
- arm64 output matches the x86 SSE4 build byte-for-byte, except kernels documented as using approximate reciprocals, which may differ by at most ±1.

### Phase 1: builds and passes everywhere (scalar)
- [x] **1.1 Arch-aware flags.** `setup.py` picks the flags from the target architecture (`sysconfig.get_platform()`, or `ARCHFLAGS` on macOS): `-msse4.1` on x86, none on arm64, none for universal2.
- [x] **1.2 `ImagingSIMD.h`.** `#include <arm_neon.h>` and define `IMAGING_SIMD_NEON` for `__aarch64__` / `_M_ARM64`.
- [x] **1.3 Scalar fallbacks** (done in `95f8f8035`, differently from plan where noted):
  - [x] AlphaComposite: upstream code.
  - [x] Bands: upstream code.
  - [x] BoxBlur: a small `px4` abstraction (`ImagingPx4.h`) with SSE4, NEON and scalar versions giving identical results, rather than a separate copy.
  - [x] ColorLUT: upstream code.
  - [x] Convert (`rgb2l`, `rgb2la`, `rgb2i`, `rgba2la`, `rgbA2rgba`, `rgba2rgbA`): upstream code.
  - [x] Filter: generic 3×3/5×5 for all modes.
  - [x] Reduce: the whole file is upstream's again; the SSE4 4-byte kernels moved to `ReduceSIMD.c`.
  - [x] Resample: portable kernels with the same 16-bit coefficients as the SIMD ones, not upstream's 32-bit ones, so results match the SIMD builds.
- [x] **1.4 Verified:**
  - x86 with `-C simd=disable` passes the full test suite.
  - arm64 (qemu container) builds and passes the full test suite with `-C simd=disable`.

### Phase 2: NEON kernels, in priority order
Each kernel has its own commit and a comparison against the x86 output on the diff harness (odd sizes, all modes).

- [x] **2.1 Resample** horizontal (1-row and 4-row) and vertical, `vmlal_n_s16` on widened pixels. **Identical to x86 SSE4.** (`a01049b05`)
- [x] **2.2 Bands.** `split`/`getband` via `vld4q_u8`, `merge` via `vst4q_u8`.
- [x] **2.3 Convert.** (`48930a083`)
  - RGB→L/LA/I and RGBA→LA: `vld4q_u8` with the SSE4 15-bit coefficients, **identical to x86**. Last pixels via padded buffers.
  - RGBA→RGBa premultiply: exact `MULDIV255` (`vraddhn_u16`).
  - RGBa→RGBA: **exact** float division (`vdivq_f32`), not `vrecpeq_f32`, so it matches upstream; x86 SSE4 still uses `rcp_ps` (±1).
- [x] **2.4 Reduce:** NEON for 1×2, 2×1 and 2×2, the common thumbnail factors (`vpaddlq`/`vpadalq` + rounding narrow, `vrhaddq`). Other factors use the scalar code. **Identical to upstream.** (`bf56b0bfd`)
- [x] **2.5 Filter:** integer 3×3/5×5 for 4-byte and 1-byte images (`FilterSIMD_NxNi.c`, shared with x86 for 1-byte). Large coefficients use the generic code. **Identical to x86 SSE4.** (`30f0a1d50`, `b94b94ee2`)
- [x] **2.6 BoxBlur:** horizontal pass via `px4`, vertical pass 16 bytes per step. **Identical to upstream.** (`e470d62b6`)
- [x] **2.7 AlphaComposite:** eight pixels per step with upstream's **exact** integer arithmetic (division in `float64x2_t`), not `vrecpeq_f32`. Verified equal to the scalar code for all 65,536 alpha pairs. (`8e2a6db5c`)
- [ ] **2.8 ColorLUT.** **Not done**; arm64 uses the upstream scalar code.
- [x] **Also:** UnsharpMask post-pass (`47dad2b12`), merge (`4d508867b`).

Remaining cross-architecture differences (arm64 vs x86 SSE4), all by design:
- `rcp_ps`-based unpremultiply and alpha composite on x86 vs exact on arm64 (±1).
- ColorLUT: SIMD on x86 vs upstream scalar on arm64 (±1).
- Float filters (only when coefficients don't fit 16 bits): GCC fuses multiply-adds on AArch64 by default (`-ffp-contract=fast`), x86 without `-mfma` doesn't (±1). Upstream Pillow behaves the same.

### Phase 3: CI and wheels
- [x] CI jobs on `ubuntu-24.04-arm` (NEON and scalar) in `test.yml`. `macos-latest` (Apple Silicon) now also builds NEON. **Prepared, not run.**
- [ ] Benchmarks on real arm64 hardware vs upstream Pillow. Drop any NEON kernel that is not faster than scalar. **Not done: no arm64 hardware here. qemu timings are meaningless.**
