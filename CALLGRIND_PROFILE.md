# Callgrind profile of pillow-simd on x86

Companion to [UPSTREAM_13_REVIEW.md](UPSTREAM_13_REVIEW.md). Goal: find what is left to optimise after step 2 (`e08321aae`), using instruction counts, cache misses and per-line costs instead of only wall time.

## Method

- Builds of `e08321aae` at `-O2 -g` with GCC 13.3: scalar (`-C simd=disable`), SSE4, AVX2. Upstream is the Pillow 12.3.0 wheel from PyPI.
- `checks/simd_callgrind.py` runs every operation of `checks/simd_bench.py` (2560×1600) under callgrind with cache and branch simulation. Only the operation is counted: instrumentation is switched on for one call after a warm-up call.
- Machine: Intel Core Ultra 9 285H. Native times come from `checks/simd_bench.py`, best of 7 rounds, pinned to one P-core (5.4 GHz, 48 KB L1d; callgrind simulated the same L1 size).
- Valgrind's limits:
  - Its branch model keeps no history, so it counts a mispredict at every exit of a short loop, which real CPUs predict.
  - Its cache model has no prefetcher.
  - So instruction counts are the primary signal here. Miss counts are a guide to where data does not fit in cache, not a time estimate.

## Results

Instructions in millions (Ir) and native time in ms.

| Operation | upstream Ir | SSE4 Ir | AVX2 Ir | upstream ms | SSE4 ms | AVX2 ms |
|---|---:|---:|---:|---:|---:|---:|
| resize RGB 1/4 bicubic | 426 | 74 | 47 | 33.4 | 5.0 | 3.3 |
| resize RGB x1.5 lanczos | 2218 | 413 | 274 | 204.4 | 58.3 | 49.8 |
| resize RGBA 1/4 bicubic | 618 | 104 | 60 | 56.4 | 7.7 | 5.3 |
| **resize L 1/4 bicubic** | 195 | 175 | 174 | 15.4 | 13.4 | 13.3 |
| **resize L x1.5 lanczos** | 1066 | 537 | 514 | 76.1 | 39.2 | 37.3 |
| reduce RGB 2 | 55 | 10 | 10 | 4.2 | 1.0 | 1.0 |
| reduce RGB 4 | 39 | 30 | 28 | 3.7 | 1.8 | 1.7 |
| **reduce L 2** | 25 | 25 | 25 | 1.6 | 1.6 | 1.6 |
| **reduce L 4** | 13 | 13 | 13 | 1.2 | 1.2 | 1.2 |
| **gaussian RGB r=5** | 1577 | 618 | 565 | 139.9 | 53.1 | 53.2 |
| gaussian L r=5 | 600 | 332 | 322 | 47.5 | 26.9 | 26.9 |
| box RGB r=2 | 556 | 195 | 178 | 54.3 | 17.5 | 17.5 |
| unsharp RGB | 1921 | 696 | 630 | 228.3 | 61.5 | 62.7 |
| filter RGB 3x3 SHARPEN | 793 | 115 | 65 | 132.5 | 8.2 | 5.4 |
| filter RGB 5x5 SMOOTH_MORE | 2093 | 311 | 151 | 168.1 | 22.5 | 18.8 |
| filter L 3x3 SHARPEN | 246 | 64 | 60 | 48.8 | 4.1 | 4.2 |
| filter L 5x5 SMOOTH_MORE | 592 | 148 | 137 | 53.8 | 10.2 | 10.3 |
| convert RGBa->RGBA | 207 | 27 | 11 | 33.2 | 2.0 | 1.7 |
| alpha_composite | 257 | 64 | 27 | 24.8 | 6.4 | 3.5 |
| split RGBA | 50 | 19 | 16 | 3.3 | 2.9 | 2.9 |
| **merge RGB** | 13 | 14 | 7 | 1.4 | 1.6 | 1.4 |
| colorlut RGB 17 | 1054 | 231 | 140 | 101.3 | 20.3 | 13.0 |
| **blend RGB** | 69 | 58 | 30 | 6.6 | 6.6 | 3.7 |
| chops multiply | 201 | 30 | 12 | 18.6 | 2.7 | 2.4 |

The full tables, including the scalar build, cache misses per operation and every remaining operation, come out of `checks/simd_callgrind.py`.

## Findings, by expected gain

### 1. Horizontal resampling of 1-band images is scalar
- `resize L 1/4 bicubic`: 97% of instructions are in `_ImagingResampleHorizontal_8bpc`'s plain tap loop, one multiply-add per tap, about 17 taps per output pixel. `resize L x1.5 lanczos`: 88%.
- The SIMD horizontal kernel (`ImagingResampleHorizontalConvolution8u4x`) only handles 4-byte pixels. The vertical pass is already SIMD for every mode.
- Fix: an SSE4/AVX2/NEON horizontal kernel for 1-byte pixels. Load 8 or 16 taps with `pmaddubsw`/`pmaddwd` against the 16-bit coefficients, then reduce horizontally, or compute 4 output pixels at once from overlapping windows.
- Expected: 2–4x on `resize L` (13 ms and 39 ms today). Same correctness rule as the RGB kernel: exact match with the scalar fixed-point code.

### 2. Box blur: vertical pass misses L1, horizontal pass is one pixel at a time
- In `gaussian RGB r=5`, the vertical pass (inlined in `ImagingBoxBlur`) has 5.4M of the 6.2M L1 read misses.
  - Each output row reads two rows and writes one (10 KB each) and updates a 40 KB row of 32-bit accumulators. That is 70 KB, more than the 48 KB L1.
  - Fix: process the vertical pass in column strips of about 512 pixels, so a strip's accumulators and rows stay in L1. Results do not change.
- The horizontal pass is 53% of the instructions. This is item 4.4 of the review, unchanged.
- Expected: 10–30% from the strips (to be measured), up to 2x with the horizontal rework. Gaussian, box blur and unsharp mask all benefit.

### 3. Blend is not vectorised on SSE4
- `ImagingBlend` computes `in1 + alpha * (in2 - in1)` in float for every byte. GCC vectorises it with `-mavx2` (3.7 ms) but not with `-msse4.1` (6.6 ms, same as upstream).
- Fix: an SSE4 kernel in float with the same order of operations. It gives the same bytes as the scalar code, since x86-64 SSE has no implicit FMA.
- Expected: about 6.6 → 3.5 ms on SSE4. The AVX2 build already gets this.

### 4. Lanczos upscaling: vertical pass misses L1
- `resize RGB x1.5 lanczos` runs at 7 instructions per ns, half of the other resizes. The vertical pass has 3.5M of the 4.3M L1 read misses.
- Each output row reads about 7 intermediate rows of 15 KB. Consecutive output rows share most of those rows, but the set doesn't fit in L1.
- Fix: column strips, as in item 2. Expected 10–20%, to be measured.

### 5. Reduce of 1-band images is scalar
- `reduce L 2` and `reduce L 4` run the same scalar code in every build, at 1.6 and 1.2 ms. RGB has SIMD (4x).
- Fix: SSE4/NEON 2×2 with `pmaddubsw` by ones and a vertical add, and 4×4 by extending that.
- Small in absolute terms, but `reduce` is used by `thumbnail()` with `reducing_gap`, where L images are common.

### 6. Merge RGB is 15% slower on SSE4 than the scalar build
- Same instruction count, but GCC builds `MAKE_UINT32` from `pmovzx*` and `psrldq` under `-msse4.1` instead of `punpck*`, and those all compete for the shuffle port.
- Fix: an explicit unpack kernel (`punpcklbw`/`punpcklwd`), as NEON already has with `vst4q_u8`. Saves about 0.4 ms.

### Measured and dropped
- **Force-inlining `ImagingFilterNxN_simd` per size.** At `-O2`, GCC shares one copy between 3×3 and 5×5, and the tap loops stay loops. Forcing the inline cut instructions by 19% (3×3) and 3% (5×5), but native time improved only 4% and 0%. The kernel is bound by its vector multiply-adds, not loop overhead, so not worth a change.

### Not worth pursuing
- `split RGBA`, `convert RGB->L` and chops are close to memory bandwidth: 15–25% of their data references miss the last-level cache, and they run at 2–3 ms for 16–48 MB of traffic.
- `colorlut` spends its time in table gathers; it is already 5–8x upstream.
