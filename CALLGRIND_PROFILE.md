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

## Outcome

Each item was implemented, checked for byte-identical output against the scalar code, and timed. Timing used alternating A/B runs: the median ratio over 10–16 pairs, pinned to one P-core. The machine was shared with other heavy jobs at the time, so single-run ratios ranged widely; medians below 15% are treated as noise.

### Kept

| Item | Change | SSE4 | AVX2 |
|---|---|---:|---:|
| 1 | `resize` on L: horizontal kernel for 1-byte pixels; SSE4 does 4 rows at once, NEON 1 row | 1/4 bicubic 8.0x, ×1.5 lanczos 2.6x, ×2 bicubic 2.4x | 8.3x, 2.8x |
| 5 | `reduce` on L: 2×2 and 4×4 kernels, SSE4 and NEON | 2×2 7.3x, 4×4 6.5x | 7.7x, 6.4x |
| 2 (part) | Box and Gaussian blur on L: the horizontal pass runs 4 rows at once through the 4-byte code (SSE4 builds) | Gaussian 2.4x, box 3.0x | 1.9x, 1.9x |

- Instruction counts: `resize L 1/4` 175M → 27M, `resize L x1.5 lanczos` 537M → 222M, `reduce L 2` 25M → 1.7M, `gaussian L r=5` 332M → 156M.
- Output:
  - x86: identical to the previous builds on about 2,800 battery outputs (all filters, boxes, odd sizes, widths 1 to 1031) and in the full test suite.
  - arm64 under emulation: the NEON resize and reduce kernels match the scalar arm64 build on the same battery.
- New tests compare each L operation with band 0 of the same operation on an RGB image made of three copies. They pass on stock Pillow 12.3 too.

### Measured and dropped

- **3, Blend on SSE4.** The premise was wrong: GCC already vectorises `ImagingBlend` at SSE4 (57.5M instructions, against 68.7M for the SSE2 build). AVX2 is faster only because of its wider vectors. Two exact float kernels were slower than the compiler's code (1.01x and 0.84x), and AVX2 got slower too (0.62x).
- **6, Merge RGB on SSE4.** An explicit unpack kernel was exact but no faster: 0.82–0.84x of scalar, against 0.86–0.90x before. The operation is limited by writing 16 MB of output, not by the shuffles.
- **2 (vertical part), column strips in the box-blur vertical pass.** 0.90–1.11x across strip sizes of 512–4096 bytes. The L1 misses valgrind counts are served from L2 with hardware prefetching, which its cache model leaves out.
- **2 (horizontal, RGB), two rows per AVX2 vector.** Gaussian 1.08x, box 0.98x, unsharp 0.84x. A 256-bit `pmulld` costs about as much per element as two 128-bit ones, and packing the two rows adds work.
- **4, column strips in the vertical resample pass, upscaling only.** Lanczos ×1.5 gained 1.18x and ×2 bicubic 1.14–1.45x, but vertical-only upscales lost 11–21% and ×2 bilinear lost 11%. Not a reliable improvement.

### Lessons for the next profile

- Compare instruction counts between builds of the same C code before writing intrinsics. A slow SSE4 time can be the compiler's best at that vector width.
- Valgrind's cache misses mark where data leaves L1, not where time goes. Cache-blocking changes need real timings on a quiet machine before they are kept.

## Operations without SIMD kernels

A second pass over common operations that have no kernel of their own, to find where one would pay. Same method and machine as above, on `79435324e`. These operations are the `extra` set of `checks/simd_bench.py` (`--set extra`; `--set all` runs both sets). Builds: upstream Pillow 12.3.0 wheel, SSE4, AVX2. Valgrind 3.22 cannot run AVX-VNNI code, so the AVX-VNNI build was timed but not counted.

Instructions in millions (Ir) and native time in ms, best of 5 rounds, 2560×1600.

| Operation | upstream ms | SSE4 ms | AVX2 ms | upstream Ir | SSE4 Ir | AVX2 Ir |
|---|---:|---:|---:|---:|---:|---:|
| rotate RGB 30 nearest | 44.0 | 29.3 | 45.9 | 121 | 113 | 121 |
| rotate RGB 30 bilinear | 99.5 | 97.5 | 107.7 | 988 | 1013 | 950 |
| rotate RGB 30 bicubic | 264.1 | 259.4 | 271.9 | 2471 | 2559 | 2350 |
| perspective RGBA bilinear | 186.1 | 152.5 | 139.7 | 1581 | 1400 | 1276 |
| transpose RGB ROTATE_90 | 5.38 | 4.88 | 4.95 | 44 | 43 | 43 |
| transpose RGB FLIP_LR | 2.35 | 2.19 | 2.20 | 37 | 33 | 33 |
| transpose L TRANSPOSE | 3.41 | 3.42 | 3.43 | 40 | 41 | 41 |
| resize RGB 1/4 nearest | 0.29 | 0.29 | 0.29 | 3 | 3 | 3 |
| resize F 1/4 bicubic | 12.7 | 13.9 | 14.1 | 212 | 211 | 190 |
| resize I;16 1/4 bicubic | 26.9 | 28.3 | 28.3 | 354 | 350 | 330 |
| convert RGB->RGBA | 7.21 | 3.64 | 3.70 | 27 | 19 | 19 |
| convert RGBA->RGB | 7.21 | 3.66 | 3.71 | 27 | 19 | 19 |
| convert L->RGB | 3.31 | 3.31 | 3.31 | 41 | 41 | 41 |
| convert P->RGB | 4.44 | 4.50 | 4.48 | 70 | 70 | 70 |
| convert RGB->YCbCr | 11.7 | 11.7 | 11.7 | 145 | 164 | 164 |
| convert YCbCr->RGB | 13.5 | 14.1 | 14.1 | 181 | 209 | 209 |
| convert RGB->HSV | 110.2 | 117.8 | 121.3 | 369 | 422 | 396 |
| convert RGB->CMYK | 9.85 | 3.67 | 4.34 | 51 | 27 | 27 |
| convert L->F | 0.97 | 0.97 | 1.00 | 7 | 6 | 3 |
| convert L->1 dither | 26.9 | 27.9 | 27.7 | 139 | 125 | 125 |
| convert RGB->P adaptive | 6875.0 | 6662.1 | 7156.3 | 6093 | 6257 | 6277 |
| quantize RGB fastoctree | 37.8 | 37.9 | 36.4 | 478 | 486 | 441 |
| paste RGB mask 1-bit L | 16.0 | 12.6 | 8.40 | 160 | 111 | 63 |
| composite RGB mask L | 15.9 | 12.7 | 8.18 | 160 | 111 | 63 |
| paste RGBA over RGB | 16.0 | 12.1 | 7.39 | 167 | 105 | 58 |
| putalpha RGBA | 9.08 | 6.91 | 7.05 | 38 | 30 | 33 |
| point RGB lut | 5.85 | 5.82 | 5.78 | 70 | 66 | 66 |
| autocontrast RGB | 20.5 | 12.7 | 12.3 | 116 | 113 | 113 |
| equalize RGB | 20.5 | 12.8 | 12.6 | 116 | 113 | 113 |
| getextrema RGB | 4.71 | 3.39 | 3.43 | 49 | 25 | 25 |
| chops add | 13.3 | 10.2 | 6.54 | 113 | 89 | 43 |
| chops difference | 8.45 | 6.69 | 4.12 | 96 | 69 | 33 |
| chops lighter | 6.79 | 5.16 | 3.55 | 73 | 52 | 25 |
| filter RGB 5x5 float | 308.7 | 40.8 | 36.0 | 2038 | 444 | 369 |
| filter L 5x5 float | 113.5 | 20.3 | 20.3 | 577 | 198 | 172 |
| filter RGB EDGE_ENHANCE | 168.5 | 15.0 | 12.1 | 776 | 180 | 98 |
| mode 3 L | 1046.1 | 1050.3 | 1048.4 | 8515 | 8499 | 8499 |
| expand RGB 20 | 2.41 | 2.52 | 2.43 | 34 | 34 | 34 |
| thumbnail RGB 1/8 | 8.81 | 4.21 | 3.99 | 91 | 53 | 49 |

A plain copy of the 16 MB image (`expand`) takes about 2.4 ms. Operations within about 2x of that are limited by memory traffic and have little to gain from wider vectors: transposes, point, L/P→RGB, getextrema, RGB→L, split, merge, pack and unpack, reduce.

### Candidates, by expected gain

1. **ModeFilter is brute force** (1.05 s, 8.5G instructions, the same as upstream). For every pixel, `ImagingModeFilter` clears a 256-entry histogram, fills it from the window and scans all 256 bins; the source says so in a FIXME. Counting only the values in the window (at most 9 for size 3) and keeping the same tie-break (lowest value) and the `maxcount > 2` rule gives the same output. Portable C, like the rank filters; expected 20x or more.
2. **Rotate and transform with bilinear or bicubic filters** (100–260 ms, 1–2.5G instructions). Each pixel calls the filter through a function pointer, loops over the bands, clips the coordinates per band and interpolates in `double`. The arithmetic is per channel and in the same order for every channel, so the four channels of a pixel fit one AVX2 `__m256d` (two SSE2 or NEON `float64x2_t`) and give bit-identical results as long as the compiler does not contract to FMA. Expected 2–3x on RGB and RGBA.
3. **Integer 3×3 and 5×5 filters on AVX2** (checklist item 4.6, still open). AVX-VNNI is 2.1x faster than AVX2 on `filter RGB 5x5 SMOOTH_MORE` (8.9 vs 18.7 ms), so the AVX2 kernel is bound by its multiply-adds: it loads 128 bits and duplicates them into both lanes. Full 256-bit loads should give 1.5–2x on AVX2. The L-image kernel (`FilterSIMD_NxNi.c`) has no 256-bit path at all; 4.1 ms (3×3) and 10.3 ms (5×5) for 4 MB of input are compute-bound too.
4. **RGB→HSV** (118 ms, 3M mispredicts per image in `rgb2hsv_row`). Float with branches per pixel; a branch-free SIMD version is possible but HSV is rarely used.
5. **YCbCr↔RGB** (12–14 ms, 5x a copy). Table lookups per channel; SIMD needs gathers or a proof that the tables are exactly linear.

Not SIMD problems:
- `convert("P", ADAPTIVE)` (6.7 s): median cut. Time is in `ImagingQuantize`, sorting by `pixel_cmp`, hash-table lookups and `malloc`/`free`. `Quantize.FASTOCTREE` takes 38 ms.
- `convert("1")` with dithering (28 ms): Floyd–Steinberg error diffusion is sequential along each row.
- `paste`/`composite` with a mask and `ImageChops` add, difference, lighter: GCC vectorises them already, and the AVX2 build is 1.5–2x faster than SSE4 from vector width alone. As with `blend`, a hand-written SSE4 kernel is unlikely to beat the compiler.
- `resize` on `F` and `I;16` (13–28 ms): float and 32-bit convolution, only worth it for scientific or medical workflows.

### SSE4-only kernels and AVX2

- Reduce, RGB→L, split, get_channel, pack and unpack run close to memory speed: AVX2 would not help.
- Box and Gaussian blur and unsharp mask are compute-bound (18–62 ms), but their horizontal pass is a running sum along each row. Two RGB rows per AVX2 vector were measured and dropped (see Outcome above).
- 5×5 float kernels on 4-byte images have no AVX2 path; 3×3 has one and is 1.24x faster than SSE4 (`FIND_EDGES`). Kernels with coefficients large enough to need the float path are uncommon.
- The integer filters are item 3 above.

### Open finding: rotate nearest is slower in the AVX2 build

Repeated with 9 alternating rounds: `rotate RGB 30 nearest` takes 24.3 ms in the SSE4 build and 33.4 ms in the AVX2 build (scalar 23.6 ms). `convert RGB->CMYK` is 3.6 against 4.3 ms. Neither function has hand-written AVX2 code, GCC reports no vectorised loop in `affine_fixed`, and its code uses no 256-bit registers; callgrind counts 7% more instructions and the same cache misses. Hardware counters were not available on this machine (`perf_event_paranoid` = 4), so the cause, probably code layout or instruction scheduling of the VEX-encoded loop, is not confirmed.
