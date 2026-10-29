# Full-API profile of pillow-simd 13: where time goes and what to optimise next

Companion to [CALLGRIND_PROFILE.md](CALLGRIND_PROFILE.md), which covered only the operations the SIMD work targeted. This profile covers 207 operations across the public API, drawing and text included. It combines four measurements for each operation:

- simulated cache and branch behaviour (callgrind),
- per-function costs,
- heap allocations,
- native time.

Branch `simd/pillow-13` at `a805ea34a`, x86-64 only.

## Method

**Builds.** `-C simd=disable` (scalar) and `-C simd=avx2`, both at `-O2 -g` with GCC 13.3. The reference is the Pillow 13.0.0 wheel from PyPI.

**Operations: 207, in three sets of `checks/simd_bench.py`.**
- `simd` (43) and `extra` (43), as before.
- The new `api` set (121):
  - ImageDraw: lines, shapes, outlines, arcs, polygons, points, floodfill, RGBA blending, text and stroked text, `textbbox`.
  - ImageOps, ImageEnhance, ImageMath, ImageMorph, ImageStat.
  - Image methods: point, matrix and dither conversions, quantize methods, getcolors, gradients, `effect_spread`, quad/mesh/extent transforms, nearest/box/hamming resizes, transposes, crop, paste, `alpha_composite` with an offset, getchannel, merge, `tobytes`/`frombytes` raw modes, pixel access, `get_flattened_data`.
  - Every ImageChops and ImageFilter operation.
  - The codecs: PNG (levels 1 and 6, RGB/RGBA/L/P), JPEG, WebP, TIFF (raw, LZW, deflate), GIF, BMP and PPM, each encode and decode.
- Drawing and text use fixed shape counts with coordinates scaled to the image.
- Codecs work on a smooth, photo-like image rather than noise.

**Image size.** 1280×800: an RGBA image (4 MB) doesn't fit in the 3 MB L2.

**callgrind** (`checks/simd_callgrind.py`):
- `--cache-sim=yes --branch-sim=yes`.
- Simulated L1d of 48 KB (12-way) and "last level" of 3 MB (12-way), the P-core's L1 and L2. So "LL miss" here means "not in L2".
- One warm-up call, then one counted call.
- Per-function costs come from `callgrind_annotate`. Inlined intrinsics are attributed to compiler headers, and they are merged back into their functions.

**Allocations** (new `checks/simd_allocs.py`): an `LD_PRELOAD` counter of malloc, calloc and realloc calls, bytes requested and peak bytes in use, over one call after a warm-up.

**Native time.** `simd_bench.py --set full --size 1280x800`, best of 5, pinned to one P-core.

**Derived metrics** (`checks/simd_profile_report.py`):
- instructions per pixel;
- SIMD/scalar instruction ratio ("v/s"): close to 1.0 means nothing vectorises the operation;
- an estimated IPC: instructions ÷ (ms × 5.4 GHz);
- mispredicts per pixel and the conditional mispredict rate;
- L1 and L2 misses per 1000 instructions;
- the split of instructions between Pillow's C code, the Python interpreter, other libraries and libc.

**Caveats.**
- Valgrind's branch model has no history, so short loops count a mispredict at every exit.
- Its cache model has no prefetcher.
- The IPC estimate assumes a fixed clock.
- So treat the rates as relative signals; instruction counts and native times are the primary data.
- arm64 was not profiled.

Reproduce:

```
checks/simd_callgrind.py --variant scalar=B1 --variant avx2=B2 --set full --size 1280x800 \
    --vg-arg=--D1=49152,12,64 --vg-arg=--LL=3145728,12,64 --top 12 --out cg --jobs 6
checks/simd_allocs.py --variant scalar=B1 --variant avx2=B2 --set full --json allocs.json
checks/simd_bench.py --variant upstream=WHEEL --variant scalar=B1 --variant avx2=B2 \
    --baseline upstream --set full --size 1280x800 --rounds 5 --cpu 5 --json times.json
checks/simd_profile_report.py --callgrind cg --times times.json --allocs allocs.json > tables.md
```

The full per-operation tables are in [PROFILE_TABLES.md](PROFILE_TABLES.md).

## Overview

The SIMD build against the 13.0 wheel at 1280×800 (206 operations with a wheel time):

| Category | Operations | Geometric mean | No real speedup (≤ 1.1x, slower ones included) | SIMD ms (sum) | Wheel ms (sum) |
|---|---:|---:|---:|---:|---:|
| Filters | 35 | 6.39x | 6 | 1747 | 3663 |
| Resampling and geometry | 47 | 3.03x | 14 | 185 | 543 |
| Compositing and channel operations | 36 | 1.91x | 10 | 65 | 127 |
| Conversion and palettes | 29 | 1.71x | 13 | 804 | 1417 |
| Codecs | 26 | 1.12x | 17 | 2271 | 2421 |
| Statistics, point operations, pixel access | 17 | 1.09x | 12 | 38 | 39 |
| Drawing and text | 16 | 1.00x | 16 | 392 | 392 |
| **All** | **206** | **2.18x** | **88** | | |

Earlier work concentrated on filters and resampling. Most of the remaining time is in:
- codecs, dominated by zlib;
- generic rank filters;
- adaptive palettes;
- drawing and text, which pillow-simd has not touched.

## Correctness finding, fixed in `a805ea34a`

The x86 SIMD unpremultiply (RGBa → RGBA), inherited from upstream pillow-simd, used an approximate reciprocal in 16-bit lanes.

**The bug.**
- With colour above alpha, the product overflowed and gave 0 where Pillow gives 255. Bicubic and Lanczos overshoot produces colour above alpha whenever an RGBA image is resized, rotated or transformed, because those go through RGBa, so almost transparent pixels turned black.
- Alpha 0 gave colour 0; Pillow keeps the colour.
- Other values could be 1 off.

**The fix.**
- Colour is clipped to alpha.
- The reciprocal quotient is corrected once with the remainder `255c − qa`.
- Every one of the 65,536 (colour, alpha) pairs now gives Pillow's bytes, as NEON already did. `test_unpremultiply_exact` checks this.

**Cost.** AVX2 1.75 → 2.84 ms and SSE4 2.0 → 5.2 ms at 2560×1600; Pillow takes 33 ms. RGBA resize, rotate and perspective change by 1–2%.

## Recommendations

Ranked by expected gain for typical use. Times are ms at 1280×800 (SIMD build / 13.0 wheel). Estimates are rough, made before implementation.

**Kind** says what work each needs:
- **SIMD**: a vector kernel.
- **branchless**: a data-dependent branch removed.
- **algorithm**: a different method.
- **memory**: fewer passes or copies.
- **Python**: work moved out of the interpreter.
- **alloc**: fewer allocations.
- **build**: a library choice.

### Tier 1: large gains

#### 1. Generic rank filters, size 5 and up
`median 5 L` 480 / 486 ms, `rank 7 L` 888 / 898 ms. Kind: SIMD, branchless, algorithm.
- 1046 and 1886 instructions per pixel, 39 and 68 mispredicts per pixel, IPC 0.4.
- 77% of instructions and 97% of mispredicts are in the quickselect (`MakeRankFunction`).
- About 10 million `memcpy` calls copy the window for every pixel (11%).
- Fix for 5×5: a sorting network evaluated on 16 or 32 pixels at a time with `pminub`/`pmaxub` (NEON `vmin`/`vmax`), as the 3×3 network already is. Since every pixel's window is sorted with the same network, there are no branches.
- Fix for larger sizes: a sliding-window histogram (Huang), O(size) per pixel.
- Exact. Expected 10–30x for 5×5.

#### 2. PNG encoding
`encode PNG RGB` 437 / 481 ms, RGBA 975 / 1011 ms, L 291 / 304 ms; level 1: 89 / 138 ms. Kind: build, SIMD.
- zlib's `longest_match`/`deflate_slow` take 55–72% at the default level.
- Building against zlib-ng (Pillow supports it; `features.check("zlib_ng")`) is the largest single gain, typically 2–4x for deflate.
- Pillow's own filter selection (`ImagingZipEncode`) is 11–17% at level 6 and 38% at level 1. It computes Sub, Up, Average and Paeth for every row and sums `|v|`.
  - For encoding, all four filters depend only on the original rows, so they vectorise fully. `psadbw` does the sums.
  - Expected 4–8x on that part.
- Exact: same filter choices and bytes.

#### 3. Adaptive palettes (median cut)
`convert RGB->P adaptive` 575 / 1120 ms, `quantize mediancut` 168 / 175, `quantize maxcoverage` 160 / 168. Kind: algorithm, branchless, SIMD.
- `ImagingQuantize` 41%.
- `qsort` comparator `pixel_cmp` 16–28%.
- Hash-table lookups 8% with 5 million mispredicts.
- `compute_distances` 29% in max coverage.
- Fixes:
  - a counting (radix) sort per channel instead of `qsort` with a comparator;
  - a branch-light open-addressing table;
  - SIMD distance computation.
- Expected 2–3x. Output must stay identical, so ties have to be ordered as now.

#### 4. ImageDraw.floodfill
86 ms for a quarter-by-quarter region (1/16 of the area). Kind: Python.
- 97% of the time is in the interpreter: `floodfill` is pure Python, with a set of tuples and per-pixel `getpixel`/`putpixel`.
- A C scanline fill would be 50–100x faster.
- Not SIMD work, but the largest single ratio found.

#### 5. Text
`draw text 100 lines` 95 / 95 ms, stroked text 89 / 88, `textbbox` 20 / 21. Kind: algorithm, SIMD.
- 75–97% is FreeType and HarfBuzz. Pillow renders every glyph from its outline on every call; there is no glyph cache.
- A cache of rendered glyph bitmaps per font, size and glyph (and subpixel offset) would make repeated text mostly cheap blending.
- `ImagingFill2`, which composites the glyph masks onto the image, is the remaining 21%. It blends one pixel at a time, and a masked-blend SIMD kernel would make it 3–6x faster.
- Exact.

#### 6. Filled shapes
`draw polygons filled` 46 ms, rectangles 6.5, ellipses 4.4; all equal to the wheel. Kind: SIMD.
- `hline32` takes 57–89% of these.
- It tests `mask == NULL || mask[x]` for every pixel, which keeps the compiler from turning the fill into a vector store.
- Fix: a separate unmasked loop, which the compiler vectorises, or an explicit 32-bit fill.
- `hline32rgba` (semi-transparent ink) blends one channel at a time; a SIMD blend of a constant colour over a span would help it.
- Expected 2–4x on filled shapes. Exact.

#### 7. Arcs, pie slices, rounded rectangles
5.7, 5.4 and 6.8 ms. Kind: alloc.
- 145,000–173,000 allocations per call, about 500 per shape.
- `clip_tree_do_clip` is 18–30% of the time, and `malloc`/`free` another 33%.
- A per-call arena (or reuse of node arrays) for the clip tree would remove most of it: expected 1.5–2x.

### Tier 2: medium gains

#### 8. Byte lookup tables (`point`)
`point RGB lut` 1.3 ms, ImageOps `invert`, `posterize`, `solarize` 1.3, `colorize` 2.3, `autocontrast`/`equalize` 2.8; all equal to the wheel. Kind: SIMD, Python.
- 16.5 instructions per pixel, unvectorised (`im_point_3x8_3x8`).
- A SIMD 256-entry byte lookup is possible: AVX2 needs eight 16-entry `vpshufb`s and blends, AVX-512 VBMI `vpermi2b`, NEON four `vqtbl4q`. Expected 2–4x.
- Several of these don't need a table at all:
  - `invert` could call `ImagingNegative`, which `chops invert` uses: 0.38 ms against 1.37.
  - `posterize` is an AND with a mask.
  - `solarize` is a compare and select.

#### 9. ImageEnhance and fused passes
`contrast` 3.6 / 5.5 ms, `brightness` 1.9 / 3.6, `color` 3.0 / 4.9, `alpha_composite` with an offset 2.0 / 7.2. Kind: memory, Python.
- Contrast and brightness blend the image with a solid image made for the purpose. Both are per-channel point operations, so one 256-entry table per channel gives the same bytes in one pass, with no temporary image: expected 2–3x.
- `alpha_composite(im, dest)` crops, composites and pastes back, three passes. An in-place composite over a sub-rectangle would halve it.

#### 10. Median and rank 3×3: the border copy
`median 3 RGB` 8.2 ms, `median 3 L` 2.4 ms. Kind: memory.
- `ImagingExpand` is 47–53% of the time. `RankFilter.filter()` copies the image with a border, then filters.
- The 3×3 kernels could read the edge pixels themselves, as the convolution filters do. Expected about 2x.

#### 11. Min and max filters of other shapes
`min 5 L` 3.7 ms, `max 3 RGB` 7.7 ms. Kind: SIMD.
- Already 80x and 37x the wheel through separable passes, but those passes are scalar (v/s 0.90–0.98).
- `pminub`/`pmaxub` over rows, and over 4-byte pixels for RGB, would give 4–8x.

#### 12. YCbCr ↔ RGB
2.9 and 3.5 ms; v/s 1.00. Kind: SIMD.
- 40–51 instructions per pixel through nine 16-bit lookup tables.
- The tables are rounded entry by entry (`T[i] ≠ i·T[1]`), so a plain multiply isn't exact.
- If each table is `round(i·K)` for a constant K, which needs checking over all 256 entries, a SIMD version can compute the same rounding in fixed point: expected 4–6x.

#### 13. F and I;16 resampling
`resize F 1/4 bicubic` 3.1 / 3.0 ms, `resize I;16` 7.2 / 6.7 ms. Kind: SIMD.
- v/s 0.90–0.94, with 0.32 mispredicts per pixel from the tap loop's exit.
- Only 8-bit modes have SIMD kernels. A float kernel for F and a 32-bit-accumulator kernel for I;16 would give 3–5x.
- The I;16 resize is currently 7% slower than the wheel.

#### 14. Transposes, 90° rotations, flips
`ROTATE_90` 1.0 ms, `TRANSVERSE` 1.05, `FLIP_LR` 0.55; v/s 1.0. Kind: SIMD.
- About 10 instructions per pixel and 12 L2 misses per 1000 instructions: tiled, but scalar.
- 4×4 32-bit transposes in registers (`unpack`), and byte-reversing shuffles for flips. Expected 1.5–2x, limited by memory.

#### 15. Nearest-neighbour resize and rotation
1.5 and 1.6 ms; v/s 1.0–1.08. Kind: SIMD.
- Per-pixel affine stepping. AVX2 gathers for 4-byte pixels, and a precomputed source-column table for axis-aligned resizes: expected about 2x.

#### 16. Raw codecs: copies and churn
TIFF/PPM/BMP encode 0.6–0.7 ms and decode 1.1–1.3 ms. Kind: memory, alloc.
- `memcpy` is 63–80% when encoding and 45% when decoding.
- Encoding a 3 MB image requests 32 MB through about 120 allocations. The Python encode loop asks the encoder for `bufsize` chunks and writes each one to the file object, which grows as it goes.
- Packing straight into one buffer of the final size would remove most of the copies: expected 2x.

#### 17. Zero-filled images that are then overwritten
Kind: memory.
- `Image.frombytes` allocates with `Image.new()`, and `ImageFile.load_prepare()` with `core.new()`. Both zero-fill (`ImagingNew`), then the decoder writes every pixel. `memset` is 63% of `frombytes` and 20% of WebP decode.
- `ImageMath` temporaries: `memset` 35%.
- A dirty allocation (`ImagingNewDirty`) where every pixel is written: expected 1.5–2x for `frombytes`, 10–20% for decoders.

#### 18. Python object creation
`get_flattened_data L` 15.4 ms (92 MB allocated), `pixel access 20k` 2.0 ms, `draw points 100k` 5.7 ms. Kind: Python.
- 62–80% interpreter.
- For 8-bit modes, building the list in C from the row bytes would be 3–5x faster.

#### 19. `effect_spread`
51.6 / 51.5 ms. Kind: algorithm.
- libc `rand()` is 74%: two calls per pixel, through `random_r`'s locking.
- An inline xorshift generator would give about 3x, but it changes the random pattern, so the output would differ from Pillow's.

### Tier 3: little to gain

- **Streaming compositing and channel operations** (chops, blend, paste, merge, split, conversions between 3- and 4-byte modes):
  - IPC 0.5–1.0 with 10–280 L2 misses per 1000 instructions: memory bandwidth limits them.
  - Most are already vectorised, by the compiler or by hand (v/s 0.1–0.5).
  - Only fusing passes helps, as in 9.
- **External codecs:** JPEG, WebP, TIFF LZW/deflate and libimagequant are 90–98% in their libraries. Only the build matters: library versions and flags.
  - The wheel is 10–25% faster on `encode JPEG L`, WebP and GIF decode. It bundles its own builds of these libraries, and the difference should be traced to versions and flags, not to Pillow's code.
- **GIF LZW** (`glzwe` 92% of encoding, decoder 97%): serial, about 1 mispredict per pixel. Hash-table tuning might give 1.3x.
- **Floyd–Steinberg dithering** (RGB→1, L→1, RGB→P web): serial error diffusion. A parallel variant would change the output.
- **Mode filter:** 13.6x the wheel already. Its remaining cost is the histogram updates.

## Tools added for this profile

- `checks/simd_bench.py`:
  - the `api` set and the `full` set (`simd` + `extra` + `api`);
  - codec decode inputs are encoded on the untimed first call.
- `checks/simd_callgrind.py`:
  - per-function costs with every event, and object names;
  - derived rates and a cycle estimate;
  - `--vg-arg` for the cache model, and `--resume`.
- `checks/simd_allocs.py`: allocation counts per operation.
- `checks/simd_profile_report.py`: the combined Markdown tables and JSON.
