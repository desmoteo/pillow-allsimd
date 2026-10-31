# Where OpenCV and libvips are still faster than Pillow-AllSIMD, and why

As of `simd/pillow-13` at 28b55224b (2026-10-31), against the libraries in
the [pillow-simd-perf](https://desmoteo.github.io/pillow-allsimd-perf/)
benchmark. The kernels themselves, including the approximate ones, are
described in [OTHER_LIBRARIES_KERNELS.md](OTHER_LIBRARIES_KERNELS.md); the
measurements come from pillow-simd-perf's `analysis/` scripts (outputs in
`analysis/results-2026-10-31/`).

Sources read: OpenCV 5.0.0 (tag 40738fb, the version in the benchmark image),
its HALs, KleidiCV 26.03 (the archive OpenCV 5.0.0 pins, MD5 b85a745b...),
Carotene (in OpenCV's tree), libvips 8.16.1. Wheel build information:

- x86-64 wheel: baseline SSE3, dispatch up to AVX2/AVX-512, **Intel IPP 2026.0**
  linked as IPP-ICV plus the `ipp` custom HAL.
- arm64 wheel: baseline NEON+FP16, dispatch to DOTPROD/FP16/BF16, custom HALs
  **Carotene** (the Tegra NEON library) and **KleidiCV 26.03** (Arm's).

## What actually runs, per benchmark case

On x86 IPP is mostly *not* used any more in 5.0.0: the inline calls for box
filter, median, morphology are commented out, Gaussian needs a build option,
and HSV/YCrCb/Gray→BGR IPP paths are disabled by `IPP_DISABLE_*`. What remains
is the IPP HAL (flip, transpose, remap) and resize (except 8-bit bilinear and
nearest), plus core arithmetic. The measured IPP on/off column below settles it.

On arm64, KleidiCV hooks (enabled by default except threshold, sum and cmp)
take precedence over Carotene, which takes precedence over OpenCV's own code.
OpenCV 5 also gates some HAL calls on an algorithm hint whose default in both
wheels is `ALGO_HINT_ACCURATE`; the benchmark uses the default:

| Case | arm64 code that runs | Why |
|---|---|---|
| GaussianBlur 8U (σ 1 → 7x7, σ 2 → 13x13) | OpenCV's own fixed point (8.8) | KleidiCV's Gaussian only with `ALGO_HINT_APPROX` (4x faster for σ 1, see OTHER_LIBRARIES_KERNELS.md) |
| erode/dilate 3x3 | Carotene | KleidiCV declines kernels under 5x5 by default |
| medianBlur 3/5 | KleidiCV sorting networks | 8U |
| add, absdiff, multiply | KleidiCV | 8U |
| max, bitwise_not, addWeighted | Carotene | 8U |
| GRAY→RGB | KleidiCV | dcn 3 |
| RGB→RGBA, RGBA→RGB, RGB→GRAY, RGB→HSV_FULL | Carotene | |
| RGB→YCrCb | Carotene (KleidiCV declines CbCr order) | |
| YCrCb→RGB | OpenCV's own NEON | neither HAL does YCrCb→RGB |
| resize RGBA bilinear (upscale) | Carotene, 7-bit weights | 1 and 4 channels only |
| resize RGB bilinear, bicubic, Lanczos | OpenCV's own | |
| resize INTER_AREA by 2 and 4 (reduce) | Carotene | ratios 2, 4, 0.5 |
| warpAffine, warpPerspective (RGB, RGBA) | OpenCV's own (new warp kernels) | KleidiCV warp: 1 channel only; Carotene hooks commented out |
| blur (box) | OpenCV's own | no HAL hook |
| LUT | OpenCV's own scalar loop | |
| flip, rotate 90 | KleidiCV rotate/transpose (flip: own) | |

## The recurring reasons

1. **3 bytes per RGB pixel against Pillow's 4.** Pillow stores RGB as RGBX.
   Every bandwidth-bound operation moves 4/3 as many bytes: add, difference,
   lighter, invert, flips, RGBA↔RGB, L→RGB, YCbCr. The measured ratios
   (OpenCV/Pillow) are close to the byte ratios (table below).
2. **One fused pass instead of whole-image passes.** OpenCV's FilterEngine
   (box, morphology, filter2D), its Gaussian (`fixedSmoothInvoker`, a ring of
   ksize rows of 16-bit horizontal results, about 107 KB here), KleidiCV's
   separable filters (vertical into one row buffer, then horizontal), and
   libvips' demand-driven regions never write a full intermediate image.
   Pillow's blur makes two sweeps (horizontal passes, then the vertical
   cascade) through a 16 MB temporary; its resize does the same with a
   horizontally resampled temporary; RGBA resize adds a premultiply and an
   unpremultiply sweep.
3. **Lower precision, so twice the lanes.** OpenCV's bit-exact Gaussian
   uses 8.8 fixed point (KleidiCV's, behind the approximate hint, 8-bit
   weights with an 8-bit intermediate); Carotene's bilinear resize 7-bit
   weights; OpenCV HSV 12-bit reciprocal tables (gathers on AVX2); warps in
   float32. Pillow's results are defined by 22-bit (resize), 24-bit (blur)
   fixed point and double (transforms) arithmetic, and the SIMD builds must
   reproduce them exactly.
4. **Different algorithms.** Downscales use INTER_AREA (box average, no
   antialiasing filter) in the benchmark; Gaussian is a true 7-tap kernel
   against Pillow's three extended box passes; ellipses are filled as a
   72-vertex polygon; text is Hershey vector strokes, not FreeType glyphs.
5. **Vectorizing across pixels, not within one.** OpenCV 5's warps compute 8
   output coordinates per step in float32 and have an all-inside fast path;
   Pillow's transforms call a filter per pixel and vectorize its 4 channels
   in doubles (2 lanes per NEON register).

## Measured on x86 (AVX2 build, Docker -O3, CPU 2, 2560x1600)

`analysis/gapcheck.py`: best of 15 calls, minor page faults per call, OpenCV with
IPP on and off. Copy bandwidth on this core: 44.9 GB/s (read + write).

**IPP changes three cases only**: flip left-right and rotate 180 (OpenCV
1.0 ms with IPP, 4.4 and 5.3 ms without) and the RGBA bicubic upscale (32.9
ms with IPP, 100.3 ms without, so OpenCV's own code is 1.9x slower than
ours there). Everything else is OpenCV's own SIMD code.

**Bandwidth-bound cases run at the same speed per byte** (GB/s moved,
Pillow / OpenCV): add 25.1 / 26.5, invert 25.2 / 27.0, flip top-bottom
26.2 / 26.7, L→RGB 26.0 / 26.4, RGBA→RGB 27.0 / 27.0. The time ratio is the
byte ratio, 4/3 for RGB: closing it would mean storing RGB in 3 bytes,
which Pillow's image layout (and its C API) doesn't allow.

**Page faults**: zero per call for nearly everything on both sides (glibc
reuses the freed blocks), except Pillow's blur and unsharp mask on RGB
(814 to 1325 per call, from the 16 MB temporary image) and the large
resizes (3400 to 5400 per call for Pillow, about 450 for OpenCV, whose
output arrays numpy places on huge pages).

`analysis/parts.py`, where blur and resize time goes (ms):

| Case | horizontal only | vertical only | both | cost of the temporary | OpenCV |
|---|---|---|---|---|---|
| box r2 | 2.94 | 4.48 | 9.00 | 1.6 | 3.19 |
| box r10 | 4.16 | 4.75 | 10.50 | 1.6 | 9.43 |
| Gaussian r1 | 7.39 | 7.89 | 18.68 | 3.4 | 7.57 |
| RGB bilinear upscale to 5478x3424 | 9.60 | 7.21 | 25.03 | 8.2 | 26.81 |

RGBA upscale: 45.0 ms, of which about 20 ms is the premultiply of the input
and the unpremultiply of the 75 MB output, two sweeps OpenCV doesn't make
(it doesn't premultiply at all).

## Measured on arm64 (NEON build, Jetson Orin, CPU 2)

Copy bandwidth on this core: 19.3 GB/s. The same pattern: add moves 15.4 GB/s
in Pillow-AllSIMD and 15.8 in OpenCV, so the time ratio is again the byte ratio.
Blur and resize split into their passes (ms):

| Case | horizontal only | vertical only | both | cost of the temporary | OpenCV |
|---|---|---|---|---|---|
| box r2 | 7.88 | 9.90 | 21.29 | 3.5 | 10.48 |
| box r10 | 18.60 | 12.17 | 33.88 | 3.1 | 18.89 |
| Gaussian r1 | 23.34 | 23.41 | 52.36 | 5.6 | 43.93 |
| RGB bilinear upscale to 5478x3424 | 33.61 | 26.79 | 91.64 | 31.2 | 69.98 |

The RGBA upscale (156.8 ms against Carotene's 43.7) adds the premultiply
(3.0 ms) and the unpremultiply, which divides: 11.1 ms for 16 MB, about 50
ms for the 75 MB output. An 8x Lanczos downscale spends 16.0 ms in the
horizontal pass and 18.8 ms in the vertical one, against 2.9 and 4.2 ms on
x86: NEON's `vmlal_s16` gives 4 multiply-accumulates per instruction where
AVX2's `pmaddwd` gives 16.

## The cases, grouped by what it would take

### Closable without changing a pixel

1. **Fuse the blur passes.** Compute each horizontally blurred row into the
   ring the vertical cascade already keeps, instead of into a whole
   temporary image. Saves 1.6 ms (box) to 3.4 ms (Gaussian r1) on x86 and
   the page faults; the arithmetic and its order are unchanged. Unsharp
   mask gains the same (it blurs first).
2. **16-bit running vertical sums for integer radii.** OpenCV's vertical
   box pass is O(1) per pixel on u16 sums held in one row; ours sums 2r+1
   rows for small radii (4.5 ms vertical for r2 against OpenCV's 3.2 ms
   whole blur). For integer radii up to 128 the box sum fits 16 bits; the
   exact 16-bit division (`BoxDirect.fast`) is verified for radii 1 to 4
   and the same start-up check could find constants for larger ones.
3. **Fuse the two resize passes**, with a ring of horizontally resampled
   rows for the vertical pass: 8 ms of the 25 ms upscale on x86. For RGBA,
   premultiply on the horizontal pass's loads and unpremultiply on the
   vertical pass's stores: the same 8-bit intermediate values, about 20 ms
   saved on x86 and 50 on the Jetson. Biggest single gap on arm64 (157 vs
   44 ms; Carotene runs a 7-bit-weight fused bilinear there).
4. **Min/max 3x3**: share the middle rows' result between two output rows
   (OpenCV's MorphColumnVec) and read the image edges in place instead of
   copying each row to a padded buffer.
5. **arm64 resize with large kernels** (Lanczos/bicubic 8x downscales: 18
   ms on the Jetson against 3.2 on x86, a much wider ratio than elsewhere):
   the NEON horizontal kernel deserves a look.
6. **YCbCr→RGB on NEON**: the per-term INT16 tables are reproduced in
   32-bit float lanes; an integer form proven equal to the tables on all
   256 values per term would use 16-bit multiplies (OpenCV's approach).

### Structural (would change Pillow's results)

The precision and algorithm items are the candidates for the opt-in fast
mode planned in [APPROXIMATE_KERNELS_PLAN.md](APPROXIMATE_KERNELS_PLAN.md).

- **4 bytes per RGB pixel**: add, difference, lighter, invert, flips,
  RGBA↔RGB, L→RGB, YCbCr; and a third more work per pixel in every
  per-channel kernel (warps, filters, rank filters).
- **Precision**: HSV (OpenCV 12-bit reciprocal tables; Pillow's float
  semantics), Gaussian (8-bit weights in KleidiCV, 8.8 fixed point in
  OpenCV, against 24-bit), bilinear resize on arm64 (7-bit weights in
  Carotene), warps (float32 against double), blend (float).
- **Algorithm**: Gaussian as a true kernel against three extended box
  passes; INTER_AREA against an antialiasing convolution for downscales;
  polygonal ellipses; Hershey text.

### Drawing outlines: bounding boxes faster, ROI polygons slower

Measured split (x86 AVX2, ms, the benchmark's 500 boxes and 100 ROI
polygons of 12 points, 2 px; pillow-simd-perf `analysis/` and
`drawprof`-style timing of each part):

| | Pillow-AllSIMD | OpenCV |
|---|---|---|
| boxes, outline 2 px | 1.80 (C) | 3.62 (thickness 2) |
| boxes, 1 px | | 1.36 |
| ROI polygons, outline 2 px | 1.86 (C) | 1.25 (polylines, thickness 2) |
| ROI polygons, 1 px | 0.32 | 0.30 |
| the edges as wide lines, no mask | 0.91 | |
| the polygons' fill (the mask's work) | 2.01 | 2.35 (fillPoly) |

- **Boxes.** Pillow draws a rectangle outline as axis-aligned runs: `width`
  horizontal spans top and bottom and vertical runs on the sides. OpenCV
  turns a thick rectangle into a polyline: each side a filled convex quad,
  each corner a round join filled as a small polygon, and centred on the
  edges, so it sets about 1.5x the pixels. Its thickness-2 rectangle costs
  2.7x its 1 px one.
- **ROI polygons.** The stroke itself is cheaper in Pillow: the twelve wide
  lines without a mask take 0.91 ms against OpenCV's 1.25 for its whole
  polyline. The rest is Pillow's rule that a wide outline stays inside the
  polygon: the outline is drawn through a mask holding the polygon's fill,
  so the cost grows with the polygon's area (these ROIs cover 7.0 Mpx, 9.7
  Mpx of bounding boxes, for 89 kpx of perimeter), where OpenCV's centred
  stroke needs no inside test. Since 7238975d1 the mask covers the
  polygon's bounding box only, not the image (17.2 -> 2.6 ms for the case).
- **Spans, exact (2026-10-31):** the mask is now each row's inside spans,
  recorded from the same scan as the fill (the same clipped hline calls),
  sorted by row and merged, and the wide lines' runs are cut to them and
  filled unmasked: no clearing and no filling of the bounding box, the same
  pixels (hashes over random polygons, modes and inks). The case went from
  2.0 to 1.8 ms, less than the 1 ms hoped: the scan itself and the twelve
  wide lines remain. Larger polygons gain more, the work now growing with
  rows instead of area: 20 polygons of radius 700 px 2.24 -> 1.34 ms.

### Where OpenCV is now behind (x86)

FIND_EDGES/EDGE_ENHANCE (4.4 vs 6.6 ms), YCbCr→RGB (2.0 vs 2.4), reduce by
2 and 4, L rotate 90, the RGB bilinear upscale (24.6 vs 26.8), the Lanczos
downscale (3.2 vs 3.6 for INTER_AREA), polygons and ellipses. On arm64:
the 3x3 kernel filters (SHARPEN 27.1 vs 39.9 ms, FIND_EDGES 26.6 vs 39.8),
the 5x5 ones (63.5 vs 102.8), the unsharp mask (74.8 vs 97.8) and
RGB→RGBA. Geomean over all 109 cases against Pillow 13, at 28b55224b:
NEON 2.40x against OpenCV's 2.06x; on x86 AVX2 5.22x, AVX-VNNI 5.34x
against 3.40x.

## Round 2: after fast mode (results at 7fab6973b)

Every case where OpenCV, libvips or ImageMagick beat the fastest
Pillow-AllSIMD build (exact or fast mode) by 3% or more in the published
run: 29 cases on x86, 42 on arm64. Grouped by cause, with what this round
changed (x86 numbers: Docker -O3, CPU 2; arm64: Jetson Orin, CPU 2).

### Benchmark artifacts, fixed in pillow-allsimd-perf

- `codec/thumbnail/jpeg`: Pillow took 0.02 ms against 9 to 21 ms for the
  others. `draft()` already gave the 320x200 size, so `thumbnail()`
  returned without decoding; the case now loads the image.
- `codec/encode/png` against libvips (1.6x, 1.5x): libvips' `pngsave`
  writes no row filter by default, a 12.2 MB file against Pillow's 7.5 MB.
  With `filter="all"`, the adaptive filtering Pillow, libpng (OpenCV) and
  ImageMagick do, it takes 671 ms against Pillow-AllSIMD's 561 and OpenCV's
  530; the benchmark now asks for it.

### Bytes per pixel (structural)

Pillow keeps RGB in 4 bytes per pixel, OpenCV in 3. The memory-bound
operations run at the same bandwidth as OpenCV's and so take about 4/3 of
its time: `chops/RGB/difference` moves 48 MB in 1.91 ms (25 GB/s), OpenCV 36
MB in 1.37 ms (26 GB/s). The same for the other chops, flips and rotate
180, `point/RGB/invert`, L -> RGB, RGB <-> RGBA, fast blend and reduce by 2
(1.1x to 1.45x). Each call also writes a freshly allocated output, as
OpenCV's do. The 3x3 min, max and median filters on RGB (1.3x to 1.7x) are
in this group too: min and max are already separable, through a ring of
row results.

### Closed or narrowed this round

| Case | Before | After | Other |
|---|---|---|---|
| `blur/box/RGB/2`, x86 | 9.56 ms | 5.71 ms | OpenCV 3.20 |
| `blur/box/RGB/1`, x86 | 8.65 ms | 4.14 ms | OpenCV 2.72 |
| `codec/decode/webp`, x86 | 105.0 ms | 76.1 ms | OpenCV 78.9 |
| `codec/decode/jpeg`, x86 | 16.7 ms | 15.9 ms | (ahead before) |
| `blur/box/RGB/2`, arm64 | 20.10 ms | 15.54 ms | OpenCV 10.43 |
| `point/L/table`, arm64 | 3.55 ms | 1.87 ms | OpenCV 2.58 |
| `point/RGB/table`, arm64 | 10.14 ms | 7.89 ms | OpenCV 7.76 |
| `codec/decode/webp`, arm64 | 108.6 ms | 93.0 ms | OpenCV 94.3 |
| `rotate/RGB/perspective/bilinear`, fast, arm64 | 60.7 ms | 50.5 ms | OpenCV 39.1 |

- **Box blur** with one pass each way: the horizontal result went through
  a full temporary image (16 MB to allocate, write and read back). Now each
  row is blurred horizontally into a ring of 2r + 3 rows just before the
  vertical pass reads it, for the radii the direct kernel takes (integers
  up to 4, and below 1). Same pixels. Radius 3 is now level with OpenCV
  (6.98 against 6.52 ms); radius 10 keeps the running sums.
- **WebP**: Pillow decodes every file through libwebp's animation decoder,
  which composes a canvas, then copies the frame into a bytes object and
  again into the image. One-frame files now decode with `WebPDecode`
  straight into the image's memory (the decoder's own first-frame path,
  so the same pixels). libwebp alone: 74.5 ms through the simple API, 94.6
  through the animation decoder.
- **JPEG**: libjpeg-turbo writes RGB images as 4-byte pixels itself
  (`JCS_EXT_RGBX`) instead of Pillow expanding each row.
- **PNG**: the "up" filter is undone 16 bytes at a time.
- **Lookup tables on arm64** (`point()` with a table, 1.3x behind): NEON
  looks up 16 bytes in a 256-entry table with one `tbl` and three `tbx`.
  RGB rows are split into band planes first, so each band's table is
  loaded once per row (looking up the three bands of 16 pixels with their
  tables reloaded gained nothing: 10.1 ms).
- **Fast-mode perspective on arm64** (1.6x behind): positions in 32-bit
  floats, four lanes per division instead of two (see below).

### Tried and not kept

- PNG sub, average and Paeth one pixel per vector step, as libpng does:
  1.4x to 2.2x slower than the scalar loops (Paeth 80 -> 152 ms per
  2560x1600 image). The left neighbour makes each pixel wait for the one
  before, a chain of about ten vector operations, where the scalar code
  runs the 3 or 4 byte positions as independent chains.
- JPEG encoding from 4-byte pixels (`JCS_EXT_RGBX` input): 3 to 7% slower
  than packing them to 3 bytes first.
- WebP encoding through libwebp's own importers instead of Pillow's
  per-pixel loop: 0.5%, within noise.
- zlib-ng: Pillow's 13.0 wheel links zlib-ng 2.3.3; against Debian's zlib
  1.3.1 it inflates rows without filters in the same 60 ms and deflates at
  level 6 in about the same time (515 against 530 ms, the same 7.51 MB
  file), so it isn't worth a build dependency for this data.

### What remains, and why

- **PNG decode** (1.15x): inflate is about 60% of it; the rest is the
  scalar Paeth and sub filters (above). **WebP on arm64** and **WebP
  encode** (1.07x to 1.14x): libwebp itself, Debian's 1.5.0 against
  OpenCV's bundled build.
- **RGBA upscales** (x86 1.4x to 1.6x, arm64 3.4x): OpenCV resizes RGBA
  without premultiplying. Pillow premultiplies the source, resizes and
  unpremultiplies the 18.8 Mpx result (about a third of the arm64 time),
  and its 4-channel bilinear upscale is itself about 2x OpenCV's on arm64,
  which fusing the two resize passes (APPROXIMATE_KERNELS_PLAN.md, item 5)
  would address.
- **`point/L/table` on x86** (1.46x): AVX2 has no 256-entry byte shuffle
  (`vpermb` needs AVX-512 VBMI), so the lookups stay scalar.
- **HSV and YCbCr -> RGB on arm64** (1.3x): HSV's two divisions per four
  lanes (a reciprocal with an integer correction was slower, round 1);
  YCbCr's 32-bit products, which a 16-bit form would make coarser.
- **Reduce by 4 and L rotate 90 on arm64** (1.7x, 1.4x): generic NEON
  paths, on operations of 1.4 to 1.6 ms. (Round 3: reduce by 4 1.15x
  faster.)
- **Drawing**: text (Hershey strokes, not comparable) and outline polygons
  (the scan and the wide lines, as before).

### FP16

Where Pillow-AllSIMD computes in floats, FP16 doesn't fit:

- The exact paths that use floats (HSV, blend, float kernel filters, the
  transforms and the resize of F, I and I;16 images in doubles) must keep
  their float arithmetic to keep their results.
- Fast mode's kernels are integer, except two divisions. HSV's must give
  exact floors of 255 n / 6d, beyond FP16's 11-bit significand.
  Perspective positions reach 512 times the image width, beyond FP16's
  largest value, 65504. Perspective now uses FP32 instead of FP64, which
  holds them within 1/512 pixel below 32768 pixels a side.
- The one place FP16 arithmetic could double throughput is the float
  kernel filters on arm64 (Cortex-A78 has FP16 vectors, 8 lanes). But x86
  has no FP16 arithmetic before AVX512-FP16 (the Core Ultra 9 has none),
  so a fast kernel that gives the same result everywhere would emulate
  FP16 rounding in FP32 there, slower than the FP32 kernel; and these
  filters already beat OpenCV on both machines.

## Round 3: NEON (results at 36580764d)

The published tables give each case's speed-up over Pillow 13.0 on both
machines. On arm64 it was often a fraction of the x86 one: alpha composite
1.01x against 6.85x, the exact transforms 1.1x to 1.25x against 2.0x to
2.9x, the conversions between 3- and 4-byte pixels about 1x against 4.7x
to 6.8x. Part of that is the baseline (Pillow 13.0's x86 wheel is slow on
those conversions, which the arm64 one compiles well and which run at
copy bandwidth there); the rest is NEON code that was missing or slow.
This round went through the cases where Pillow-AllSIMD on arm64 is behind
OpenCV or libvips, or barely ahead of Pillow 13.0.

### Profiling on the Jetson

`perf` works there: `perf_event_paranoid` is 2 and the kernel exposes the
Cortex-A78 PMU, so a process can count its own user-space events. Docker's
default seccomp profile blocks `perf_event_open` unless the container has
`--cap-add PERFMON`. The `pillow-simd-prof` image adds Debian's
`linux-perf` and `valgrind` to the benchmark image. `perf annotate`
crashes (perf 6.12) on `ImagingGenericTransform`. A histogram of the
addresses from `perf script`, mapped onto `objdump`, gives the same
listing (`bench-results/neon/prof/hist.sh`).

### Changed: the same pixels

Jetson Orin, CPU 2, 2560x1600, best of 7, NEON build before and after:

| Case | before (ms) | after (ms) | | OpenCV | fast mode |
|---|---|---|---|---|---|
| rotate RGBA 30 bicubic | 334.3 | 210.8 | 1.59x | 102.4 | 96.4 |
| rotate RGB 30 bicubic | 264.8 | 201.2 | 1.32x | 80.3 | 79.4 |
| rotate RGB 30 bilinear | 108.0 | 73.3 | 1.47x | 35.3 | 40.5 |
| rotate RGBA 30 bilinear | 130.5 | 84.7 | 1.54x | | |
| perspective RGB bilinear | 121.1 | 96.8 | 1.25x | 39.0 | 52.0 |
| perspective RGB bicubic | 317.7 | 244.3 | 1.30x | | |
| quad RGB bilinear | 125.4 | 91.2 | 1.37x | | |
| alpha composite RGBA | 21.9 | 17.6 | 1.25x | 241 | |
| reduce RGB by 4 | 1.64 | 1.43 | 1.15x | 1.06 | |

- **Exact transforms.** 81% of rotate bicubic was in the scalar
  `bicubic_filter32RGB`: 590 instructions and about 180 cycles per pixel,
  3.2 per cycle, so bound by instruction count. The x86 vector filters
  convert each byte to double. On arm64 that form was slower than the
  scalar code (round 1). The new NEON filters keep the scalar code's
  arithmetic:
  - `BILINEAR` and `BICUBIC` get bytes there, so `b - a` and `p1` to `p4`
    are integers. They are computed for the 4 bands of two rows at once
    in 16-bit lanes, and converted to double through float, which holds
    them exactly.
  - The rest is `vfmaq_f64`, the fused multiply-adds that GCC and clang
    make of the scalar expressions on arm64.

  That halved the instructions (33.2G to 17.8G) but gained only 4% for
  RGB bicubic: IPC fell to 1.9. Each pixel is one chain of about 100
  cycles of dependent operations (conversions, three fused multiply-adds
  each way), and its 190 instructions overflow the A78's reorder window,
  so successive pixels hardly overlapped. On top of that, the 3-byte RGB
  store went from a vector register through the stack to two general
  loads, which held 35% of the samples.

  The fix has three parts:
  - Pixels go in pairs, when both have their whole source window inside
    the image (the others take the one-pixel filter).
  - Bytes are stored straight from the vector register.
  - The pair's coordinates are checked in double before any conversion.

  The way GCC compiled the pair was worth 4% to 8%. Bicubic runs better
  written without a loop over the two pixels, bilinear with one, and a
  `tbl` beat narrowing with `uzp` for the bytes of two pixels. The exact
  path is still 2.5x OpenCV: per pixel it converts 64 integers to double
  and runs about 30 two-lane fused multiply-adds. Fast mode's fixed-point
  filters are level with OpenCV.
- **Alpha composite.** One division per pixel, `coef1 = sa 255² 128 /
  outa`, bound the scalar loop at about 12 cycles a pixel. NEON with a
  vector division and an exact correction, or with a reciprocal estimate
  and a correction, was slower than the scalar loop (30.9 and 29.8 against
  26.5 ms in a microbenchmark), and so was a table with scalar channel
  arithmetic (24.6 ms). The division depends only on the two alphas, so
  NEON now reads `coef1` from a table of all 65536 pairs and does the
  channel products in vectors:
  - 128 KB, built at first use in 0.33 ms;
  - an atomic state makes it thread-safe, and a call that finds another
    thread building it runs the scalar loop;
  - `test_alpha_composite_every_alpha` checks every pair against the
    scalar formula.
- **Reduce by 4x4** of 4-byte images: pairwise sums (`vpaddlq`,
  `vpadalq`) over the 4 rows, 8 output pixels per step, rounded with
  `vrshrn`.

### Tried and not kept

- Resize: a horizontal kernel for 4 rows at once, sharing the coefficient
  loads: no gain.
- Transposes of 1-byte images in 16x16 blocks: slower than the 8x8 ones.
- The NEON transform filters a pixel at a time: 1.04x to 1.29x
  (latency, above).
- "transpose L rotate 90" was 20% slower after the changes in every A/B
  run, but by its place in the sequence: run alone, both builds take
  1.48 ms.

### What remains on arm64

- **YCbCr <-> RGB** (2.1x and 1.7x OpenCV, exact) and **RGB -> HSV**
  (2.6x): IPC 1.3 to 1.8 with backend stalls. YCbCr converts each term to
  float and back. A 16-bit integer form that reproduces Pillow's tables
  exactly would avoid that, but it isn't written. Fast mode is at OpenCV's
  level for both.
- **Exact transforms** (2.3x to 2.5x OpenCV): doubles, above.
- **Reduce by 4** (1.35x) and **box blur** radius 2 and 10 (1.5x).
- **RGBA upscales** (3.4x): as in round 2.

## Round 4: fast mode by default, and perf on both machines (2026-10-31)

Fast mode became the default (`754ef42ea`): exact mode is
`PILLOW_ALLSIMD_FAST=0`. So this round compares the default, the published
`-fast` variants, with OpenCV. With the results at `36580764d`, fast mode was
behind OpenCV in 25 of 104 cases on x86 and 37 on arm64. The largest gaps were
the RGBA upscale (arm64 3.4x, x86 1.4x to 1.7x), the 3x3 minimum and maximum
(1.7x on both), the box blur at radius 2 (x86 1.8x) and fast HSV and
YCbCr -> RGB on arm64 (1.3x).

### Profiling on x86

The laptop's `perf_event_paranoid` is 4, which blocks `perf` without
privileges, but a container with `--cap-add PERFMON` may count its own
events, as on the Jetson. `bench-results/prof-x86/` holds a Dockerfile
(`pillow-simd-prof`: the benchmark image plus Debian's `linux-perf` and
`valgrind`) and `prof.sh`, the x86 version of the Jetson scripts:
symbols and counters per case, or `prof.sh hist BUILD CASE SYMBOL` for
the samples per instruction. CPU 2 is a P-core, so the counters come from
the `cpu_core` PMU. A/B timings now use `-O3` builds in the benchmark
image on both machines (`bench-results/round4/x86ab.sh`, `jetson.sh`).
The local `-O2` builds don't unroll some of these loops, so they can hide
or invert a change.

### Changed: the same pixels

Fast mode (the default) before and after, best of 3 rounds, 2560x1600:

| Case | x86 before | x86 after | x86 OpenCV | arm64 before | arm64 after | arm64 OpenCV |
|---|---|---|---|---|---|---|
| RGBA to 2048x1280 bilinear | 11.04 | 6.56 | 24.20 | 36.49 | 27.84 | 47.59 |
| RGBA to 5478x3424 bilinear | 44.67 | 36.30 | 31.27 | 146.47 | 123.34 | 43.80 |
| RGBA to 5478x3424 bicubic | 53.12 | 47.39 | 32.78 | 184.60 | 161.47 | 216.67 |
| RGBA to 320x200 bilinear | 2.70 | 2.01 | 4.81 | 9.83 | 9.72 | 8.86 |
| min 3x3 RGB | 2.08 | 1.72 | 1.21 | 5.18 | 3.76 | 3.08 |
| max 3x3 RGB | 2.07 | 1.87 | 1.21 | 5.22 | 3.69 | 3.11 |
| box blur RGB radius 2 | 5.61 | 3.83 | 2.79 | 15.55 | 15.13 | 10.44 |
| RGB -> HSV | | | | 17.74 | 10.63 | 13.60 |
| YCbCr -> RGB | | | | 7.17 | 4.68 | 5.69 |

- **RGBA and LA resize**: Pillow converts to RGBa, resizes and converts
  back, two sweeps of whole images (the RGBa result of an upscale is 75 MB).
  `ImagingResampleAlpha` premultiplies each source row inside the
  horizontal pass, into a scratch of four rows, since each source row is
  read by one horizontal convolution. It removes the premultiplication
  from the rows of the last pass while they are in the cache. When the
  vertical pass comes first, or alone, it reads each row several times,
  so it gets a premultiplied copy, as before. `perf` on x86 after the
  change (bilinear upscale): unpremultiply 33%, vertical pass 27%,
  horizontal 21%, premultiply 5%, at 4.5 instructions per cycle.
- **Unpremultiply on arm64**: 31% of the same upscale there. The NEON
  kernel made one float division per pixel, but its widenings and
  conversions, not the divider, set the time. The scalar code's
  `CLIP8(255 * c / a)` for every alpha and colour fits a 64 KB byte table.
  Looking up two pixels per 64-bit word, four per iteration, takes 1.39
  ns/px against 2.35. All checked against the scalar code over every pair.
- **Min and max**: `memmove` was 31% of the 3x3 minimum on x86, the copy
  of each row into a padded buffer. The inner bytes now read the row
  itself, and the few at each end clamp the pixel index. The 3x3 filters
  also skip the loop over the sources. They are now bound by memory: 32 MB
  moved per call against OpenCV's 24 MB, the 4/3 of the RGB layout.
- **Box blur, radius 1 to 4** (no running sums): the loop over the box's
  sources reloaded each row pointer for every vector, at IPC 6 and 228
  instructions per 32 bytes. With the count a constant per radius, GCC
  unrolls it at `-O3` (not at `-O2`). On NEON it was already unrolled.
  Its time there goes to loads that miss L1.
- **Fast HSV on arm64**: the hue's numerator and denominator in 16-bit
  lanes, one float division for the hue. The saturation floor(255 d / v)
  is the unpremultiply table's entry for (v, d). Checked against the
  scalar fast kernel on all 2^24 colours.
- **Fast YCbCr -> RGB on arm64**: R and B have one product each, and
  `vqrdmulh` of 2 (C - 128) and K is exactly ((C - 128) K + 2^13) >> 14 in
  16-bit lanes. G sums two products before its one rounding, so it stays
  32-bit, with `vrshrn`.

### Tried and not kept

- Unpremultiply on NEON with a reciprocal estimate and two Newton steps
  (2.84 ns/px), or in 16-bit integers with a remainder correction (2.74):
  both exact, both slower than the division (2.35) and the table (1.39).
- The unpremultiply table on x86: 1.29 ns/px against 0.66 for the AVX2
  kernel.
- The HSV hue by reciprocal estimate on NEON: 2.75 ns/px against 2.48 with
  the division.

### What remains

- **RGBA upscales**: x86 1.16x (bilinear) and 1.45x (bicubic) behind
  OpenCV, arm64 2.8x (bilinear; the bicubic and Lanczos ones are ahead).
  OpenCV doesn't premultiply. Pillow's resample passes compute in 32 bits
  per channel; OpenCV's bilinear uses 11-bit weights in 16 bits. A fast
  kernel with 16-bit arithmetic would close it, as in
  APPROXIMATE_KERNELS_PLAN.md item 5.
- **Layout**: copies, flips, chops, L -> RGB and the min, max and box
  filters move 4 bytes per RGB pixel against OpenCV's 3, 1.3x to 1.6x.
- **Median 3x3** (x86 1.3x, arm64 1.6x), **reduce by 2 and 4** on arm64
  (1.4x), `point/L/table` on x86 (1.46x, no byte shuffle over 256
  entries without AVX-512 VBMI), and text (Hershey strokes, not glyphs).
