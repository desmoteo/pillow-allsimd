# Where OpenCV and libvips are still faster than Pillow-SIMD, and why

As of `simd/pillow-13` at 28b55224b (2026-10-31), against the libraries in
the [pillow-simd-perf](https://desmoteo.github.io/pillow-simd-perf/)
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
in Pillow-SIMD and 15.8 in OpenCV, so the time ratio is again the byte ratio.
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

| | Pillow-SIMD | OpenCV |
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
