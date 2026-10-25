Changelog (Pillow-SIMD)
=======================

Unreleased
----------

Based on Pillow 13.0.0

- Merged Pillow 13.0.0: security fixes, free-threading locks (the SIMD
  paths run inside them), vertical-first resampling and the MKS2013 and
  MKS2021 filters, which use the SIMD resample kernels (MKS2021 RGB 1/4:
  about 10x the 13.0 wheel on AVX2)
- arm64: NEON again for the horizontal box blur of 4-byte images and the
  unsharp mask row, which beat the compiler's code with GCC 12 and 14
  (gaussian RGB 1.27x, unsharp RGB 1.31x on Cortex-A78AE); 13.0's
  histogram loops on arm64 (1.1x)
- Rank, median, minimum and maximum filters of 1-byte images up to 11x11
  use a sorting network pruned for the rank, vectorised by the compiler:
  median 5 65x, rank 7 50x faster than Pillow 13.0 (2560x1600)
- Rank filters read the image's edges instead of an expanded copy (median 3
  2.5x); ImageDraw: vector span fill and RGBA ink blend (filled shapes
  1.6x to 2.3x), masked fills for text and bitmap(); floodfill in C (60x)
- point(): XOR, AND and solarize tables as byte operations (ImageOps
  invert, posterize, solarize 3.3x), output not zero-filled; ImageEnhance
  contrast and brightness as lookup tables (1.5x)
- PNG encoding: vector row filters (level 1 1.7x); RGB <-> YCbCr without
  table lookups (1.6x, 3.1x)
- Text: rendered glyphs and their boxes are cached per font (text 5.9x,
  stroked text 17x, textbbox 2.1x)
- Palette mapping (quantize, convert to an adaptive palette): a colour cache
  and, for max coverage, a vector nearest-entry search (1.2x to 1.7x)
- Transposes and 90 degree rotations in register blocks (L 3.2x, RGB 1.4x);
  flips and rotate 180 by vector row reversal (RGB 1.4x, L 10x)
- Resampling of F, I and I;16 images in vectors of doubles, every sum in
  the scalar code's order (1.8x to 5.5x)
- Arcs, chords and pie slices are clipped without allocating (1.25x);
  alpha_composite() with an offset composites in place (2.2x); tobytes()
  encodes in one chunk (1.9x)
- Resize: the AVX-VNNI build no longer puts vpdpwssd in the horizontal
  pass, where each pixel is one long chain of additions: shrinks with long
  kernels (2560x1600 to 26x16) 1.3x to 1.6x faster, now level with AVX2;
  RGB in the portable build sums three channels, not four (1.2x, level with
  Pillow 13.0 for big shrinks)
- RGB -> HSV: AVX2 kernel for 8 pixels, the same operations per lane as
  the SSE4 one; identical for all 16.7 million RGB values; 2.2x
- RGB -> CMYK, L -> RGB/LA, RGB <-> RGBA a pixel at a time, which the
  compiler vectorizes (1.6x to 2.9x; the AVX2 build's CMYK was 15% slower
  than SSE4); ImageChops add and subtract with scale 1 in integers, other
  scales by table, lighter and darker without clipping (add 1.6x)
- BoxBlur, GaussianBlur, UnsharpMask: the passes of each direction run
  together, every horizontal pass on a row while it is in cache, the
  vertical ones in one sweep with a ring of rows per pass; AVX2 blurs two
  rows at once horizontally and 32 bytes per step vertically. Same
  output. AVX2: Gaussian RGB 2x, box blur 2.3x, unsharp mask 1.8x
- Rank filters (min, max, median, any rank up to 11x11) on RGB, RGBA,
  CMYK, LA and other 4-byte modes run all bands in one pass instead of
  splitting and merging; min and max keep their row results in a ring of
  rows and vectorize both passes; 3x3 and 5x5 medians in registers
  (SSE4, AVX2, NEON; Devillard's 99-comparator network for 5x5). RGB
  min/max 3x3 11x, median 3x3 5.3x, median 5x5 2.4x, L median 3x3 2.8x,
  median 5x5 2.5x (AVX2)
- Images take transparent huge pages on Linux, as numpy arrays do: box
  blur 1.5x (its vertical pass), ImageChops lighter 1.6x, most passes
  over large images a few percent
- x86: RGBa -> RGBA is exact (colour above alpha gave 0, alpha 0 gave
  colour 0); slightly slower
- ``checks/simd_test_remote.sh`` runs the test suite per SIMD build in a
  container, on a remote host or locally, so the host's Python version does
  not matter

Build

- ``-C simd=auto|sse4|avx2|avxvnni|native|native512|neon|disable``
  option; flags chosen per target architecture; SSE4.1 baseline on x86
  (was SSE4.2). ``native`` builds for the build machine's CPU
  (``-march=native``, float contraction off), ``native512`` also prefers
  512-bit vectors; both only run on CPUs like the build machine's
- AVX-VNNI build (``-C simd=avxvnni``): the 16-bit multiply-add of the
  resample and integer filter kernels becomes one vpdpwssd. Same output;
  integer filters on RGB 1.3x to 2.1x faster than AVX2
- arm64: NEON kernels; other architectures build portable C code

New and faster kernels

- Resize of 1-byte images (L): SIMD horizontal pass (SSE4, NEON), 2.4x to
  8x faster on x86, 2.5x to 3.3x on arm64
- Reduce: 1x2, 2x1, 2x2 about 2x faster on x86; 2x2 and 4x4 for 1-byte
  images on SSE4 and NEON, 6x to 10x faster; NEON for 4-byte images with
  horizontal factors up to 512 (1.6x to 4.5x on arm64); restored 4x4 and 5x5
  for 1-byte images
- Filter: integer kernel for L images, up to 2x faster; on arm64 NEON
  integer kernels and NEON float kernels for 4-byte images (FIND_EDGES on
  RGB 1.7x). Like the x86 float kernels, the float ones can differ by 1
  from the scalar code for inexact coefficients
- BoxBlur, GaussianBlur: vertical pass without transposes, up to 1.8x
  faster; on x86 1-byte images are blurred four rows at a time, 1.9x to 3x
- UnsharpMask: SIMD post-pass on x86, about 1.7x faster
- ColorLUT: NEON kernel (1.5x), output identical to the scalar code
- ImageChops.multiply, screen: exact division by 255 without a divide;
  SSE4, AVX2 and NEON. 6.4x to 12x faster on x86, 3.4x on arm64
- RankFilter, MinFilter, MaxFilter, MedianFilter on 1-byte images: a
  sorting network for 3x3 windows (any rank) and row-then-column minimum
  and maximum for any size, 14x to 130x faster; the GIL is released
- Filter with integer kernels on L images: AVX2 path, 32 pixels per step,
  2x faster than SSE4 (2.2x with AVX-VNNI); output unchanged
- Quantize with median cut or maximum coverage, convert("P", ADAPTIVE):
  the color hash table takes its nodes from blocks instead of one malloc
  each; the same chains, so the same palette and pixels. 1.7x faster on a
  2560x1600 image of random colors
- RGB to HSV: SSE4 kernel for 4 pixels with the float and double steps of
  the scalar code, and fmod(h, 1.0) replaced by its exact equivalent for
  the range h is in. 5.4x faster than upstream on SSE4, 1.4x in the scalar
  build; output identical for all 16.7 million RGB values
- Filter with 3x3 float kernels on L images (CONTOUR, FIND_EDGES): AVX2
  path, 8 pixels per step, 1.9x faster than SSE4; output unchanged
- Rotate and transform (affine, perspective, quad, mesh): the output is
  computed in 64x64 tiles, which keeps the source rows in cache; bilinear
  and bicubic filters for 3- and 4-band images take all channels in one
  AVX2 vector of doubles (two with SSE4). Output identical. On x86, rotate
  30 degrees 2560x1600 RGB: nearest 3x to 4.6x, bilinear 1.6x (SSE4) to 2x
  (AVX2), bicubic 1.4x to 2.2x; perspective 2x to 2.5x. The tiles alone
  make the scalar build 1.2x (bilinear, bicubic) to 4x (nearest) faster
- ModeFilter: the window histogram is updated by columns, and windows
  of fewer than 16 values check only their own values; 13x faster for
  size 3, 3.4x to 4.8x for sizes 5 to 17; output unchanged; the GIL is
  released
- Histogram of 8-bit images: counts split over several tables, 1.2x to
  2.4x faster
- ImageChops.offset: two row copies instead of a modulo per pixel, 13x
  faster
- RGB pack and unpack (frombytes, tobytes, codecs): SSE4 and NEON,
  1.1x to 1.7x faster
- arm64: alpha composition and the vertical blur pass use the portable C
  code, which GCC vectorises better than the previous NEON kernels did
- Released the GIL in AlphaComposite, Bands, Blend and Chops

Fixes

- BoxBlur, GaussianBlur: fixed swapped channels for single-axis blurs on
  4-byte modes; fixed reads past 1-pixel lines
- Filter: fixed I;16 images going to the 8-bit kernel
- Reduce: fixed 1x3 rounding
- AlphaComposite, Convert: results no longer depend on the pixel's position
  in the row
- ImageChops.offset: fixed reads and writes past the end of each row for
  I;16 images
- From Pillow 13.0: I;16 filters keep the high byte and clamp to 0..65535;
  BoxBlur and GaussianBlur reject NaN, infinite and >= 2**31 radii;
  ImagingExpand's size check no longer overflows or divides by zero; masked
  histograms of LA images count the alpha band

9.5.0.post2 & 9.4.0.post2 & 9.3.0.post2 & 9.2.0.post2 & 9.1.1.post2 & 9.0.0.post3
---------------------------------------------------------------------------------

- Fixed color overflow in LUT processing

9.0.0.post1
-----------

- Fixed possible overflow in LUT processing
- Restored compatibility with Visual C Compiler

7.0.0.post4
-----------

- Filter: fixed wrong offset handling for 3x3 single-band version

7.0.0.post3
-----------

- ColorLUT: fixed potential access violation, up to 2x faster

7.0.0.post2
-----------

- ColorLUT: SSE4 & AVX2

7.0.0.post1 & 6.2.2.post1 & 6.1.0.post1 & 6.0.0.post2
-----------------------------------------------------

- Bands: access violation in ``getband`` in some environments

7.0.0.post0
-----------

- Reduce: SSE4

6.0.0.post1
-----------

- GCC 9.0+: fixed unaligned read for ``_mm_cvtepu8_epi32`` and
  ``_mm256_cvtepu8_epi32`` functions.

6.0.0.post0 and 5.3.0.post1
---------------------------

- Resampling: correct max coefficient calculation. Some rare combinations of
  initial and requested sizes lead to black lines.

4.3.0.post0
-----------

- Float-based filters, single-band: 3x3 SSE4, 5x5 SSE4
- Float-based filters, multi-band: 3x3 SSE4 & AVX2, 5x5 SSE4
- Int-based filters, multi-band: 3x3 SSE4 & AVX2, 5x5 SSE4 & AVX2
- Box blur: fast path for radius < 1
- Alpha composite: fast division approximation
- Color conversion: SSE4 RGB-to-L conversion, fast division in RGBa-to-RGBA
  conversion
- Resampling: optimized coefficients loading
- Split and get_channel: SSE4

3.4.1.post1
-----------

- Fixed a critical memory error for some combinations of source and destination
  sizes.

3.4.1.post0
-----------

- Numerous resampling optimizations, including a 16-bit intermediate color
  representation and heavy unrolling.

3.3.2.post0
-----------

- Maintenance release

3.3.0.post2
-----------

- Fixed an error in RGBa -> RGBA conversion

3.3.0.post1
-----------

Alpha compositing
~~~~~~~~~~~~~~~~~

- SSE4 and AVX2 fixed-point full-loading implementations.
  Up to 4.6x faster.

3.3.0.post0
-----------

Resampling
~~~~~~~~~~

- SSE4 and AVX2 fixed-point full-loading horizontal pass.
- SSE4 and AVX2 fixed-point full-loading vertical pass.

Conversion
~~~~~~~~~~

- RGBA -> RGBa SSE4 and AVX2 fixed-point full-loading implementations.
  Up to 2.6x faster.
- RGBa -> RGBA AVX2 implementation using gather instructions.
  Up to 5x faster.


3.2.0.post3
-----------

Resampling
~~~~~~~~~~

- SSE4 and AVX2 float full-loading horizontal pass.
- SSE4 float full-loading vertical pass.


3.2.0.post2
-----------

Resampling
~~~~~~~~~~

- SSE4 and AVX2 float full-loading horizontal pass.
- SSE4 float per-pixel loading vertical pass.


2.9.0.post1
-----------

Resampling
~~~~~~~~~~

- SSE4 and AVX2 float per-pixel loading horizontal pass.
- SSE4 float per-pixel loading vertical pass.
- SSE4: up to 2x for downscaling and 3.5x for upscaling.
- AVX2: up to 2.7x for downscaling and 3.5x for upscaling.


Box blur
~~~~~~~~

- Simple SSE4 fixed-point implementations with per-pixel loading.
- Up to 2.1x faster.
