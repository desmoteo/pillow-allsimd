Changelog (Pillow-SIMD)
=======================

Unreleased
----------

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
- arm64: blur, unsharp mask and alpha composition use the portable C
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
