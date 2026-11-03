.. _allsimd-kernels:

How the kernels work
====================

This page explains Pillow-AllSIMD's SIMD kernels, operation by operation:
what Pillow computes, the trick that makes it fast, how the x86-64 (SSE4.1,
AVX2, AVX-VNNI) and arm64 (NEON) versions do it, why the result is the same
on every build, and what it gains. It is written for a C programmer who
knows what a SIMD register is but not Pillow's internals, so that a
contributor or a reviewer can read a kernel without reverse-engineering its
intrinsics. Code excerpts are short and name their file and function; the
files are in ``src/libImaging/``.

The speed-ups quoted are from the
`benchmark page <https://desmoteo.github.io/pillow-allsimd-perf/>`_: one
thread, a 2560×1600 image, the best of three rounds, on an Intel Core
Ultra 9 285H (x86-64, GCC 14) and an NVIDIA Jetson Orin (arm64,
Cortex-A78AE, GCC 14). Unless a section says "fast mode", they are for
exact mode, Pillow's arithmetic.

The page is written section by section. Written so far: the ground rules
and resampling. To come: blurs, kernel and rank filters, colour
conversions, compositing and channel operations, geometry, drawing and
text, codecs and packing, what was tried and dropped, where OpenCV and
libvips are still faster, and the fast-mode kernels (meanwhile,
``design/SIMD_FAST_KERNELS.md`` in the repository describes those).

Ground rules
------------

Every kernel on this page was shaped by a few facts about Pillow, the
compilers and the two architectures.

**The same pixels on every build.** A SIMD build must produce the pixels
of the portable C build, bit for bit, so that an image processed on an
AVX2 laptop, an arm64 server and a RISC-V board is the same image. The
portable code is the definition: each kernel's scalar version does the
arithmetic in a form the vector version can reproduce exactly, usually
integer sums of products with one rounding at the end, rather than the
floating-point loops Pillow once had. The exceptions are documented in
``checks/simd_hashes.py``: the x86 alpha composite rounds a division
differently from the portable code by one level, and on arm64 the
compilers fuse floating-point multiply-adds, so the kernels that must
match double sums (the F, I and I;16 resample, the exact transforms) use
fused multiply-adds explicitly and differ from x86 only where the scalar
code would too. Against Pillow itself, resize uses 16-bit coefficients
where Pillow uses 32-bit ones and RGB to L uses 15-bit weights, each off
by one level in a few places; everything else in exact mode is Pillow's
arithmetic.

The checks: ``checks/simd_hashes.py`` prints one hash per kernel area over
many sizes and modes, and the hashes of a scalar, SSE4, AVX2, AVX-VNNI and
NEON build must agree; ``Tests/test_simd_consistency.py`` checks that the
pixels a vector loop leaves for its scalar tail get the same arithmetic;
exhaustive checks cover the conversions whose input space is small enough
(every RGB value for HSV, every colour and alpha for unpremultiplication);
and Pillow's test suite runs on every build.

**Compile-time instruction sets, no run-time dispatch.** The build picks
the kernels with ``-C simd=`` (``sse4``, ``avx2``, ``avxvnni``, ``neon``,
``native``, ``disable``; :ref:`allsimd-install`), and ``ImagingSIMD.h``
turns the compiler's macros into ``IMAGING_SIMD_SSE4``,
``IMAGING_SIMD_AVX2``, ``IMAGING_SIMD_AVXVNNI`` and ``IMAGING_SIMD_NEON``.
Every kernel file is one ``#if`` ladder: the SSE4 version, with ``#if
defined(IMAGING_SIMD_AVX2)`` blocks inside it where 256-bit code exists;
the NEON version; and the portable version under ``#else``. A wheel can't
carry an AVX2 build, since pip would install it on a CPU without AVX2
(the plan for run-time dispatch is in ``design/PLAN.md``). On arm64, NEON
is part of the baseline, so the wheels have it.

``ImagingSIMD.h`` also holds the helpers the kernels share:
``SIMD_FORCE_INLINE`` for kernels that must see a shift count as a
constant; ``mm_dpwssd`` and ``mm256_dpwssd``, one ``vpdpwssd`` with
AVX-VNNI and ``pmaddwd`` plus ``paddd`` without, and ``mm_madd_add``,
always the pair, for sums that are one long dependency chain (below);
``permute4``, which shuffles the bytes of 4-byte pixels with constant
indices; and ``ImagingPx4.h``, one pixel as four 32-bit lanes with the
same modulo-2³² arithmetic in SSE4, NEON and plain C.

**Pillow's memory layout.** An image is an array of row pointers
(``im->image[y]``). Modes of one 8-bit band (L, P) are stored as bytes
(``image8``); every other 8-bit mode is stored as 4 bytes per pixel
(``image32``): RGB with a padding byte of 255, RGBA, RGBX, CMYK, YCbCr,
LAB, HSV, and LA and La as L, L, L, A. So "the kernel for 4-byte pixels"
serves all of them, and RGB pays for a fourth channel it doesn't use.
Rows of all sizes live in blocks of up to 16 MB with transparent huge
pages where the kernel offers them (``madvise``). I, F and I;16 images use
4 or 2 bytes per pixel and their own kernels.

**What the compilers do and don't do.** Three facts decided more designs
than any intrinsic:

* Every common Python passes ``-fno-strict-overflow``, so GCC can't
  vectorize a loop whose address is ``in + x * 4`` with an ``int`` ``x``:
  the multiplication may wrap. Portable loops meant to vectorize use
  pointer increments, or fixed-size chunks with ``restrict`` pointers (the
  ``CONVERT_PIXELS`` pattern in ``Convert.c``).
* Debian, Ubuntu and conda build Python with ``-O2``, the Docker images
  with ``-O3``. At ``-O2`` GCC vectorizes only loops with a fixed trip
  count and no aliasing, so the portable kernels are written that way,
  and the hot paths have explicit SIMD.
* On arm64 GCC fuses ``a * b + c`` into one rounding by default. Kernels
  whose results must match scalar double sums use ``vfmaq_f64``
  explicitly, so the result is the same whatever the compiler fuses.

And two about the hardware. On the Cortex-A78AE, a 32-bit vector multiply
and even a vector division are cheap, and a hand-written NEON loop beats
the compiler's vectorization of the portable code by a smaller margin than
SSE4 beats scalar on x86, sometimes by none: several x86 tricks (a 12-bit
weight split to keep products in 16 bits, a reciprocal estimate with a
correction step) were slower than the plain instruction on arm64 and
weren't ported. And the two instruction sets count differently: x86's
``pmaddwd`` multiplies 16-bit pairs and sums them into 32 bits, where NEON
has ``vmlal`` (multiply-accumulate with widening) but no pair sum; x86's
``pshufb`` and NEON's ``tbl`` both permute bytes but zero a lane on
different index values; and AVX2 operates on two 128-bit halves, so data
that must cross between them costs an extra instruction.

Resampling
----------

``Image.resize()`` with any filter but ``NEAREST`` (``Resample.c``). This
is the kernel Pillow-SIMD was written for, and the one with the most
variants: 8-bit images with 4-byte pixels, 8-bit images with 1-byte
pixels, and I, F and I;16 images, each with a horizontal and a vertical
pass.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

A resize is a separable convolution: a horizontal pass makes an image of
the output width and the input height, and a vertical pass brings that to
the output height. For each output column, ``precompute_coeffs()`` finds
the input pixels under the filter's support (``xmin``, and ``xmax`` of
them) and their weights, evaluates the filter in doubles and normalizes
the weights to sum to 1. When the image shrinks, the filter's support is
scaled by the shrink factor, so a 1/8 Lanczos downscale has 48 taps per
output pixel and a 2x upscale has 6.

For 8-bit images the weights become 16-bit integers.
``normalize_coeffs_8bpc()`` picks the largest precision, up to 22 bits,
at which the largest weight times 2 still fits a signed 16-bit value, and
rounds each weight to that many fractional bits. A pixel is then the sum
of ``pixel * weight`` in a 32-bit accumulator that starts at half a unit
(``1 << (coefs_precision - 1)``), shifted right by the precision and
clipped to 0 to 255 (``clip8()``, a table indexed by the shifted sum,
since a filter with negative lobes can give sums outside the range). The
portable kernel is that, written out:

.. code-block:: c

    /* ResampleSIMDHorizontalConv.c, the portable
       ImagingResampleHorizontalConvolution8u */
    ss0 = ss1 = ss2 = ss3 = 1 << (coefs_precision - 1);
    for (x = 0; x < xmax; x++) {
        ss0 += in[x * 4 + 0] * k[x];
        ss1 += in[x * 4 + 1] * k[x];
        ss2 += in[x * 4 + 2] * k[x];
        ss3 += in[x * 4 + 3] * k[x];
    }
    out[0] = clip8(ss0, coefs_precision);

Pillow's own resize does the same sums with 32-bit weights at a fixed
precision. The 16-bit weights are what let a vector instruction multiply
8 or 16 of them at once, and they differ from Pillow's result by one
level in a few pixels.

The idea
~~~~~~~~

Three things, on top of the vectors themselves:

* **Two taps per multiply.** ``pmaddwd`` multiplies eight 16-bit pairs
  and adds each pair into a 32-bit lane. If a register holds the bytes of
  two neighbouring input pixels interleaved, channel by channel (``r0 r1
  g0 g1 b0 b1 a0 a1``, each widened to 16 bits), and another holds their
  two weights repeated (``k0 k1 k0 k1 …``), one instruction adds two taps
  of all four channels to the accumulator.
* **Four rows at once.** The horizontal pass has the same bounds and
  weights for every row, so a kernel that resamples four rows in one loop
  shares the loads of the weights and the loop control between them, and
  keeps four independent accumulators in flight.
* **A pass order that reads less.** When the height shrinks more than
  twice as much as the width, the vertical pass runs first
  (``ImagingResampleInner()``, ``horizontal_first``), so the horizontal
  pass runs over the shorter image; otherwise the horizontal pass runs
  first and only over the source rows the vertical pass will read.

x86-64
~~~~~~

The horizontal kernel for 4-byte pixels
(``ImagingResampleHorizontalConvolution8u4x``, SSE4 with AVX2 blocks)
loads 16 bytes, four pixels, of each of two rows into the two halves of a
256-bit register, then two ``pshufb`` shuffles spread them into the
interleaved form, one for pixels 0 and 1, one for pixels 2 and 3, each
followed by a multiply-add with the matching pair of weights:

.. code-block:: c

    mmk0 = _mm256_set1_epi32(*(INT32 *)&k[x]);      /* k[x], k[x+1] */
    mmk1 = _mm256_set1_epi32(*(INT32 *)&k[x + 2]);  /* k[x+2], k[x+3] */
    source = _mm256_inserti128_si256(                /* row 0 low, row 1 high */
        _mm256_castsi128_si256(_mm_loadu_si128((__m128i *)&lineIn0[x + xmin])),
        _mm_loadu_si128((__m128i *)&lineIn1[x + xmin]), 1);
    pix = _mm256_shuffle_epi8(source, /* bytes 0,4 1,5 2,6 3,7 widened */);
    sss0 = mm256_madd_add(sss0, pix, mmk0);
    pix = _mm256_shuffle_epi8(source, /* bytes 8,12 9,13 10,14 11,15 */);
    sss0 = mm256_madd_add(sss0, pix, mmk1);

so each step consumes four taps of two rows. Rows 2 and 3 take the same
steps into a second accumulator. Shorter loops handle the taps left over
two at a time and one at a time; then the accumulators are shifted right
by the precision (``srai``), narrowed with saturation to 16 bits
(``packs_epi32``) and to bytes (``packus_epi16``), which is the clip, and
stored as one 32-bit pixel per row. Without AVX2 the same happens in
128-bit registers, one row per register.

Here ``mm256_madd_add`` is deliberately the ``pmaddwd`` plus ``paddd`` pair
even on AVX-VNNI, where one ``vpdpwssd`` could do both. Each output pixel
is a single chain of additions into one accumulator, and ``vpdpwssd`` puts
its multiply inside that chain, about five cycles per step instead of
one; with the long kernels of a 1/100 downscale the AVX-VNNI build was
1.5x slower than AVX2 until this was changed. The vertical kernel, with
four to eight independent accumulators per step, does use ``vpdpwssd``.

The vertical kernel (``ImagingResampleVerticalConvolution8u``,
``ResampleSIMDVerticalConv.c``) is simpler, because the taps of an output
row are whole source rows: every byte of a row is multiplied by the same
weight, whatever its channel. It takes 32 bytes of two source rows,
interleaves them byte by byte (``unpacklo_epi8``, ``unpackhi_epi8``),
widens to 16 bits with another unpack against zero, and multiply-adds
with the two rows' weights, 8 pixels per step in four accumulators; then
the same shift and saturating packs. The 1-byte kernel for L images
(``ImagingResampleHorizontalConvolution1u4x``) resamples four rows at
once and takes its taps 8, 4 and 2 at a time as 16-bit products; the
vertical pass serves L images unchanged, since it works on bytes.

arm64
~~~~~

NEON has no pair-summing multiply, so the horizontal kernel takes one tap
per instruction and four channels per lane group:
``neon_resample_horiz_px()`` loads 16 bytes, widens them to two vectors of
eight 16-bit values (``vmovl_u8``), and multiply-accumulates each pixel's
four channels by its weight as a scalar (``vmlal_n_s16``), alternating
between two accumulators so that the two chains of dependent adds run in
parallel:

.. code-block:: c

    uint8x16_t source = vld1q_u8(in + x * 4);
    int16x8_t lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(source)));
    int16x8_t hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(source)));
    sss = vmlal_n_s16(sss, vget_low_s16(lo), k[x]);
    sss1 = vmlal_n_s16(sss1, vget_high_s16(lo), k[x + 1]);
    sss = vmlal_n_s16(sss, vget_low_s16(hi), k[x + 2]);
    sss1 = vmlal_n_s16(sss1, vget_high_s16(hi), k[x + 3]);

The sums are added at the end, shifted (``vshlq_s32`` by a negative
count) and narrowed with saturation to 16 bits and to bytes
(``vqmovn_s32``, ``vqmovun_s16``), the same clip as ``packs`` and
``packus``. The four-row variant calls this once per row. The vertical
kernel widens 32 bytes of each source row and accumulates them in eight
vectors, 32 bytes of output per step, with a 16-byte and a 4-byte tail.
Rows are read in order, as on x86.

The portable vertical kernel is written to be vectorized at ``-O2``: it
sums chunks of 64 bytes into a fixed array of 32-bit sums, and GCC turns
the inner loop into vector code since its trip count is fixed. The
portable horizontal kernel for RGB skips the padding byte, a quarter of
the work, and writes 255 to it, as the decoders do.

Why the builds agree
~~~~~~~~~~~~~~~~~~~~

All three versions compute the same integers: the same 16-bit weights,
32-bit sums in any order (integer addition commutes), the same initial
half unit, the same arithmetic shift and the same saturation to 0 to
255. Saturating a 32-bit sum to 16 bits before saturating to a byte gives
the byte clip too, because a sum that overflows 16 bits is far outside 0
to 255 either way. So the SSE4, AVX2, AVX-VNNI, NEON and portable builds
produce identical pixels, and ``checks/simd_hashes.py`` checks it.

I, F and I;16 images (``ResampleSIMD32bpc.c``) are the exception that
proves the rule: Pillow sums their taps in doubles, in order, and a
vector of doubles can only reproduce that if each lane computes one
pixel's sum in the same order. So the vertical kernel's lanes are
neighbouring pixels of a row, which share their taps, and the horizontal
kernel's lanes are the same pixel of neighbouring rows, read from a copy
of the rows with their pixels interleaved. Each lane adds the same
products in the same order as the scalar code, and the results match it;
on arm64 both use fused multiply-adds, so they match the arm64 scalar
build and differ from x86 as Pillow's own builds do.

RGBA and LA images are resized premultiplied by alpha, as Pillow does by
converting to RGBa or La and back. The passes premultiply each source row
as they read it and unpremultiply each output row as they write it, while
the rows are in the cache (``struct resample_alpha``), which gives the
same pixels as the two whole-image conversions without them. Fast mode
replaces the two passes with one fused kernel for 4-byte pixels
(``ResampleFast.c``), described with the other fast kernels.

Reduce
~~~~~~

``Image.reduce()`` and the first step of a resize with ``reducing_gap``
average boxes of whole pixels. ``Reduce.c`` has a kernel per small factor
(1×2, 2×1, 2×2, 1×3, 3×1, 3×3, 4×4, 5×5) and general N×N, N×1 and 1×N
ones; each box's sum is scaled by a fixed-point reciprocal
(``division_UINT32()``) rather than divided, and rounded. ``ReduceSIMD.c``
has the versions for 4-byte pixels, which use the same arithmetic on four
channels per lane group, so the scalar and vector builds agree.

Numbers
~~~~~~~

2560×1600 RGB, exact mode, times faster than Pillow 13.0.0; Pillow 13's
own time in milliseconds for scale. The arm64 column is the NEON build
on the Jetson Orin.

.. list-table::
   :header-rows: 1

   * - Resize
     - Pillow 13 (x86, ms)
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - To 320×200, bilinear
     - 11.9
     - 6.9x
     - 10.8x
     - 15.4
     - 2.4x
   * - To 320×200, bicubic
     - 22.3
     - 7.0x
     - 10.9x
     - 30.2
     - 2.5x
   * - To 320×200, Lanczos
     - 33.1
     - 6.7x
     - 10.2x
     - 46.3
     - 2.6x
   * - To 2048×1280, bicubic
     - 46.5
     - 5.6x
     - 8.4x
     - 58.5
     - 2.0x
   * - To 5478×3424, bicubic
     - 225.7
     - 4.8x
     - 6.4x
     - 259.6
     - 2.0x
   * - RGBA to 320×200, bicubic
     - 37.8
     - 7.6x
     - 12.6x
     - 44.8
     - 2.9x
   * - L to 320×200, bicubic
     - 10.8
     - 9.5x
     - 10.1x
     - 11.9
     - 3.7x
   * - Reduce by 2
     - 3.3
     - 4.5x
     - 4.5x
     - 5.7
     - 3.8x

AVX-VNNI adds up to 4% over AVX2 on these. Against Pillow-SIMD 12.1.1
with AVX2 on the same machine, the downscales are equal and the upscales
to 5478×3424 are 1.8x faster. Against OpenCV, the AVX2 build is faster on
every downscale here (1.8x on the 320×200 bicubic one, 1.1x on the
Lanczos one), and slower on the RGBA upscales to 5478×3424, by up to
1.45x; on arm64 OpenCV, with its KleidiCV kernels, is 1.8x faster on the
320×200 bicubic downscale. ``design/PERFORMANCE_GAPS.md`` discusses the
gaps. Fast mode
(the default) adds a fused two-pass kernel and the ``reducing_gap``
default for large downscales: to 320×200 bicubic 15.8x on x86 and 6.0x
on arm64; see :ref:`fast-mode`.

Code map
~~~~~~~~

* ``Resample.c``: the filters, ``precompute_coeffs()``,
  ``normalize_coeffs_8bpc()``, the pass drivers
  (``_ImagingResampleHorizontal_8bpc`` and friends),
  ``ImagingResampleInner()`` with the pass order and the fast-mode entry,
  ``ImagingResampleAlpha()``.
* ``ResampleSIMDHorizontalConv.c``: the horizontal kernels for 4-byte and
  1-byte pixels, SSE4/AVX2, NEON and portable, included twice by
  ``Resample.c`` (the second time with ``RESAMPLE_OUT16`` for fast mode's
  16-bit intermediate rows).
* ``ResampleSIMDVerticalConv.c``: the vertical kernel, three versions.
* ``ResampleSIMD32bpc.c``: I, F and I;16 in vectors of doubles.
* ``ResampleFast.c``: fast mode's fused resize.
* ``Reduce.c`` and ``ReduceSIMD.c``.

Pitfalls met on the way: ``vpdpwssd`` in a single dependency chain
(above); a vertical-first pass order for every downscale was measured and
not adopted, because the result would then depend on the build; and the
horizontal kernel file is compiled twice, so a change to the 8-bit
kernels changes fast mode's 16-bit ones with it, which is intended, and
their tests cover both.
