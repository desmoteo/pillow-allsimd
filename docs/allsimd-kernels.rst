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

The sections: the ground rules; then, operation by operation, resampling,
the blurs, the kernel filters, the rank filters, the colour conversions,
compositing and the channel operations, geometry, drawing and text, and
codecs and packing; then what was tried and dropped, where OpenCV and
libvips are still faster, the fast-mode kernels, and three appendices:
the overall numbers, a file map and the tools.

Ground rules
------------

Every kernel on this page was shaped by a few facts about Pillow, the
compilers and the two architectures.

**The same pixels on every build.** A SIMD build must produce the pixels
of the portable C build, bit for bit, so that an image processed on an
AVX2 laptop, an arm64 server and a RISC-V board is the same image, and
the tiers of one x86-64 build must agree with one another. The
portable code is the definition: each kernel's scalar version does the
arithmetic in a form the vector version can reproduce exactly, usually
integer sums of products with one rounding at the end, rather than the
floating-point loops Pillow once had. The exceptions are documented in
``checks/simd_hashes.py``: the x86 alpha composite rounds a division
differently from the portable code by one level, the x86 3D colour
lookup table interpolates at Pillow-SIMD's 16-bit precision, up to two
levels from Pillow's, and on arm64 the compilers fuse floating-point multiply-adds, so the kernels that must
match double sums (the F, I and I;16 resample, the exact transforms) use
fused multiply-adds explicitly and differ from x86 only where the scalar
code would too. Against Pillow itself, resize uses 16-bit coefficients
where Pillow uses 32-bit ones and RGB to L uses 15-bit weights, each off
by one level in a few places, and the kernel filters use 16-bit
coefficients, which can't hold 1/13 or 1/100 exactly (so ``SMOOTH_MORE``
differs by one level in a few percent of the pixels); everything else in
exact mode is Pillow's arithmetic.

The checks: ``checks/simd_hashes.py`` prints one hash per kernel area over
many sizes and modes, and the hashes of a scalar, SSE4, AVX2, AVX-VNNI and
NEON build must agree; ``Tests/test_simd_consistency.py`` checks that the
pixels a vector loop leaves for its scalar tail get the same arithmetic;
exhaustive checks cover the conversions whose input space is small enough
(every RGB value for HSV, every colour and alpha for unpremultiplication);
and Pillow's test suite runs on every build.

**One instruction set per build, through a table of kernels.** The build
picks the instruction set with ``-C simd=`` (``sse4``, ``avx2``,
``avxvnni``, ``neon``, ``native``, ``disable``; :ref:`allsimd-install`),
and ``ImagingSIMD.h`` turns the compiler's macros into
``IMAGING_SIMD_SSE4``, ``IMAGING_SIMD_AVX2``, ``IMAGING_SIMD_AVXVNNI`` and
``IMAGING_SIMD_NEON``. The SIMD code of each operation lives in a
companion file, ``<Area>SIMD.c``, one ``#if`` ladder: the SSE4 version,
with ``#if defined(IMAGING_SIMD_AVX2)`` blocks inside it where 256-bit
code exists, and the NEON version. Every companion is compiled into one
translation unit, ``SIMDKernels.c``, whose only export is a table of
kernel pointers (``SIMDKernels.h``, one group per area); Pillow's own
files keep the portable code and call a kernel through ``ImagingSIMD``,
the table of the tier chosen when ``_imaging`` loads (``SIMDDispatch.c``;
a scalar table, every pointer NULL, for ``disable`` and for
architectures without kernels). A row kernel does the pixels its vector
loop can and returns how many, and the scalar file finishes the row; a
kernel for a whole operation returns 1 when it did the work. The call
is once per row or per operation, never per pixel: the fast transforms
were first called per tile row of 64 pixels, and the SSE4 build lost
14% to the calls and the vector constants set up on each, so they take
the whole region.

**Run-time dispatch on x86-64.** The ``dispatch`` build (``auto`` there)
compiles ``SIMDKernels.c`` three times, through the ``SIMDKernels_sse4.c``,
``_avx2.c`` and ``_avxvnni.c`` stubs, each with its instruction set as a
per-file compiler flag (``setup.py`` wraps the compiler's per-source
step; MSVC needs no flag and has the SSE4.1 and AVX2 tiers), and every
other file for plain x86-64; the stub's ``IMAGING_SIMD_TIER`` tells
``ImagingSIMD.h`` which macros to define. ``SIMDDispatch.c`` reads
``cpuid`` (SSE4.1; AVX2 with the OS saving the YMM state; AVX-VNNI) and
``ImagingSIMDInit()`` points ``ImagingSIMD`` at the highest tier built
that the CPU has, capped by ``PILLOW_ALLSIMD_ISA`` if set; a build for
one instruction set the CPU lacks fails to import with a message. The
tiers of one build give the same pixels, which ``Tests/test_simd_tiers.py``
checks through ``Image.core.set_simd()``; the scalar table of a dispatch
build is the common files' portable code compiled for x86-64. On arm64,
NEON is part of the baseline, so the wheels have it and the build has
one tier.

``ImagingSIMD.h`` also holds the helpers the kernels share:
``SIMD_FORCE_INLINE`` for kernels that must see a shift count as a
constant; ``mm_dpwssd`` and ``mm256_dpwssd``, one ``vpdpwssd`` with
AVX-VNNI and ``pmaddwd`` plus ``paddd`` without, and ``mm_madd_add``,
always the pair, for sums that are one long dependency chain (below);
and ``ImagingPx4.h``, one pixel as four 32-bit lanes with the same
modulo-2³² arithmetic in SSE4, NEON and plain C.

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

.. generated: kernels-resize (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

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
     - 226
     - 4.8x
     - 6.4x
     - 260
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
     - 3.7x

.. generated-end

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
* ``ResampleSIMD.c``, the companion: it includes
  ``ResampleSIMDHorizontalConv.c`` (the horizontal kernels for 4-byte
  and 1-byte pixels, SSE4/AVX2 and NEON) twice, the second time with
  ``RESAMPLE_OUT16`` for fast mode's 16-bit intermediate rows,
  ``ResampleSIMDVerticalConv.c`` (the vertical kernel),
  ``ResampleSIMD32bpc.c`` (I, F and I;16 in vectors of doubles) and
  ``ResampleFastSIMD.c`` (fast mode's vertical row), and defines the
  table's ``resample`` group.
* ``ResampleHorizontal.c``: the portable horizontal kernels, included
  twice by ``Resample.c`` the same way; the portable vertical kernel is
  in ``Resample.c`` itself.
* ``ResampleFast.c``: fast mode's fused resize; ``ResampleFast.h``: the
  constants it shares with its SIMD loop.
* ``Reduce.c`` and ``ReduceSIMD.c`` (the ``reduce`` group).

Pitfalls met on the way: ``vpdpwssd`` in a single dependency chain
(above); a vertical-first pass order for every downscale was measured and
not adopted, because the result would then depend on the build; and the
horizontal kernel file is compiled twice, so a change to the 8-bit
kernels changes fast mode's 16-bit ones with it, which is intended, and
their tests cover both.

Box blur, Gaussian blur and unsharp mask
----------------------------------------

``ImageFilter.BoxBlur``, ``ImageFilter.GaussianBlur`` and
``ImageFilter.UnsharpMask`` (``BoxBlur.c``, ``UnsharpMask.c``). All three
are box blurs underneath: a Gaussian blur is three box passes, and an
unsharp mask is a Gaussian blur followed by one pass over the pixels. The
speed-up here comes as much from the order the passes run in as from the
vector code.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

A box blur of radius *R* (a float) averages, along one axis, the 2\ *R* +
1 pixels around each pixel: the 2\ *r* + 1 whole pixels, *r* = ⌊\ *R*\ ⌋,
with the weight ``ww`` = ⌊2²⁴ / (2\ *R* + 1)⌋, and the two pixels just
beyond them with the fractional weight ``fw`` = (2²⁴ − (2\ *r* + 1)
``ww``) / 2, so that the weights add up to 2²⁴ (less a rounding). With
``acc`` the sum of the whole pixels and ``left`` and ``right`` the two
beyond, each channel is::

    out = (acc * ww + (left + right) * fw + (1 << 23)) >> 24

in 32-bit unsigned arithmetic (``acc * ww`` is at most 255 · 2²⁴, so
nothing wraps). Pixels beyond the edges repeat the edge pixel. A blur is
*n* such passes along x, then *n* along y (``BoxBlur`` takes one each;
the C function takes any number).

``GaussianBlur(radius)``, whose radius is the standard deviation σ, runs
three box passes per axis, with the box radius that makes three boxes
approximate the Gaussian (Gwosdek et al., *Theoretical foundations of
Gaussian convolution by extended box filtering*, 2011, in
``_gaussian_blur_radius()``): for σ = 1 each pass has radius 0.25, three
taps with the weights 1/6, 2/3, 1/6; for σ = 10, 9.475. The passes cost
the same for any radius, so the Gaussian's cost doesn't grow with σ.

``UnsharpMask(radius, percent, threshold)`` blurs the image with that
Gaussian, then, for every byte with ``diff = in - blurred``, writes
``clip(in + diff * percent / 100)`` where ``|diff| > threshold`` and
``in`` elsewhere, with C's integer division (truncating toward zero).

Pillow 13 runs every pass over the whole image, one pixel at a time, with
the running sum in scalar variables (four of them for a 4-byte pixel),
and blurs vertically by transposing the image, blurring rows and
transposing back: three horizontal passes, a transpose, three passes and
a transpose for a Gaussian, each a full read and write of the image.

The idea
~~~~~~~~

Four changes, each keeping the arithmetic above:

* **Vertical passes on rows.** The vertical pass keeps one 32-bit running
  sum per *byte of the row* (``linesize`` of them), adds the row entering
  the window and subtracts the one leaving it, and applies the weights to
  the whole row at once: ``acc[b] += add[b] - sub[b]`` for every byte,
  then the formula. It is the horizontal pass's arithmetic with the loop
  over pixels turned into a loop over rows, so it gives the pixels of the
  transposed computation (``test_vertical_matches_transposed_horizontal``
  checks this), without the transposes, and it is as wide as the row.
* **The passes of a direction together.** Horizontally, all *n* passes run
  on one row before the next, alternating between two row buffers, so
  the row stays in cache. Vertically, the passes form a cascade: pass *k*
  keeps its last 2\ *r* + 3 output rows in a ring, which is all pass
  *k* + 1 reads, and ``BoxBlurEnsure()`` computes each pass's rows as the
  next needs them; one sweep down the image writes the output. The
  intermediate images are gone: a Gaussian reads and writes the image
  twice instead of six times plus two transposes. Rings above 8 MB (very
  large radii) fall back to one pass at a time.
* **One pass each way, fused.** ``BoxBlur``'s default, one pass per axis,
  fuses further: each row is blurred horizontally into a ring of
  2\ *r* + 4 rows just before the vertical pass first reads it
  (``box_fused_row()``), so there is no temporary image at all, 16 MB to
  allocate, write and read for a 2560×1600 RGB image. Rings up to 4 MB.
* **Small radii without running sums.** A running sum is a serial
  dependency, each pixel waiting for the previous one, and in the
  horizontal pass it is four lanes wide at most. For radii below 2, and
  for the integer radii 1 to 4, the box is summed directly from its 2\ *r*
  + 3 sources, 16 or 32 bytes of the row per step: the sources are rows
  (vertical pass) or the row shifted by whole pixels (horizontal pass,
  with the edge pixels repeated in a padded copy of the row's ends). For
  the integer radii 1 to 4 ``fw`` is 0 (2²⁴ − 1 is divisible by 3, 5, 7
  and 9) and the division by ``ww`` is done in 16 bits:
  ``(((c << s) + k) * m >> 16) >> t`` with constants per radius found by
  a search around *m* = ``ww`` · 2\ :sup:`t − s − 8`. ``BoxDirectInit()``
  checks the 16-bit form against the 24-bit one for every possible sum
  *c* (at most 255 · 9 of them) at the start of each blur, and uses it
  only if they all agree. The radii between (2.5, say) take the running
  sums.

Two more things decide the speed: 1-byte images are blurred four rows at
a time on x86, the bytes of rows *y* to *y* + 3 interleaved into one row
of 4-byte pixels, blurred with the 4-byte kernel (the same arithmetic
per byte) and de-interleaved, the rows left over taking the scalar
1-byte running sum (``test_l_matches_rgb_band``); and the vertical pass,
which touches rows 2\ *r* + 3 rows apart, is the one that gained 1.5x from
transparent huge pages.

x86-64
~~~~~~

*Horizontal running sums* (``ImagingLineBoxBlur32``) are the form
Pillow-SIMD has had since 2015: one pixel as four 32-bit lanes
(``ImagingPx4.h``: ``pmovzxbd`` to load, ``paddd`` and ``psubd`` for the
sum, ``pmulld`` by the weights, a shift and a ``pshufb`` to pack the four
bytes). The loop body is one macro, ``BOX_LINE_BLUR``, that takes the
vector type's operations as macros; with AVX2 the same macro runs on two
rows at once, one per 128-bit half (``ImagingLineBoxBlur32x2``), so two
rows cost the instructions of one; 1-byte images then take eight rows
per step.

*Vertical running sums* (``ImagingBoxBlurColumns``) take 32 bytes of the
row per step with AVX2, 16 with SSE4: four groups of 8 bytes of ``add``,
``sub`` and the far row widened to 32 bits, the sums updated and stored,
``pmulld`` by ``ww`` and ``fw``, the rounding shift, then the four results
packed to bytes. ``packssdw`` and ``packuswb`` pack within 128-bit lanes,
so the AVX2 version ends with a ``vpermd`` that puts the four groups back
in order.

*Direct passes* (``box_direct()``) sum the inner sources in 16-bit lanes
(``pmovzxbw``, ``paddw``), 32 bytes per step with AVX2, 16 with SSE4.
The exact form then needs ``acc * ww + e * fw`` with 24-bit weights,
which ``pmulld`` would compute in 32 bits at two µops and 10 cycles of
latency each; instead the weights are split into 12-bit halves, ``ww`` =
``wh`` · 2¹² + ``wl``, and ``pmaddwd`` on the pairs (``c``, ``e``)
against (``wh``, ``fh``) and (``wl``, ``fl``) gives::

    ww * c + fw * e = ((wh * c + fh * e) << 12) + (wl * c + fl * e)

two ``pmaddwd``, a shift and an add, with nothing above 2³² (``c`` is at
most 765 here, the weights at most 2¹²). The 16-bit form for the integer
radii stays in 16-bit lanes throughout, its multiply-high being
``pmulhuw``. The number of sources is a compile-time constant for each
radius (``ImagingBoxBlurDirect()`` is a ``switch`` into an
``always_inline`` function), so the sum unrolls and the source pointers
stay in registers: 1.46x at radius 2 over the loop that reloaded them.

*Unsharp mask* (``ImagingUnsharpMaskRow``): 16 bytes per step, widened to
32 bits; ``diff * percent`` in 32 bits, converted to float, divided by
100 and truncated with ``cvttps2dq``, which equals C's integer division
as long as ``diff * percent`` is below 2²⁴ in magnitude, so the vector
loop takes ``percent`` between −65535 and 65535 only; ``|diff| > threshold`` as a mask for
``pblendvb``; and the signed and unsigned saturating packs are
``clip8()``.

arm64
~~~~~

The arm64 version is a different mix, decided by measurement against
GCC's own vectorization of the portable loops, and it has changed with
the compiler.

*Horizontal running sums*: the ``px4`` NEON version (``vmovl`` widenings,
``vmulq_u32``, ``vmovn`` narrowings). It is 1.27x faster than the
compiler's code for the portable loop with GCC 12 and 14 and was slower
with GCC 11, so it was removed for a while and restored. 1-byte images
keep the scalar running sum: the four-row interleave was 1 to 5% slower
than the compiler's 1-byte loop on the Cortex-A78AE.

*Vertical running sums*: the portable loop. GCC vectorizes it at ``-O3``,
with a run-time check that the rows don't alias, and leaves it scalar at
``-O2``; a NEON version was measured slower than the compiler's code.

*Direct passes*: 16 bytes per step. The sum uses the widening adds
(``vaddw_u8``), and the exact form multiplies in 32 bits with
``vmulq_u32`` and ``vmlaq_u32``, one instruction each on arm64, where the
12-bit split was slower; ``vaddhn_u32`` adds 2²³ and keeps the high 16
bits of each sum, which is the ``>> 16`` of the rounding, and ``vshrn``
by 8 completes the ``>> 24``. The 16-bit form has no multiply-high
instruction: ``vmull_u16`` makes the 32-bit products and ``vuzp2q`` keeps
their high halves.

*Unsharp mask*: the same steps as x86 with ``vdivq_f32``, a real vector
division, cheap on this core; 1.31x over the compiler's multiply-high
loop with GCC 12 and 14, not with GCC 11.

Why the builds agree
~~~~~~~~~~~~~~~~~~~~

Everything is integer arithmetic modulo 2³², and every path computes the
same integers: a running sum and a direct sum of the same box are the
same number; the weights come from one float expression, evaluated the
same way on every build; the 16-bit form of the direct passes is checked
against the 24-bit form at run time, for every sum it can meet, before it
is used; the cascade and the fused pass change where the intermediate
rows live, not their values (``test_box_blur_fused_matches_passes``
compares the fused blur with the two passes run separately); and each
vector loop leaves the end of the row to the scalar loop with the same
formula (``Tests/test_simd_consistency.py``). The unsharp mask's float
division is exact against the integer division in the range the vector
loop accepts. ``checks/simd_hashes.py``'s ``blur/unsharp`` area agrees on
every build, and so does Pillow's ``Tests/test_box_blur.py``, extended
with the channel-order, one-pixel-line and transposition checks.

Numbers
~~~~~~~

2560×1600, exact mode, times faster than Pillow 13.0.0, which took the
milliseconds shown.

.. generated: kernels-blur (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Blur
     - Pillow 13 (x86, ms)
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - Gaussian RGB, σ 1
     - 121
     - 4.4x
     - 7.3x
     - 162
     - 3.1x
   * - Gaussian RGB, σ 10
     - 121
     - 2.7x
     - 4.6x
     - 163
     - 1.8x
   * - Gaussian RGB, σ 30
     - 121
     - 2.7x
     - 4.6x
     - 163
     - 1.8x
   * - Gaussian RGBA, σ 10
     - 121
     - 2.7x
     - 4.7x
     - 163
     - 1.8x
   * - Gaussian L, σ 10
     - 37.0
     - 3.5x
     - 6.2x
     - 53.5
     - 1.9x
   * - Box RGB, radius 2
     - 53.8
     - 8.1x
     - 14.2x
     - 65.6
     - 4.4x
   * - Box RGB, radius 10
     - 53.6
     - 3.9x
     - 7.1x
     - 65.7
     - 2.3x
   * - Unsharp mask RGB
     - 197
     - 5.0x
     - 7.4x
     - 236
     - 3.2x
   * - Unsharp mask L
     - 54.4
     - 6.0x
     - 8.9x
     - 69.6
     - 4.0x

.. generated-end

The portable build alone is 1.4 to 2.1x Pillow 13 on x86 and 1.4 to 2.7x
on arm64: that is the share of the passes' order. The σ 1 Gaussian and
the radius-2 box take the direct passes, the others the running sums,
which is why they gain more; and σ 10 and σ 30 cost the same. Gaussian L
on arm64 gains nothing from NEON (the scalar build is 1.9x too): its
horizontal pass is the scalar 1-byte running sum and its vertical pass is
the compiler's code. Against Pillow-SIMD 12.1.1 with AVX2, the σ 1
Gaussian is 3.1x faster, the radius-2 box 8.2x and the unsharp mask 5.0x.
AVX-VNNI changes nothing here: there is no ``pmaddwd`` chain to fuse.

Against OpenCV on x86, a true Gaussian with one kernel of about 6σ + 1
taps per axis: OpenCV is 2.1x faster at σ 1 (7 taps against three passes of 3),
and the AVX2 build 3.3x faster at σ 10 and 13x at σ 30, where OpenCV's
kernel keeps growing. On the box blur OpenCV's radius 2 is 1.2x faster
(it keeps 16-bit running column sums; a 16-bit vertical sum would be an
exact-compatible change here, listed in ``design/PERFORMANCE_GAPS.md``),
and its radius 10 is 1.16x slower. On arm64 OpenCV is ahead on the small
radii (1.2x at σ 1, 1.4 to 1.5x on the box blur) and 4.5x behind at
σ 10. Fast mode
(the default) replaces the three passes with one kernel for σ up to 6:
the σ 1 Gaussian 21.6x on x86 and 9.7x on arm64 (3.0x and 3.2x over the
exact kernels, and ahead of OpenCV on both), the RGB unsharp mask 12.5x
and 5.9x; see :ref:`fast-mode` and, until the fast-mode section is
written, ``design/SIMD_FAST_KERNELS.md``.

Code map
~~~~~~~~

* ``BoxBlur.c``: ``line_blur32()`` and ``ImagingLineBoxBlur8``, the
  horizontal running sums; ``BoxDirectInit()``, ``box_direct()``,
  ``ImagingBoxBlurDirect()`` and ``ImagingLineBoxBlurDirect()``, the direct
  passes; ``ImagingHorizontalBoxBlur()``; ``ImagingBoxBlurColumns()`` and
  ``ImagingVerticalBoxBlur()``; the cascade (``BoxBlurCascade``,
  ``BoxBlurEnsure()``, ``ImagingVerticalBoxBlurPasses()``); the fused pass
  (``BoxFusedRows``, ``box_fused_row()``, ``ImagingBoxBlurFused()``);
  ``ImagingBoxBlur()``, which chooses between them;
  ``_gaussian_blur_radius()`` and ``ImagingGaussianBlur()``, the fast-mode
  entry.
* ``BoxBlur.h``: ``BOX_LINE_BLUR`` and its px4 instance
  ``box_line_blur32()``, ``BoxDirect``; ``ImagingPx4.h``: one pixel as four
  32-bit lanes, SSE4, NEON and portable.
* ``BoxBlurSIMD.c``, the ``blur`` group: the row blur in SIMD lanes and
  ``box_line_blur32x2()`` (AVX2), ``box_direct_simd()`` with the source
  count a constant per radius, ``box_interleave4()`` and
  ``box_deinterleave4()``, ``box_columns_simd()``.
* ``GaussianFast.c``: fast mode's one-kernel Gaussian;
  ``GaussianFastSIMD.c``: its taps (the ``gauss`` group).
* ``UnsharpMask.c``; ``UnsharpMaskSIMD.c`` (the ``unsharp`` group).
* Tests: the end of ``Tests/test_box_blur.py``,
  ``test_box_blur_fused_matches_passes`` in
  ``Tests/test_simd_consistency.py``, the ``blur/unsharp`` and ``fast:
  gaussian`` areas of ``checks/simd_hashes.py``.

Pitfalls met on the way: Pillow-SIMD's SSE4 store mask wrote lane 3 to
byte 0, which a two-axis blur reversed twice and hid, so that
``BoxBlur((r, 0))`` swapped the channels (``test_channel_order``); the
old kernel read past the end of 1-pixel lines; two rows per AVX2 vector
were measured and dropped once (Gaussian 1.08x, unsharp 0.84x, with
``pmulld`` costing as much per element at 256 bits as at 128 and the
packing of two rows adding work) and adopted after the passes were
restructured, when the rows were in cache and the AVX2 vertical kernel
came with them (Gaussian 57 to 28 ms); column strips to keep the vertical
pass's accumulators in L1 measured 0.90 to 1.11x, because the misses the
cache model counted are served by the hardware prefetcher, and were
dropped; and the NEON decisions above each went the other way with
another GCC, so they are re-measured when the compiler changes.

Kernel filters
--------------

``ImageFilter.Kernel`` and the built-in 3×3 and 5×5 filters (``SMOOTH``,
``SHARPEN``, ``FIND_EDGES``, ``SMOOTH_MORE`` and the others), on 8-bit
images (``Filter.c`` and the ``FilterSIMD_*.c`` files it includes). I and
I;16 images keep Pillow's code.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

A kernel is given as integers with a scale and an offset; the
coefficients are the integers divided by the scale (``SMOOTH``: 1/13 and
5/13; ``FIND_EDGES``: −1 and 8; ``SMOOTH_MORE``: 1/100, 5/100 and
44/100). For every pixel and channel, Pillow sums the 9 or 25 products of
the coefficients and the neighbouring pixels in single-precision floats,
adds the offset and 0.5, clamps to 0..255 and truncates; the border rows
and columns, where the kernel doesn't fit, are copied from the input.
Kernel row 0 applies to the image row *below* the pixel (``y + 1``), the
last row to the row above. Pillow 13's loop is scalar, one channel at a
time, and the portable build is no faster than it.

Pillow-SIMD's method, which this fork keeps, is the fixed-point one:
the coefficients as 16-bit integers with 13 fractional bits
(``PRECISION_BITS`` = 8 + 5: 8 for the result and 5 for a sum of up to 25
coefficients), ``pmaddwd`` into 32-bit sums, the rounding as an added
half (``offset += 1 << (precision - 1)``), an arithmetic shift and a
saturating pack to bytes. A coefficient of 4 or more doesn't fit 16 bits
at that precision, and Pillow-SIMD sent those kernels to float versions
of the same loops.

The idea
~~~~~~~~

Four changes since Pillow-SIMD:

* **The lower exact precision.** When a coefficient doesn't fit at 13
  fractional bits, ``ImagingFilterSIMD()`` tries 12, 11, …, and takes the
  highest precision at which every coefficient *and the offset* are
  exact integers once scaled. ``FIND_EDGES`` and ``CONTOUR`` (−1 and 8,
  with the offset 0 or 255) take 11 bits, ``EDGE_ENHANCE`` (−1/2, 5) 12,
  ``EDGE_ENHANCE_MORE`` (−1, 9) 11. The sums are then exact integers, and
  adding the half and shifting rounds them half up exactly as Pillow's
  ``+ 0.5`` and truncation do, so these kernels give Pillow's pixels and
  run at the integer kernels' speed: ``FIND_EDGES`` on RGB from 9.5 to
  4.3 ms with AVX2, 53 to 38 on the Jetson. Only a kernel whose
  coefficients are inexact at every precision down to 1 bit (a custom
  one with 5.5 in it, say) still takes the float kernels.
* **Integer kernels for 1-byte images** (``FilterSIMD_NxNi.c``). Pillow-SIMD
  filtered L images with float kernels and horizontal adds, 2.7x the cost
  per byte of the RGB kernel. The integer kernel for 1-byte images is one
  function for both sizes and both instruction sets, written over two
  primitives (a multiply-accumulate of 16 bytes by a pair of taps and a
  shift-and-saturate pack), 16 bytes per step with SSE4 and NEON, 32 with
  AVX2, and the same arithmetic as the 4-byte kernels; L images filter
  1.4x (3×3) to 2x (5×5) faster than with the float kernels, and x86 and
  arm64 give the same L pixels.
* **Constants where the compiler needs them.** The shift count is
  ``PRECISION_BITS`` as a constant wherever that is the precision (a
  wrapper calls the kernel once with the constant and once with the
  variable, ``FILTER_NxN``), because a count in a register costs a µop
  per shift on x86 (``SHARPEN`` was 12% slower with AVX-VNNI) and NEON
  then fuses the shift and the narrowing into one ``sqshrn`` (6% on the
  Jetson); the kernel bodies are ``SIMD_FORCE_INLINE``, so that the two
  calls don't stop GCC inlining them; and the 3×3 kernel for 1-byte
  images copies its nine coefficients to a local array, because the
  byte stores to the output could alias the coefficient pointer and GCC
  reloaded all nine on every step (L ``SHARPEN`` 11% on NEON, 8% with
  AVX2).
* **AVX-VNNI.** Every ``add(acc, madd(a, b))`` of the integer kernels is
  ``mm_dpwssd`` or ``mm256_dpwssd``: one ``vpdpwssd`` with AVX-VNNI, the
  pair without, the same sum modulo 2³². The integer filters are where
  it pays most: 1.35x (3×3) and 2.1x (5×5) over AVX2 on RGB.

x86-64
~~~~~~

*4-byte pixels, integer kernels* (``FilterSIMD_3x3i_4u8.c``,
``FilterSIMD_5x5i_4u8.c``, Pillow-SIMD's). One ``pshufb`` turns a load of
four pixels into channel-major pairs, so that after widening to 16 bits
each 32-bit lane of ``pmaddwd`` holds two horizontally adjacent pixels of
one channel; the kernel vector holds a row's taps in that pair layout
with zeros, and ``pshufd`` of it selects which two taps multiply which
pair. So one load of four pixels per kernel row serves four output
pixels, each ``pmaddwd`` making one output pixel's four channels for two
taps, and a 3×3 kernel is six ``pmaddwd`` per output pixel. The AVX2
version copies the four pixels into both 128-bit halves and applies a
differently aligned kernel in each half: two output pixels per
multiply-add. The assessment that this "isn't really 256-bit" and eight
consecutive pixels per step would be better was wrong: that saves
shuffles but not multiply-adds, and measured 0.93x (3×3) and 1.02x (5×5)
with AVX2, 0.89x and 0.46x with AVX-VNNI.

*1-byte pixels, integer kernels* (``FilterSIMD_NxNi.c``). A step loads 16
bytes at each tap's offset, widens them to 16 bits and interleaves the
two loads of a pair of taps (``punpcklwd``/``punpckhwd``), so that
``pmaddwd`` computes ``a[i] * k0 + b[i] * k1`` for 16 bytes into four
accumulators; the odd last tap of a row is paired with 0. With AVX2, 32
bytes per step in the same way, the pairs interleaved within each 128-bit
lane and a ``vpermq`` after the packs putting the 32 results back in
order.

*Float kernels* (``FilterSIMD_3x3f_4u8.c``, ``5x5f_4u8``, ``3x3f_u8``,
``5x5f_u8``). For 4-byte pixels, one pixel's four channels are one float
vector (``pmovzxbd``, ``cvtdq2ps``) and the sum is nine (or 25)
multiply-adds per pixel, three pixels per step; the AVX2 3×3 version
takes eight pixels per step with ``vperm2f128`` sliding the window. For
1-byte pixels the three taps of a row are one dot product with
horizontal adds (``haddps``), two rows at a time; the AVX2 version runs
the same operations in the same order for eight pixels. The rounding is
``cvtps2dq``, to nearest even, where Pillow adds 0.5 and truncates, and
the summation order differs from Pillow's, so these kernels can differ
from the scalar build by one level (below).

arm64
~~~~~

*Integer kernels*: the generic kernel of ``FilterSIMD_NxNi.c`` for both
pixel sizes, with NEON's primitives: ``vmlal_n_s16``, a widening
multiply-accumulate of 8 lanes by one coefficient, one per tap (NEON has
no pair sum), into four 32-bit accumulators, and ``vqmovn_s32`` plus
``vqmovun_s16`` to pack, which with the constant shift become
``sqshrn``. The 4-byte pixels go through the same loop, 4 pixels per
step; the same arithmetic as x86's, and the same pixels.

*Small coefficients (2026-11-05)*. The Cortex-A78 runs vector multiplies
on one pipe, and the generic kernel spends four of them per tap and 16
bytes, 9 cycles per pixel of a 3×3 kernel on RGB, which is most of what
it costs: the reason NEON gains 4 to 5x here where AVX2, with ``pmaddwd``
on two ports, gains 13x and more. For a kernel whose fixed-point
coefficients and offset share a power of two, the sum is that power
times a small sum, so ``filter_small_kernel()`` in ``Filter.c`` divides
it out, and where the coefficients left are at most 255 in size and
every sum of a window fits 16 bits, ``ImagingFilterSmall_simd()`` adds
them in 16-bit lanes, eight per operation: a coefficient of 1 or -1 is
a widening add or subtract, any other one widening multiply-accumulate
of bytes. Shifting the small sum by the precision that is left (with its
own rounding; none when the whole precision divided out, where the
half can never carry) gives the same bytes as the generic kernel, which
keeps every other kernel. Seven of Pillow's ten built-in filters reduce:
``SHARPEN`` to -1 and 16 at 3 bits, ``FIND_EDGES`` and ``CONTOUR`` to -1
and 8 with no shift, ``EMBOSS`` to -1, 0 and 1, ``BLUR`` to ones at 4
bits, ``EDGE_ENHANCE`` and ``EDGE_ENHANCE_MORE``; ``SMOOTH``,
``SMOOTH_MORE`` and ``DETAIL`` (scales 13, 100 and 6) do not. On the
Jetson, 2560×1600 RGB: ``SHARPEN``, ``FIND_EDGES`` and ``EDGE_ENHANCE``
27 to 14.7 ms (1.8x), ``BLUR`` 24 ms, ``EMBOSS`` 10; L ``SHARPEN`` 6.0
to 4.0 ms. The NEON and scalar builds hash every filter the same before
and after, and ``test_integer_kernels_with_power_of_two_scale`` checks
ten kernels against the definition at widths around the vector's.
Splitting the adds into two chains was measured and is slower (0.8x),
as it was for the generic kernel.

*Float kernels* (``FilterSIMD_NxNf.c``): 4-byte pixels only, 16 bytes per
step as four float vectors. On arm64 GCC compiles Pillow's float loop
with fused multiply-adds, so the "scalar" result there is already
contracted; the NEON kernel reproduces GCC's contraction (each kernel
row as ``vmulq_n_f32`` then a chain of ``vfmaq_n_f32``, rows added to the
sum in order, clamp, truncate) and the leftover bytes use ``fmaf()``. It
matches the arm64 scalar build bit for bit for exact coefficients, and
in about 0.05% of bytes for an inexact one GCC's own loop fuses
differently. 1-byte images keep the generic code: GCC vectorizes it
well, and the NEON float kernel was 1.5x slower (``CONTOUR`` on L 15.5
against 22.9 ms).

Why the builds agree, and where they don't
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integer kernels are the same arithmetic on SSE4, AVX2, AVX-VNNI and
NEON, 16-bit products summed in 32 bits modulo 2³² and one rounding, so
all four give the same pixels; and when the coefficients are exact at
the precision used (the built-in kernels whose scale is a power of two,
1 or 16, and the lower-precision ones above) those pixels are Pillow's,
the scalar build's. When a coefficient is inexact at 13 fractional bits
it is rounded (1/13 becomes 630/8192), and the SIMD builds differ from
the scalar build by at most one level: on the test image ``DETAIL`` in
9% of the bytes, ``SMOOTH_MORE`` in 4%, ``SMOOTH`` in none (its rounded
coefficients sum to 8191/8192). This is the one place in exact mode
where a SIMD build doesn't give Pillow's pixels on purpose; it is
Pillow-SIMD's behaviour and was kept. The float kernels differ from the
scalar build by one level in about 0.2% of bytes, and agree between the
x86 tiers: the 3x3 kernel for 4-byte pixels had an AVX2 version that
summed two columns in the two 128-bit halves and added them last, 0.2%
of bytes off the SSE4 kernel's rounding, and it was dropped (it was also
20% slower than the SSE4 kernel on the Core Ultra 9 285H); the 1-byte
AVX2 kernel equals SSE4 by construction.

The checks: ``test_filter`` in ``Tests/test_simd_consistency.py`` (flat
images stay flat inside the border for integer and float kernels, every
width and mode, so the vector loop and its scalar tail agree),
``test_float_kernel_l_matches_rgb_band`` and
``test_i16_uses_generic_kernel`` in ``Tests/test_image_filter.py``, and,
when the kernels were written, 3,268 filter outputs compared across the
SSE4, AVX2 and AVX-VNNI builds. The kernel filters have no area in
``checks/simd_hashes.py``, whose areas are those where the scalar build
agrees.

Numbers
~~~~~~~

2560×1600, times faster than Pillow 13.0.0, which took the milliseconds
shown.

.. generated: kernels-filters (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Filter
     - Pillow 13 (x86, ms)
     - SSE4
     - AVX2
     - AVX-VNNI
     - Pillow 13 (arm64, ms)
     - NEON
   * - RGB 3×3 SMOOTH
     - 51.4
     - 8.0x
     - 12.3x
     - 16.4x
     - 98.2
     - 3.6x
   * - RGB 3×3 SHARPEN
     - 52.9
     - 8.2x
     - 12.6x
     - 16.9x
     - 99.0
     - 3.7x
   * - RGB 3×3 EDGE_ENHANCE
     - 55.9
     - 8.2x
     - 12.7x
     - 16.3x
     - 100
     - 3.8x
   * - RGB 3×3 FIND_EDGES
     - 147
     - 21.4x
     - 33.5x
     - 42.9x
     - 145
     - 5.5x
   * - RGB 5×5 SMOOTH_MORE
     - 134
     - 7.5x
     - 8.9x
     - 18.9x
     - 227
     - 3.6x
   * - L 3×3 SHARPEN
     - 16.5
     - 5.8x
     - 12.1x
     - 14.8x
     - 33.1
     - 5.6x
   * - L 5×5 SMOOTH_MORE
     - 42.5
     - 5.6x
     - 12.6x
     - 15.5x
     - 75.1
     - 4.6x

.. generated-end

The SIMD time of a 3×3 kernel on RGB is the same whatever the kernel,
4.2 to 4.4 ms with AVX2; ``FIND_EDGES``'s larger factor is Pillow 13's
own time on it, 147 ms against 51 for the other kernels. Against
Pillow-SIMD 12.1.1 with AVX2, the kernels it ran as integers are equal
(``SMOOTH`` 12.5x there), ``FIND_EDGES``, which it ran in float, is 2.2x
faster, and L images, which it filtered in float, 3.4x (3×3) and 3.6x
(5×5). Against OpenCV's ``filter2D``, which accumulates in 32-bit floats,
the AVX2 build is 1.5 to 1.6x faster on the 3×3 kernels and 1.1x on
5×5, the AVX-VNNI build 1.9 to 2.1x and 2.3x, and the NEON build 1.4x
to 2.2x on all of them; libvips' convolution is slower than Pillow 13 on 5×5.

Code map
~~~~~~~~

* ``Filter.c``: Pillow's ``ImagingFilter3x3()`` and ``ImagingFilter5x5()``
  (the definition, and the I, I;16 and portable paths),
  ``filter_int_kernel()`` and the precision search in ``filter_simd()``,
  which picks the kernel from the active tier's table.
* ``FilterSIMD.c``, the companion: it includes the eight files below and
  defines the ``filter`` group.
* ``FilterSIMD_3x3i_4u8.c``, ``FilterSIMD_5x5i_4u8.c``: the x86 integer
  kernels for 4-byte pixels, SSE4 and AVX2.
* ``FilterSIMD_NxNi.c``: the integer kernel for 1-byte pixels on x86 (and
  for both pixel sizes on arm64), with the SSE4/AVX2 and NEON primitives;
  ``ImagingFilterSmall_simd()``, the arm64 kernel for small coefficients
  (``filter_small_kernel()`` in ``Filter.c`` reduces them).
* ``FilterSIMD_3x3f_4u8.c``, ``FilterSIMD_5x5f_4u8.c``,
  ``FilterSIMD_3x3f_u8.c``, ``FilterSIMD_5x5f_u8.c``: the x86 float
  kernels; ``FilterSIMD_NxNf.c``: the NEON one.
* ``ImagingSIMD.h``: ``mm_dpwssd``, ``mm256_dpwssd``,
  ``SIMD_FORCE_INLINE``.
* Tests: ``test_filter`` in ``Tests/test_simd_consistency.py``; the end of
  ``Tests/test_image_filter.py``.

Pitfalls met on the way: Pillow-SIMD sent I;16 images to the 8-bit
kernels, which filtered the bytes of the 16-bit samples over half of
each row (``test_i16_uses_generic_kernel``); passing the precision as a
variable cost 12% on x86 and un-inlined the 1-byte kernel, and the
inlining then exposed the aliasing reloads, so one change needed two
follow-ups; AVX2 5×5 float kernels gained nothing for 4-byte pixels and
lost (0.57x) for 1-byte ones, and were dropped; and on arm64 the
decision per pixel size went opposite ways for the float kernels, as it
did for the blurs.

Rank filters
------------

``ImageFilter.MedianFilter``, ``MinFilter``, ``MaxFilter`` and
``RankFilter`` (``RankFilter.c``), and ``ImageFilter.ModeFilter``
(``ModeFilter.c``). These have the largest factors on this page, 50x to
500x, and most of each factor is the algorithm: the vector code is a
minimum and a maximum of bytes, and the compiler writes much of it.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

A rank filter of size *s* replaces each pixel with the *k*-th smallest of
the *s*\ ² pixels around it (*k* = 0 for the minimum, *s*\ ² − 1 for the
maximum, *s*\ ² / 2 for the median), with the pixels beyond the edges
repeating the edge pixels: ``RankFilter.filter()`` expands the image by
*s* / 2 pixels on each side (``ImagingExpand()``, a copy) and filters the
copy. Multi-band images are filtered band by band: ``Image.filter()``
splits them, filters each band and merges. A rank filter has no
arithmetic, so there is nothing to round: any method that selects the
*k*-th value gives the same pixels.

Pillow 13 selects with Wirth's quickselect on a copy of each window
(``memcpy`` of each row of the window, then a partitioning loop whose
branches depend on the data: 40 to 70 mispredictions per pixel). A 3×3
median of a 2560×1600 RGB image takes 1.2 s, a 5×5 one 4 s.

``ModeFilter`` writes the most frequent value of the window if it occurs
more than twice (the lowest of equally frequent ones) and the pixel
itself otherwise; Pillow 13 builds a 256-bin histogram from scratch for
every pixel and scans all of it.

The idea
~~~~~~~~

* **Sorting networks over rows of pixels.** A sorting network is a fixed
  sequence of compare-exchanges, the same for every input; applied to
  *s*\ ² rows of 256 pixels at once (one row per window position), each
  compare-exchange is a minimum and a maximum of two rows of bytes,
  which has no branches and which every compiler vectorizes (x86-64's
  baseline has ``pminub``; the portable build is 50 to 85x Pillow 13 on
  the medians and 270x on minimum and maximum). The network is an optimal 25-comparator one for nine values,
  and for larger windows Batcher's odd-even merge sort for the next
  power of two with the comparators that touch wires past *s*\ ² left
  out (those wires can be taken as holding the largest value, which no
  comparator moves); and it is pruned to the comparators that the wire
  of the requested rank depends on, 113 of 140 for the 5×5 median, 319
  of 394 for 7×7. Sizes up to 11×11 use this; larger ones keep
  quickselect.
* **No expanded copy.** The fast paths read the image itself
  (``rankfilter(size, rank, True)``) and repeat the edges as they load:
  rows beyond the image clamp to the first or last row, and the chunks
  at a row's ends are filled with the edge pixel: a 3×3 median of a
  1280×800 L image went from 2.4 to 0.95 ms.
* **All bands in one pass.** For 4-byte pixels the neighbours of a byte
  are 4 bytes apart and the edges repeat whole pixels, so filtering
  every byte of the row with the same byte of its neighbours filters
  every band at once (``ImagingRankFilterSameSize()``), and
  ``Image.filter()`` hands those images over instead of splitting and
  merging. It also filters the padding byte of RGB, a quarter more work,
  so ``ImagingRankFilterAllBands()`` says when it pays: minimum,
  maximum, the register medians, and 4-band images; a 3-band image with
  the general network goes band by band, as it was 0.81x on arm64
  otherwise. A subclass that overrides ``filter()`` keeps the
  band-by-band path, so that its override still runs.
* **Medians in registers.** For 3×3 and 5×5 the nine or 25 neighbours
  are loaded straight from the rows into vectors and go through a
  median network there, Paeth's 19 comparators for nine values and
  Devillard's 99 for 25 (checked for all 2²⁵ inputs of 0 and 1, which by
  the 0-1 principle covers every input); the compiler drops the half of
  each comparator whose result isn't used. 32 output bytes per step with
  AVX2, 16 with SSE4 and NEON. The 5×5 network needs more than SSE4's 16
  registers, which spill: 10.3 ms for L against 4.6 with AVX2.
* **Minimum and maximum separably.** The minimum of an *s*\ ×\ *s*
  window is the minimum over *s* rows of the minimum over *s* pixels of
  each row, 2\ *s* operations per pixel instead of *s*\ ². The row
  results go to a ring of *s* rows (they went to a whole-image buffer,
  16 MB for RGB, freshly allocated on every call), and ``RankMinMaxLine()``
  takes all *s* sources of a line in one vector pass, with the 3×3 case
  unrolled; the row ends clamp the pixel index, where a padded copy of
  each row had been 31% of a 3×3 minimum's time. Any size.
* **The mode filter by columns.** The histogram is kept for the row and
  updated by one column entering and one leaving per pixel; windows of
  fewer than 16 values look up the counts of their own values only, and
  larger ones take the maximum of the 256 counts (a vectorized loop)
  and its first index. The same tie-break and the same "more than
  twice" rule: 2560×1600 L, size 3, 1195 to 94 ms, sizes 5 to 17 3.4x
  to 4.8x.
* **The 3×3 mode in vectors (2026-11-04).** ``ModeFilterSIMD.c`` does
  the inner rows' pixels 16 or 32 at a time: the nine window bytes are
  loaded from the three rows, each is compared with the other eight
  (the 36 pairs once, each result subtracted from both counts), and the
  winner is chosen by packing the count above the inverted value in a
  16-bit lane and taking the maximum of the nine, which is the highest
  count and, on a tie, the lowest value, as the scalar scan chooses;
  where the count is two or less the centre pixel is kept. The first and
  last pixel of a row, the row tails and the first and last rows go
  through a per-pixel count in ``ModeFilter.c``. 2560×1600 L: 63 to
  4.5 ms (14x) on AVX2; the rank area of the hash set, which now holds
  the mode filter at sizes 3 and 5 on noise and on a flat-region image,
  agrees between the tiers and with the histogram code, and
  ``Tests/test_simd_consistency.py`` compares every width from 1 up
  against a per-pixel definition.

The vector code
~~~~~~~~~~~~~~~

There is little of it, and it is the same on both architectures. The
network path (``RankFilterNetwork_u8()``) is portable C,
``RankCompareExchange()`` over ``RANK_CHUNK`` = 256 bytes with
``restrict`` pointers, which GCC turns into ``pminub``/``pmaxub`` or
``umin``/``umax`` loops at ``-O2``; a chunk of every window position is
copied into a buffer first (``memcpy``, or the edge fill). The register
medians and the min/max line are written with four macros, ``RANK_LOAD``,
``RANK_STORE``, ``RANK_VMIN`` and ``RANK_VMAX``, defined for AVX2 (32
bytes), SSE4 and NEON (16); ``RANK_SORT(a, b)`` is a minimum and a
maximum of two vectors, and the two networks are written out as lists of
``RANK_SORT``.

Two details are for ``-O2``, where Debian, Ubuntu and conda build
Python. The register medians index their vectors with constants only
(``RANK_LOAD3``, ``RANK_LOAD5``): with the loads in a loop that GCC
doesn't unroll at ``-O2``, one variable index kept the whole array in
memory, 5.0 ms instead of 2.8 for a 3×3 median of RGB. And the min/max
line loops take ``restrict`` sources, since GCC's ``-O2`` cost model
leaves a loop scalar rather than add the run-time check that the rows
don't overlap.

Why the builds agree
~~~~~~~~~~~~~~~~~~~~

Nothing is computed, so nothing is rounded: every path returns the
*k*-th value of the window, and a sorting network returns it whatever
the input, which the 0-1 principle lets the median networks be checked
for exhaustively. The checks compare the paths against each other and
against Python references: ``test_rankfilter_3x3_all_ranks``,
``test_rankfilter_network_ranks`` (sizes 5 to 11), ``test_min_max_filter_any_size``,
``test_rankfilter_same_size`` (no expanded copy against the expanded
copy, images smaller than the window included),
``test_rankfilter_all_bands_at_once`` (ten modes against band by band),
``test_median3_matches_reference`` at every width around the vector
width, ``test_modefilter_matches_reference`` and
``test_rankfilter_subclass_filter_is_used``, all in
``Tests/test_image_filter.py``; ``checks/simd_hashes.py``'s ``rank
filters`` area (L and RGB, sizes 3 to 11, median, minimum, maximum and a
rank of *s*\ ² / 3) agrees on every build.

Numbers
~~~~~~~

2560×1600, times faster than Pillow 13.0.0, which took the milliseconds
shown. The "portable" column is the fork's build without SIMD, where
the compiler vectorizes the networks.

.. generated: kernels-rank (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Filter
     - Pillow 13 (x86, ms)
     - Portable
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - Median 3×3 RGB
     - 1,183
     - 50x
     - 218x
     - 424x
     - 1,262
     - 154x
   * - Median 5×5 RGB
     - 4,023
     - 54x
     - 128x
     - 252x
     - 3,933
     - 82x
   * - Median 3×3 L
     - 383
     - 85x
     - 280x
     - 541x
     - 410
     - 191x
   * - Minimum 3×3 RGB
     - 818
     - 273x
     - 439x
     - 479x
     - 884
     - 298x
   * - Maximum 3×3 RGB
     - 811
     - 272x
     - 386x
     - 432x
     - 880
     - 253x

.. generated-end

In milliseconds, the AVX2 build's 3×3 median of RGB takes 2.8 ms, the
5×5 one 16, the L one 0.7, the minimum and maximum 1.7 and 1.9; the NEON
build 8.2, 48, 2.1, 3.0 and 3.5. Pillow-SIMD has no rank filter kernels
(1.0x). OpenCV's ``medianBlur`` (sorting networks too, KleidiCV's on
arm64) is 1.3x faster on the 3×3 RGB median on x86 and 1.6x on arm64,
level with the 5×5 and L medians on x86 and 1.1x to 1.3x ahead on them
on arm64; its ``erode`` and ``dilate`` are 1.4x and 1.5x faster on x86
and within 10% on arm64. What remains of the gap is
structural: OpenCV moves 3 bytes per RGB pixel where Pillow's layout has
4, and its morphology makes two output rows per step, sharing the rows
they have in common (``design/PERFORMANCE_GAPS.md``). The mode filter
isn't in the benchmark; its figures are above (the 3×3 kernel's are
from ``checks/simd_bench.py``'s "mode 3 L").

Code map
~~~~~~~~

* ``RankFilter.c``: ``RankNetworkBatcher()`` and ``RankNetworkPrune()``,
  ``RankFilterNetwork_u8()`` (any rank, sizes 3 to 11),
  ``rank_minmax_line()`` and ``RankFilterMinMax_u8()``,
  ``ImagingRankFilter()`` (Pillow's entry, on an expanded image),
  ``ImagingRankFilterAllBands()`` and ``ImagingRankFilterSameSize()`` (the
  fast paths on the image itself, 1-byte and 4-byte pixels);
  ``RankFilter.h``: ``RankFillPixels()``.
* ``RankFilterSIMD.c``, the ``rank`` group: ``RankFilterMedianVec_u8()``
  with the two networks and the ``RankFilterMedian3_u8()``/``Median5``
  entries, ``rank_minmax_line_simd()``.
* ``ModeFilter.c``: ``ImagingModeFilter()`` with the column-updated
  histogram, ``mode_window_pixel()`` for the pixels the kernel leaves;
  ``ModeFilterSIMD.c``, the ``mode`` group: ``mode3_u8_simd()``.
* ``src/PIL/ImageFilter.py``: ``RankFilter.filter()`` and
  ``RankFilter._filter_bands()``; ``Image.filter()`` in
  ``src/PIL/Image.py``, which calls the latter for multi-band images.
* Tests: the rank and mode tests at the end of
  ``Tests/test_image_filter.py``.

Pitfalls met on the way: the first version of "all bands at once" made
the 5×5 median of RGB 0.81x on arm64, from the padding byte, hence the
rule for when it pays; the register median was 1.8x slower at ``-O2``
than at ``-O3`` until the loads used constant indices; the min/max row
results in a whole-image buffer cost page faults on every call; and a
rank filter size set after ``__init__`` must still raise from
``expand()`` as Pillow's does, so ``filter()`` checks the size again
before taking the path without the copy.

Colour conversions
------------------

``Image.convert()`` between the 8-bit modes (``Convert.c``,
``ConvertYCbCr.c``): the whole-pixel conversions (L to RGB, RGB to and
from RGBA, RGB to CMYK), RGB to L, LA and I, premultiplying and
unpremultiplying alpha (RGBA to and from RGBa, which every resize,
rotation and transform of an RGBA image goes through), RGB to HSV, and
RGB to and from YCbCr. Each is a row function that ``ImagingConvert()``
runs over the image.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

The whole-pixel conversions copy bytes and set the padding or alpha byte
(CMYK is the bitwise inverse of RGB, with no undercolour removal). RGB
to L is ITU-R 601-2 luma in 16-bit fixed point, ``(r * 19595 + g * 38470
+ b * 7471 + 0x8000) >> 16``. Premultiplying is ``c * a / 255`` rounded
(``MULDIV255``, an exact ``/ 255`` as ``(x + 128 + ((x + 128) >> 8)) >>
8``); unpremultiplying is ``255 * c / a`` in C's integer division,
clipped to 255, and ``c`` unchanged where ``a`` is 0 or 255. RGB to HSV
follows ``colorsys.py`` in single-precision floats, with a few steps in
double, and truncates ``h * 255`` and ``s * 255``. RGB to and from YCbCr
are JPEG's BT.601 formulas through twelve tables of 16-bit terms at 6
fractional bits, ``T[v] = (int)(v * K * 64 + 0.5)`` truncated toward
zero, each sum of three terms shifted right by 6 (a floor); to RGB the
result is clipped, to YCbCr it is taken modulo 256 as a ``(UINT8)`` cast
does. Pillow 13's loops are scalar on x86; on arm64 GCC vectorizes the
byte loops of the whole-pixel conversions, which matters below.

The idea
~~~~~~~~

Most of these are bandwidth-bound: a 2560×1600 RGB image is 16 MB in
and 16 MB out, and the AVX2 build and OpenCV both run L to RGB and RGBA
to RGB at about 26 GB/s on the x86 machine (``design/PERFORMANCE_GAPS.md``).
The work is to make every loop vectorize, and for the four that compute
something (luma, unpremultiplication, HSV, YCbCr), to reproduce Pillow's
arithmetic in lanes.

* **Loops that vectorize under Python's flags.** The whole-pixel
  conversions are one ``pixel(out, in)`` function each, run by
  ``CONVERT_PIXELS``: chunks of 16 pixels with ``restrict`` pointers,
  which GCC vectorizes at ``-O2`` with no alias check and no remainder
  loop, and pointer increments rather than ``in + x * 4``, which
  ``-fno-strict-overflow`` lets wrap and GCC then refuses at any level.
  On arm64 the compiler's code for these loops changed with the
  optimization level (whole pixels built with widening multiplies, byte
  loops stored with ``st4`` at one level and scalar at the other), so
  the NEON build has explicit loops instead: ``vst4q`` of four planes
  for L to RGB, a vector and/or over whole pixels for RGB to RGBA and
  CMYK; and the arm64 portable build keeps Pillow's byte loop for L to
  RGB, which GCC does vectorize at ``-O3`` (the whole-pixel form was
  0.79x there).
* **Luma with 15-bit weights.** Pillow-SIMD's ``pmaddwd`` kernel needs
  signed 16-bit coefficients, and 38470 doesn't fit, so RGB to L, LA and
  I use ``(r * 9798 + g * 19235 + b * 3735 + 0x4000) >> 15``, which
  rounds one level away from Pillow's in a few pixels (an exception the
  ground rules list). The NEON kernel computes the same formula with the
  coefficients split into bytes (9798 = 38 · 256 + 70), so that
  ``vmull_u8``/``vmlal_u8`` products fit 16 bits: ``(hi * 256 + lo +
  0x4000) >> 15`` equals ``(hi + (lo >> 8) + 64) >> 7`` for every input,
  and the three builds give the same bytes.
* **Exact unpremultiplication.** Pillow-SIMD divided with a reciprocal
  estimate and was wrong at the edges: colour above alpha, which bicubic
  and Lanczos overshoot produce in every RGBA resize, overflowed to 0
  where Pillow clips to 255 (nearly transparent pixels went black), alpha
  0 gave colour 0, and other values were off by one. The x86 kernel now
  clips ``c`` to ``a`` first (which gives 255 where Pillow's clip would),
  takes ``rcpps`` of alpha scaled to a 16-bit factor, the quotient with
  ``pmulhuw``, and corrects it by one step from the remainder ``255 * c
  - q * a`` (the estimate is within one of the exact quotient); every
  value fits 16 bits, four pixels per SSE4 step, eight per AVX2 step. On
  arm64 one real division per pixel, ``(255 + 2⁻¹²) / a`` times ``c``,
  truncated, is exact for every pair (the two roundings stay within
  2⁻²² of the product, and a non-integer quotient is at least ``1 / a``
  from an integer), and faster still, 1.7x, is a 64 KB table of the
  scalar code's results, built on first use and read two pixels per
  64-bit word; the float kernel finishes rows while another thread
  builds it. On x86 the table is slower than the vector kernel.
  Exactness cost 1.6x against Pillow-SIMD's approximate kernel.
* **HSV lane for lane.** The scalar code's float and double steps are
  repeated per lane in the same order (``rgb2hsv_4``, ``rgb2hsv_8``,
  ``rgb2hsv_neon4``): three float divisions, the hue case selected with
  blends in the scalar code's priority, the ``2.0 + rc - bc`` terms in
  double and back, and ``fmod(h / 6 + 1, 1)``, which has no vector form,
  replaced by ``hd >= 1 ? hd - 1 : hd``, exact because ``h`` lies in
  [−1, 5]. No step is a multiply followed by an add, so arm64's fused
  multiply-adds can't change a result, and all 16.7 million RGB values
  give the same bytes on SSE4, AVX2, NEON and the portable build (which
  the ``fmod`` replacement alone made 2x Pillow 13). A reciprocal in place
  of the divisions was not exact; a multiplication by 1/6 was exact but
  no faster.
* **YCbCr without lookups.** A vector can't index twelve tables cheaply,
  but it can compute the entries: ``T[v] = (int)(v * K * 64 + 0.5)`` in
  32-bit floats gives every entry of every table (checked for all 256
  values, with or without a fused multiply-add), so the kernels convert
  the channels to floats, compute the three terms, sum them as 32-bit
  integers and shift, as the tables do. To RGB, the saturating packs are
  the clip; to YCbCr, masking to a byte is the ``(UINT8)`` cast. Four
  pixels per SSE4 step, eight per AVX2 step; on NEON 16 pixels as planes
  (``vld4q``), in four quarters of 4 lanes. Also the premultiply, four,
  eight or 16 pixels per step with ``MULDIV255`` in 16-bit lanes (NEON's
  ``vraddhn`` does the ``+ 128 >> 8`` twice in two instructions).

Why the builds agree, and where they don't
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unpremultiplication and YCbCr are integer results, exact on every path:
``test_unpremultiply_exact`` and ``test_unpremultiply_every_value`` run
all 65,536 (colour, alpha) pairs at several row widths, including colour
above alpha, and ``test_ycbcr_conversions_exact`` checks the three
pixels-at-a-time paths against the table formula in Python. HSV is
checked for all 2²⁴ colours when a kernel changes, and the ``chops/convert``
and ``ycbcr`` areas of ``checks/simd_hashes.py`` agree on every build,
YCbCr input outside the RGB range (which clips) included. RGB to L, LA
and I are the exception: the SIMD builds, which agree with each other,
round with 15-bit weights where Pillow and the portable build use 16,
one level apart in a few pixels; the hash tool makes its inputs with
``frombytes`` for that reason, and ``test_consistency_i16`` builds its
reference from the same conversion. Each vector kernel's leftover
pixels go through the same kernel on a zero-padded buffer
(``CONVERT_ROW``), so a pixel's result doesn't depend on its position in
the row (``test_convert`` in ``Tests/test_simd_consistency.py``).

Numbers
~~~~~~~

2560×1600, exact mode, times faster than Pillow 13.0.0, which took the
milliseconds shown.

.. generated: kernels-conversions (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Conversion
     - Pillow 13 (x86, ms)
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - RGB to L
     - 2.1
     - 1.9x
     - 1.9x
     - 2.5
     - 1.1x
   * - L to RGB
     - 2.7
     - 3.6x
     - 3.5x
     - 0.9
     - 1.0x
   * - RGB to RGBA
     - 5.6
     - 4.6x
     - 4.7x
     - 1.7
     - 1.0x
   * - RGB to CMYK
     - 7.4
     - 6.1x
     - 6.2x
     - 1.7
     - 1.0x
   * - RGBA to RGBa
     - 8.9
     - 4.9x
     - 6.7x
     - 3.4
     - 1.1x
   * - RGBa to RGBA
     - 26.6
     - 6.4x
     - 12.1x
     - 38.8
     - 5.4x
   * - RGB to HSV
     - 59.7
     - 3.7x
     - 6.7x
     - 81.4
     - 2.3x
   * - RGB to YCbCr
     - 9.2
     - 1.6x
     - 3.1x
     - 18.9
     - 1.1x
   * - YCbCr to RGB
     - 10.8
     - 3.0x
     - 5.7x
     - 14.1
     - 1.2x

.. generated-end

The arm64 column shows what the x86 one hides: on arm64 Pillow 13's
whole-pixel loops already ran vectorized (its L to RGB takes 0.9 ms
there against 2.7 on the faster x86 machine), so the gain is in writing
loops that vectorize *on x86 under Python's flags*; the gain on arm64 is
nil, and both are at the memory's speed. The portable build is 3.5 to 6x
on the whole-pixel conversions on x86, 2x on HSV and 1.0 to 1.3x on the
others. On the benchmark revision it was 0.32x on RGBA to RGBa on arm64: the
tail loop of that conversion, which is the whole loop in the portable
build, indexed with ``x * 4``, the pattern the first bullet describes;
it has pointer increments now, as Pillow's.

Against Pillow-SIMD 12.1.1 with AVX2: the whole-pixel conversions 2.9
to 3.5x faster, HSV 6.7x and YCbCr 2.8x and 4.9x (it has no kernels
there), and RGBa to RGBA 0.64x, the price of the exact division (its
kernel takes 1.4 ms, the exact one 2.2). Against OpenCV, the whole-pixel
conversions and RGB to L are within 1.3x either way, the difference
being Pillow's 4 bytes per RGB pixel against OpenCV's 3; RGB to HSV is 2.5x slower on x86 and 2.6x on arm64 (OpenCV divides by 12-bit
reciprocal tables, a different result, half of its hues one level from
the true value as half of Pillow's are); RGB to YCbCr is 1.2x slower on
x86 and 1.7x on arm64, YCbCr to RGB 1.3x faster on x86 and 2.1x slower
on arm64 (OpenCV uses 14-bit integer coefficients with one rounding,
where the exact kernels reproduce Pillow's tables in floats). Fast mode
(the default) computes HSV and YCbCr with one rounding instead: HSV 20x
on x86 and 7.7x on arm64, RGB to YCbCr 5.0x and 2.4x, YCbCr to RGB 7.2x
and 3.0x, all ahead of OpenCV; see :ref:`fast-mode` and, until the
fast-mode section is written, ``design/SIMD_FAST_KERNELS.md``.

**The dithers (2026-11-04).** The Floyd-Steinberg conversions to mode
1 and to a palette are a serial chain of about ten dependent operations
per pixel, which sets their speed, so they are not vector code; what
changed is the schedule. ``tobilevel()`` and ``topalette()`` do two
rows at a time, the second two pixels behind the first: a pixel needs
the errors of the row above at x + 1, which the first row has written
by then, so the two chains overlap, and the rows alternate between two
error buffers. The per-pixel body is Pillow's, so the pixels are the
same (checked over fourteen widths and eight heights, noise and
gradients, with the palette dither's closing write of the blue terms
into the slots past the row's end kept as it was): L to 1 1.74x, RGB
to 1 1.60x, RGB to the web palette 1.62x on 2560×1600.

Code map
~~~~~~~~

* ``Convert.c``: ``CONVERT_PIXELS`` and the ``*_pixel`` functions (whole
  pixels); the scalar ``rgb2l``, ``rgb2la``, ``rgb2i``, ``rgba2la``,
  ``rgb2hsv_row()`` and ``rgb2hsv_fast_pixel()``, ``rgbA2rgba()``
  (premultiply) and ``rgba2rgbA()``, each calling the table's ``convert``
  group first.
* ``ConvertSIMD.c``, the companion: ``rgb2l_4px()`` and ``rgb2l_8px()``
  with the ``rgb2l``, ``rgb2la``, ``rgb2i`` and ``rgba2la`` kernels;
  ``UNPREMULTIPLY`` and ``rgba2rgbA_4()``/``_8()``,
  ``neon_unpremultiply_16()``, the ``unpremultiply_table`` and
  ``unpremultiply_lookup()``; ``neon_muldiv255()``;
  ``rgb2hsv_4()``/``_8()``/``_neon4()`` and fast mode's loops;
  ``CONVERT_ROW``, which runs a kernel over a row with the zero-padded
  tail; ``convert_mask16()`` for the NEON and/or loops.
* ``ConvertYCbCr.c``: the tables and the scalar loops of
  ``ImagingConvertRGB2YCbCr()`` and ``ImagingConvertYCbCr2RGB()`` and of
  the fast-mode conversions; ``ConvertYCbCrSIMD.c``: ``ycc_term()`` and
  the four vector loops; ``ConvertYCbCr.h``: the constants both use.
* Tests: ``test_convert`` in ``Tests/test_simd_consistency.py``;
  ``test_unpremultiply_exact`` and ``test_ycbcr_conversions_exact`` in
  ``Tests/test_image_convert.py``; ``test_unpremultiply_every_value``.

Pitfalls met on the way: Pillow-SIMD's unpremultiplication (above);
results that depended on a pixel's position in the row, when the vector
loop and its tail computed differently (fixed by ``CONVERT_ROW``'s
padded tail); the AVX2 build's byte-by-byte CMYK being 15% slower than
SSE4 until the loop worked on whole pixels; the whole-pixel loops being
3.4x slower than byte loops on arm64 with one GCC and the reverse with
another, so arm64 got its own loops; and the tail loop with ``x * 4``
indexing that the benchmark caught after the page's own ground rules
had described the problem.

Compositing and channel operations
----------------------------------

``Image.alpha_composite()`` (``AlphaComposite.c``), ``Image.blend()``
(``Blend.c``), ``Image.composite()`` and ``paste()`` with a mask
(``Paste.c``), ``ImageChops`` (``Chops.c``), ``Image.point()`` and
``ImageOps``' invert, posterize and solarize and ``ImageEnhance``
(``Point.c``, ``ImageEnhance.py``), ``ImageFilter.Color3DLUT``
(``ColorLUT.c``), and ``split()``, ``merge()`` and ``getchannel()``
(``Bands.c``). Small kernels, each a few instructions per pixel, and
mostly bandwidth-bound once written right.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

Alpha compositing is Porter-Duff *over* in integers with 7 extra bits:
``outa255 = sa * 255 + da * (255 - sa)``, ``coef1 = sa * 255 * 255 * 128
/ outa255`` (one integer division per pixel), ``coef2 = 255 * 128 -
coef1``, each colour ``(s * coef1 + d * coef2 + 0x80 * 128) / 255 >> 7``
with ``/ 255`` as ``(x + (x >> 8)) >> 8``, and pixels with ``sa`` = 0 keep
the destination. Blend is ``in1 + alpha * (in2 - in1)`` in floats,
truncated. Masked paste and ``composite()`` are ``BLEND(m, out, in) =
(out * (255 - m) + in * m + 128) / 255`` with the same ``/ 255``. The
chops: ``multiply`` is ``a * b / 255`` and ``screen`` its inverse, in C's
integer division; ``add`` and ``subtract`` are ``(a ± b) / scale +
offset`` in float, clipped; ``lighter``, ``darker`` and ``difference``
need no arithmetic. ``point()`` looks every byte up in a 256-entry table
per band, into a zero-filled image; ``ImageEnhance`` blends the image
with a "degenerate" one (grey, black, desaturated, blurred). The 3D
colour LUT interpolates trilinearly between eight table entries, 16-bit
values with 6 fractional bits, the weights at 15 bits. Pillow 13's loops
for all of these are scalar on x86 except where GCC vectorizes them
(``lighter``, ``difference``, blend with AVX2).

The idea
~~~~~~~~

* **Alpha composite.** Pillow-SIMD's kernel, kept on x86: four (SSE4) or
  eight (AVX2) pixels per step, ``coef1`` from ``rcpps``, a reciprocal
  estimate, which rounds one level away from the integer division in 6%
  of the bytes, the one approximation left in exact mode on x86 (an
  exception the ground rules list; ``test_alpha_composite_every_alpha``
  is skipped there). On arm64 the division bound the loop at about 12
  cycles a pixel, a vector division or a corrected reciprocal was no
  faster, and GCC's vectorization of the portable loop beat the first
  NEON kernel; what won, 1.25x over that, is a 128 KB table of ``coef1``
  for every (source alpha, destination alpha) pair, built on first use,
  with NEON for the blends (``ac_blend8``, the ``/ 255`` as two shifts
  and adds in 32-bit lanes): the same pixels as the scalar code. Where
  ``alpha_composite()`` takes a destination offset, the overlay is
  composited straight onto the destination's rows in place
  (``ImagingAlphaCompositeOnto()``) instead of through a crop, a
  composite and a paste, which is no SIMD but 2.2x.
* **Multiply and screen without a division.** For every product ``x`` of
  two bytes, ``x / 255`` is exactly ``(x + 1 + (x >> 8)) >> 8``, and no
  step leaves 16 bits; screen is multiply with every byte inverted on the
  way in and out (``255 - (255 - a)(255 - b) / 255``). 32 bytes per AVX2
  step, 16 per SSE4 or NEON step, and the portable loop uses the same
  formula, so even the portable build is 6x Pillow 13. ``add`` and
  ``subtract`` with scale 1 and offset 0 are one saturating add or
  subtract per vector (``paddusb``, ``vqaddq_u8``), which compilers don't
  find; with scale 1 and another offset an integer loop that
  vectorizes; with any other scale a 511-entry table of the same float
  expression for every possible sum, in place of a float division per
  byte. ``lighter`` and ``darker`` can't leave 0..255, so they skip the
  clipping that stopped them vectorizing.
* **Tables that are a byte operation.** ``point()`` recognizes a table
  that equals ``v ^ c`` (invert is ``v ^ 255``), ``v & c`` (posterize) or
  ``v < t ? v : 255 - v`` (solarize) for all 256 values, in every band,
  and transforms whole rows with the byte operation instead (solarize as
  ``v ^ (v >= t ? 255 : 0)``, the comparison a ``pmaxub`` and
  ``pcmpeqb``); other tables keep the lookup, and the output is no
  longer zero-filled first, since every byte is written. A vector table
  lookup on x86 (``pshufb`` over 16 groups of 16 entries) was measured
  slower than the scalar lookup, 0.56 cycles per byte; on arm64 NEON's
  ``tbl``/``tbx`` look 16 bytes up in 64-entry groups held in registers,
  four instructions for a 256-entry table, and a row of 4-byte pixels is
  split into band planes so that each band's table is loaded once per
  row (L 1.9x, RGB 1.3x).
* **ImageEnhance as a table.** Contrast and Brightness blend the image
  with an image that is constant in each band (alpha excepted), so each
  output byte is a function of the input byte: ``enhance()`` now blends a
  0..255 gradient with those constants, through the same ``Image.blend``,
  and applies the result with ``point()``, instead of allocating a
  full-size degenerate image and blending it; the degenerate image is
  still made on request, for the subclasses that set it. The pixels are
  the same, since the same C code computes each byte.
* **Masked fills.** ``ImageDraw.text()`` and ``bitmap()`` put ink through
  an L mask one byte at a time; ``fill_mask_L_simd()`` blends 16 bytes
  (or four 4-byte pixels) with ``BLEND``'s rounding in 16-bit lanes, the
  colour channels of transparent pixels taking a mask of 255 as the
  scalar expression gives; the SSE4 and NEON kernels exist, AVX2 uses
  SSE4's. The drawing section returns to it.
* **The 3D colour LUT.** Pillow-SIMD's kernel on x86: the index and the
  interpolation weight of each axis in one 16-bit value (6 bits of
  index, 9 of weight), the eight entries interpolated with ``pmaddwd``
  against (1 − *s*, *s*) pairs, one pixel per 128-bit vector and two
  per AVX2 vector, with ``vpdpwssd`` in the AVX-VNNI build. It uses its
  own precision, not Pillow's: the output differs from the scalar
  build's by up to two levels in about 10% of the bytes for a random
  table (none for a smooth one), the other exception the ground rules
  list. The NEON kernel computes Pillow's arithmetic instead, one pixel
  as four 32-bit lanes, and gives the scalar pixels.
* **Bands.** ``split()`` and ``getchannel()`` transpose four pixels into
  planes with one ``pshufb`` on x86 and ``vld4q`` on NEON; ``merge()``
  uses ``vst4q`` on NEON, and on x86 the compiler's code, since an
  explicit unpack kernel measured no better.

Why the builds agree, and where they don't
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chops, ``point()``, ``ImageEnhance``, the masked pastes and fills,
blend and the band operations give the same bytes on every build: the
``paste/composite``, ``point/enhance`` and ``chops/convert`` areas of
``checks/simd_hashes.py`` agree, and ``test_multiply_screen_exact``,
``test_byte_operation_tables`` (every byte value, each pattern and a
near miss), ``test_lookup_table_matches_blend``, ``test_point_table``,
``test_bitmap_mask_blend_exact`` and ``test_alpha_composite`` (flat
images stay flat, vector loop and tail) check the paths against
references. The two exceptions are on x86 only: ``alpha_composite()``,
one level in 6% of the bytes (its hash area is named "x86 SIMD vs
scalar: known"), and ``Color3DLUT``, up to two levels in about 10%, both
Pillow-SIMD's kernels kept as they were. On arm64 both are exact, the
alpha composite checked for every alpha pair
(``test_alpha_composite_every_alpha``).

Numbers
~~~~~~~

2560×1600, exact mode, times faster than Pillow 13.0.0, which took the
milliseconds shown.

.. generated: kernels-compositing (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Operation
     - Pillow 13 (x86, ms)
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - alpha_composite RGBA
     - 19.6
     - 3.8x
     - 6.7x
     - 22.1
     - 1.2x
   * - blend RGB, 50%
     - 5.2
     - 1.0x
     - 1.7x
     - 13.0
     - 1.0x
   * - composite RGB, L mask
     - 11.7
     - 1.2x
     - 1.9x
     - 11.1
     - 1.0x
   * - ImageChops.multiply RGB
     - 14.7
     - 6.6x
     - 7.6x
     - 14.0
     - 3.8x
   * - ImageChops.screen RGB
     - 18.5
     - 8.4x
     - 9.5x
     - 17.8
     - 4.9x
   * - ImageChops.add RGB
     - 10.4
     - 5.5x
     - 5.6x
     - 15.5
     - 5.0x
   * - ImageChops.lighter RGB
     - 5.3
     - 2.8x
     - 2.8x
     - 6.4
     - 2.1x
   * - ImageOps.invert RGB
     - 4.1
     - 3.2x
     - 3.2x
     - 9.9
     - 5.3x
   * - point() RGB, a table
     - 4.1
     - 1.1x
     - 1.1x
     - 9.9
     - 1.3x
   * - point() L, a table
     - 1.7
     - 1.1x
     - 1.1x
     - 3.6
     - 1.9x
   * - Color3DLUT RGB, 16³
     - 80.0
     - 5.1x
     - 7.5x
     - 103
     - 2.0x
   * - Color3DLUT RGBA, 16³
     - 98.3
     - 6.5x
     - 9.7x
     - 118
     - 2.4x

.. generated-end

Blend, the masked composite and the plain table lookups are the
compiler's code in exact mode (blend's 1.7x is GCC's AVX2 vectorization
of Pillow's loop; an SSE4 kernel for it was measured and dropped), so
their columns say what the compiler does; fast mode's blend, which
interpolates in integers with ``pmulhrsw``, is 2.7x on x86 and 4.2x on
arm64. The chops run at the memory's speed: against OpenCV, ``multiply``
is level, ``screen`` 3.8x faster (OpenCV has no screen and composes it),
and ``add``, ``lighter``, ``difference`` and invert are 1.3 to 1.4x
slower on both machines, the 4/3 byte ratio of Pillow's RGB layout.
OpenCV's ``LUT``, a scalar loop over the whole image as one row, is 1.5x
faster than ``point()`` with a table on x86 and level on arm64 for RGB,
where NEON's lookups put the L table 1.4x ahead. Against Pillow-SIMD
12.1.1 with AVX2, the alpha composite and the LUT are its kernels and
equal; ``multiply`` is 6.3x faster, ``add`` 9x and invert 3.3x.

Code map
~~~~~~~~

* ``AlphaComposite.c``: ``alpha_composite_row()`` with the scalar
  definition, ``ImagingAlphaCompositeOnto()``; ``AlphaCompositeSIMD.c``
  (the ``composite`` group): ``alpha_composite_4px()`` and ``_8px()``
  (x86), ``ac_coef1_table`` and ``alpha_composite_16px_neon()``;
  ``AlphaComposite.h``: the pixel and the precision.
* ``Chops.c``: ``chop_multiply_row()``, ``chop_saturate_row()``,
  ``chop_add_subtract()``, with ``ChopsSIMD.c`` (the ``chops`` group);
  ``Blend.c``: ``blend_fast_row()`` and Pillow's loops, with
  ``BlendSIMD.c``; ``Paste.c`` with ``PasteSIMD.c``: ``fill_blend16()``
  and ``fill_mask_L_simd()``.
* ``Point.c``: ``point_byte_op()`` and ``point_byte_row()``;
  ``PointSIMD.c`` (the ``point`` group): the byte operations' loops,
  ``neon_lut16()``, ``neon_lut_row()`` and NEON's whole-image lookups;
  ``Point.h``: the operation enum; ``src/PIL/ImageEnhance.py``:
  ``_Enhance._lut()`` and ``_levels()``.
* ``ColorLUT.c``: the portable ``ImagingColorLUT3D_linear()``;
  ``ColorLUTSIMD.c`` (the ``colorlut`` group): ``mm_lut_pixel()`` and
  ``mm256_lut_pixel()`` with the x86 lookup, ``neon_lut_pixel()`` with the
  NEON one; ``ColorLUT.h``: the portable layout.
* ``Bands.c``: ``ImagingGetBand()``, ``SPLIT_LOAD``/``SPLIT_STORE``,
  ``ImagingMerge()``; ``BandsSIMD.c`` (the ``bands`` group): the band
  extraction, split and NEON merge of a row.
* Tests: ``test_alpha_composite``, ``test_alpha_composite_every_alpha``
  and ``test_point_table`` in ``Tests/test_simd_consistency.py``;
  ``test_multiply_screen_exact`` in ``Tests/test_imagechops.py``;
  ``test_byte_operation_tables`` in ``Tests/test_image_point.py``;
  ``test_lookup_table_matches_blend`` in ``Tests/test_imageenhance.py``.

Pitfalls met on the way: Pillow-SIMD's alpha composite gave a pixel a
different value depending on its column, because the vector loop and
its scalar tail rounded differently (the tail now runs through the
vector code on a padded buffer); the first NEON alpha composite divided
in double and lost to GCC's code with GCC 11, the second took a
corrected reciprocal and lost to the table; the SSE4 blend and merge
kernels measured no gain over GCC's; and ``pshufb`` table lookups on
x86 lost to the scalar loop, where NEON's ``tbl`` won.

Geometry
--------

``Image.transpose()`` (the flips, the 90° rotations, transpose and
transverse), ``rotate()`` and ``transform()`` with the affine,
perspective and quad methods (``Geometry.c``). The transposes move
pixels; the transforms compute each output pixel from a filter over the
source, in doubles.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

The 90° transposes move one pixel at a time, in 8×8 tiles within
512×512 chunks for the cache; the flips reverse rows, and flip top-bottom
is a ``memcpy`` per row. A transform maps every output pixel's centre
(``x + 0.5``, ``y + 0.5``) through the transform's equations in double
(affine: two linear forms; perspective: two linear forms over a third;
quad: bilinear in *x* and *y*), then samples the source there with a
filter: nearest truncates the coordinate; bilinear interpolates the 2×2
neighbours, with indices clamped at the edges and a missing second row
repeating the first; bicubic interpolates the 4×4 neighbours with a
cubic in each direction (``p1 + d (p2 + d (p3 + d p4))`` with the
``p``\ s differences of the samples), rows missing at the edge repeating
the one before. Each band goes through the filter separately, in double,
and is truncated to a byte (the bicubic clamped first). ``rotate()`` is
an affine transform; an affine transform with the nearest filter runs in
16.16 fixed point. Everything is one pixel at a time through a function
pointer for the transform and one for the filter.

The idea
~~~~~~~~

* **Transposes in register blocks.** The 8×8 tiles of the old loop
  wrote 32 bytes to each of their output rows, rows whose stride maps
  them to a few cache sets, so the lines were evicted and fetched again
  about three times before they were complete. Now a block of 8×8 bytes
  or 4×4 32-bit pixels is transposed in registers (SSE unpacks, NEON
  ``vtrn``), the mirrored variants of rotate 90 and 270 and transverse
  choose the order of the rows read and written, and the blocks go in
  columns of 64 bytes' worth of rows, so that each output row receives a
  whole cache line at once. The flips and rotate 180 reverse rows with
  ``pshufb``/``pshufd`` and ``vrev64``/``vext``, as fast as the
  ``memcpy`` of flip top-bottom. I;16 and the portable build keep
  Pillow's loops.
* **Transforms in tiles.** A rotation crosses a new source row every few
  pixels of an output row, so an output row walks over hundreds of source
  rows and nothing stays in cache. The output is computed in 64×64 tiles
  (``TRANSFORM_TILES``), whose source rows stay in cache; every pixel is
  computed from the same coordinates as before. The tiles alone make the
  portable build 4x faster on nearest rotation and 1.2x on bilinear and
  bicubic, and the fixed-point nearest path is tiled too. Other tile
  shapes were measured: 256×32 and 512×16 up to 3x slower than 64×64 on
  x86; 128×64 gives back the 4% the tiles cost a near-identity
  perspective transform on arm64, whose reads were sequential already,
  but brings nearest rotation from 10.9 to 25 ms. 64×64 stayed.
* **A pixel's channels in one vector of doubles** (x86). The filters for
  3- and 4-band images take the four channels together, one ``__m256d``
  with AVX2 or two ``__m128d`` with SSE4, with the operations of the
  scalar filter in the same order: the bilinear ``a + (b - a) d``, the
  bicubic's four terms (``-a + b`` as ``b - a``, which rounds the same),
  the clamp and the truncation (``cvttpd2dq``). x86 compilers don't
  contract these to fused multiply-adds, so the pixels are the scalar
  filter's. The common cases call the transform and the filter directly
  instead of through pointers, so they are inlined (quad with bilinear
  1.2x from that alone).
* **A tile row's coordinates at once.** A perspective transform divides
  twice per pixel. ``perspective_row()`` computes the coordinates of a
  tile row of 64 pixels in a loop of fixed length, with the per-pixel
  expressions unchanged, which the compiler vectorizes (``vdivpd`` with
  AVX2: four divisions at a time); IEEE division is exact, so the
  coordinates are the same. RGB bilinear 62 to 44 ms, bicubic 131 to 90,
  L nearest 23 to 13.
* **arm64: the filters' integer parts in lanes, two pixels at a time.**
  On arm64 GCC contracts the scalar filters to fused multiply-adds, so
  "the scalar result" there is the contracted one, and a NEON filter
  must contract the same way (the ground rules' exception). The first
  NEON filters, two doubles per vector with GCC's vector types so that
  the compiler fused them alike, were exact and no faster than the
  scalar code (bilinear 112 to 115 ms, bicubic 286 to 313) and were
  removed. The second kind won: the samples' differences (``b - a``,
  ``p2`` to ``p4``) are integers in the scalar filters, so they are
  computed for the four bands of two rows at once in 16-bit lanes,
  converted to double exactly through float (which holds them), and
  only then go through the same fused multiply-adds; and two output
  pixels whose source windows are both inside the image go through the
  filter together (``bilinear_filter32RGB_simd2``), so that their long
  chains of dependent operations overlap, the pixels at the edges one at
  a time. Rotate bicubic RGBA 1.59x, RGB 1.32x, bilinear 1.47x,
  perspective 1.25 to 1.30x, quad 1.37x on the Cortex-A78AE.

Why the builds agree, and where they don't
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transposes move bytes, and ``test_rotations_match_columns`` and
``test_flips_match_rows`` compare every variant with Python slicing at
sizes that exercise the blocks, the bands and the edges. For the
transforms, every build computes the same coordinates (the same
expressions per pixel, or a vector division of them) and the same
filter arithmetic in the same order, so all x86 builds give the same
pixels, and so do all arm64 builds; x86 and arm64 differ from each
other by the contraction, which is Pillow's own behaviour on arm64 (the
``transform`` area of ``checks/simd_hashes.py`` is one hash on x86 and
another on arm64). ``test_multiband_matches_single_band`` checks each
band of a transformed RGB, RGBX and CMYK image against the same
transform of an L image, across several tiles;
``test_transform_pairs_match_single_pixels`` checks the arm64 pair
filters against the single-pixel ones for every column of the output.

Numbers
~~~~~~~

2560×1600, exact mode, times faster than Pillow 13.0.0, which took the
milliseconds shown.

.. generated: kernels-geometry (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Operation
     - Pillow 13 (x86, ms)
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - FLIP_LEFT_RIGHT RGB
     - 2.0
     - 1.5x
     - 1.5x
     - 2.1
     - 1.2x
   * - ROTATE_180 RGB
     - 2.1
     - 1.6x
     - 1.6x
     - 2.1
     - 1.2x
   * - ROTATE_90 RGB
     - 3.6
     - 1.9x
     - 2.1x
     - 5.8
     - 1.2x
   * - ROTATE_270 RGB
     - 4.1
     - 2.2x
     - 2.2x
     - 7.3
     - 1.5x
   * - TRANSPOSE RGB
     - 3.6
     - 2.0x
     - 2.0x
     - 5.9
     - 1.2x
   * - ROTATE_90 L
     - 2.7
     - 6.2x
     - 6.2x
     - 5.3
     - 3.6x
   * - rotate 30°, nearest, RGB
     - 8.7
     - 1.9x
     - 1.8x
     - 40.2
     - 3.6x
   * - rotate 30°, bilinear, RGB
     - 73.9
     - 1.6x
     - 2.1x
     - 128
     - 1.7x
   * - rotate 30°, bicubic, RGB
     - 185
     - 1.3x
     - 2.0x
     - 291
     - 1.4x
   * - rotate 30°, bicubic, RGBA
     - 283
     - 1.8x
     - 2.9x
     - 391
     - 1.9x
   * - perspective, bilinear, RGB
     - 83.7
     - 1.8x
     - 2.6x
     - 147
     - 1.6x

.. generated-end

Nearest rotation's gain is the tiles (the portable build has the same
1.9x and 3.6x). Flip top-bottom, a ``memcpy`` per row, is 1.0x
everywhere, and the flips are at the memory's speed. Pillow-SIMD has no
geometry kernels. Against OpenCV: the 90° rotations and transpose of RGB
are 1.1 to 1.5x faster than OpenCV's on x86 and 2 to 2.4x on arm64;
the flips are 1.4x slower on x86, where OpenCV's are IPP's (the only
places IPP still runs for it), and 1.2x and 1.8x faster on arm64; the L
rotation is 1.4x behind
KleidiCV's transpose on arm64. The exact transforms are 1.2 to 1.5x
slower than ``warpAffine`` and ``warpPerspective`` on x86 and 2.1 to
2.5x on arm64: OpenCV computes in single-precision floats, eight or 16
output pixels per step across pixels, with an all-inside fast path,
where Pillow's result is defined in double per pixel. Fast mode (the
default) takes positions in fixed point and integer filters with one
rounding, within 2 levels of the exact result: rotate bilinear 6.2x on
x86 and 3.2x on arm64, bicubic 6.3x and 3.7x, RGBA bicubic 7.7x and
4.2x, perspective 6.0x and 2.9x, ahead of OpenCV on x86 and within 1.3x
either way on arm64; see :ref:`fast-mode` and, until the fast-mode section is
written, ``design/SIMD_FAST_KERNELS.md``.

Code map
~~~~~~~~

* ``Geometry.c``: the six ``Imaging*`` transpose and flip entries, which
  call the table's ``geometry`` group first and keep Pillow's loops for
  the other types; the scalar filters (``bilinear_filter*``,
  ``bicubic_filter*``); ``ImagingGenericTransform()`` with the tile loops;
  the tiled ``affine_fixed()``; the fast-mode entry.
* ``GeometryTransform.h``: ``affine_transform()``,
  ``perspective_transform()`` and ``quad_transform()``,
  ``perspective_row()``, ``TRANSFORM_TILES`` and ``PERSPECTIVE_TILES``,
  the clamps; shared with the companion so the filters inline.
* ``GeometrySIMD.c``: ``transpose_4x4_32()``, ``transpose_8x8_8()`` and the
  ``TRANSPOSE_BLOCKS`` driver, ``flip_rows()``; the x86 vector filters
  (``transform_vec``, ``tv_bilinear()``, ``tv_bicubic()``,
  ``bilinear_filter32RGB_simd()``, ``bicubic_filter32RGB_simd()``) and the
  NEON ones (``tn_*``, the ``*_simd2`` pair filters);
  ``geometry_transform_simd()``, the tile loops with those filters and
  the arm64 ``TRANSFORM_PAIRS``.
* ``TransformFast.c``: fast mode's map and tables; ``TransformFast.h``:
  the tile loops, ``tf_row()`` and the pixel kernels, instantiated
  twice, scalar in ``TransformFast.c`` and with the vector loops in
  ``TransformFastSIMD.c`` (the ``transform_fast`` group).
* Tests: ``test_rotations_match_columns`` and ``test_flips_match_rows``
  in ``Tests/test_image_transpose.py``;
  ``test_multiband_matches_single_band`` in
  ``Tests/test_image_transform.py``;
  ``test_transform_pairs_match_single_pixels`` in
  ``Tests/test_simd_consistency.py``.

Pitfalls met on the way: the AVX2 build's nearest rotation was slower
than the SSE4 build's (33 against 24 ms) with no AVX2 code in it, from
the order of memory accesses alone, and both became 8.8 ms with the
tiles, a reminder that a rotation is limited by where it reads, not by
its arithmetic (the vector filters cut instructions by a third to a
half and time much less, until the output was tiled); the NEON filters
went in, came out and went back in a different form; converting each
byte to double on arm64, as x86 does, was slower than the scalar
filters; and three bytes of an RGB result copied from a general
register went through the stack and stalled, so they are stored from
the vector register (``vst1_lane``).

Drawing and text
----------------

``ImageDraw``'s shapes, outlines, flood fill and text (``Draw.c``,
``Fill.c``, ``Paste.c``, ``_imagingft.c``, ``ImageDraw.py``,
``ImageFont.py``). Pillow-SIMD never touched these, and a profile of the
whole API (``design/PROFILE_ASSESSMENT.md``) found most of their time
outside any pixel loop: in allocations, in calls through function
pointers, in FreeType. The vector code here is small, the span fill and
two blends; the rest is where the time was.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

Shapes are rasterized by scan lines: a polygon's edges are intersected
with each row, the intersections sorted, and the runs between them
filled with a horizontal-line primitive chosen for the pixel type
(``DRAW`` holds a ``point``, an ``hline`` and a ``line`` per type). An
RGBA ink on an RGB image blends each channel, ``BLEND(a, out, in) = (out
* (255 - a) + in * a + 128) / 255``. A polygon outline wider than one
pixel must stay inside the polygon, so ``ImageDraw.polygon()`` filled an
image-sized ``1`` mask with the polygon and drew the wide edges through
it. ``text()`` renders each glyph with FreeType and blends it through its
8-bit bitmap as a mask (``ImagingFill2``, the masked fill of the
compositing section). ``floodfill()`` searched from the seed in Python,
pixel by pixel.

The idea
~~~~~~~~

* **Span fill.** ``hline32`` stored one pixel at a time behind a per-pixel
  mask test, which GCC doesn't vectorize at ``-O2`` even without the test
  (the length is unknown). Unmasked spans go through ``fill32()``, a
  loop over fixed chunks of four pixels, whose four stores every
  compiler merges into one 16-byte store at ``-O2`` on any architecture.
  Filled rectangles 6.45 to 2.85 ms at 1280×800. It is plain C in
  ``Draw.c`` and not a kernel of the table: it was one for a day
  (2026-11-04, 8 pixels per AVX2 store), and the call per span, with the
  caller's saved registers around it, cost 30% of the filled rectangles
  and 12% of the wide lines on arm64 and 7% of the lines on x86-64,
  for a loop that the stores bound either way
  (``design/NEON_ROUND_2026-11-04.md``).
* **RGBA ink in lanes.** ``hline32rgba`` computes the ink's terms once
  (``in * a`` per channel) and blends four pixels per step in 16-bit
  lanes, ``out * (255 - a) + c + 128`` then the ``/ 255`` as two shifts
  and an add, the alpha byte kept; the scalar loop finishes the row and
  takes the masked case (``test_rgba_ink_blend_exact`` checks every
  alpha at several widths). AVX2 builds use the SSE4 kernel. Text and
  ``bitmap()`` got the same treatment in ``fill_mask_L_simd()``.
* **Lines through direct calls.** Pillow 13 merged ``line8``, ``line32``
  and ``line32rgba`` into one ``line()`` that calls ``point`` and
  ``hline`` through pointers, an indirect call per pixel of every thin
  line and of a rectangle outline's sides. ``line_generic()`` is
  force-inlined back into one function per pixel type, dispatched like
  the others: 500 bounding boxes with 2-pixel outlines 4.03 to 2.53 ms.
* **Scan lines sorted in place.** The few intersections of a scan line
  went through ``qsort``, a call and a comparison through a pointer each,
  a large part of filling a polygon beyond writing its pixels. An
  insertion sort with the same comparison takes up to 32 values (equal
  values are interchangeable, so the order is the same): 200 polygons
  32.7 to 27.1 ms, against 19.2 to write the same area as rectangles.
* **Wide outlines without the image-sized mask.** The mask was 14 of the
  16.5 ms of drawing 100 ROI polygons on 2560×1600: allocated and cleared
  for each. ``ImagingDrawPolygonInside()`` records the polygon's fill as
  spans per row from the *same* scan (``hline_record`` in place of the
  mask's ``hline``), sorts them by row with a counting sort and by *x*
  with an insertion sort, merges them, and cuts the wide lines' runs to
  them, filled unmasked (``hline_spans``): the same pixels (checked
  against Pillow's mask on random polygons in nine modes, partly outside
  the image, degenerate and self-intersecting, widths 2 to 15), 17.2 to
  1.8 ms, and the work grows with the rows rather than the area, 1.7x
  more for polygons 1,400 pixels across.
* **Arcs, chords and pie slices without allocating.** Each row went
  through the clipping tree as linked lists of segment ends, two
  ``malloc``\ s per segment at every node, about 500 per shape; a tree
  has at most 7 nodes, so a row has at most 16 ends, which now sit in
  fixed arrays merged by the same comparisons in the same order: 1.23 to
  1.29x.
* **Flood fill in C.** The result of ``floodfill()`` is the 4-connected
  region of pixels that match (within ``thresh`` of the seed's colour,
  or neither the fill nor the border value), and a filled pixel never
  matches again, so the order of the search doesn't change the pixels.
  ``ImagingFloodFill()`` is a scan-line fill in C for 8-bit modes, run
  only where it compares exactly as the Python search did (the seed
  reads back as the value, the arguments have a pixel's shape, the seed
  isn't negative), the search kept otherwise: 86 to 1.4 ms for a
  320×200 region.
* **Glyphs cached.** ``font_render()`` loaded and rendered every glyph
  twice per call, and every call did it again; ``getbbox()`` and the
  layout loaded each glyph for its box and its advance. What FreeType
  gives for a glyph index and load flags depends on nothing else (the
  size is fixed, no transform is set, glyphs sit on whole pixels), so
  each font keeps the rendered bitmaps as 8-bit grey, their offsets,
  the boxes and the advances in an open-addressing hash table, cleared
  when the variation changes or at 8 MB. 100 lines at 1280×800: 87.8 to
  14.9 ms from the bitmaps and boxes, then 11.0 to 3.3 from the
  advances. Not SIMD at all, and the largest single gain in this
  section.

Why the builds agree
~~~~~~~~~~~~~~~~~~~~

Nothing here changes an algorithm's output: the fills store the same
ink, the blends keep ``BLEND``'s rounding in 16-bit lanes where every
term fits, and the structural changes (the span mask, the fixed arrays,
the C flood fill, the glyph cache) are checked against the code they
replace: ``test_polygon_outline_matches_full_mask``,
``test_floodfill_c_matches_search`` (seven modes, and 1,500 random
cases when it was written), ``test_rgba_ink_blend_exact`` and
``test_bitmap_mask_blend_exact``, ``test_glyph_cache_variation``, and
renders and boxes compared for every font in ``Tests/fonts`` at three
sizes, stroked and not. The ``draw`` area of ``checks/simd_hashes.py``
(rectangles, ellipses, pie slices, arcs, rounded rectangles and polygons
in three modes, RGBA ink, flood fill) agrees on every build.

Numbers
~~~~~~~

2560×1600, times faster than Pillow 13.0.0, which took the milliseconds
shown. The "portable" column is the build without SIMD: most of these
gains are algorithmic.

.. generated: kernels-drawing (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Drawing
     - Pillow 13 (x86, ms)
     - Portable
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - 200 filled polygons
     - 61.7
     - 2.3x
     - 2.4x
     - 2.5x
     - 86.6
     - 1.5x
   * - 200 filled ellipses
     - 3.3
     - 1.5x
     - 1.4x
     - 1.5x
     - 5.9
     - 0.9x
   * - 500 filled rectangles
     - 5.5
     - 1.7x
     - 1.7x
     - 1.7x
     - 11.2
     - 1.2x
   * - 500 lines, 5 px wide
     - 3.1
     - 1.3x
     - 1.6x
     - 1.5x
     - 5.5
     - 1.6x
   * - 500 boxes, 2 px outline
     - 4.2
     - 1.7x
     - 1.7x
     - 1.7x
     - 3.7
     - 1.0x
   * - 100 ROI polygons, 2 px outline
     - 19.1
     - 7.2x
     - 7.9x
     - 7.9x
     - 21.7
     - 6.0x
   * - 100 lines of text
     - 56.5
     - 7.5x
     - 17.5x
     - 17.5x
     - 113
     - 19.6x

.. generated-end

Pillow-SIMD is 1.0x on all of these. The ellipses and the boxes gain
nothing on arm64, where Pillow 13's loops were already as fast as the
new ones for their short spans. Against OpenCV, Pillow's filled shapes
are 1.1 to 1.7x faster on x86 (ellipses 1.25x: OpenCV fills them as
polygons of up to 72 vertices) and the ROI outlines 1.4x slower, from
the rule that keeps a wide outline inside the polygon, which OpenCV's
centred stroke doesn't have; on arm64 the lines and boxes are 1.8x and
2.4x faster, the ellipses 1.4x and the ROI outlines 1.2x slower. Text
isn't comparable: OpenCV's is Hershey vector strokes, not glyphs.

Code map
~~~~~~~~

* ``Draw.c``: ``fill32()``, ``hline32()`` and ``hline32rgba()``;
  ``line_generic()`` and ``line8``/``line16``/``line32``/``line32rgba``;
  ``sort_intersections()`` and ``polygon_generic()``; ``SpanMask``,
  ``hline_record()``, ``span_mask_build()``, ``hline_spans()`` and
  ``ImagingDrawPolygonInside()``; the clipping tree's fixed arrays
  (``event_list``, ``clip_tree_do_clip()``); ``ImagingFloodFill()``.
  ``DrawSIMD.c`` (the ``draw`` group): the fill and the RGBA blend loops.
* ``Paste.c`` and ``PasteSIMD.c``: ``fill_blend16()`` and
  ``fill_mask_L_simd()`` (text and ``bitmap()``).
* ``_imagingft.c``: ``GlyphCacheEntry``, ``glyph_cache_slot()``,
  ``glyph_advance()``, ``glyph_cache_clear()``.
* ``src/PIL/ImageDraw.py``: ``polygon()``'s call to
  ``draw_polygon_inside``, ``floodfill()``'s conditions for the C fill.
* Tests: ``Tests/test_imagedraw_simd.py``; ``test_rgba_ink_blend_exact``,
  ``test_bitmap_mask_blend_exact`` and ``test_floodfill_c_matches_search``
  in ``Tests/test_imagedraw.py``; ``test_glyph_cache_variation`` in
  ``Tests/test_imagefont.py``.

Pitfalls met on the way: a refactoring upstream (one ``line()`` through
pointers) cost 1.5x on thin lines without anyone noticing, and the
benchmark found it; the span mask gained less than hoped on the
benchmark's small ROIs (2.0 to 1.8 ms), because the scan and the twelve
wide lines remain, and more on large polygons; and the glyph cache had
to keep FreeType's unsupported pixel modes as such, so that they still
raise where they did.

Codecs and packing
------------------

The PNG encoder's row filters and the decoder's unfiltering
(``ZipEncode.c``, ``ZipDecode.c``), the raw codec's packing and
unpacking (``Pack.c``, ``Unpack.c``), ``tobytes()``, WebP and JPEG
decoding (``_webp.c``, ``JpegDecode.c``), and two neighbours: palette
mapping (``Quant.c``, ``QuantHash.c``) and the histogram (``Histo.c``).
The codecs themselves (zlib, libjpeg-turbo, libwebp) are untouched;
what changed is what Pillow does around them, with one exception: a
build with libdeflate (the wheels) compresses the PNG encoder's output
with it instead of zlib, as below.

What Pillow computes
~~~~~~~~~~~~~~~~~~~~

The PNG encoder tries the None, Up, Sub and Paeth filters on every row
(Average too with ``optimize``), keeps the one whose bytes have the
smallest sum of absolute values as signed bytes, libpng's heuristic,
and hands it to zlib; the decoder undoes the row's filter byte by byte.
The raw codec moves pixels between Pillow's layout and a "raw mode":
``RGB`` is three bytes per pixel, ``BGRX``, ``ABGR`` and the like are
four in another order, and each has a packer and an unpacker.
``tobytes()`` ran the encoder in 64 KB chunks and joined them. Every
WebP file went through libwebp's animation decoder, which decodes into
its own canvas, and from there through a ``bytes`` object into the
image. JPEG RGB rows came out of libjpeg-turbo as three bytes per
pixel, expanded to four by Pillow. Palette mapping (``quantize()`` and
``convert("P")`` with an adaptive palette) builds the palette with
median cut or maximum coverage, then maps each pixel to its nearest
entry, remembering each colour's entry in a hash table and searching
the palette from entry 0 in the order of the entries' distance from it,
as far as one can still be nearer. The histogram counts each byte.

The idea
~~~~~~~~

* **PNG filters 16 bytes at a time.** When encoding, each predictor reads
  only original bytes, so the filters vectorize across the row: Up and
  Sub are a byte subtraction; Average's ``(a + b) / 2`` rounded down is
  ``pavgb``, which rounds up, minus ``(a ^ b) & 1`` (NEON's ``vhadd``
  rounds down); Paeth computes its three distances in 16-bit lanes
  (``|a + b - 2c|`` doesn't fit a byte) and picks ``a``, ``b`` or ``c``
  with masks in the same order of preference; and the cost is ``min(v,
  -v)`` as unsigned bytes, summed with ``psadbw`` (NEON: widening
  pairwise adds). The bytes and the costs are the scalar loops', so the
  same filter is chosen and the file is byte for byte the same: PNG at
  level 1 89 to 51 ms at 1280×800, at the default level 436 to 399,
  zlib's matching being most of the rest. Decoding undoes only the Up
  filter in vectors: Sub, Average and Paeth depend on the pixel to the
  left, and one pixel per step, libpng's way, was 1.4 to 2.2x slower
  than the scalar loops, whose three or four byte positions run as
  independent chains (Paeth 80 to 152 ms).
* **Byte-order shuffles.** RGBX to RGB and back are one ``pshufb`` per
  four pixels (the 16-byte store writes four bytes past them, so the
  loop stops six pixels early) or ``vld4q``/``vst3q`` on NEON; the eight
  4-byte reorderings (BGRX, XRGB, XBGR, BGXR, BGRA, ABGR, ARGB, BGAR) go
  through one helper, ``permute4()``, which builds the shuffle mask and
  the fill of 0 or 255 from the permutation at run time. ``tobytes()``
  with the image's own raw mode knows its size, so the encoder gets room
  for all of it: one chunk, which ``join`` returns as it is, 1.9x.
* **WebP stills straight into the image.** A one-frame file whose frame
  covers the canvas is what the animation decoder decodes into a cleared
  canvas without blending; ``decode_still()`` calls ``WebPDecode`` into
  the image's own memory (when its rows are in one block), without the
  canvas and the two copies: 105 to 76 ms for a 2560×1600 RGB file.
  Animations and the other cases keep the frame decoder.
* **JPEG to 4-byte pixels.** With libjpeg-turbo, RGB images are read with
  the raw mode ``RGBX`` (``JCS_EXT_RGBX``), so its SIMD colour conversion
  writes Pillow's layout itself, 16.7 to 15.9 ms; ``draft()`` takes it.
  Writing from RGBX was 3 to 7% slower than packing to RGB first, so
  saving is unchanged.
* **Palette mapping.** Three changes, each keeping the palette and the
  pixels: the colour hash table's nodes come from 4,096-node blocks
  instead of one ``malloc`` each (a fifth of the time; the chains, table
  sizes and iteration order are the same); a direct-mapped cache of 2¹⁸
  (colour, entry) words replaces the memo hash table; and with maximum
  coverage the search from entry 0 is a vector scan of every entry for
  the smallest of ``dist * 512 + rank``, ``rank`` being an entry's place
  in the scalar search's order, so that ties go where they did (the
  keys fit 27 bits), eight entries per AVX2 step, four per SSE4 or NEON
  step. The median-box search stays scalar: it starts near the answer,
  and a full scan was slower with many colours. Then (2026-11-04) the
  colour hash table itself, where the median cut spent most of its
  time: the nodes keep their hash, and the pixel table, whose
  comparison orders by that hash, walks its chains on integers instead
  of calling ``pixel_cmp``, which recomputed both pixels' packed values
  at every step; the bucket is a multiply-shift modulus instead of a
  division by the table's prime; a run of equal pixels is one insertion
  counting them all; and the passes that looked every pixel up again
  take the box from the leaves' pixel lists (each colour with its count;
  the hash is injective at every scale, checked over all 16.7 million
  colours) or, with a scaled table, from a direct map of scaled cell to
  box. Chains, merge points and iteration order are unchanged, so the
  palettes are bit-identical: median cut 2.1x, the adaptive conversion
  1.7x on a 2560×1600 photo.
* **The histogram in four tables.** On x86, neighbouring pixels count
  into four separate tables, so that the increments of equal bytes
  don't serialize, and the padding bytes of 2- and 3-band images are
  skipped: 13.8 to 5.8 ms against Pillow 12.3. Pillow 13 then rewrote
  its own loops, which are level with the tables on x86 and 1.1x faster
  on arm64, where they are used. The masked histogram (2026-11-04) counts
  the same way, with the mask byte as the increment (0 or 1) instead of
  a branch per pixel, which on a random mask mispredicted: 12.5 to 2.1 ms
  on x86, the same counts.

Why the builds agree
~~~~~~~~~~~~~~~~~~~~

The PNG files are identical because the filters' bytes and costs are:
the ``png filters`` area of ``checks/simd_hashes.py`` hashes the files
of six modes at two levels, and ``test_png_every_filter`` decodes every
filter type on the six test shapes against a byte-by-byte reference.
``test_webp_still_matches_the_frame_decoder`` and
``test_jpeg_rgbx_decode_matches_rgb`` compare the new decoder paths with
the old; the ``pack/unpack`` area hashes every raw mode at every width
from 1 to 69 and at 1,280; the ``quantize`` area hashes the palettes and
pixels of five images at six colour counts with both methods and the
adaptive conversion, and agrees on every build, the vector search
included (``Tests/test_codecs_simd.py`` holds the codec tests).

Numbers
~~~~~~~

2560×1600 RGB, times faster than Pillow 13.0.0, which took the
milliseconds shown; the "portable" column is the build without SIMD.

.. generated: kernels-codecs (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Operation
     - Pillow 13 (x86, ms)
     - Portable
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - Encode PNG, level 6
     - 789
     - 1.2x
     - 1.6x
     - 1.6x
     - 1,015
     - 1.5x
   * - Decode PNG
     - 100
     - 1.0x
     - 1.0x
     - 1.0x
     - 109
     - 1.1x
   * - Decode WebP
     - 112
     - 1.5x
     - 1.5x
     - 1.5x
     - 108
     - 1.2x
   * - Decode JPEG
     - 16.2
     - 1.0x
     - 1.0x
     - 1.0x
     - 28.3
     - 1.0x
   * - Median cut, 256 colours
     - 2,603
     - 4.2x
     - 4.2x
     - 4.2x
     - 5,898
     - 5.1x

.. generated-end

PNG encoding at level 6 is zlib's: the filters are 1.2x in the portable
build (the arithmetic alone) and 1.6x in vectors, and the encoder is
now level with OpenCV's and ahead of libvips', which use libpng; at
level 1 the filters are a larger share and the gain 1.7x.

**libdeflate (2026-11-04).** The table above is a zlib build. Where
Pillow is built with libdeflate (``features.check("libdeflate")``; the
wheels are), the encoder keeps the filtered rows of the whole image and
compresses them in one piece after the last row with
``libdeflate_zlib_compress``, which has no streaming interface but a
better match finder: on a 1280x800 photo-like image at the default
level, 5x faster for RGB (1,177 to 232 ms), 6.4x for RGBA, 5.4x for L
and 1.9x for P, with files within 2% of zlib's for RGB and L and 5%
smaller for RGBA; ``optimize`` uses libdeflate's level 10, whose files
are 3 to 9% smaller than zlib's level 9 and take 2 to 9x less time for
RGB and RGBA (a little more for L and P). ``compress_level`` 0 to 9
keep their meaning. The pixels are the same; the bytes differ, as a
zlib-ng build's already did. zlib is still used for a preset
dictionary, an explicit ``compress_type`` (a zlib strategy), images
whose filtered rows exceed 256 MB (the one-piece path holds them and
the output bound in memory; ``PILLOW_LIBDEFLATE_MAX`` moves the cap)
and builds without the library. Decoding is unchanged: zlib's inflate
and the vector unfiltering above. JPEG encoding
and decoding, WebP encoding and JPEG thumbnails are the libraries' own
time, 1.0x; WebP decoding is 1.5x on x86 and 1.2x on arm64 from the
copies saved, now ahead of OpenCV. The histogram is 1.0x against Pillow
13 (above). Pillow-SIMD is 1.0x on all of these.

Code map
~~~~~~~~

* ``ZipEncode.c``: ``png_cost_none()``, ``png_filter_diff()`` (Up and
  Sub), ``png_filter_average()``, ``png_filter_paeth()``;
  ``ZipEncodeSIMD.c`` (the ``zip`` group): ``png_cost16()`` and their
  vector loops, and the decoder's ``png_unfilter_up_simd()``;
  ``ZipEncode.h``: the cost; ``ZipDecode.c``.
* ``Pack.c``: ``ImagingPackRGB()``, ``ImagingPackBGR()`` and the
  ``permute4`` callers; ``Unpack.c``: the RGB and BGR unpackers and the
  ``permute4`` callers; ``PackSIMD.c`` (the ``pack`` group, with
  ``pack_permute4_simd()``) and ``UnpackSIMD.c``;
  ``src/PIL/Image.py``: ``tobytes()``'s one-chunk path.
* ``_webp.c``: ``_anim_decoder_decode_still()``;
  ``src/PIL/WebPImagePlugin.py``; ``JpegDecode.c`` and
  ``src/PIL/JpegImagePlugin.py`` (``_EXT_RGBX``).
* ``Quant.c``: ``nearest_palette_new()`` and the colour cache;
  ``QuantSIMD.c`` (the ``quant`` group): ``nearest_entry_simd()``;
  ``QuantNearest.h``: ``NearestPalette``; ``QuantHash.c``: the node
  blocks, ``_hashtable_find()`` and ``hashtable_order_by_hash()``;
  ``Quant.c``: ``sum_box_leaves()``, ``build_box_map()``. ``Histo.c``:
  ``histogram_u8()``, two versions, and ``histogram_u8_masked()``.
* Tests: ``Tests/test_codecs_simd.py``.

Pitfalls met on the way: vectorizing the PNG decoder's Sub, Average and
Paeth as libpng does lost to the scalar loops; writing JPEG from
4-byte pixels lost to packing first; a build against zlib-ng was
measured and dropped; and the four-table histogram, a clear gain over
Pillow 12.3, was matched by Pillow 13's own rewrite, so it is kept
where it still wins and not where it doesn't.

What was tried and dropped
--------------------------

Reviewers ask about these first. Each was measured on the machine named
and dropped for the reason given; the sections above tell the rest.

On both architectures:

* **Column strips** for the blur's and the resize's vertical passes, to
  keep their accumulators in L1: 0.90 to 1.11x on the blur, and on the
  resize 1.14 to 1.45x on some upscales but 11 to 21% slower on others.
  Valgrind's cache model counted L1 misses that the hardware prefetcher
  serves from L2.
* **A different resize pass order per build** (vertical first for every
  downscale): faster, but the result would then depend on the build.
  The order is Pillow 13's.
* **Fusing the two exact resize passes** (an 8-bit intermediate in a
  ring): done only in fast mode, where the intermediate keeps 2⁻⁶ levels;
  the exact version is still open (below).
* **The horizontal resize pass in 16-bit products** for fast mode: faster
  on the Jetson, 2.5 to 2.9x slower than the exact AVX2 kernel on x86;
  one result on both machines rules out a kernel per architecture.
* **A sampled Gaussian** for fast mode, in place of Pillow's three boxes
  convolved: under-blurs at small σ (σ 0.3 is nearly the identity, where
  Pillow blurs) and differs by up to 25 levels at edges and 30 at corners.
* **``reducing_gap`` 2.0** for fast mode's large downscales: up to 6 levels
  from the exact result at 16x; 3.0 stays within 2 on photos.
* **Per-pixel PNG unfiltering** for Sub, Average and Paeth, as libpng does
  it: 1.4 to 2.2x slower than the scalar loops, whose byte positions run
  as independent chains. **JPEG encoding from 4-byte pixels**: 3 to 7%
  slower than packing first. **WebP encoding through libwebp's
  importers**: 0.5%. **zlib-ng**: the same time as Debian's zlib on this
  data.
* **ModeFilter variants**: keeping the highest count up to date was 2x
  faster on noise and slower wherever modes occur; searching the counts
  in blocks of 16 was slower than the plain loop.
* **FP16** for any kernel: the exact paths must keep their float
  arithmetic, fast mode's divisions need more than 11 bits, and x86 has
  no FP16 arithmetic before AVX-512, so a kernel that gives the same
  result everywhere would emulate it.
* **Tiling transforms only when a row spans many source rows**, and other
  tile shapes: no change, or slower (256×32 up to 3x slower than 64×64).

On x86:

* **Eight consecutive pixels per AVX2 step** for the integer 4-byte
  kernel filters: 0.93 and 1.02x with AVX2, 0.89 and 0.46x with
  AVX-VNNI; the duplicated lanes already make two pixels per
  multiply-add. **AVX2 5×5 float kernels**: no gain (4-byte) and 0.57x
  (1-byte). **The AVX2 3×3 float kernel for 4-byte pixels**, two columns
  in the two 128-bit halves added last: 0.63x the SSE4 kernel, and 0.2%
  of bytes off its rounding, which a build with several tiers can't
  have; dropped in favour of the SSE4 kernel on every x86 tier.
  **Force-inlining the filter kernel per size** for its own sake: 4% and
  0%.
* **Two rows per AVX2 vector** in the horizontal blur, the first time
  (Gaussian 1.08x, unsharp 0.84x); adopted once the passes kept rows in
  cache.
* **``vpdpwssd`` in the horizontal resize pass**, where each pixel is one
  long chain of additions: 1.55x slower than AVX2 on extreme downscales;
  the pair instructions stay there, ``vpdpwssd`` in the vertical pass
  and the filters.
* **A reciprocal in place of the divisions** in the exact HSV: not exact.
  **A multiplication by 1/6**: exact, no faster. **``pshufb`` table
  lookups** for ``point()``: slower than the scalar lookup. **An SSE4
  blend kernel** and **an SSE4 merge kernel**: no gain over GCC's code.
* **Huge pages**: without ``madvise`` the AVX2 build was 2x slower on big
  resizes and the box blur, so the advice stays.

On arm64 (Cortex-A78AE, GCC 11, 12 and 14):

* **NEON float filters for 1-byte images**: 1.5x slower than GCC's
  vectorization of the generic loop. **NEON transform filters converting
  each byte to double**, as x86 does: slower than the scalar filters;
  the integer-differences form replaced them.
* **The 12-bit weight split** for the blur's direct passes: slower than
  the 32-bit multiply. **The 1-byte blur's four-row interleave**: 1 to
  5% slower than the compiler's 1-byte loop.
* **A corrected reciprocal** for the alpha composite and for
  unpremultiplication, **a vector division** for the alpha composite,
  **a 16-bit integer unpremultiplication**: all slower than the table
  (alpha composite) or the float division (unpremultiplication). **A
  reciprocal with a Newton step** for fast mode's HSV: 24.1 against 17.8
  ms.
* **Whole-pixel conversion loops** (the x86 form): 3.4x slower than the
  byte loops with GCC 14 and faster with another GCC; the NEON loops
  replaced both. **The NEON vertical blur pass** and **the first NEON
  alpha composite**: 5% and 15% slower than the compiler's code.
* **Reloading the ``point()`` tables per 16 pixels** instead of splitting
  the row into planes: no gain.

Where OpenCV and libvips are still faster
-----------------------------------------

Over the benchmark's 111 cases, against Pillow 13.0.0: on x86 the AVX2
build is 5.44x (geometric mean), the AVX-VNNI build 5.56x, fast mode
6.13x, the SSE4 build 4.18x and the portable build 1.71x, where
Pillow-SIMD 12.1.1 with AVX2 is 2.13x and OpenCV 3.60x; on arm64 the
NEON build is 2.52x, fast mode 3.08x, the portable build 1.49x and
OpenCV 2.17x (libvips 1.18x and 0.70x, ImageMagick 0.28x and 0.20x). The
cases where OpenCV or libvips are ahead fall into a few classes
(``design/PERFORMANCE_GAPS.md`` has the per-case accounting over four
rounds, ``design/OTHER_LIBRARIES_KERNELS.md`` the study of their code).

Structural, and left alone because closing them would change Pillow's
pixels or its layout:

* **Four bytes per RGB pixel.** Pillow stores RGB as RGBX; OpenCV as
  three bytes. Every bandwidth-bound operation moves 4/3 of the bytes
  and runs at the same gigabytes per second (26 GB/s on the x86
  machine, both libraries), so takes 4/3 of the time: the chops, flips
  and rotate 180, invert, L to RGB, RGB to and from RGBA, YCbCr; and
  every per-channel kernel does a third more work, which the 3×3 and 5×5
  filters, the rank filters and the warps pay.
* **Precision.** OpenCV's HSV divides by 12-bit reciprocal tables, its
  Gaussian runs in 8.8 fixed point (KleidiCV's, behind the approximate
  hint, in 8-bit weights with an 8-bit intermediate), Carotene's bilinear
  resize uses 7-bit weights, the warps compute in single precision and
  ``addWeighted`` in floats rounded once. Pillow's results are defined
  by 22-bit (resize), 24-bit (blur) and double (transforms) arithmetic.
  Fast mode takes the ones where a single rounding is both faster and
  closer to the formula; the exact kernels keep Pillow's.
* **Algorithms.** A true Gaussian kernel against three extended box
  passes (OpenCV wins at small σ, Pillow at large); ``INTER_AREA``, a
  box average, against an antialiasing convolution for downscales;
  polygonal ellipses; Hershey stroke text against FreeType glyphs; a
  centred stroke for polygon outlines against Pillow's rule that a wide
  outline stays inside the polygon.
* **Vendor kernels.** On x86, IPP's flips and 180° rotation and its RGBA
  bicubic upscale (OpenCV's own code is 1.9x slower than Pillow's there);
  on arm64, KleidiCV's transpose on L images and its median, Carotene's
  erode and dilate.

Closable without changing a pixel, in the order the gaps document lists
them: fusing the two exact resize passes through a ring of horizontally
resampled rows, with the RGBA premultiplication folded into the loads
and its removal into the stores (the largest single gap on arm64, the
RGBA upscale, 157 against 44 ms; done for fast mode, not yet for the
exact kernels); 16-bit running vertical sums for the box blur's integer
radii, with the 16-bit division checked at start-up as the direct passes
do; sharing the middle rows' minimum or maximum between two output rows
of the 3×3 rank filters; the NEON horizontal resample kernel for long
filters (an 8x Lanczos downscale is 18 ms on the Jetson against 3.2 on
x86, a wider ratio than elsewhere); and an integer form of YCbCr to RGB
on NEON proven equal to the tables on all 256 values per term.

What remains otherwise is the libraries' own: PNG decoding is 60% inflate
and the rest the scalar Paeth and Sub filters; WebP encoding and
decoding on arm64 are libwebp's, Debian's build against OpenCV's bundled
one; ``point()`` with a table on x86 has no 256-entry byte shuffle below
AVX-512 VBMI.

Exact and fast modes
--------------------

Everything above is exact mode, Pillow's arithmetic, and it is what
``PILLOW_ALLSIMD_FAST=0`` or ``Image.set_fast(False)`` selects. Fast
mode, on by default, lets seven operations use kernels whose results
differ from Pillow's by small documented amounts; :ref:`fast-mode` is the
user documentation (the switch, the guarantees, the table of operations)
and ``design/SIMD_FAST_KERNELS.md`` the full description of each kernel.
This section gives the shape of each kernel and the rule they share.

The rule: a fast kernel sits beside the exact path, in its own function
that the operation's entry point calls when the thread's or the
process's setting says so, read once per call; a portable C function
defines its result, and the SSE4, AVX2 and NEON versions compute the
same integers, so every build on every architecture gives the same
pixels (each kernel has a ``fast:`` area in ``checks/simd_hashes.py``,
one hash on x86 and arm64); its largest difference from the exact
result and its error against a float reference are measured and
asserted by ``Tests/test_fast_mode.py`` on six image shapes as noise and
as photographs; arguments it doesn't take go to the exact path, which
also raises the errors; and the output's mode, size and edge handling
are the exact path's. The kernels, with their speed-ups over the exact
kernels (x86 AVX2, arm64 NEON) and their difference from the exact
result:

* **Gaussian blur and unsharp mask**, σ up to 6, 8-bit modes
  (``GaussianFast.c``): Pillow's three extended boxes, built with the
  exact path's radius and weights, convolved into one kernel whose
  weights are rounded to 1/256, applied in one sweep (vertical taps into
  one row, then horizontal, every sum in 16 bits); the bands within the
  kernel's reach of the edges are taken from the exact blur of strips
  twice as wide, so that the exact path's per-pass clamping is
  reproduced there. 3.3x and 3.2x at σ 1, 1.5x and 2.3x at σ 4; at most
  3 levels, and closer to a true Gaussian than the exact result (one
  rounding instead of three).
* **RGB to and from YCbCr** (``ConvertYCbCr.c``): JPEG's formulas with
  14-bit coefficients and one rounding, as OpenCV computes them, with
  ``pmaddwd`` and ``phaddd`` or 16 pixels per NEON step. 1.5x and 2.2x
  to YCbCr, 1.3x and 1.7x back; at most 1 level, and 0.3% and 0.06% of
  values off by one against the float formulas where the exact tables
  have 47% and 32%.
* **RGB to HSV** (``rgb2hsv_fast()``): the same formulas truncated
  exactly, ``floor(255 n / 6d)`` and ``floor(255 d / max)``, each one
  single-precision division, which gives the floor exactly because a
  non-integer quotient is at least 1/1530 from an integer; on arm64 the
  saturation comes from a 64 KB table and the hue's division takes
  16-bit lanes. 2.9x and 3.3x; hue 1 above the exact result for 0.3% of
  colours, saturation and value the same.
* **Affine, perspective and quad transforms and ``rotate()``**, bilinear
  and bicubic, L and the 4-byte modes (``TransformFast.c``): positions in
  exact integer arithmetic rounded to 1/256 pixel (perspective's two
  divisions as one IEEE division per pixel in single precision below
  32,768 pixels a side, four or eight lanes), the bilinear filter
  vertical-first in 16-bit lanes with one rounding, the bicubic as a
  table of 256 weight sets at 12 fractional bits; four pixels per AVX2
  step. Rotate bilinear 2.7x and 2.6x, bicubic 2.9x and 3.4x; at most 2
  levels (1 on photos), 0.6% of values off by one against float
  interpolation where the exact path's truncation has 41%; the pixels
  whose source position falls exactly on the image's edge may count as
  inside in one result and outside in the other.
* **Blend**, 0 < alpha < 1 (``blend_fast_row()``): alpha in 2⁻¹⁵ units,
  ``in1 + ((in2 - in1) a + 2¹⁴) >> 15``, which is what ``pmulhrsw`` and
  ``vqrdmulh`` compute. 1.7x and 4.1x; at most 1 level, the float result
  rounded rather than truncated.
* **Resize** of 8-bit images with 4-byte pixels, any filter but nearest,
  when the vertical pass comes second (``ResampleFast.c``): the two
  passes fused through a ring of ``ksize + 3`` rows, the horizontal
  kernels compiled a second time to keep 2⁻⁶ levels instead of whole
  ones (``RESAMPLE_OUT16``), the vertical pass in 16-bit products with
  ``pmulhrsw``/``vqrdmulh``. 1.2 to 1.3x on x86 upscales, 1.3 to 1.5x on
  arm64, 2 to 4% slower on moderate x86 downscales; at most 1 level (2
  for RGBA and LA, premultiplied), 0.4 to 1.5% of values off by one
  against a float resize where the exact result has 8 to 25%.
* **Large downscales** with the bicubic, Lanczos and MKS filters
  (``Image.resize()``): ``reducing_gap=3.0``, so a downscale of 6x or
  more first shrinks by an integer factor with ``reduce()``, exact box
  averages, and convolves the last 3x to 6x; a different algorithm, not a
  rounding, so bounded by measurement. 1/8 Lanczos 1.9x and 2.5x, 1/16
  2.0x and 4.5x; at most 2 levels at photo sizes, more at very small
  sizes, where ``reduce()`` averages partial boxes at the edges.

With fast mode on, the 320×200 bicubic downscale of the resampling
section is 15.8x and 6.0x, the σ 1 Gaussian 21.6x and 9.7x, HSV 20x and
7.7x, the 30° bicubic rotation 6.3x and 3.7x; the benchmark page shows
both modes side by side.

Appendix A: the numbers in one place
------------------------------------

Geometric means over the benchmark's 111 cases, times faster than Pillow
13.0.0, the published run of 2026-10-31 (the per-operation tables are in
the sections above and on the `benchmark page
<https://desmoteo.github.io/pillow-allsimd-perf/>`_):

.. generated: kernels-appendix-a (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Build
     - x86-64 (Core Ultra 9 285H)
     - arm64 (Jetson Orin)
   * - Portable (no SIMD)
     - 1.71x
     - 1.49x
   * - SSE4
     - 4.18x
     -
   * - AVX2
     - 5.44x
     -
   * - AVX-VNNI
     - 5.56x
     -
   * - NEON
     -
     - 2.52x
   * - Fast mode (AVX2, NEON)
     - 6.13x
     - 3.08x
   * - Pillow-SIMD 12.1.1 (AVX2)
     - 2.13x
     -
   * - OpenCV 5.0 (single-threaded)
     - 3.60x
     - 2.17x
   * - libvips 8.16
     - 1.18x
     - 0.70x

.. generated-end

The portable build's gain is the share of the algorithmic changes: the
pass orders, the rings, the sorting networks, the caches. The arm64
factors are smaller than the x86 ones for two reasons the sections
explain: Pillow 13's own code is already vectorized by GCC on arm64 for
many loops that stay scalar on x86, and NEON's 128-bit lanes and
instruction mix gain less over that than AVX2 does.

Appendix B: file map
--------------------

Where the kernels live. Pillow's own files hold the portable code and
the drivers; each area's SIMD code is in a companion, ``<Area>SIMD.c``,
with the SSE4 version (AVX2 blocks inside it), the NEON version, and
the area's group of the kernel table at the end. Unless noted, an area
has all three of SSE4, AVX2 and NEON.

* ``SIMDKernels.h``: the table of kernel groups and ``IMAGING_SIMD_CALL``;
  ``SIMDKernels.c``: the one translation unit, which includes every
  companion and defines the tier's table; ``SIMDKernels_sse4.c``,
  ``SIMDKernels_avx2.c`` and ``SIMDKernels_avxvnni.c``: the stubs that
  compile it once per tier in a dispatch build, each with its own flags;
  ``SIMDDispatch.c``: the scalar table, ``ImagingSIMD``, the CPU
  detection, ``ImagingSIMDInit()`` and ``ImagingSIMDSelect()``;
  ``ImagingSIMD.h``: the tier and ISA macros, ``SIMD_FORCE_INLINE``,
  ``mm_dpwssd`` and ``mm256_dpwssd``, ``mm_madd_add``, the widening loads;
  ``ImagingPx4.h``: one pixel as four 32-bit lanes.
* Resampling: ``Resample.c`` (the drivers) and ``ResampleHorizontal.c``
  (the portable horizontal kernels); ``ResampleSIMD.c``, which includes
  ``ResampleSIMDHorizontalConv.c``, ``ResampleSIMDVerticalConv.c``,
  ``ResampleSIMD32bpc.c`` (I, F, I;16) and ``ResampleFastSIMD.c``;
  ``ResampleFast.c`` and ``ResampleFast.h`` (fast mode); ``Reduce.c`` and
  ``ReduceSIMD.c``.
* Blurs: ``BoxBlur.c``, ``BoxBlur.h`` and ``BoxBlurSIMD.c``;
  ``GaussianFast.c`` and ``GaussianFastSIMD.c`` (fast mode);
  ``UnsharpMask.c`` and ``UnsharpMaskSIMD.c``.
* Kernel filters: ``Filter.c``; ``FilterSIMD.c``, which includes
  ``FilterSIMD_3x3i_4u8.c``, ``FilterSIMD_5x5i_4u8.c``,
  ``FilterSIMD_3x3f_4u8.c``, ``FilterSIMD_5x5f_4u8.c``,
  ``FilterSIMD_3x3f_u8.c``, ``FilterSIMD_5x5f_u8.c`` (x86),
  ``FilterSIMD_NxNi.c`` (x86 and NEON) and ``FilterSIMD_NxNf.c`` (NEON).
* Rank filters: ``RankFilter.c``, ``RankFilter.h`` and
  ``RankFilterSIMD.c``; ``ModeFilter.c`` and ``ModeFilterSIMD.c``.
* Conversions: ``Convert.c`` and ``ConvertSIMD.c``; ``ConvertYCbCr.c``,
  ``ConvertYCbCr.h`` and ``ConvertYCbCrSIMD.c``.
* Compositing and channels: ``AlphaComposite.c``, ``AlphaComposite.h``
  and ``AlphaCompositeSIMD.c``; ``Blend.c`` and ``BlendSIMD.c``;
  ``Paste.c`` and ``PasteSIMD.c``; ``Chops.c`` and ``ChopsSIMD.c``;
  ``Point.c``, ``Point.h`` and ``PointSIMD.c`` (byte operations; NEON
  lookups); ``ColorLUT.c``, ``ColorLUT.h`` and ``ColorLUTSIMD.c``;
  ``Bands.c`` and ``BandsSIMD.c``.
* Geometry: ``Geometry.c``, ``GeometryTransform.h`` and
  ``GeometrySIMD.c``; ``TransformFast.c``, ``TransformFast.h`` and
  ``TransformFastSIMD.c`` (fast mode).
* Drawing and text: ``Draw.c`` and ``DrawSIMD.c``, ``Paste.c`` and
  ``PasteSIMD.c`` (masked fills), ``_imagingft.c`` (the glyph cache,
  portable).
* Codecs and packing: ``ZipEncode.c``, ``ZipEncode.h``, ``ZipDecode.c``
  and ``ZipEncodeSIMD.c``; ``Pack.c`` and ``PackSIMD.c``; ``Unpack.c`` and
  ``UnpackSIMD.c``; ``_webp.c``, ``JpegDecode.c``; ``Quant.c``,
  ``QuantNearest.h`` and ``QuantSIMD.c`` (the nearest search),
  ``QuantHash.c`` and ``Histo.c`` (portable).
* ``Fast.c``: the fast-mode setting, process-wide and per thread.

Appendix C: the tools
---------------------

* ``checks/simd_hashes.py``: one hash per kernel area over many sizes
  and modes; run with each build on ``PYTHONPATH`` and compare. The areas
  agree on every build except where this page says so. On a dispatch
  build, ``PILLOW_ALLSIMD_ISA=sse4 python3 checks/simd_hashes.py`` hashes
  one tier (``scalar``, ``sse4``, ``avx2``, ``avxvnni``); the tiers'
  hashes equal the single-tier builds'.
* ``Tests/test_simd_tiers.py``: 75 operations on two image shapes, exact
  and fast, through every tier of the build (``Image.core.set_simd()``,
  ``Image.core.simd_tiers``): the SIMD tiers give the same pixels, and
  the scalar table differs only in the areas listed above. The x86 CI
  jobs also run the whole suite under ``PILLOW_ALLSIMD_ISA`` set to
  ``sse4``, ``avx2`` and ``scalar``.
* ``Tests/test_simd_consistency.py`` and ``Tests/simd_shapes.py``: the
  vector loops and their scalar tails give the same pixels, on six image
  shapes; ``Tests/test_fast_mode.py``: the fast kernels' bounds;
  ``Tests/conftest.py``'s ``EXACT_KERNEL_TESTS``: the Pillow tests that
  compare with Pillow's own pixels and so run with the exact kernels
  (an entry that matches no test fails the collection, so a renamed
  test can't slip through).
* ``checks/simd_test_remote.sh``: the test suite per SIMD build in a
  container, on this machine or over SSH, with its own Python and
  libraries, for other platforms too (QEMU);
  ``checks/simd_bench_remote.sh``: builds and benchmarks a git ref on a
  remote x86 or arm64 machine.
* ``checks/simd_dbench.sh``: builds the working tree in the benchmark
  image, at the ``-O3`` of the published wheels, and runs a script with
  it on one core; ``checks/simd_counters.py``: user-mode cycles,
  instructions, cache and branch misses per run of sixty cases, read
  with ``perf_event_open`` around each run, so they hold on a loaded
  machine; its ``--table`` compares builds or tiers.
* ``checks/simd_bench.py``: times operations across several builds, each
  in its own process; ``checks/simd_callgrind.py`` and
  ``checks/simd_allocs.py``: instructions, cache misses and allocations
  per operation; ``checks/simd_profile_report.py``: the three combined
  into a report that ranks operations by the kind of work left in them
  (``design/PROFILE_ASSESSMENT.md`` came from it).
* The benchmark repository, `pillow-allsimd-perf
  <https://github.com/desmoteo/pillow-allsimd-perf>`_: 111 cases against
  Pillow, Pillow-SIMD, OpenCV, libvips and ImageMagick in Docker, with
  ``analysis/`` scripts for accuracy against references;
  ``checks/simd_speedups_svg.py`` draws the chart of the PyPI description
  from its results, and ``checks/simd_tables.py`` writes the "Numbers"
  tables of this page, Appendix A, the PyPI description's table and
  figures and the fast-mode page's speed-ups from them, into regions
  marked with the benchmark commit they come from, which the lint
  workflow checks (``make tables``; ``design/GENERATED_TABLES_PLAN.md``).
* ``design/OPTIMIZATION_METHOD.md``: the order in which these tools are
  used on the next optimization (timings and callgrind, then the
  hardware counters, then the compiler's remarks, then SIMD), and what
  each stage rules out.
