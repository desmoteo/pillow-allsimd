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

The page is written section by section. Written so far: the ground rules,
resampling, the blurs, the kernel filters and the rank filters. To come:
colour conversions, compositing and channel operations, geometry, drawing and
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

.. list-table::
   :header-rows: 1

   * - Blur
     - Pillow 13 (x86, ms)
     - SSE4
     - AVX2
     - Pillow 13 (arm64, ms)
     - NEON
   * - Gaussian RGB, σ 1
     - 120.8
     - 4.4x
     - 7.3x
     - 162.2
     - 3.1x
   * - Gaussian RGB, σ 10
     - 120.9
     - 2.7x
     - 4.6x
     - 162.5
     - 1.8x
   * - Gaussian RGB, σ 30
     - 120.7
     - 2.7x
     - 4.6x
     - 162.6
     - 1.8x
   * - Gaussian RGBA, σ 10
     - 120.9
     - 2.7x
     - 4.7x
     - 162.6
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
     - 197.0
     - 5.0x
     - 7.4x
     - 236.2
     - 3.2x
   * - Unsharp mask L
     - 54.4
     - 6.0x
     - 8.9x
     - 69.6
     - 4.0x

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

* ``BoxBlur.c``: ``BOX_LINE_BLUR`` and ``ImagingLineBoxBlur32``,
  ``ImagingLineBoxBlur32x2`` (AVX2) and ``ImagingLineBoxBlur8``, the
  horizontal running sums; ``BoxDirectInit()``, ``box_direct()``,
  ``ImagingBoxBlurDirect()`` and ``ImagingLineBoxBlurDirect()``, the direct
  passes; ``interleave4()`` and ``deinterleave4()``;
  ``ImagingHorizontalBoxBlur()``; ``ImagingBoxBlurColumns()`` and
  ``ImagingVerticalBoxBlur()``; the cascade (``BoxBlurCascade``,
  ``BoxBlurEnsure()``, ``ImagingVerticalBoxBlurPasses()``); the fused pass
  (``BoxFusedRows``, ``box_fused_row()``, ``ImagingBoxBlurFused()``);
  ``ImagingBoxBlur()``, which chooses between them;
  ``_gaussian_blur_radius()`` and ``ImagingGaussianBlur()``, the fast-mode
  entry.
* ``ImagingPx4.h``: one pixel as four 32-bit lanes, SSE4, NEON and
  portable.
* ``GaussianFast.c``: fast mode's one-kernel Gaussian.
* ``UnsharpMask.c``.
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
scalar build by one level in about 0.2% of bytes, the 4-byte ones also
between SSE4 and AVX2, whose summation orders differ; the 1-byte AVX2
kernel equals SSE4 by construction.

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
     - 100.4
     - 3.8x
   * - RGB 3×3 FIND_EDGES
     - 146.9
     - 21.4x
     - 33.5x
     - 42.9x
     - 145.3
     - 5.5x
   * - RGB 5×5 SMOOTH_MORE
     - 133.6
     - 7.5x
     - 8.9x
     - 18.9x
     - 227.3
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
  ``filter_int_kernel()`` and the precision search in
  ``ImagingFilterSIMD()``, which chooses the kernel.
* ``FilterSIMD_3x3i_4u8.c``, ``FilterSIMD_5x5i_4u8.c``: the x86 integer
  kernels for 4-byte pixels, SSE4 and AVX2.
* ``FilterSIMD_NxNi.c``: the integer kernel for 1-byte pixels on x86 (and
  for both pixel sizes on arm64), with the SSE4/AVX2 and NEON primitives.
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
     - 1,184
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
     - 83x
   * - Median 3×3 L
     - 384
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
     - 433x
     - 881
     - 253x

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
isn't in the benchmark; its figures are above.

Code map
~~~~~~~~

* ``RankFilter.c``: ``RankNetworkBatcher()`` and ``RankNetworkPrune()``,
  ``RankFilterNetwork_u8()`` (any rank, sizes 3 to 11),
  ``RankFilterMedianVec_u8()`` with the two networks and the
  ``RankFilterMedian3_u8()``/``Median5`` entries, ``RankMinMaxLine()`` and
  ``RankFilterMinMax_u8()``, ``ImagingRankFilter()`` (Pillow's entry, on an
  expanded image), ``ImagingRankFilterAllBands()`` and
  ``ImagingRankFilterSameSize()`` (the fast paths on the image itself,
  1-byte and 4-byte pixels).
* ``ModeFilter.c``.
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
