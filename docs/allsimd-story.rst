.. _allsimd-story:

Six times Pillow: notes from Pillow-AllSIMD
===========================================

Pillow-AllSIMD is `Pillow <https://python-pillow.org>`_ 13 with SIMD
kernels: the same ``PIL`` package and the same API, installed with
``pip install pillow-allsimd``. On one thread and a 2560×1600 image, the
geometric mean of 111 operations is 6.3 times Pillow 13's speed on an x86-64
laptop with AVX2 and 3.3 times on an arm64 board with NEON. In exact mode,
which follows Pillow's arithmetic, it is 5.6 and 2.7 times. It started as a
fork of `Pillow-SIMD <https://github.com/uploadcare/pillow-simd>`_, which
brought SIMD resampling and filters to Pillow on x86 a decade ago.

This page is about where the speed comes from, and more about what didn't
work the way it looked like it would. The numbers are from the
`benchmark page <https://desmoteo.github.io/pillow-allsimd-perf/>`_ (an Intel
Core Ultra 9 285H with turbo off, and an NVIDIA Jetson Orin's Cortex-A78AE)
and from the design notes in the repository, where every change is written
down with its measurements, including the ones that were reverted.

How it is measured
------------------

Each change is timed against the build before it at ``-O3``, both builds
interleaved in one run on a pinned core, with a hash of the outputs in the
same run, so that a speed-up that changes pixels shows up next to the time.
Before any SIMD code is written, hardware counters say whether the loop is
waiting for memory (wider vectors won't help) or for the core. When the
arm64 board was shared with other people's containers at a load average of
15, wall-clock times were useless, but user-mode cycles per run, read with
``perf_event_open`` on the measuring thread, repeated to 0.1%.

Every SIMD kernel is compared with the portable C code on every build, and
the builds for different instruction sets must give the same pixels. Pillow's
own test suite runs on all of them. Where fast mode differs from Pillow (by
a level or two in a few operations, always the same way on every CPU), the
difference is listed in the documentation, and ``PILLOW_ALLSIMD_FAST=0``
turns it off.

One wheel for every x86-64 CPU
------------------------------

Pillow-SIMD is built for one instruction set: SSE4, or AVX2 with
``CC="cc -mavx2"``, and a binary built for AVX2 crashes on a CPU without it.
A wheel on PyPI has to run everywhere, so Pillow-AllSIMD compiles its kernel
file three times, for SSE4.1, AVX2 and AVX-VNNI, and picks the best table of
kernels when ``PIL`` loads.

The rest of the library is then compiled for baseline x86-64, and that cost
more than expected. Timing all 216 operations of the fork's own benchmark,
the wheel against a build for AVX2, found 33 slower in the wheel, some by
3.4 times. None of them had a SIMD kernel: they were plain C loops that GCC
had vectorized because the whole build was given ``-mavx2``. ImageChops'
soft light, hard light and overlay, the extrapolating blend behind
``ImageEnhance``, the colour matrix, and the general rank filter's sorting
network. The fix was to put such loops in one header of plain C that every
tier compiles for itself, called once per row through the kernel table. On
all the large ones the wheel now matches the build for AVX2.

That kind of regression is invisible to a benchmark of the build you happen
to test, and timings on CI runners are noise. Instruction counts under
callgrind are not: CI now counts the instructions of 59 operations in each
tier of the wheel's build and fails when one rises by more than 3%. It
passed its first run on a GitHub runner with all 177 counts unchanged.

PNG: the time wasn't in the filters
-----------------------------------

Decoding the benchmark's PNG took 97 ms, and 67 of them were zlib inflating
the image data. `libdeflate <https://github.com/ebiggers/libdeflate>`_
inflates the same stream in 28 ms, but only in one piece, while Pillow feeds
its decoder a block at a time. So the PNG plugin now walks the chunk headers
first (a seek and an 8-byte read per chunk) to learn how long the compressed
stream is, the decoder gathers it, inflates it at once and unfilters the
rows where they lie. 97 ms became 62 ms.

The careful part was everything that isn't the happy path. Animation frames,
``LOAD_TRUNCATED_IMAGES``, files whose chunks can't be walked, and any stream
that libdeflate refuses or that doesn't hold exactly the image's rows go
through zlib's stream decoder as before, replaying the gathered bytes from
the start, so a damaged file raises the same error and leaves the same
partial image. Every PNG in Pillow's test suite, whole, cut at random offsets
and with single bits flipped, 15,666 cases, gives the same pixels or the
same error text with the new path, without it, and in the old build.

Encoding went the same way: libdeflate compresses the filtered rows of the
whole image in one call, 5 times as fast as zlib at the default level for an
RGB photo, with files within 2% of zlib's.

The quantizer: removing the obvious cost gained nothing
-------------------------------------------------------

``quantize(method=FASTOCTREE)`` copied the image into an array of pixels and
returned the palette indices in another, 16 MB each at 2560×1600, allocated
on every call. Removing both arrays went from 15.6 to 15.3 ms. Timing the
phases showed the passes over the pixels were the cost: counting each pixel
into its bucket took 11 ms.

A bucket was a count and four 64-bit sums, five additions to memory per
pixel. Keeping instead four 32-bit sums per bucket, flushed to the 64-bit
ones every 2²⁴ pixels, made the count one vector addition per pixel, in a
table less than half the size. The lookup pass got a 16 KB table of palette
indices in place of the 160 KB of buckets. That gave 10 ms. The last step
was surprising: the bucket of a pixel is built from shifts whose counts are
read from the cube, and on x86 without BMI2 (not part of any tier's flags)
a shift by a variable count is three micro-operations. The cube of an RGB
image always has four bits per channel; with that as a constant, 7.9 ms,
against 25 ms in Pillow 13. Same palette, same indices.

A faster path that took twice as long
-------------------------------------

Drawing a wide line fills a polygon, and every row of it has exactly two
intersections to sort. A special case for two values cut the instructions
by 6% and doubled the time: 85 million cycles instead of 43. GCC compiled
the compare-and-swap as one 8-byte load of the two floats that had just been
stored one by one, and a load that spans two smaller stores is not
forwarded from the store buffer; the core waits for the stores to reach the
cache. The special case is gone, and a comment says why.

arm64: one multiplier
---------------------

The Cortex-A78 has one pipe for vector multiplies. Kernels with many
multiply-accumulates are bound by it however the instructions are scheduled,
and splitting a sum into two accumulator chains measured slower each of the
three times it was tried.

So the gains came from multiplying less. The 3D colour LUT interpolates
eight table entries per pixel with seven 32-bit multiplies; where the table
values fit 15 bits, as they do for any table that doesn't extrapolate far,
``sqdmulh`` on 16-bit lanes gives exactly ``((b - a) · s) >> 15`` for two
pixels at once: 1.63 times as fast, the same pixels. The box blur multiplied
the two bytes just outside the box by a weight that is zero for the common
radii (1 to 4, 6, 7, 8, 10, …): skipping it made radius 10 1.35 times as
fast, on x86 too.

The resampling kernel was the opposite case: four instructions per cycle,
limited by the number of instructions. Its tap loop used an ``int`` index,
and GCC computed four sign-extended addresses and loaded each coefficient on
its own: 17 instructions for four taps. With pointers that advance and the
four coefficients in one load, multiplied by lane, it is 10, and resizes
take 8 to 15% fewer cycles. Now it waits for the multiplier.

What CI found on other people's machines
----------------------------------------

Two failures that only showed up on platforms nobody develops on:

- **Cygwin**: since the wheels' build includes AVX2 kernels everywhere, the
  Cygwin job crashed intermittently in every operation with an AVX2 kernel.
  GCC for 64-bit Windows keeps 256-bit values in stack slots that it assumes
  are 32-byte aligned, on a stack that the ABI aligns to 16 (GCC bug 54412,
  open since 2012). MSYS2 and Debian patch their compilers; Cygwin's doesn't.
  Asking the assembler to encode aligned moves as unaligned ones
  (``-muse-unaligned-vector-move``, binutils 2.38) fixed it at no cost.
- **RHEL 9**: GCC 11 accepts ``-mavxvnni``, and binutils 2.35 then rejects
  the AVX-VNNI instructions. The build now compiles a three-line probe and
  leaves that tier out where the toolchain can't make it.

Try it
------

.. code-block:: bash

    python3 -m pip uninstall -y pillow pillow-simd
    python3 -m pip install pillow-allsimd

Wheels are on PyPI for Linux on x86-64 and arm64; elsewhere it builds from
source. The :doc:`kernels page
<allsimd-kernels>` has the design of every kernel, and the
`repository <https://github.com/desmoteo/pillow-allsimd>`_ has the design
notes, the benchmark harness and the issue tracker.
