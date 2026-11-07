Changelog (Pillow-AllSIMD)
==========================

Pillow-AllSIMD's changes, then those of Pillow-SIMD, which it forks.

Unreleased
----------

- JPEG loading: for L and RGB images, libjpeg-turbo writes the image's
  rows itself instead of one line at a time into a buffer that Pillow
  copied. The pixels are the same; averaged over heap layouts, RGB files
  decode 4 to 10% faster, and the time moves less with where the heap
  puts the buffers

13.0.0.post4 (2026-11-06)
-------------------------

Based on Pillow 13.0.0

- Building from source on RHEL 9, CentOS Stream 9 and their relatives
  works again: their GCC 11 accepts ``-mavxvnni`` and their assembler
  (binutils 2.35) rejects the instructions, so the default build of
  13.0.0.post3 failed there. The build now tries the toolchain first and
  leaves the AVX-VNNI kernels out where it cannot make them; SSE4.1 and
  AVX2 are chosen at run time as elsewhere. ``-C simd=avxvnni`` says what
  it needs instead of failing in the assembler
- Cygwin, and MinGW with an unpatched GCC: the AVX2 and AVX-VNNI kernels
  no longer crash. GCC for 64-bit Windows keeps 256-bit values in stack
  slots that it takes to be 32-byte aligned on a stack the ABI aligns to
  16 (GCC bug 54412), so since the build chooses its kernels at run time
  (13.0.0.post3), about every other call of an AVX2 kernel on a CPU with
  AVX2 ended in a segmentation fault there. The build now asks the
  assembler to encode aligned vector moves as unaligned ones
  (``-Wa,-muse-unaligned-vector-move``, binutils 2.38 or later), which
  cost the same. MSVC builds and MSYS2's patched GCC were not affected
- PNG loading inflates the image data with libdeflate where Pillow was
  built with it: the plugin walks the headers of the ``IDAT`` chunks for
  the length of the compressed stream, the decoder gathers the stream
  and decompresses the image in one piece, which libdeflate does 2.4x
  faster than zlib's stream, and the rows are then unfiltered where they
  lie. On 2560x1600 images: 1.6x for RGB and RGBA, 1.4x for L, 4x for P;
  1.6 to 1.8x at 512x512 and 128x128, no slower on icons. The pixels are
  the same. zlib's stream still decodes the frames of an animation,
  files loaded with ``LOAD_TRUNCATED_IMAGES``, images above the 256 MB
  cap of the encoder (``PILLOW_LIBDEFLATE_MAX``), and any stream that
  libdeflate refuses or that does not hold exactly the image's rows, so
  that damaged and truncated files raise the errors they raised
- ``quantize(method=Image.Quantize.FASTOCTREE)`` on RGB and RGBA images
  reads the image's rows and writes the result's, without the two
  per-pixel work arrays, counts an RGB pixel into its bucket with one
  vector addition and maps it through a table that fits the first-level
  cache: 7.9 ms on a 2560x1600 photo, from 15.6 ms in a build for AVX2
  and 25.1 ms in the x86-64 wheel and in Pillow 13. The same palette
  and indices
- ``ImageFilter.BoxBlur`` with a radius whose box divides 2^24 with a
  remainder of 0 or 1 (1 to 4, 6, 7, 8, 10, 17, ...) skips the product
  with the far weight, which is zero there: 1.35x at radius 10 on
  x86-64 with AVX2 (7.5 to 5.5 ms on 2560x1600 RGB) and on arm64, 1.4x
  with SSE4.1. The same pixels
- arm64: 3D colour LUTs whose values stay within about twice the 0..1
  range are interpolated two pixels at a time in 16-bit lanes
  (``sqdmulh``), with the scalar code's pixels: 1.63x (52 to 32 ms on a
  2560x1600 image on a Cortex-A78)
- arm64: resizing RGB and RGBA images takes 8 to 15% fewer cycles (15%
  when enlarging with Lanczos): the horizontal kernel's tap loop went
  from 17 to 10 instructions per four taps, and fast mode's vertical
  kernel keeps its row pointers and weights in registers for up to 16
  taps. The same pixels
- Pasting through a mask has vector kernels on x86-64 and arm64:
  ``paste(im, box, mask)`` with an L mask, ``Image.composite()``, and
  an RGBA image pasted through its own alpha. On a 2560x1600 RGB image
  the mask composite takes 3.0 ms, from 11.7 ms in the x86-64 wheel and
  6.3 ms in a build for AVX2, and RGBA over RGB 2.9 ms, from 12.0 and
  5.8 ms; the same pixels
- x86-64 wheels: the loops that only a build for AVX2 ran fast, because
  the compiler vectorized them there, are compiled once per instruction
  set (``PlainLoops.h``) and reached through the kernel table, so the
  wheel runs them as that build does: ``ImageChops.soft_light``,
  ``hard_light`` and ``overlay`` 3.4x faster, ``ImageEnhance.Color`` and
  ``Sharpness`` (and every ``Image.blend`` outside 0..1, or in exact
  mode) 1.7x, ``convert()`` with a colour matrix 1.9x, I to L 1.5x, the
  rank filters of sizes 7 to 11 and non-median ranks 1.7x. The same
  pixels in every tier
- The quantizers' two per-pixel work arrays (16 MB each for a 2560x1600
  image, allocated and freed on every call) get the huge-page advice the
  image blocks have, on Linux where transparent huge pages are in
  ``madvise`` mode: when the allocator maps them afresh, which depends on
  what the process allocated before, their page faults were half of a
  fast-octree quantization (30 to 19.6 ms; 15.6 ms when the allocator
  recycles the arrays, as before)

13.0.0.post3 (2026-11-04)
-------------------------

Based on Pillow 13.0.0

- PNG saving compresses with libdeflate where Pillow was built with it
  (the wheels, and source builds on a machine with ``libdeflate.h``):
  the filtered rows of the image are compressed in one piece after the
  last row, which libdeflate does several times faster than zlib's
  stream (on a 1280x800 photo at the default level: 5x for RGB, 6.4x
  for RGBA, 5.4x for L, 1.9x for P, the files within 2% of zlib's or
  smaller). ``optimize`` uses libdeflate's level 10: files 3 to 9%
  smaller than zlib's level 9, in 2 to 9x less time for RGB and RGBA
  (a little more for L and P). The pixels are the same; the
  bytes differ from zlib's, as a zlib-ng build's already did. zlib is
  still used for a ``dictionary``, an explicit ``compress_type``, images
  whose filtered data is above 256 MB (``PILLOW_LIBDEFLATE_MAX`` moves
  the cap, in bytes; 0 turns libdeflate off), and builds without it
  (``-C libdeflate=disable``). ``features.check("libdeflate")`` tells
- The masked histogram (``Image.histogram(mask)``) counts without a
  branch per pixel, into four tables like the unmasked one: 6x faster on
  x86-64, the same counts
- The median-cut quantizer (``quantize()``'s default,
  ``convert("P", palette=ADAPTIVE)``) spends less time in its colour
  hash table: chain walks compare the hashes kept in the nodes, the
  bucket needs no division, runs of one colour count at once, and the
  passes over the image that looked every pixel up again read the box
  from the pixel lists or a direct map instead. The palettes and indices
  are bit-identical; 2.1x on a 2560x1600 photo, 1.7x for the adaptive
  conversion
- ``ImageFilter.ModeFilter(3)`` has a vector kernel on x86-64 and arm64
  (``ModeFilterSIMD.c``): the nine window bytes counted against each
  other and the winner picked by a packed maximum, the same pixels as the
  histogram code; 14x on a 2560x1600 L image on AVX2
- arm64: kernel filters with a power-of-two scale (``SHARPEN``,
  ``FIND_EDGES``, ``EDGE_ENHANCE``, ``EMBOSS``, ``CONTOUR``, ``BLUR`` and
  kernels of one's own like them) are computed with their small integer
  coefficients in 16-bit sums, adds for coefficients of one, where the
  general kernel pays four multiplies per tap on a core with one
  multiplier pipe: 1.8x on RGB, 1.5x on L on a Jetson Orin, the same
  pixels
- Drawing: the fill of a span is a loop in ``Draw.c`` again, four pixels
  per step, instead of a kernel called through the table once per span,
  which the arm64 measurement of this release found to cost 30% of the
  filled rectangles and 12% of the wide lines there (and 7% of the lines
  on x86-64); the same pixels. The polygon code's rounding no longer
  calls ``floor`` and ``ceil``, which are library calls where the file is
  compiled without SSE4.1, as in the x86-64 wheels: another 5% on lines
  and outlines there, the same pixels
- The dithered conversions (``convert("1")`` and ``convert("P")`` with
  Floyd-Steinberg dithering) do two rows at a time, the second two
  pixels behind the first, so that the two rows' chains of error terms
  overlap; the same pixels, 1.6 to 1.7x
- Wheels for x86-64 Linux (manylinux_2_28 and musllinux_1_2 x86_64), built
  and tested as the arm64 ones are. Each holds the SSE4.1, AVX2 and
  AVX-VNNI kernels and picks the set the CPU has when Pillow loads (the
  dispatch build below), so pip installs a wheel on x86-64 Linux now and
  ``-C simd=avx2`` is no longer needed; ``--no-binary pillow-allsimd``
  still builds from source, with the same three sets
- The benchmark tables and figures of the documentation (the PyPI
  description's table and headline figures, the kernels page's nine
  "Numbers" tables and Appendix A, the fast-mode page's speed-ups, the
  README's Gaussian figure) are written by ``checks/simd_tables.py`` from
  the benchmark repository's data, like the chart, into marked regions
  that name the benchmark commit; the lint workflow checks them against
  it, so they can't drift (``make tables``). The hand-typed figures were
  off by one in the last digit in a few cells; the fast-mode page's
  speed-ups are now the published benchmark's, in their own table
- ``Image.fast_mode()`` sets the mode of the thread it runs in, and
  restores that thread's previous setting: before, it set the process-wide
  flag, so two threads' blocks could restore it in the wrong order.
  ``set_fast()`` and the environment variable still set the process-wide
  default, which threads outside a block use; ``get_fast()`` returns the
  calling thread's effective setting. ``resize()`` passes the setting it
  read for its ``reducing_gap`` default to the C resize, so a change from
  another thread meanwhile can't mix the exact and fast kernels
- The test suite fails if an entry of ``EXACT_KERNEL_TESTS``
  (``Tests/conftest.py``, the upstream tests that need the exact kernels)
  names no test of a collected file, so a test renamed upstream can't run
  with the fast kernels unnoticed
- The portable RGBA -> RGBa loop (the tail of the SIMD builds, the whole
  conversion of the portable one) indexed its rows with ``x * 4``, which GCC
  can't vectorize under ``-fno-strict-overflow``: the portable build was
  0.32x Pillow 13 on arm64. Pointer increments, as in Pillow; same pixels
- The SIMD code of every operation now lives in a companion file,
  ``<Area>SIMD.c`` next to Pillow's own file, and the companions are
  compiled into one translation unit, ``SIMDKernels.c``, whose only
  export is a table of kernel pointers; Pillow's files keep the portable
  code and call the kernels through the table of the tier chosen when
  ``_imaging`` loads (``SIMDDispatch.c``). No file of Pillow's own carries
  an instruction-set guard any more, so merging a Pillow release no
  longer conflicts on them, and one build can later hold several tiers
  (``design/SIMD_DISPATCH_PLAN.md``, phase A). The same kernels, the same
  pixels: the hashes of every area are unchanged on the SSE4, AVX2 and
  scalar builds. ``_imaging.simd_kernels`` names the active tier
- Run-time dispatch on x86-64: ``-C simd=dispatch``, now what ``auto``
  means there, compiles the kernels three times, for SSE4.1, AVX2 and
  AVX-VNNI (SSE4.1 and AVX2 with MSVC), and the rest of ``_imaging`` for
  plain x86-64; at import, ``cpuid`` picks the highest tier the CPU and
  OS support, so one build runs on every x86-64 CPU, which is what the
  x86 wheels need (``design/SIMD_DISPATCH_PLAN.md``, phase B). The
  ``PILLOW_ALLSIMD_ISA`` environment variable (``scalar``, ``sse4``,
  ``avx2``, ``avxvnni``) caps the tier, ``Image.core.set_simd(name)``
  changes it, ``Image.core.simd_tiers`` lists the tiers built,
  ``features.check("simd")`` is true when a SIMD tier is built in, and
  ``python3 -m PIL`` prints ``--- SIMD kernels: avx2 (built: scalar, sse4,
  avx2, avxvnni)``. A build for one instruction set now fails to import
  with a message on a CPU without it, instead of an illegal instruction.
  The tiers of one build give the same pixels (``Tests/test_simd_tiers.py``),
  and each tier runs within noise of the single-tier build of the same
  instruction set
- The AVX2 build's 3x3 float kernel filters on 4-byte images (a
  ``Kernel`` whose coefficients don't fit 16 bits) use the SSE4 kernel:
  the AVX2 one was 1.6 times slower and rounded 0.2% of the bytes
  differently, so an AVX2 build's pixels now equal an SSE4 build's for
  every operation
- Documentation: "How the kernels work" (``docs/allsimd-kernels.rst``),
  the explanation of the SIMD kernels for contributors planned in
  ``design/SIMD_EXPLAINER_PLAN.md``: the ground rules, then each
  operation (what Pillow computes, the idea, the x86 and arm64 kernels,
  why the builds agree and the exceptions, the numbers, a code map),
  what was tried and dropped, where OpenCV and libvips are still faster,
  the fast-mode kernels, and the overall numbers, a file map and the
  tools

13.0.0.post2 (2026-11-03)
-------------------------

Based on Pillow 13.0.0; the code is that of 13.0.0.post0

- Wheels for arm64 Linux (manylinux_2_28 and musllinux_1_2 aarch64), for
  CPython 3.11 to 3.15 and the free-threaded 3.14t and 3.15t. They have the
  NEON kernels and bundle the image libraries as Pillow's wheels do, each
  carries a CycloneDX SBOM of them (PEP 770; ``.github/allsimd_sbom.py``,
  Pillow's SBOM renamed for this package), and each is tested with Pillow's
  wheel tests on its platform. Wheels for macOS 11 or later on Apple
  silicon are built the same way and join this release when GitHub's macOS
  arm64 runners are available (they weren't on the release day). x86 and
  the other architectures install the source package, as before (``-C
  simd=avx2`` on x86)
- The release workflow builds the platforms its tag names
  (``allsimd-<version>``, ``-linux`` or ``-macos``), so one platform's
  runners can't block the others; the test workflows run once per push
  (``simd/master`` mirrors ``simd/pillow-13``)
- ``design/REVIEW_2026-11.md``: a review of the fork's code, documentation
  and process, with the next work

13.0.0.post1 (2026-11-03)
-------------------------

Based on Pillow 13.0.0; the code is that of 13.0.0.post0

- The PyPI description shows the benchmarks: a chart of speed-ups over
  Pillow 13.0.0 (``docs/resources/allsimd-speedups.svg``, drawn by
  ``checks/simd_speedups_svg.py`` from the benchmark results) and a table of
  selected operations on x86-64 and arm64. It also lists the ``simd`` build
  options and how to choose exact mode

13.0.0.post0 (2026-11-03)
-------------------------

Based on Pillow 13.0.0

- First release on PyPI, as a source package (``pip install -C simd=avx2
  pillow-allsimd``), uploaded by the release workflow with trusted
  publishing

- The project is Pillow-AllSIMD (``pillow-allsimd``), a fork of Pillow-SIMD:
  the distribution name, the documentation and the credits say so; the fast
  mode variable is ``PILLOW_ALLSIMD_FAST`` (``PILLOW_SIMD_FAST``, its former
  name, still works)
- Fast mode is on by default: the operations with a fast kernel (Gaussian
  blur, RGB <-> YCbCr, RGB -> HSV, transforms and rotate(), blend(), large
  downscales) give results within a few documented levels of Pillow's.
  ``PILLOW_ALLSIMD_FAST=0`` or ``Image.set_fast(False)`` selects the exact
  kernels, as before. Large downscales take the reduce() shortcut on 8-bit
  images only (32-bit sums overflow on I and F images), and leave a
  malformed box to resize()'s own error

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
- Perspective transforms: the coordinates of a tile row are computed in
  one loop the compiler vectorizes, so the two divisions per pixel run
  several at once (1.4x RGB bilinear and bicubic, 1.8x L nearest, AVX2);
  quad transforms with the bilinear filter inlined (1.2x). Same output
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
- Fast mode switch: ``PILLOW_ALLSIMD_FAST=1``, ``Image.set_fast()``,
  ``Image.get_fast()``, ``Image.fast_mode()`` and the ``fast`` feature.
  Off by default; it lets operations use kernels whose results may differ
  from Pillow's by small, documented amounts (``design/SIMD_FAST_KERNELS.md``)
- Fast mode: GaussianBlur and UnsharpMask up to radius 6 apply Pillow's three
  box passes as one kernel with 8-bit weights in one pass, with exact bands
  near the edges: within 3 levels of the exact result, closer to a true
  Gaussian, 3.3x (x86) and 3.2x (arm64) faster at radius 1
- Fast mode: RGB, RGBA, RGBX <-> YCbCr with JPEG's formulas at 14-bit
  coefficients and one rounding: within 1 level of the exact result and
  closer to the formulas (0.3% of values off by one, against 47%); to YCbCr
  1.5x (x86) and 2.2x (arm64), back 1.3x and 1.7x
- Fast mode: RGB, RGBA, RGBX -> HSV with the hue and saturation truncated
  exactly (one 32-bit float division each): the hue is 1 above the exact
  result for 0.3% of colours, saturation and value are the same; 2.9x (x86),
  2.0x (arm64)
- Fast mode: affine, perspective and quad transforms and rotate() with the
  bilinear and bicubic filters (L and 4-byte modes) take positions in fixed
  point and integer filters with one rounding: within 2 levels of the exact
  result, 0.6% of values off by one against float interpolation (exact:
  41%); rotate bilinear 2.7x (x86) and 2.6x (arm64), bicubic 2.9x and 3.4x,
  perspective 2.1x and 1.9x, L rotate bilinear 2.8x and 1.9x (vertical
  first in 16-bit lanes, four pixels per AVX2 step, eight L pixels per
  step)
- Fast mode: blend() with 0 < alpha < 1 interpolates in integers, alpha in
  2^-15 units, rounded (pmulhrsw, vqrdmulh): within 1 level of the exact
  result, 1.7x (x86) and 4.1x (arm64)
- Fast mode: resize() with the bicubic, Lanczos and MKS filters and no
  reducing_gap uses 3.0, so downscales of 6x and more shrink with reduce()
  first: within 2 levels at photo sizes, 1/8 Lanczos 1.9x (x86) and 2.5x
  (arm64)
- BoxBlur with one pass each way and radii the direct kernel takes
  (integers up to 4, below 1): each row is blurred horizontally into a ring
  of rows just before the vertical pass reads it, instead of through a full
  temporary image; the same pixels (radius 1 2.1x, 2 1.7x, 3 1.5x, AVX2)
- arm64: point() tables on 8-bit bands with NEON lookups (tbl, tbx) in the
  256-entry tables, a row's bands as planes so each table loads once (L
  1.9x, RGB 1.3x, both now level with or ahead of OpenCV)
- Fast mode: perspective positions in 32-bit floats from the exact integer
  sums (images below 32768 pixels a side), four lanes per division on SSE4
  and NEON, eight on AVX2; still the same on every build (arm64 1.2x)
- WebP: one-frame files decode straight into the image with libwebp's
  simple API, without the animation decoder's canvas and two copies (the
  same pixels; 1.38x, 2560x1600 RGB, now ahead of OpenCV)
- JPEG: RGB images decode to 4-byte pixels in libjpeg-turbo's colour
  conversion (JCS_EXT_RGBX) instead of being expanded row by row (1.05x);
  PNG: the "up" filter undone 16 bytes at a time (1.1x on such rows)
- ImageDraw polygon outlines wider than 1 px keep the polygon's inside as
  spans per row, from the same scan as its fill, instead of a mask of its
  bounding box: the same pixels, 1.1x for the benchmark's ROI polygons and
  1.7x for polygons 1400 px across
- arm64: RGBa -> RGBA (and so RGBA resize and transforms) divides once per
  pixel instead of once per channel, with the same result for every colour
  and alpha (1.12x)
- Pixel conversions written so that they vectorize under Python's
  ``-fno-strict-overflow`` (RGBA -> RGB 1.9x, RGB -> RGBA 1.9x, L -> RGB
  2.1x with AVX2); NEON loops for L -> RGB, RGB <-> RGBA and CMYK
- ImageChops ``add`` and ``subtract`` with scale 1 and offset 0 as saturating
  vector additions (AVX2 1.8x, NEON 2.4x)
- Rank filters: the 3x3 median in registers also at -O2 (Debian, Ubuntu and
  conda Pythons: 1.75x), minimum and maximum in one pass (1.4x AVX2, 1.3x
  NEON)
- Box and Gaussian blur: radii below 2 and integer radii up to 4 without
  running sums, in SSE4, AVX2 and NEON (Gaussian RGB radius 1 1.7x on
  both; box radius 2 1.2x AVX2, 1.6x NEON; unsharp mask 1.2x AVX2, 1.5x
  NEON)
- RGB <-> YCbCr: AVX2, and 16 pixels per step on NEON (AVX2 2.0x and 1.9x);
  RGB -> HSV on NEON (1.4x)
- Kernel filters: kernels whose coefficients don't fit 16 bits at 13
  fractional bits (FIND_EDGES, CONTOUR, EDGE_ENHANCE) use the integer
  kernels at the highest precision that keeps them exact (2.2x AVX2, 2.0x
  NEON), and EDGE_ENHANCE and other kernels with half-integer coefficients
  now give the scalar build's result on x86; 3x3 kernels for L images keep
  their coefficients in registers (1.1x AVX2)
- Text: glyph advances cached for the basic layout (3.3x x86, 4.2x arm64);
  polygons: scan-line intersections sorted without qsort (1.25x)
- ImageDraw: thin lines and rectangle outlines write their pixels directly
  again, as before Pillow 13.0, instead of through a function pointer per
  pixel (500 bounding boxes 2 px wide: 1.6x)
- ImageDraw.polygon() with an outline wider than 1 px: the mask that keeps
  the outline inside the polygon covers only the polygon, not the whole
  image (100 ROI polygons on 2560x1600: 6.6x). Same pixels
- arm64: rotate() and the affine, perspective and quad transforms with the
  bilinear and bicubic filters on RGB and RGBA images in NEON doubles: the
  scalar filters' integer differences for the 4 bands of two rows in
  16-bit lanes, converted exactly, then the same fused multiply-adds; two
  pixels at a time where their source pixels are inside the image. The
  same pixels; rotate bicubic RGBA 1.59x, RGB 1.32x, bilinear RGB 1.47x,
  RGBA 1.54x, perspective 1.25x to 1.30x, quad 1.37x (Cortex-A78AE)
- arm64: alpha_composite() looks the scalar code's per-pixel division up in
  a table of every source and destination alpha (128 KB, built on first
  use), with NEON for the rest: the same pixels, 1.25x; reduce() by 4x4 of
  4-byte images in NEON pairwise sums (1.15x)
- resize() of RGBA and LA images premultiplies the source rows inside the
  first resample pass and removes it from the rows of the last, where they
  are in the cache, instead of converting the whole image to RGBa and back.
  The same pixels; x86 1.1x to 1.7x (2048x1280 bilinear 1.68x), arm64
  1.1x to 1.3x
- arm64: RGBa -> RGBA (and so RGBA resizes) looks up 255 * c / alpha in a
  64 KB table of the scalar code's results, two pixels per 64-bit word:
  1.7x on Cortex-A78AE, where the float kernel was bound by conversions
  (on x86 the table is slower than the vector kernels, which stay)
- arm64, fast mode: RGB -> HSV computes the hue in 16-bit lanes with one
  division and takes the saturation from that table (1.67x); YCbCr -> RGB
  multiplies Cr and Cb in 16-bit lanes with rounding doubling multiplies
  (1.53x). The same pixels as before; both now faster than OpenCV
- MinFilter and MaxFilter of the same size: the row ends clamp the pixel
  index instead of copying each row into a padded buffer, and the 3x3
  filters skip the loop over the sources (3x3 RGB: 1.1x to 1.2x AVX2,
  1.4x NEON)
- Box blur radii 1 to 4 without running sums: the sum's sources unrolled
  per radius (radius 2 RGB: 1.46x AVX2 at -O3)
- 32-bit x86: the SIMD builds do floating point in SSE registers
  (``-mfpmath=sse``); on the x87 unit a transform's pixels could depend on
  the code around them. ``-C simd=native`` passes ``-mcpu=native`` on POWER
  and is refused on RISC-V with a hint. The portable build passes the test
  suite on riscv64, ppc64le, s390x, armv7 and 32-bit x86
  (``design/OTHER_ARCHITECTURES.md``); ``checks/simd_test_remote.sh`` takes those
  platforms, and ``checks/simd_hashes.py`` hashes I and F pixels
  little-endian, so big-endian builds compare
- Fast mode: resize() of 8-bit images with 4-byte pixels fuses the two
  passes through a ring of rows, keeps the horizontal sums to 2^-6 levels
  and multiplies in 16 bits in the vertical pass (pmulhrsw, vqrdmulh): within
  1 level of the exact result and closer to a float resize; upscales 1.16x
  to 1.28x (x86) and 1.31x to 1.47x (arm64), downscales to 1.20x on arm64
  and within 4% on x86. The same pixels on every build

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

Pillow-SIMD
===========

The changelog of Pillow-SIMD (https://github.com/uploadcare/pillow-simd), by
Alexander Karpinsky and contributors, up to the fork.

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
