# Pillow-SIMD

Pillow-SIMD is highly optimized version of [Pillow library][original-docs]
for x86 architecture (mainly Intel and AMD CPUs).

Pillow-SIMD is "following" Pillow which means it is
drop-in replacements for Pillow of the same version.
For example, `Pillow-SIMD 3.2.0.post3` is a drop-in replacement for
`Pillow 3.2.0`, and  `Pillow-SIMD 3.3.3.post0` — for `Pillow 3.3.3`.

For more information on the original Pillow, please refer to:
[read the documentation][original-docs],
[check the changelog][original-changelog] and
[find out how to contribute][original-contribute].


## Why SIMD

There are multiple ways to tweak image processing performance.
To name a few, such ways can be: utilizing better algorithms, optimizing existing implementations,
using more processing power and/or resources.
One of the great examples of using a more efficient algorithm is [replacing][gaussian-blur-changes]
a convolution-based Gaussian blur with a sequential-box one.

Such examples are rather rare, though. It is also known, that certain processes might be optimized
by using parallel processing to run the respective routines.
But a more practical key to optimizations might be making things work faster
using the resources at hand. For instance, SIMD computing might be the case.

SIMD stands for "single instruction, multiple data" and its essence is
in performing the same operation on multiple data points simultaneously
by using multiple processing elements.
Common CPU SIMD instruction sets are MMX, SSE-SSE4, AVX, AVX2, AVX512, NEON.

Currently, Pillow-SIMD can be [compiled](#installation) with SSE4 (default on x86), AVX2, AVX-VNNI or NEON (default on arm64) support, or for the build machine's own CPU.


## Status

Pillow-SIMD project is production-ready.
The project is supported by Uploadcare, a SAAS for cloud-based image storing and processing.

[![Uploadcare][uploadcare.logo]][uploadcare.com]

In fact, Uploadcare has been running Pillow-SIMD since 2015.

The following image operations are currently SIMD-accelerated:

- Resize (convolution-based resampling), including L images: SSE4, AVX2, AVX-VNNI, NEON;
  F, I and I;16 images in vectors of doubles: SSE4, AVX2, NEON
- Reduce: SSE4 (any factor), NEON (horizontal factors up to 512); 2x2 and 4x4 for L images on both
- Gaussian and box blur: running sums in SSE4, AVX2 (two rows at a time) and
  NEON for the horizontal pass, SSE4 and AVX2 for the vertical pass (L images
  four rows at a time on SSE4, eight on AVX2); radii below 2 and integer radii
  up to 4 without running sums: SSE4, AVX2, NEON. The vertical passes of a
  Gaussian blur run in one sweep down the image
- Unsharp mask: SSE4, NEON
- Alpha composition: SSE4, AVX2
- RGBA → RGBa (alpha premultiplication): SSE4, AVX2, NEON
- RGBa → RGBA (division by alpha): SSE4, AVX2, NEON
- RGB → L, LA and I, RGBA → LA (grayscale): SSE4, NEON
- L → RGB, RGB ↔ RGBA, RGB → CMYK: NEON (portable C that compilers vectorize
  elsewhere)
- RGB → HSV: SSE4, AVX2, NEON
- RGB ↔ YCbCr: SSE4, AVX2, NEON
- 3x3 and 5x5 kernel filters with integer kernels (exact in 16-bit fixed point,
  at the highest precision that keeps them exact): SSE4, AVX2, AVX-VNNI, NEON
- 3x3 and 5x5 kernel filters with float kernels (large coefficients): SSE4,
  AVX2 (3x3 only), NEON on 4-byte images; SSE4, AVX2 (3x3 only) on L images
- Rotate and transforms with bilinear and bicubic filters, 3- and 4-band
  images: SSE4, AVX2; every architecture transforms in cache-friendly tiles
- Transposes, 90 and 270 degree rotations, flips and 180 degree rotation:
  SSE4, NEON
- 3D color lookup tables: SSE4, AVX2, NEON
- `ImageChops.multiply` and `screen`, and `add` and `subtract` with scale 1
  and offset 0: SSE4, AVX2, NEON
- Rank filters on 1- and 4-byte images: median 3x3 and 5x5 in registers, min
  and max of any size in one pass: SSE4, AVX2, NEON; other ranks up to 11x11
  through sorting networks in portable C that compilers vectorize
- `point()` tables that are byte operations (invert, posterize, solarize):
  SSE4, AVX2, NEON
- Palette mapping (quantize, convert to an adaptive palette), nearest-entry
  search: SSE4, AVX2, NEON
- ImageDraw span fill and RGBA ink blend: SSE4, AVX2, NEON; masked fills for
  text and `bitmap()`: SSE4, NEON
- PNG encoding row filters: SSE4, NEON
- RGB pack and unpack (`frombytes`, `tobytes`, codecs): SSE4, NEON
- Split and get_channel: SSE4, NEON; merge: NEON

On arm64, alpha composition and the vertical running-sum blur pass use the
portable C code, which GCC vectorizes better than hand-written NEON on a
Cortex-A78AE.
For the horizontal blur of 4-byte images and the unsharp mask row, NEON is
faster with GCC 12 and 14 (not with GCC 11).
The mode filter, histograms, `ImageChops.offset`, floodfill, polygon fills
and text rendering (per-font caches of glyphs and advances) have faster
portable C code on every architecture.
Other architectures build the portable C code.


## Design documents

Notes on the SIMD work, kept with the code they describe:

- [X86_SIMD_ASSESSMENT.md](X86_SIMD_ASSESSMENT.md),
  [ARM64_SIMD_ASSESSMENT.md](ARM64_SIMD_ASSESSMENT.md) and
  [ACCELERATION_OPTIONS_ASSESSMENT.md](ACCELERATION_OPTIONS_ASSESSMENT.md):
  the starting assessments of the x86 kernels, the arm64 port and other
  acceleration options.
- [UPSTREAM_13_REVIEW.md](UPSTREAM_13_REVIEW.md): Pillow 12.3 and 13.0 as a
  source of fixes and optimizations.
- [CALLGRIND_PROFILE.md](CALLGRIND_PROFILE.md),
  [PROFILE_ASSESSMENT.md](PROFILE_ASSESSMENT.md) and
  [PROFILE_TABLES.md](PROFILE_TABLES.md): where the time goes, from
  instruction counts, cache misses and timings over the whole API.
- [PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md): where OpenCV and libvips are
  still faster, and why.
- [OTHER_LIBRARIES_KERNELS.md](OTHER_LIBRARIES_KERNELS.md): how OpenCV,
  KleidiCV, Carotene and libvips implement those kernels, approximate ones
  included, measured for speed and accuracy.
- [APPROXIMATE_KERNELS_PLAN.md](APPROXIMATE_KERNELS_PLAN.md): a proposed
  opt-in fast mode with approximate kernels beside the exact ones, chosen at
  run time.
- [SIMD_EXPLAINER_PLAN.md](SIMD_EXPLAINER_PLAN.md): the outline of a detailed
  explanation of every optimization on both architectures.


## Benchmarks

Tons of tests can be found on the [Pillow Performance][pillow-perf-page] page.
There are benchmarks against different versions of Pillow and Pillow-SIMD
as well as ImageMagick, Skia, OpenCV and IPP.

The results show that for resizing Pillow is always faster than ImageMagick,
Pillow-SIMD, in turn, is even faster than the original Pillow by the factor of 4-6.
In general, Pillow-SIMD with AVX2 is always **16 to 40 times faster** than
ImageMagick and outperforms Skia, the high-speed graphics library used in Chromium.

Current results for this version, on x86-64 (SSE4, AVX2, AVX-VNNI) and arm64
(NEON), against Pillow 13.0, the last Pillow-SIMD release, OpenCV, libvips
and ImageMagick, are at [pillow-simd-perf][pillow-simd-perf-page]. They cover
111 operations, each library single-threaded, and the benchmark runs in Docker
on any machine.

[pillow-simd-perf-page]: https://desmoteo.github.io/pillow-simd-perf/


## Why Pillow itself is so fast

No cheats involved. We've used identical high-quality resize and blur methods for the benchmark.
Outcomes produced by different libraries are in almost pixel-perfect agreement.
The difference in measured rates is only provided with the performance of every involved algorithm.


## Why Pillow-SIMD is even faster

Because of the SIMD computing, of course. But there's more to it:
heavy loops unrolling, specific instructions, which aren't available for scalar data types.


## Why do not contribute SIMD to the original Pillow

Well, it's not that simple. First of all, the original Pillow supports
a large number of architectures, not just x86.
But even for x86 platforms, Pillow is often distributed via precompiled binaries.
In order for us to integrate SIMD into the precompiled binaries
we'd need to execute runtime CPU capabilities checks.
To compile the code this way we need to pass the `-mavx2` option to the compiler.
But with the option included, a compiler will inject AVX instructions even
for SSE functions (i.e. interchange them) since every SSE instruction has its AVX equivalent.
So there is no easy way to compile such library, especially with setuptools.


## Installation

If there's a copy of the original Pillow installed, it has to be removed first
with `$ pip uninstall -y pillow`.
Please install [prerequisites](https://pillow.readthedocs.io/en/stable/installation.html#building-from-source) for your platform.
The installation itself is simple just as running `$ pip install pillow-simd`,
and if you're using SSE4-capable CPU everything should run smoothly.
If you'd like to install the AVX2-enabled version,
pass the `simd` build option:

```bash
$ pip uninstall pillow
$ pip install -U --force-reinstall --no-binary pillow-simd -C simd=avx2 pillow-simd
```

The `simd` option accepts:

- `auto` (default): SSE4.1 on x86, NEON on arm64, portable C elsewhere.
- `sse4`: SSE4.1 (x86 only).
- `avx2`: SSE4.1 and AVX2 (x86 only).
- `avxvnni`: SSE4.1, AVX2 and AVX-VNNI (x86 only: Intel Alder Lake,
  Sapphire Rapids and newer, AMD Zen 5). Faster integer filters and resize.
- `native`: everything the build machine's CPU supports (`-march=native`),
  including AVX-512 in compiler-generated code. Only runs on similar CPUs.
- `native512`: like `native`, with 512-bit vectors preferred (x86 only).
- `neon`: NEON (arm64 only).
- `disable`: portable C code only, on any architecture.

Defining the `CC` variable (`CC="cc -mavx2"`) also works.
Don't use `CFLAGS` for this: setting it replaces Python's default
compiler flags, including the optimization level.


## Contributing to Pillow-SIMD

Please be aware that Pillow-SIMD and Pillow are two separate projects.
Please submit bugs and improvements not related to SIMD to the [original Pillow][original-issues].
All bugfixes to the original Pillow will then be transferred to the next Pillow-SIMD version automatically.


  [original-homepage]: https://python-pillow.org/
  [original-docs]: https://pillow.readthedocs.io/
  [original-issues]: https://github.com/python-pillow/Pillow/issues/new
  [original-changelog]: https://github.com/python-pillow/Pillow/blob/master/CHANGES.rst
  [original-contribute]: https://github.com/python-pillow/Pillow/blob/master/.github/CONTRIBUTING.md
  [gaussian-blur-changes]: https://pillow.readthedocs.io/en/stable/releasenotes/2.7.0.html#gaussian-blur-and-unsharp-mask
  [pillow-perf-page]: https://python-pillow.github.io/pillow-perf/
  [pillow-perf-repo]: https://github.com/python-pillow/pillow-perf
  [uploadcare.com]: https://uploadcare.com/?utm_source=github&utm_medium=description&utm_campaign=pillow-simd
  [uploadcare.logo]: https://ucarecdn.com/8eca784b-bbe5-4f7e-8cdf-98d75aab8cec/logotransparent.svg
