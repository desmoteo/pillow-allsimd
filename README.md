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

- Resize (convolution-based resampling), including L images: SSE4, AVX2, AVX-VNNI, NEON
- Reduce: SSE4 (any factor), NEON (horizontal factors up to 512); 2x2 and 4x4 for L images on both
- Gaussian and box blur: SSE4 (L images four rows at a time); portable C on arm64
- Unsharp mask: SSE4; portable C on arm64
- Alpha composition: SSE4, AVX2; portable C on arm64
- RGBA → RGBa (alpha premultiplication): SSE4, AVX2, NEON
- RGBa → RGBA (division by alpha): SSE4, AVX2, NEON
- RGB → L, LA and I, RGBA → LA (grayscale): SSE4, NEON
- 3x3 and 5x5 kernel filters with integer kernels (coefficients that fit 16 bits):
  SSE4, AVX2, AVX-VNNI, NEON on 4-byte images; SSE4 (AVX-VNNI when built
  with it), NEON on L images
- 3x3 and 5x5 kernel filters with float kernels (large coefficients): SSE4,
  AVX2 (3x3 only), NEON on 4-byte images; SSE4 on L images
- 3D color lookup tables: SSE4, AVX2, NEON
- `ImageChops.multiply` and `screen`: SSE4, AVX2, NEON
- RGB pack and unpack (`frombytes`, `tobytes`, codecs): SSE4, NEON
- Split and get_channel: SSE4, NEON; merge: NEON

On arm64, blur, unsharp mask and alpha composition use the portable C code
because on a Cortex-A78AE it was faster than hand-written NEON.
Rank filters (min, max, median), the mode filter, histograms and
`ImageChops.offset` have faster portable C code on every architecture.
Other architectures build the portable C code.


## Benchmarks

Tons of tests can be found on the [Pillow Performance][pillow-perf-page] page.
There are benchmarks against different versions of Pillow and Pillow-SIMD
as well as ImageMagick, Skia, OpenCV and IPP.

The results show that for resizing Pillow is always faster than ImageMagick,
Pillow-SIMD, in turn, is even faster than the original Pillow by the factor of 4-6.
In general, Pillow-SIMD with AVX2 is always **16 to 40 times faster** than
ImageMagick and outperforms Skia, the high-speed graphics library used in Chromium.


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
