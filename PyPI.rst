Pillow-AllSIMD
==============

**Pillow, 6 times as fast on x86-64 and 3 times as fast on arm64.**

Pillow-AllSIMD is a drop-in replacement for `Pillow <https://python-pillow.org>`_
13.0.0, the friendly PIL fork: the same ``PIL`` package and the same API, with
SIMD kernels for x86-64 (SSE4, AVX2, AVX-VNNI) and arm64 (NEON). It is a fork
of `Pillow-SIMD <https://github.com/uploadcare/pillow-simd>`_, which brought
SIMD to Pillow on x86.

.. image:: https://raw.githubusercontent.com/desmoteo/pillow-allsimd/allsimd-13.0.0.post2/docs/resources/allsimd-speedups.svg
   :alt: Bar chart: Pillow-AllSIMD is 6.1 times as fast as Pillow 13.0.0 on
         x86-64 with AVX2 and 3.1 times on arm64 with NEON, as the geometric
         mean of 111 operations; resizes 3.6 to 15.8 times, Gaussian blur at
         radius 1 9.7 to 21.6 times, text drawing 17.5 to 19.7 times.
   :target: https://desmoteo.github.io/pillow-allsimd-perf/

- **Fast**: as the geometric mean of 111 operations on one thread, 6.1 times
  as fast as Pillow 13.0.0 on x86-64 with AVX2 and 3.1 times on arm64 with
  NEON. That is 2.9 times as fast as Pillow-SIMD 12.1.1 with AVX2, Pillow-SIMD's
  last release. Of the 104 operations that OpenCV 5.0 also has, Pillow-AllSIMD
  is faster in 78 on x86-64 and in 68 on arm64.
- **Drop-in**: your code doesn't change. Pillow's test suite passes on every
  build.
- **Exact when you ask**: in fast mode (the default) a few operations give
  results that differ from Pillow's by a few levels at most, the same on every
  build and CPU. Exact mode follows Pillow's arithmetic and is still 5.4
  times as fast on x86-64 and 2.5 times on arm64.
- **arm64 too**: NEON kernels for resizes, blurs, filters, conversions,
  transforms and more, tested on an NVIDIA Jetson Orin, in wheels for Linux
  and macOS. Other CPUs build portable C that compilers vectorize.

Benchmarks
----------

Selected operations, one thread, a 2560×1600 RGB image. Each cell shows
Pillow 13.0.0's time, then Pillow-AllSIMD's, then the speed-up:

=====================================  =========================  ========================
Operation                              x86-64, AVX2               arm64, NEON
=====================================  =========================  ========================
Resize to 320×200, bicubic             22.3 → 1.41 ms, **15.8×**  30.2 → 5.02 ms, **6.0×**
Resize RGBA to 2048×1280, Lanczos      119 → 9.95 ms, **12.0×**   145 → 40.7 ms, **3.6×**
Enlarge to 5478×3424, bicubic          226 → 27.7 ms, **8.2×**    260 → 87.3 ms, **3.0×**
Gaussian blur, radius 1                121 → 5.59 ms, **21.6×**   162 → 16.7 ms, **9.7×**
Gaussian blur, radius 10               121 → 25.9 ms, **4.7×**    163 → 91.2 ms, **1.8×**
Unsharp mask                           197 → 15.8 ms, **12.5×**   236 → 40.1 ms, **5.9×**
SHARPEN filter                         52.9 → 4.21 ms, **12.6×**  99.0 → 27.1 ms, **3.7×**
Median filter, 3×3                     1183 → 2.79 ms, **425×**   1262 → 8.20 ms, **154×**
RGB to HSV                             59.7 → 2.99 ms, **20.0×**  81.4 → 10.6 ms, **7.7×**
Unpremultiply alpha                    26.6 → 2.20 ms, **12.1×**  38.8 → 7.21 ms, **5.4×**
Rotate 30°, bicubic                    185 → 29.3 ms, **6.3×**    291 → 79.3 ms, **3.7×**
Alpha composite                        19.6 → 2.90 ms, **6.8×**   22.1 → 18.0 ms, **1.2×**
Draw 100 lines of text                 56.5 → 3.23 ms, **17.5×**  113 → 5.72 ms, **19.7×**
Decode JPEG                            16.2 → 15.5 ms, **1.0×**   28.3 → 28.4 ms, **1.0×**
Encode PNG                             789 → 509 ms, **1.6×**     1015 → 711 ms, **1.4×**
JPEG load, rotate, resize, blur, save  86.1 → 28.6 ms, **3.0×**   112 → 66.4 ms, **1.7×**
=====================================  =========================  ========================

Codecs gain little: decoding and encoding are mostly libjpeg's, libpng's and
libwebp's own work. The machines are an Intel Core Ultra 9 285H laptop (with
turbo off) and an NVIDIA Jetson Orin (Cortex-A78AE). Each time is the best of
3 interleaved rounds, in Docker with Python 3.12 and GCC 14, measured on
2026-10-31 on the same x86-64 and arm64 code as this release. The
`benchmark page <https://desmoteo.github.io/pillow-allsimd-perf/>`_ has all
111 operations, in exact and fast mode. It also compares Pillow-SIMD, OpenCV,
libvips and ImageMagick, with notes where another library's operation
differs.

Installing
----------

On arm64 Linux (glibc 2.28 or later, or musl) and macOS 11 or later, pip
installs a wheel with the NEON kernels and the image libraries bundled, as
Pillow's wheels have them. Elsewhere, x86 included, Pillow-AllSIMD is a
source package, so pip compiles it on installation: install the libraries
that `building Pillow from source
<https://desmoteo.github.io/pillow-allsimd/installation/building-from-source.html>`_
needs first. It installs the ``PIL`` package, as Pillow does. Install your
other packages first, then replace Pillow with it::

    python3 -m pip uninstall -y pillow pillow-simd
    python3 -m pip install pillow-allsimd               # arm64: a wheel
    python3 -m pip install -C simd=avx2 pillow-allsimd  # x86: built with AVX2

Uninstall Pillow before installing, not after: both own the files in
``PIL``. The ``-C simd=`` options below apply to source builds only; to build
from source on arm64 anyway, add ``--no-binary pillow-allsimd``. To rebuild
with another ``simd`` option, add ``--no-cache-dir --force-reinstall``.
Otherwise pip reuses the wheel it built the first time. With ``-v``, pip shows
the line ``--- SIMD kernels: ...`` from the build, which names the kernels you
got.

====================  ==============================================================
``-C simd=``          Kernels
====================  ==============================================================
``auto`` (default)    SSE4.1 on x86, NEON on arm64, portable C elsewhere
``sse4``              SSE4.1
``avx2``              SSE4.1 and AVX2: x86 CPUs since Intel Haswell and AMD
                      Excavator, except some Pentium, Celeron and Atom models
``avxvnni``           SSE4.1, AVX2 and AVX-VNNI, for faster integer filters and
                      resize: Intel Alder Lake, Sapphire Rapids and newer, AMD Zen 5
``native``            Everything the build machine's CPU supports
                      (``-march=native``); the build runs only on similar CPUs
``native512``         Like ``native``, preferring 512-bit vectors (x86)
``neon``              NEON (arm64)
``disable``           Portable C only
====================  ==============================================================

Exact and fast modes
--------------------

Fast mode is on by default. The Gaussian blur, RGB ↔ YCbCr, RGB → HSV, the
affine, perspective and quad transforms, ``rotate()``, ``blend()``, the resize
of RGB, RGBA and other 4-byte images and large downscales then use faster
kernels. Their results differ from Pillow's by a few levels at most and are
the same on every build and architecture. Exact mode follows Pillow's
arithmetic for every operation. Use it to compare pixels with images made by
Pillow::

    PILLOW_ALLSIMD_FAST=0 python3 app.py

Or, in code:

.. code-block:: python

    from PIL import Image, ImageFilter

    Image.set_fast(False)  # every operation from now on

    with Image.fast_mode(False):  # one block
        blurred = im.filter(ImageFilter.GaussianBlur(2))

`Fast mode <https://desmoteo.github.io/pillow-allsimd/reference/fast_mode.html>`_
in the documentation lists each operation and how far its results may
differ.

Testing
-------

Pillow's test suite runs on every build, and in every wheel on its own
platform. Consistency tests check each SIMD kernel against the scalar code,
and per-kernel output hashes compare the builds with each other. The
hardware tests run on x86-64 (SSE4, AVX2, AVX-VNNI) and on arm64 (NEON). The
portable build also passes the suite on riscv64, ppc64le, s390x, armv7 and
32-bit x86, under QEMU. Status: beta.

Links
-----

- `Repository and readme <https://github.com/desmoteo/pillow-allsimd>`_
- `Documentation <https://desmoteo.github.io/pillow-allsimd/>`_
- `Benchmarks <https://desmoteo.github.io/pillow-allsimd-perf/>`_
- `Changelog <https://github.com/desmoteo/pillow-allsimd/blob/simd/pillow-13/CHANGES.SIMD.rst>`_
- `Issues <https://github.com/desmoteo/pillow-allsimd/issues>`_: SIMD kernels,
  builds and fast mode. Report other bugs to
  `Pillow <https://github.com/python-pillow/Pillow/issues>`_.

Pillow is by Jeffrey 'Alex' Clark and contributors, on PIL by Fredrik Lundh
and Secret Labs AB. Pillow-SIMD is by Alexander Karpinsky and contributors,
supported by Uploadcare. All are licensed under the MIT-CMU License, and so
is Pillow-AllSIMD.
