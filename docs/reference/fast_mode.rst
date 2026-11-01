.. _fast-mode:

Fast mode (Pillow-AllSIMD)
==========================

Fast mode lets operations use kernels that are faster and whose results may
differ from Pillow's by small, documented amounts. It is on by default: the
operations listed below use their fast kernels. The other operations, and
every operation in exact mode (fast mode off), compute with Pillow's
arithmetic, or an integer form of it that rounds differently by one level at
most in a few places (resize with 16-bit coefficients, RGB to L with 15-bit
ones). Fast mode was off by default until October 2026.

This is a Pillow-AllSIMD feature: neither Pillow nor Pillow-SIMD has a fast
mode. Check for it with ``features.check_feature("fast")``.

Turning it off
--------------

Set the ``PILLOW_ALLSIMD_FAST`` environment variable before
:py:mod:`PIL.Image` is imported (``PILLOW_SIMD_FAST``, its former name, is
read when ``PILLOW_ALLSIMD_FAST`` isn't set). ``0``, ``false``, ``off``,
``no`` and an empty value turn fast mode off; ``1``, ``true``, ``on`` and
``yes`` turn it on. Any other value gives a warning and leaves the setting
unchanged::

    PILLOW_ALLSIMD_FAST=0 python3 app.py

Or call :py:func:`~PIL.Image.set_fast`, or use the
:py:class:`~PIL.Image.fast_mode` context manager for one block::

    from PIL import Image, ImageFilter, features

    if features.check_feature("fast"):
        Image.set_fast(False)

    with Image.fast_mode(False):
        blurred = im.filter(ImageFilter.GaussianBlur(2))

The environment variable and ``set_fast()`` set the process-wide default,
for every thread. A ``fast_mode()`` block sets the mode of the thread it
runs in, and restores it on exit: other threads keep their own blocks or
the default, blocks nest, and threads started inside a block don't inherit
it. Each operation reads the setting once, when it starts, so a call never
mixes exact and fast kernels.

What it guarantees
------------------

* **Exact when off.** With fast mode off, the operations run the exact
  kernels, the same as before fast mode existed.
* **Deterministic.** A fast kernel's result depends only on its input and
  arguments, and is the same on every build and architecture of a given
  Pillow-AllSIMD version. (Large downscales use the exact resize and reduce
  kernels in another order, so they are as reproducible as those.)
* **Bounded.** For each fast kernel, the largest difference from the exact
  result and its error against a float reference of the operation are
  documented below and checked by the tests.
* **Same shape.** Output mode, size and edge handling are those of the exact
  operation; only pixel values differ.
* **Not stable across versions.** A later release may change a fast kernel's
  results when the kernel improves. The exact path follows Pillow.

Turn fast mode off if you compare pixels with reference images made by
Pillow, or cache or deduplicate images by their content across versions.

Operations with a fast kernel
-----------------------------

.. list-table::
   :header-rows: 1

   * - Operation
     - Fast kernel
     - Difference from the exact result
   * - :py:class:`~PIL.ImageFilter.GaussianBlur` and
       :py:class:`~PIL.ImageFilter.UnsharpMask`, radius up to 6, 8-bit
       modes (L, LA, La, RGB, RGBA, RGBa, RGBX, CMYK)
     - Pillow's three box passes as one kernel with 8-bit weights, in one
       pass over the image; the bands near the edges are the exact result
     - at most 3 levels, 0.1 to 0.7 on average; closer to a true Gaussian
       than the exact result
   * - :py:meth:`~PIL.Image.Image.convert` RGB, RGBA, RGBX → YCbCr and
       YCbCr → RGB
     - JPEG's formulas with 14-bit coefficients and one rounding, in place
       of tables that floor each term
     - at most 1 level; against the float formulas 0.3% and 0.06% of values
       off by one, against 47% and 32% for the exact result
   * - :py:meth:`~PIL.Image.Image.convert` RGB, RGBA, RGBX → HSV
     - the same formulas, truncated exactly: one division in 32-bit floats
       per channel (arm64: for the hue, with the saturation from a table of
       its 65,536 values)
     - hue 1 above the exact result for 0.3% of colours; saturation and value
       the same
   * - :py:meth:`~PIL.Image.Image.transform` (affine, extent, perspective,
       quad, mesh) and :py:meth:`~PIL.Image.Image.rotate`, bilinear and
       bicubic, modes L, RGB, RGBA, RGBa, RGBX, CMYK
     - positions in fixed point (1/256 pixel), integer filters with one
       rounding
     - at most 2 levels, 1 on photos (RGBA: more where alpha is low, from
       the premultiplication), but for the few pixels whose source position
       is on the image's edge, which may fall inside in one result and
       outside in the other; against float interpolation 0.6% of values off
       by one, against 41% for the exact result
   * - :py:func:`~PIL.Image.blend`, alpha between 0 and 1
     - integer interpolation with alpha in 2^-15 units, rounded
     - at most 1 level; the float result rounded, but within 0.004 of a
       half; the exact result is truncated
   * - :py:meth:`~PIL.Image.Image.resize` of 8-bit images with 4-byte
       pixels (RGB, RGBA, RGBX, CMYK, LA), any filter but nearest, when the
       vertical pass comes second
     - the two passes fused through a ring of rows; the horizontal sums
       kept to 2^-6 levels instead of whole ones; the vertical pass in
       16-bit products
     - at most 1 level (RGBA and LA: 2, premultiplied); against a float
       resize 0.4% to 1.5% of values off by one, against 8% to 25% for the
       exact result
   * - :py:meth:`~PIL.Image.Image.resize` with the bicubic, Lanczos and MKS
       filters and no ``reducing_gap``, downscales of 6x and more, 8-bit
       modes
     - ``reducing_gap=3.0``: an integer shrink with
       :py:meth:`~PIL.Image.Image.reduce` first
     - at most 2 levels at photo sizes; more for small results (32x32:
       4, 9 at the edges)

Other operations, and other arguments of these, use the exact kernels in
fast mode too.

How much faster the fast kernels are than the exact ones, for 2560×1600 RGB
images on one core of an Intel Core Ultra 9 285H (AVX2) and an NVIDIA Jetson
Orin (NEON), from the published benchmark (the exact kernels' own speed over
Pillow is on the :doc:`kernels page </allsimd-kernels>`):

.. generated: fast-mode-speedups (checks/simd_tables.py, pillow-allsimd-perf@0d7d352f2d047c9ff9daffe99d7314c197cd2ab2)

.. list-table::
   :header-rows: 1

   * - Operation
     - x86-64 (AVX2)
     - arm64 (NEON)
   * - Gaussian blur, radius 1
     - 2.96x
     - 3.18x
   * - Unsharp mask, radius 2
     - 1.68x
     - 1.82x
   * - RGB to YCbCr
     - 1.62x
     - 2.23x
   * - YCbCr to RGB
     - 1.25x
     - 2.60x
   * - RGB to HSV
     - 3.00x
     - 3.35x
   * - rotate 30°, bilinear
     - 2.97x
     - 1.84x
   * - rotate 30°, bicubic
     - 3.10x
     - 2.54x
   * - rotate 30°, bicubic, RGBA
     - 2.68x
     - 2.23x
   * - perspective, bilinear
     - 2.30x
     - 1.85x
   * - blend, 50%
     - 1.64x
     - 4.20x
   * - resize to 5478×3424, bilinear
     - 1.17x
     - 1.43x
   * - resize to 5478×3424, bicubic
     - 1.28x
     - 1.47x
   * - resize RGBA to 5478×3424, bilinear
     - 1.17x
     - 1.31x
   * - resize to 2048×1280, bicubic
     - 0.96x
     - 1.20x
   * - resize to 320×200, Lanczos (``reducing_gap``)
     - 1.72x
     - 2.61x
   * - resize to 26×16, Lanczos (``reducing_gap``)
     - 2.97x
     - 5.75x

.. generated-end

Functions
---------

.. autofunction:: PIL.Image.set_fast
   :noindex:
.. autofunction:: PIL.Image.get_fast
   :noindex:
.. autoclass:: PIL.Image.fast_mode
   :noindex:
