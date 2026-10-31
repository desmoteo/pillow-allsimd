.. _fast-mode:

Fast mode (Pillow-AllSIMD)
==========================

By default Pillow-AllSIMD computes each operation with Pillow's arithmetic,
or an integer form of it that rounds differently by one level at most in a
few places (resize with 16-bit coefficients, RGB to L with 15-bit ones).
Fast mode lets operations use kernels that are faster and whose results may
differ from Pillow's by small, documented amounts. It is off unless you turn
it on.

This is a Pillow-AllSIMD feature: neither Pillow nor Pillow-SIMD has a fast
mode. Check for it with ``features.check_feature("fast")``.

Turning it on
-------------

Set the ``PILLOW_ALLSIMD_FAST`` environment variable before
:py:mod:`PIL.Image` is imported (``PILLOW_SIMD_FAST``, its former name, is
read when ``PILLOW_ALLSIMD_FAST`` isn't set). ``1``, ``true``, ``on`` and ``yes`` turn fast mode on; ``0``,
``false``, ``off``, ``no`` and an empty value turn it off. Any other value
gives a warning and leaves the setting unchanged::

    PILLOW_ALLSIMD_FAST=1 python3 app.py

Or call :py:func:`~PIL.Image.set_fast`, or use the
:py:class:`~PIL.Image.fast_mode` context manager::

    from PIL import Image, ImageFilter, features

    if features.check_feature("fast"):
        Image.set_fast(True)

    with Image.fast_mode():
        blurred = im.filter(ImageFilter.GaussianBlur(2))

The setting is process-wide: it applies to every thread, including while a
``fast_mode()`` block runs. Each operation reads it once, when the operation
starts, so a call never mixes exact and fast kernels.

What it guarantees
------------------

* **Off by default.** With fast mode off, nothing changes: the operations
  run the kernels they run without fast mode.
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

Leave fast mode off if you compare pixels with reference images, or cache or
deduplicate images by their content.

Operations with a fast kernel
-----------------------------

Speed-ups are for 2560x1600 RGB images on one core of an Intel Core Ultra 9
285H (AVX2) and an NVIDIA Jetson Orin (NEON).

.. list-table::
   :header-rows: 1

   * - Operation
     - Fast kernel
     - Speed-up (x86-64, arm64)
     - Difference from the exact result
   * - :py:class:`~PIL.ImageFilter.GaussianBlur` and
       :py:class:`~PIL.ImageFilter.UnsharpMask`, radius up to 6, 8-bit
       modes (L, LA, La, RGB, RGBA, RGBa, RGBX, CMYK)
     - Pillow's three box passes as one kernel with 8-bit weights, in one
       pass over the image; the bands near the edges are the exact result
     - radius 0.5: 4.0x, 4.4x; 1: 3.3x, 3.2x; 2: 2.5x, 2.6x; 4: 1.5x, 2.3x
     - at most 3 levels, 0.1 to 0.7 on average; closer to a true Gaussian
       than the exact result
   * - :py:meth:`~PIL.Image.Image.convert` RGB, RGBA, RGBX → YCbCr and
       YCbCr → RGB
     - JPEG's formulas with 14-bit coefficients and one rounding, in place
       of tables that floor each term
     - to YCbCr: 1.5x, 2.2x; to RGB: 1.3x, 1.7x
     - at most 1 level; against the float formulas 0.3% and 0.06% of values
       off by one, against 47% and 32% for the exact result
   * - :py:meth:`~PIL.Image.Image.convert` RGB, RGBA, RGBX → HSV
     - the same formulas, truncated exactly with one division in 32-bit
       floats per channel
     - 2.9x, 2.0x
     - hue 1 above the exact result for 0.3% of colours; saturation and value
       the same
   * - :py:meth:`~PIL.Image.Image.transform` (affine, extent, perspective,
       quad, mesh) and :py:meth:`~PIL.Image.Image.rotate`, bilinear and
       bicubic, modes L, RGB, RGBA, RGBa, RGBX, CMYK
     - positions in fixed point (1/256 pixel), integer filters with one
       rounding
     - rotate bilinear: 2.7x, 2.6x; bicubic: 2.9x, 3.4x; perspective: 2.1x,
       2.3x; L rotate bilinear: 2.8x, 1.9x
     - at most 2 levels, 1 on photos (RGBA: more where alpha is low, from
       the premultiplication), but for the few pixels whose source position
       is on the image's edge, which may fall inside in one result and
       outside in the other; against float interpolation 0.6% of values off
       by one, against 41% for the exact result
   * - :py:func:`~PIL.Image.blend`, alpha between 0 and 1
     - integer interpolation with alpha in 2^-15 units, rounded
     - 1.7x, 4.1x
     - at most 1 level; the float result rounded, but within 0.004 of a
       half; the exact result is truncated
   * - :py:meth:`~PIL.Image.Image.resize` with the bicubic, Lanczos and MKS
       filters and no ``reducing_gap``, downscales of 6x and more, all modes
       but I;16
     - ``reducing_gap=3.0``: an integer shrink with
       :py:meth:`~PIL.Image.Image.reduce` first
     - 1/8 Lanczos: 1.9x, 2.5x; 1/16: 2.0x, 4.5x
     - at most 2 levels at photo sizes; more for small results (32x32:
       4, 9 at the edges)

Other operations, and other arguments of these, use the exact kernels in
fast mode too.

Functions
---------

.. autofunction:: PIL.Image.set_fast
   :noindex:
.. autofunction:: PIL.Image.get_fast
   :noindex:
.. autoclass:: PIL.Image.fast_mode
   :noindex:
