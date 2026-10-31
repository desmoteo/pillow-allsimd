.. _fast-mode:

Fast mode (Pillow-SIMD)
=======================

By default every Pillow-SIMD build, whatever its instruction set, produces
the same pixels as Pillow. Fast mode lets operations use kernels that are
faster and whose results may differ from Pillow's by small, documented
amounts. It is off unless you turn it on.

This is a Pillow-SIMD feature: upstream Pillow has no fast mode. Check for it
with ``features.check_feature("fast")``.

Turning it on
-------------

Set the ``PILLOW_SIMD_FAST`` environment variable before :py:mod:`PIL.Image`
is imported. ``1``, ``true``, ``on`` and ``yes`` turn fast mode on; ``0``,
``false``, ``off``, ``no`` and an empty value turn it off. Any other value
gives a warning and leaves the setting unchanged::

    PILLOW_SIMD_FAST=1 python3 app.py

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

* **Off by default.** With fast mode off, nothing changes.
* **Deterministic.** A fast kernel's result depends only on its input and
  arguments, and is the same on every build and architecture of a given
  Pillow-SIMD version.
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

None yet: this release adds the switch. The first fast kernels planned are
the Gaussian blur, RGB↔YCbCr and RGB→HSV conversions, and the affine,
perspective and quad transforms. Each will be listed here with its speed-up
on x86-64 and arm64, its largest difference from the exact result, and its
error against a float reference.

Functions
---------

.. autofunction:: PIL.Image.set_fast
   :noindex:
.. autofunction:: PIL.Image.get_fast
   :noindex:
.. autoclass:: PIL.Image.fast_mode
   :noindex:
