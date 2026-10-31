Pillow-AllSIMD
==============

Pillow-AllSIMD is a fork of `Pillow-SIMD <https://github.com/uploadcare/pillow-simd>`_,
the SIMD build of `Pillow <https://python-pillow.org>`_, the friendly fork of
the Python Imaging Library (PIL). It is a drop-in replacement for the Pillow
version it follows, now 13.0.0: the same ``PIL`` package and API, built with
SIMD kernels for x86-64 (SSE4, AVX2, AVX-VNNI) and arm64 (NEON).

* **Aim:** Pillow's results, faster, on every common CPU; by default the
  operations follow Pillow's arithmetic, and an opt-in :ref:`fast mode
  <fast-mode>` trades a few documented levels for more speed.
* **State:** beta, version |release| on Pillow 13.0.0, built from source;
  tested on x86-64 and arm64. See :doc:`allsimd`.
* **Credits:** PIL is by Fredrik Lundh and contributors; Pillow by Jeffrey
  'Alex' Clark and contributors, whose documentation these pages are, with
  Pillow-AllSIMD's additions; Pillow-SIMD by Alexander Karpinsky and
  contributors, supported by Uploadcare. See :doc:`about`.

Links: `repository <https://github.com/desmoteo/pillow-allsimd>`_,
`benchmarks <https://desmoteo.github.io/pillow-allsimd-perf/>`_,
:doc:`changelog <allsimd-changelog>`, and Pillow's own documentation at
`pillow.readthedocs.io <https://pillow.readthedocs.io/>`_.

Overview
========

The Python Imaging Library adds image processing capabilities to your Python interpreter.

This library provides extensive file format support, an efficient internal representation, and fairly powerful image processing capabilities.

The core image library is designed for fast access to data stored in a few basic pixel formats. It should provide a solid foundation for a general image processing tool.

.. toctree::
   :maxdepth: 2

   allsimd.rst
   installation/index.rst
   handbook/index.rst
   reference/index.rst
   porting.rst
   about.rst
   allsimd-changelog.rst
   releasenotes/index.rst
   deprecations.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
