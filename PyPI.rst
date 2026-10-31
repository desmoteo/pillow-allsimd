Pillow-AllSIMD
==============

Pillow-AllSIMD is a fork of `Pillow-SIMD <https://github.com/uploadcare/pillow-simd>`_,
the SIMD build of `Pillow <https://python-pillow.org>`_, the friendly PIL fork.
It is a drop-in replacement for the Pillow version it follows (now 13.0.0):
the same ``PIL`` package and API, with SIMD kernels for x86-64 (SSE4, AVX2,
AVX-VNNI) and arm64 (NEON). By default a fast mode trades a few levels of
precision for speed in some operations; exact mode follows Pillow's
arithmetic.

Status: beta, built from source; see the repository for what is
accelerated, how it is tested and the benchmarks.

- `Pillow-AllSIMD repository and readme <https://github.com/desmoteo/pillow-allsimd>`_
- `Documentation <https://desmoteo.github.io/pillow-allsimd/>`_
- `Changelog <https://github.com/desmoteo/pillow-allsimd/blob/simd/pillow-13/CHANGES.SIMD.rst>`_
- `Benchmarks <https://desmoteo.github.io/pillow-allsimd-perf/>`_

Pillow is by Jeffrey 'Alex' Clark and contributors, on PIL by Fredrik Lundh
and Secret Labs AB; Pillow-SIMD by Alexander Karpinsky and contributors,
supported by Uploadcare. Licensed under the MIT-CMU License, like them.
