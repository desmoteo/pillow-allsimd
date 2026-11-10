.. _allsimd:

About Pillow-AllSIMD
====================

Pillow-AllSIMD is Pillow with SIMD kernels: a fork of
`Pillow-SIMD <https://github.com/uploadcare/pillow-simd>`_, which is a fork of
`Pillow <https://github.com/python-pillow/Pillow>`_. Everything in these pages
about Pillow's API applies to it unchanged; this page and :ref:`fast-mode` are
its own.

Aim
---

* **SIMD on every common CPU.** Pillow-SIMD targets x86; Pillow-AllSIMD
  carries its kernels further on x86-64 (SSE4, AVX2, AVX-VNNI) and ports them
  to arm64 (NEON), with portable C that compilers vectorize elsewhere.
* **Fast by default.** :ref:`fast-mode` uses kernels whose results differ
  from Pillow's by a few documented levels at most, the same on every build
  and architecture.
* **Pillow's results when asked.** In exact mode the operations follow
  Pillow's arithmetic, or an integer form of it that rounds differently by one
  level at most in a few places (resize with 16-bit coefficients, RGB to L
  with 15-bit ones), and each SIMD kernel is checked against the scalar code
  on every build.
* **Following Pillow.** Each Pillow release is merged, and changes that suit
  Pillow-SIMD are offered to it.

State
-----

* Version |release|, on Pillow 13.0.0 (the ``simd/pillow-13`` branch).
* Beta. On PyPI as wheels for Linux on x86-64 and arm64 and for macOS 11
  or later on Apple silicon, and as a source package, compiled on
  installation, everywhere else (below).
* Tested on x86-64 (Intel Core Ultra 9 285H) and arm64 (NVIDIA Jetson Orin):
  Pillow's test suite on every build, the SIMD consistency tests, and output
  hashes per kernel that compare the builds (``checks/simd_hashes.py``). The
  portable build also passes the suite on riscv64, ppc64le, s390x, armv7 and
  32-bit x86, under QEMU (``design/OTHER_ARCHITECTURES.md`` in the repository).
* `Benchmarks <https://desmoteo.github.io/pillow-allsimd-perf/>`_ against
  Pillow, Pillow-SIMD, OpenCV, libvips and ImageMagick, exact and fast mode.

.. _allsimd-install:

Installing
----------

On Linux (glibc 2.28 or later, or musl), on x86-64 and arm64, pip
installs a wheel with the image libraries bundled, as Pillow's wheels
have them. The x86-64 wheel holds SSE4.1, AVX2 and AVX-VNNI kernels and
picks the set the CPU has when Pillow loads; the arm64 wheel has the NEON
kernels, and so has the macOS wheel, for macOS 11 or later on Apple
silicon. Elsewhere,
Pillow-AllSIMD is a source package, which pip compiles on installation,
so install the libraries that :doc:`building Pillow from source
<installation/building-from-source>` needs first; a source build on
x86-64 holds the same three sets as the wheel, so no option is needed.
It installs the ``PIL`` package, like Pillow: install your other packages
first, then uninstall Pillow (or Pillow-SIMD) and install
Pillow-AllSIMD::

    python3 -m pip uninstall -y pillow pillow-simd
    python3 -m pip install pillow-allsimd

``python3 -m PIL`` shows the kernels in use (``--- SIMD kernels: avx2
(built: scalar, sse4, avx2, avxvnni)``).

Uninstall Pillow before, not after: both own the files in ``PIL``. To build
from source on Linux anyway, add ``--no-binary pillow-allsimd``. To
rebuild with another ``simd`` option, add ``--no-cache-dir
--force-reinstall``, or pip reuses the wheel it built first. The development
branch installs with ``"pillow-allsimd @
git+https://github.com/desmoteo/pillow-allsimd@simd/pillow-13"``.

The ``simd`` build option chooses the instruction sets of a source build:

* ``auto`` (the default): ``dispatch`` on x86-64, NEON on arm64, SSE4.1 on
  32-bit x86, portable C elsewhere.
* ``dispatch``: SSE4.1, AVX2 and AVX-VNNI kernels in one build, the set
  the CPU supports chosen when Pillow loads (x86-64 only; with MSVC,
  and with a compiler or an assembler too old for AVX-VNNI, which needs
  GCC 11 or clang 12 and binutils 2.36, SSE4.1 and AVX2: RHEL 9 and its
  relatives have binutils 2.35). The ``PILLOW_ALLSIMD_ISA`` environment variable
  (``scalar``, ``sse4``, ``avx2``, ``avxvnni``) caps the choice, to compare
  the tiers on one machine.
* ``sse4``: SSE4.1 only, in every file (x86 only). This and ``avx2``,
  ``avxvnni``, ``native`` and ``native512`` build for one instruction set,
  which the CPU must have, or Pillow fails to import with a message.
* ``avx2``: SSE4.1 and AVX2 (x86 only).
* ``avxvnni``: SSE4.1, AVX2 and AVX-VNNI (x86 only: Intel Alder Lake,
  Sapphire Rapids and newer, AMD Zen 5); faster integer filters and resize.
* ``native``: everything the build machine's CPU supports
  (``-march=native``), AVX-512 included in compiler-generated code; the build
  runs only on similar CPUs.
* ``native512``: like ``native``, with 512-bit vectors preferred (x86 only).
* ``neon``: NEON (arm64 only).
* ``disable``: portable C only, on any architecture.

Don't pass the instruction set in ``CFLAGS``: setting it replaces Python's
compiler flags, the optimization level included.

What is accelerated
-------------------

Resampling and reduce, box and Gaussian blur, the unsharp mask, alpha
composition and premultiplication, colour conversions (grayscale, RGB and
RGBA, CMYK, HSV, YCbCr), kernel and rank filters, rotations, transposes and
transforms, colour lookup tables, channel operations, point tables, palette
mapping, drawing fills, PNG row filters and pixel packing, each with the
instruction sets the
`readme <https://github.com/desmoteo/pillow-allsimd#what-is-accelerated>`_
lists. :ref:`allsimd-kernels` explains how they work, operation by
operation.

Credits
-------

* **PIL**, the Python Imaging Library, by Fredrik Lundh and contributors
  (Secret Labs AB, 1995-2011).
* **Pillow**, the friendly PIL fork, by Jeffrey 'Alex' Clark and
  contributors (2010-): the library as a whole, its tests and this
  documentation.
* **Pillow-SIMD**, by Alexander Karpinsky and contributors, supported by
  Uploadcare, which has run it in production since 2015: the SIMD
  resampling, blur, filter, conversion and composition kernels for x86 that
  Pillow-AllSIMD starts from, and the idea of a drop-in SIMD build.
* **Pillow-AllSIMD**'s own changes: Matteo Ferrabone and contributors.

To cite Pillow-AllSIMD, use its
`CITATION.cff <https://github.com/desmoteo/pillow-allsimd/blob/simd/pillow-13/CITATION.cff>`_,
which cites Pillow and Pillow-SIMD too. All four are under the MIT-CMU
License (:doc:`about`).

Reporting issues
----------------

Bugs that aren't about the SIMD code or fast mode are Pillow's: please report
them to `Pillow <https://github.com/python-pillow/Pillow/issues>`_, whose
fixes reach Pillow-AllSIMD with each merged release. Issues with the SIMD
kernels, the builds or fast mode go to
`Pillow-AllSIMD's issues <https://github.com/desmoteo/pillow-allsimd/issues>`_.
