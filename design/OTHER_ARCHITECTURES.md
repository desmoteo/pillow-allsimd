# Other architectures: RISC-V, POWER, s390x, 32-bit ARM and x86

As of `simd/pillow-13` on 2026-11-01 (the portable builds tested at
`98d26558d` and `d67202ab7`).

Pillow-AllSIMD has SIMD kernels for x86 (SSE4.1, AVX2, AVX-VNNI) and arm64
(NEON). Every other architecture builds the portable C code. This document
records what upstream users have asked for, why the RISC-V vector pull
request to Pillow-SIMD isn't ported, and how the portable builds behave on
riscv64, ppc64le, s390x, 32-bit ARM and 32-bit x86.

## Summary

- **The portable build passes the full test suite** on riscv64, ppc64le
  (POWER), s390x (big-endian), armv7 and 32-bit x86. The only failures
  are two animated-WebP tests on s390x, which upstream Pillow 13.0.0 fails
  there too.
- **It gives the same pixels as x86-64** in 21 of the 23 areas of
  `checks/simd_hashes.py`, fast mode included. The two others depend on
  double-precision arithmetic: float resampling (F, I and I;16 images) and
  the exact transforms. arm64 differs in the same two.
- **Fixed while testing:** the SIMD builds on 32-bit x86 now do floating
  point in SSE registers (`-mfpmath=sse`). The x87 unit had made a
  transform's pixels depend on the code around them. `-C simd=native` now
  works on POWER (`-mcpu=native`) and explains itself on RISC-V. The test
  harness runs on all these platforms, and the hash script no longer
  depends on the machine's byte order.
- **No RISC-V or POWER kernels.** There is no hardware to measure them on,
  and few users. Whatever gets added to the code would need a third SIMD
  version of every kernel. GCC already vectorizes the portable loops for
  POWER's VSX.

## What upstream users asked for

From the issues and pull requests of
[uploadcare/pillow-simd](https://github.com/uploadcare/pillow-simd) as of
2026-11-01:

| Item | Ask | Outcome |
|---|---|---|
| [#158](https://github.com/uploadcare/pillow-simd/pull/158) (open PR, 2026-05) | RISC-V vector (RVV) kernels | No maintainer review; only automated review bots. Conflicts with `simd/master` |
| [#43](https://github.com/uploadcare/pillow-simd/issues/43) (open since 2018) | ARM support; one comment asks for ppc64le (AltiVec) | The maintainer turned down a translation of the SSE code through SSE2NEON and asked for optimized NEON instead. Pillow-AllSIMD's NEON kernels, offered as [#162](https://github.com/uploadcare/pillow-simd/pull/162), are that |
| [#80](https://github.com/uploadcare/pillow-simd/issues/80), [#116](https://github.com/uploadcare/pillow-simd/issues/116), [#143](https://github.com/uploadcare/pillow-simd/issues/143) | The build fails on Raspberry Pi and Apple silicon (`-msse4`) | Closed: Pillow-SIMD is for x86 |
| [#156](https://github.com/uploadcare/pillow-simd/issues/156) (open) | Feedback on the wheel variants proposal (wheels per CPU level: `x86_64_v4`, `armv8.1a`, AVX-512) | No reply. It would fit Pillow-AllSIMD's builds per instruction set |
| [#3](https://github.com/uploadcare/pillow-simd/issues/3) (closed, 2016) | A portable SIMD library (libsimdpp) | Declined: it couldn't match hand-written kernels |

Nothing asks for WebAssembly, LoongArch, s390x or SVE.

## The RISC-V vector pull request (#158)

**What it is.** Pillow-SIMD's `simd/master` plus RVV intrinsics for filters,
reduce and resample, colour conversions, alpha composition, box blur,
ColorLUT and band operations, in 18 C sources and headers. Its size is
+6,543 and −1,682 lines in 32 files. It also changes `setup.py` for
riscv64 cross builds, two CI workflows, the version, and how the FPX and
MIC plugins close their files. It was validated with Pillow's test suite
under QEMU (4,026 passed) and a cross build with clang. It was never run
on RISC-V hardware.

**Notes from reading the diff.**

- The vector code is compiled only when the compiler targets the V
  extension (`#if defined(__riscv_vector)`). The default riscv64 build
  passes `-march=rv64gc`, so it gets none of it. A build for `rv64gcv`
  then needs vector hardware: the description mentions runtime detection
  (`ImagingRVVDetect.c`), but no such file or `getauxval` call is in the
  diff.
- Some kernels follow the SSE code's structure, the box blur through
  helpers named after SSE intrinsics (`rvv_cvtepu8_epi32`). That is the
  translation approach the maintainer turned down for NEON in #43.
- Whether the kernels give the scalar code's pixels isn't shown by the
  tests the PR ran. Pillow's own suite tolerates differences, and the PR
  adds no consistency tests or hashes. The alpha composition, for one,
  divides in floats and then corrects the quotient. That may be exact, but
  nothing checks it.

**Why it isn't ported.**

1. **It can't be measured.** QEMU checks results, not speed, and no
   RVV 1.0 hardware is at hand. Kernels that can't be timed can't be tuned
   or even shown to be faster than the portable code. The maintainer's
   objection to translated NEON was exactly that.
2. **Few users.** Boards with the vector extension are rare, and Python
   image processing on them rarer. Every upstream request was about ARM.
3. **It's a port, not a merge.** The PR is against Pillow-SIMD's older
   code. Pillow-AllSIMD has merged Pillow 13 and rewritten many of those
   kernels (resample, blur, conversions, composition, filters). It has
   added fast mode, and fast mode's promise of the same pixels on every
   build would cover RVV versions too. Each of the 15+ kernels would have
   to be rewritten onto the current code and checked against the scalar
   code.
4. **The ongoing cost.** Every later kernel would need an RVV version as
   well, untested on hardware.

**When to revisit.** With RVV 1.0 hardware to time on, and users asking.
Then the place to start would be the operations that matter most (resize,
conversions, blur), written for RVV's variable vector length rather than
translated from SSE. They would need the consistency tests and hashes the
other kernels have.

## The portable builds, tested

**Method.** `checks/simd_test_remote.sh --platform linux/<arch> local` on
the x86-64 laptop. It builds a `python:3.13-slim-trixie` container for the
platform with Debian's codec libraries and the optional test dependencies
that install there. It copies the commit and the extra test images, builds
Pillow-AllSIMD, and runs Pillow's whole test suite with pytest-xdist.
riscv64, ppc64le, s390x and armv7 run under QEMU user emulation. 386 runs
natively, so its SSE4 and AVX2 builds were tested as well.

| Platform | Builds | Passed | Skipped | Failed | Time |
|---|---|---|---|---|---|
| riscv64 | portable | 8105 | 60 | 0 | 9 min (QEMU) |
| ppc64le (POWER8) | portable | 8104 | 61 | 0 | 8 min (QEMU) |
| s390x (big-endian) | portable | 8103 | 60 | 2 | 12 min (QEMU) |
| armv7 (armhf) | portable | 8104 | 61 | 0 | 7 min (QEMU) |
| 386 (i686) | scalar, SSE4, AVX2, AVX-VNNI | 8103 each | 62 | 0 | 5 min each |

The suites skip more here than on x86-64 (36): Qt, Arrow and some other
wheels don't exist for these platforms.

**s390x: animated WebP.** `test_write_animation_L` and `_RGB` fail: the
decoded frames are far from the originals. Pillow 13.0.0 itself, built from
source in the same container, fails the same two tests with Debian's
libwebp 1.5.0. Pillow-AllSIMD's only WebP change is the still-image decode
path. It's an upstream or libwebp issue on big-endian, not this fork's.

**32-bit x86: the x87 unit.** Without SSE, GCC does double arithmetic on
the x87 unit, whose registers keep 80 bits. A value then rounds to 64 bits
or not depending on whether the compiler spills it, so the same pixel
could differ between a whole image and one column of it. The SSE4 and AVX2
builds failed `test_transform_pairs_match_single_pixels` that way.

- The SIMD builds require SSE4.1 anyway, so they now pass `-mfpmath=sse`
  and round as x86-64 does. A 32-bit Python in a container on a 64-bit
  kernel reports `linux-x86_64`, so `setup.py` recognizes the target by
  its pointer size. MSVC uses SSE2 for floating point already.
- The scalar build keeps the x87, since Debian's i686 baseline doesn't
  have SSE2. There the exact HSV test (against IEEE doubles) is skipped,
  and the fast transform test allows more pixels on the source's edge to
  fall inside in one result and outside in the other.

**armv7.** The test image's `libglib2.0-0` is called `libglib2.0-0t64`
since Debian's 64-bit `time_t` transition. The harness now installs
whichever exists.

**Output hashes.** `checks/simd_hashes.py` hashes the output of each SIMD
area over many sizes and modes. Portable builds compared with the x86-64
portable build:

| Platform | Equal | Different |
|---|---|---|
| riscv64 | 21 of 23 | resample F/I/I;16, transform |
| ppc64le | 21 of 23 | resample F/I/I;16, transform |
| s390x | 21 of 23 | resample F/I/I;16, transform |
| armv7 | 21 of 23 | resample F/I/I;16, transform |
| 386, AVX2 build | 22 of 23 | alpha_composite (the x86 SIMD kernel, known) |
| 386, scalar build | 14 of 23 | the floating-point areas, among them the fast Gaussian, transform and resize, whose weights come from doubles |

- The two areas that differ everywhere are the ones that differ on arm64:
  F, I and I;16 resampling with double coefficients, and the exact
  transforms. Turning off fused multiply-adds (`-ffp-contract=off`) on
  ppc64le didn't remove the difference. Both take values from `sin` and
  `cos` (Lanczos weights, rotation matrices), so the math library's
  rounding probably contributes. That wasn't investigated further.
- So fast mode's promise of the same pixels on every build holds on all
  these platforms, except 32-bit x86 scalar builds: there the x87 changes
  the weights. On 32-bit x86, build with `-C simd=sse4`.
- s390x first differed in two more areas, transposes and fast downscales.
  The cause was the hash script: it took I and F pixels in the machine's
  byte order. It now hashes them little-endian, and s390x matches.

## Compilers and vectors on these targets

The portable code is written so that compilers can vectorize its loops
(`-ftree-vectorize`, Python's `-O3`). Whether they do depends on the
instruction set the build targets. Debian's GCC 14 defaults:

| Platform | Vector unit in the default target | `-C simd=native` |
|---|---|---|
| ppc64le | VSX and AltiVec (POWER8 baseline): the loops are vectorized | `-mcpu=native` (GCC takes no `-march` there; fixed) |
| riscv64 | none (`rv64gc`) | refused, with a hint: GCC 14 has no `-march=native` for RISC-V; use `CC="cc -march=rv64gcv"` |
| s390x | none (no vector facility in the baseline) | `-march=native`: z13 and later get vector code |
| armv7 | none (VFPv3-D16, no NEON in the baseline) | `-march=native` |
| 386 | none (i686); `-C simd=sse4` gives SSE4.1 | `-march=native` |

Speed on these platforms wasn't measured: QEMU's timings say nothing
about hardware.

## How to reproduce

```bash
for p in riscv64 ppc64le s390x arm/v7 386; do
    checks/simd_test_remote.sh --platform linux/$p \
        --workdir pillow-simd-test-${p//\//-} -n 3 --no-compare local
done
```

Hashes: build with `-C simd=disable` in the same images and run
`PYTHONPATH=<build> python3 checks/simd_hashes.py` from the repository
root (`bench-results/archs/hashes.sh` did it for all platforms).
