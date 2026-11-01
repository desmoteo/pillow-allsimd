# Plan: releases, reports and open points

As of 2026-11-01: `simd/pillow-13` and `simd/master` at `a689617c3` on
GitHub, `142881bb3` (design documents moved to `design/`) local.

## Publishing on PyPI

The name `pillow-allsimd` is free on PyPI and TestPyPI. Pillow-SIMD has
never published a wheel: its 83 releases are source packages, and users
compile them (`pip install pillow-simd`, or `CC="cc -mavx2" pip install
-U --force-reinstall pillow-simd` for AVX2).

**Decision: two steps, with wheels only where they lose nothing.**

1. **The source package first.** It's what Pillow-SIMD users know. On x86
   it gives the best build (`-C simd=avx2`), and it links the system's
   image libraries, so the distribution's security updates cover them.
2. **Then wheels for arm64 only** (Linux aarch64 and Apple-silicon macOS).
   NEON is the arm64 baseline, so a wheel is as fast as a source build, and
   ARM users are the ones a source build stops (every ARM issue on
   Pillow-SIMD). No x86 wheel yet: pip would choose it over the source
   package without saying so, and x86 users would get SSE4.1, 4.18x
   Pillow 13 instead of AVX2's 5.44x (exact mode, x86 benchmark).

### Step 1: the source package

- [ ] Run the test workflows on the current code: they last ran at
      `1881401a5`, because every push since ended with a `[ci skip]`
      commit (`gh workflow run test.yml -R desmoteo/pillow-allsimd --ref
      simd/pillow-13`, and the same for `test-windows.yml`).
- [ ] Fix or mark `Tests/test_imagefont.py::test_glyph_cache_variation[0]`.
      It fails on Windows, on macOS with Python 3.15 and on Intel macOS
      (at `1881401a5`); not SIMD code.
- [ ] Build and check the source package locally: `make sdist`,
      check-manifest, `twine check`, then install it in a clean venv with
      `-C simd=avx2`, with `CC="cc -mavx2"` and with the default.
- [ ] Choose how to upload: from the wheel workflow with trusted publishing
      (no token stored; step 2 then only adds jobs), or by hand with
      `twine` and a token scoped to the project. Recommended: the workflow.
- [ ] Workflow (`.github/workflows/wheels.yml`): the publish job runs only
      when `github.event.repository.fork == false`, which never holds here
      (the repo is a GitHub fork of uploadcare/pillow-simd). Condition on
      `github.repository == 'desmoteo/pillow-allsimd'`; PyPI link
      `https://pypi.org/p/pillow-allsimd`; turn off the wheel jobs, the
      iOS builds and the nightly upload for now; fix `EXPECTED_DISTS` (83)
      and check the SBOM file name the release job expects
      (`pillow-*.cdx.json`).
- [ ] PyPI and TestPyPI accounts with two-factor login. Under Publishing,
      add a pending publisher on each: project `pillow-allsimd`, owner
      `desmoteo`, repository `pillow-allsimd`, workflow `wheels.yml`,
      environment `release-pypi` (`release-testpypi` on TestPyPI).
- [ ] GitHub environments `release-pypi` and `release-testpypi`, with a
      required reviewer, so nothing uploads without a click.
- [ ] Dry run to TestPyPI; install from it on the x86 laptop and the
      Jetson.
- [ ] Install instructions (README, `PyPI.rst`, `docs/allsimd.rst`):
      `python3 -m pip install pillow-allsimd`, with `-C simd=...` passed
      through pip; and the `PIL` conflict. Packages that depend on Pillow
      (matplotlib, torchvision…) make pip install Pillow too, and the last
      one installed overwrites `PIL`: install everything else first, then
      `python3 -m pip uninstall -y pillow` and install pillow-allsimd.
- [ ] A release tag that clashes with neither Pillow's nor Pillow-SIMD's
      tags (both are in this repo), e.g. `allsimd-13.0.0.post0`. The
      version comes from `src/PIL/_version.py`, not the tag. A version
      number can't be reused on PyPI, even after deleting it.
- [ ] `CHANGES.SIMD.rst` entry for the first PyPI release.

### Step 2: arm64 wheels

- [ ] Wheel matrix: manylinux and musllinux aarch64, macOS arm64. Update
      `EXPECTED_DISTS`.
- [ ] The tests that run in the wheel builds pass on those platforms.
- [ ] Install instructions: arm64 gets a wheel, x86 builds from source.
- [ ] The ongoing duty: wheels bundle libjpeg, libwebp, libtiff, FreeType
      and others. When one of them has a security fix, rebuild and
      release. In practice, release after merging each Pillow release,
      which updates those libraries (`.github/dependencies.json`,
      `.github/workflows/wheels-dependencies.sh`).

### Later: x86 and Windows wheels

- [ ] One build that chooses SSE4.1 or AVX2 kernels at run time, before
      any x86 wheel (or per-CPU wheels, if the wheel variants proposal
      lands in pip: uploadcare/pillow-simd#156).
- [ ] A check when `_imaging` loads, so a CPU without SSE4.1 gets a clear
      error instead of an illegal instruction.
- [ ] Windows x64 wheels: building from source is hardest there. Windows
      arm64 with MSVC is untested.

## Issues and reports on the fork

- [ ] Turn on issues: `gh repo edit desmoteo/pillow-allsimd
      --enable-issues`. They are off, so problems can only be reported on
      uploadcare's tracker.
- [ ] Issue template and contributing guide (`.github/ISSUE_TEMPLATE/`,
      `.github/CONTRIBUTING.md`): they point to Pillow's tracker. Send
      Pillow-AllSIMD problems here; problems that Pillow itself has go to
      Pillow.
- [ ] Security policy (`.github/SECURITY.md`): it sends reports to Pillow's
      private advisories and Tidelift. Turn on private vulnerability
      reporting on the fork and write a policy for this code, pointing to
      Pillow for Pillow's own code.
- [ ] README: say where to report problems.

## Outreach

- [ ] Reply on the ARM issues of uploadcare/pillow-simd: one comment on #43
      mentioning everyone (recommended), or a reply in each of #80, #116,
      #118, #123 and #143. Drafts in `.pr-drafts/arm-issue-replies.md`
      (local, git-ignored). Before posting: the CI run above (the drafts
      say macOS passes in CI), issues turned on (then add a line inviting
      reports), and preferably step 1 (then the install line becomes
      `python3 -m pip install pillow-allsimd`).
- [ ] Upstream pull requests: #160 (CI), #161 (x86), #162 (NEON) and #164
      (docs) are open with no maintainer review; #163 is closed. Follow up
      or decide to leave them.

## Pushes and pages

- [ ] Push `142881bb3` to `simd/pillow-13` and `simd/master`; republish the
      API docs (the About page names `design/OTHER_ARCHITECTURES.md`).
- [ ] Push so that CI runs: a push whose last commit says `[ci skip]` skips
      the tests for every commit in it. Push code commits on their own, or
      start the workflows by hand after such a push.

## Other open points

Performance (details in [PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md)):

- [ ] arm64: the RGBA bilinear upscale, about 2.3x behind OpenCV.
- [ ] arm64: YCbCr ↔ RGB 2.1x and 1.7x, HSV 2.6x behind OpenCV. An exact
      16-bit integer YCbCr form is untried.
- [ ] Median 3x3 and 5x5 (column-sort reuse); reduce 2 and 4 on arm64.
- [ ] Gaussian blur at small radii: OpenCV's direct convolution gives
      different output, so only a fast-mode version could match it.

Correctness and upstream:

- [ ] Resizing I;16B and I;16N images on little-endian machines reads the
      bytes in the wrong order (the `bigendian` test in `Resample.c`). It's
      Pillow's bug too: report it to Pillow, and fix it here.
- [ ] s390x: two animated-WebP tests fail, as they do with Pillow 13.0.0
      and libwebp 1.5.0 ([OTHER_ARCHITECTURES.md](OTHER_ARCHITECTURES.md)).
      Report to Pillow.

Later, on conditions:

- [ ] RISC-V vector kernels: with RVV 1.0 hardware to time on, and users
      asking ([OTHER_ARCHITECTURES.md](OTHER_ARCHITECTURES.md)).
- [ ] The detailed explanation of the optimizations: a proposal in
      [SIMD_EXPLAINER_PLAN.md](SIMD_EXPLAINER_PLAN.md), its reader and
      location still to confirm.
