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

- [x] (2026-11-02, at `da7a94ac6`: all pass except the font test below) Run the test workflows on the current code: they last ran at
      `1881401a5`, because every push since ended with a `[ci skip]`
      commit (`gh workflow run test.yml -R desmoteo/pillow-allsimd --ref
      simd/pillow-13`, and the same for `test-windows.yml`).
- [x] `Tests/test_imagefont.py::test_glyph_cache_variation[0]` failed on
      Windows and Intel macOS: FreeType 2.14 renders a named instance set
      after axes and a render differently from one set on a new face
      (stock Pillow 12.3 too). The test's fresh font now does the same
      steps (2026-11-02).
- [x] (2026-11-02) Build and check the source package locally: `make sdist`,
      check-manifest, `twine check`, then install it in a clean venv with
      `-C simd=avx2`, with `CC="cc -mavx2"` and with the default. All
      three build the expected kernels and pass the selftest and the SIMD,
      fast-mode, resize, filter, font and conversion tests. The sdist had
      included a local `docs/_build`; MANIFEST.in now prunes it.
- [x] (2026-11-02) Upload from a workflow with trusted publishing (no token
      stored): `.github/workflows/release.yml` builds and checks the
      source package; a tag `allsimd-<version>` uploads it to PyPI (the job
      checks that the tag matches `src/PIL/_version.py`), a manual run to
      TestPyPI. Pillow's `wheels.yml` stays as it is (its publish jobs
      never run in a fork) and ignores `allsimd-*` tags; step 2 adds the
      arm64 wheels.
- [ ] **You:** PyPI and TestPyPI accounts with two-factor login. Under
      Publishing, add a pending publisher on each: project
      `pillow-allsimd`, owner `desmoteo`, repository `pillow-allsimd`,
      workflow `release.yml`, environment `release-pypi` on PyPI and
      `release-testpypi` on TestPyPI.
- [x] (2026-11-02) GitHub environments `release-pypi` (tags `allsimd-*`
      only) and `release-testpypi` (branch `simd/pillow-13` only), each
      with you as required reviewer, so nothing uploads without a click.
- [ ] Dry run to TestPyPI (`gh workflow run release.yml -R
      desmoteo/pillow-allsimd --ref simd/pillow-13`, then approve); install
      from it on the x86 laptop and the Jetson.
- [ ] Install instructions (README, `PyPI.rst`, `docs/allsimd.rst`):
      `python3 -m pip install pillow-allsimd`, with `-C simd=...` passed
      through pip; and the `PIL` conflict. Packages that depend on Pillow
      (matplotlib, torchvision…) make pip install Pillow too, and the last
      one installed overwrites `PIL`: install everything else first, then
      `python3 -m pip uninstall -y pillow` and install pillow-allsimd.
- [ ] The release: push the tag `allsimd-13.0.0.post0` (it clashes with
      neither Pillow's nor Pillow-SIMD's tags, both in this repo; never
      `git push --tags`), then approve the upload. The tag
      must match `src/PIL/_version.py`. A version number can't be reused
      on PyPI, even after deleting it.
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

- [x] Turn on issues (done 2026-11-02): `gh repo edit desmoteo/pillow-allsimd
      --enable-issues`. They are off, so problems can only be reported on
      uploadcare's tracker.
- [x] (2026-11-02) Issue template and contributing guide (`.github/ISSUE_TEMPLATE/`,
      `.github/CONTRIBUTING.md`): they point to Pillow's tracker. Send
      Pillow-AllSIMD problems here; problems that Pillow itself has go to
      Pillow.
- [x] (2026-11-02) Security policy (`.github/SECURITY.md`): it sends reports to Pillow's
      private advisories and Tidelift. Turn on private vulnerability
      reporting on the fork and write a policy for this code, pointing to
      Pillow for Pillow's own code.
- [x] README: say where to report problems (its Contributing section already does).

## Outreach

- [x] (2026-11-02: per-issue replies posted on #80, #116, #118, #123, #143; #43 not posted) Reply on the ARM issues of uploadcare/pillow-simd: one comment on #43
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

- [x] Push `142881bb3` to `simd/pillow-13` and `simd/master`; republish the
      API docs (the About page names `design/OTHER_ARCHITECTURES.md`).
- [x] Push so that CI runs: a push whose last commit says `[ci skip]` skips
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

Python overhead (details in
[PYTHON_TO_C_ASSESSMENT.md](PYTHON_TO_C_ASSESSMENT.md); no wholesale
conversion to C):

- [ ] Python fixes, also for Pillow: TIFF directories parsed three times
      per page, `point()` rounding integer tables, `open()` probing order.
- [ ] C: WebP stills opened without the animation decoder; `ImageStat` in
      one C call; a `text()` fast path.
- [ ] Kernels: `histogram()` (1.3 ms on 1080p RGB) and general-table
      `point()` (1.1 ms).

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
