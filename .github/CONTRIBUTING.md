# Contributing to Pillow-AllSIMD

Pillow-AllSIMD is Pillow with SIMD kernels (SSE4, AVX2, AVX-VNNI, NEON) and
a fast mode. Contributions to those are welcome here, as
[issues](https://github.com/desmoteo/pillow-allsimd/issues) and
[pull requests](https://github.com/desmoteo/pillow-allsimd/pulls).

Bugs and features that are about Pillow itself (they also happen with
Pillow) belong to [Pillow](https://github.com/python-pillow/Pillow/issues).
Its fixes reach Pillow-AllSIMD with each merged release.

## Bug fixes, feature additions, etc.

Please send pull requests to the `simd/pillow-13` branch, with
[tests](../Tests/README.rst) for new code and an entry in
[CHANGES.SIMD.rst](../CHANGES.SIMD.rst).

- Fork the repository and create a branch from `simd/pillow-13`.
- Run the test suite. You can enable GitHub Actions on your fork to catch
  failures before the pull request.
- Create a pull request to `simd/pillow-13`.

### Guidelines

- Separate code commits from reformatting commits.
- Provide tests for any newly added code.
- **A SIMD kernel matches the scalar code.** In exact mode, every SIMD
  version gives the portable C code's pixels; in fast mode, every version
  gives the same pixels as the fast kernel's portable C definition. Add
  cases to `Tests/test_simd_consistency.py`, and run
  `checks/simd_hashes.py` on the builds you touched.
- **Time it.** Include before and after timings, on the machines the
  kernel is for, with the build options used. A kernel that isn't faster
  isn't merged.
- Follow PEP 8; `make lint` runs the formatters and linters.
- When committing only documentation changes please include `[ci skip]` in
  the commit message to avoid running extra tests. A push whose last commit
  says `[ci skip]` skips the tests for every commit in it.

The design notes in [design/](../design/) explain how the kernels and fast
mode work.

## Reporting issues

When reporting issues, please include code that reproduces the issue and
whenever possible, an image that demonstrates the issue. Please upload
images to GitHub, not to third-party file hosting sites. If necessary, add
the image to a zip or tar archive.

### Provide details

- What did you do?
- What did you expect to happen?
- What actually happened?
- Does it also happen in exact mode (`PILLOW_ALLSIMD_FAST=0`)?
- How did you install Pillow-AllSIMD (the `-C simd=...` option or `CC`),
  and on which CPU?
- What versions of Pillow-AllSIMD and Python are you using?

## Security vulnerabilities

Please see the [security policy](SECURITY.md).
