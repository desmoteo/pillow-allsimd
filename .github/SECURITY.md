# Security policy

## Reporting a vulnerability

Pillow-AllSIMD is Pillow with SIMD kernels and a fast mode. Where you report
a vulnerability depends on where it is:

- **In this project's code** (the SIMD kernels, fast mode, the build), or if
  you aren't sure: report it
  [privately on GitHub](https://github.com/desmoteo/pillow-allsimd/security/advisories/new).
- **In Pillow's own code**, which this project shares (it also happens with
  Pillow): report it to Pillow, as its
  [security policy](https://github.com/python-pillow/Pillow/security/policy)
  describes. Its fixes reach Pillow-AllSIMD with each merged release.

**DO NOT report sensitive vulnerability information in public.**

Pillow-AllSIMD has one maintainer, so there is no guaranteed response time.
Reports about the latest commit of `simd/pillow-13` are the ones that get
fixed.

## Threat model

Pillow's primary attack surface is parsing untrusted image data, and that is
the same here. A full STRIDE threat model covering spoofing, tampering, repudiation, information disclosure, denial of service, and elevation of privilege is maintained in the [Security handbook page](https://pillow.readthedocs.io/en/latest/handbook/security.html).

Key risks to be aware of when using Pillow or Pillow-AllSIMD to process untrusted images:

- **Decompression bombs** — do not set `Image.MAX_IMAGE_PIXELS = None` in production.
- **EPS files invoke Ghostscript** — block EPS input at the application layer unless strictly required.
- **`ImageMath.unsafe_eval()`** — never pass user-controlled strings to this function; use `lambda_eval` instead.
- **C extension memory safety** — keep Pillow-AllSIMD and the C libraries it uses (libjpeg, libpng, libtiff, libwebp, etc.) up to date.
- **Sandboxing** — for high-risk deployments, run image processing in a sandboxed subprocess.
