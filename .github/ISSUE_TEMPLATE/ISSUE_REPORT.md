---
name: Issue report
about: Report a problem with Pillow-AllSIMD
---

<!--
Thank you for reporting an issue.

Pillow-AllSIMD is Pillow with SIMD kernels (SSE4, AVX2, NEON) and a fast
mode. Problems that Pillow itself has belong to Pillow:

    https://github.com/python-pillow/Pillow/issues

Three quick checks tell them apart:

* Does it happen in exact mode? Run with PILLOW_ALLSIMD_FAST=0 in the
  environment, or call Image.set_fast(False) first.
* Does it happen with a portable build (-C simd=disable)?
* Does it happen with Pillow (python3 -m pip install pillow)?

If it happens with Pillow too, please report it to Pillow.

Include a self-contained, copy-pastable example, and an image if the
problem depends on one (upload it here, not to another site; zip it if
needed).
-->

### What did you do?

### What did you expect to happen?

### What actually happened?

### Fast mode and exact mode

<!-- Does it also happen with PILLOW_ALLSIMD_FAST=0? -->

### How did you install Pillow-AllSIMD?

<!--
The pip command, including any -C simd=... option or CC variable, and the
commit or branch.
-->

### What are your CPU, OS and Python?

* CPU (e.g. Intel Core i7-12700, Apple M2, Cortex-A78AE):
* OS:
* Python:

```text
Please paste here the output of running:

python3 -m PIL --report
```

```python
code goes here
```
