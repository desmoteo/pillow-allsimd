# Other acceleration options for pillow-simd (beyond SSE4/AVX2 and NEON)

Companion to [ARM64_SIMD_ASSESSMENT.md](ARM64_SIMD_ASSESSMENT.md). Branch `simd/master` (HEAD `788bf910f`).

Scope: wider x86 instruction sets, Intel/AMD integrated GPUs and NPUs, NVIDIA GPUs, and non-SIMD wins (threading, codecs, memory).

Basis: code review of this repo plus general knowledge of the hardware and libraries. **Nothing here was benchmarked.** Speed-up figures are expectations, not measurements. The dev machine is an Intel Core Ultra 9 285H (16 threads, AVX2/VNNI/GFNI; no AVX-512), so AVX-512 could not be tested locally.

## Summary and recommendation

| Option | Expected gain | Effort | Recommend |
|---|---|---|---|
| Runtime CPU dispatch (SSE4 / AVX2 / AVX-512) | High for AVX2 users, since wheels contain none today | Medium | **Yes, do first** |
| Multithreading (GIL already released) | Near-linear on large images | Low–medium | **Yes** |
| Faster codecs (libjpeg-turbo, zlib-ng/libdeflate, libwebp) | Often larger than any kernel work | Low (packaging) | **Yes** |
| AVX-512 / VNNI / GFNI kernels | Moderate, only on some CPUs | Medium | Selectively |
| NVIDIA GPU (CUDA/NPP/nvJPEG/DALI) | Large only for big batches | High | Only as a separate opt-in backend |
| Intel/AMD iGPU (oneAPI/SYCL, OpenCL, Vulkan) | Small to moderate | High | No |
| NPU / AMX / coprocessors | Negligible for this workload | High | No |

Main finding: the largest cheap wins are not new instruction sets. They are shipping AVX2 properly, using more cores, and faster codecs.

## 1. Gaps in the current CPU path

### 1.1 No runtime dispatch (highest value)
- [setup.py:1082](setup.py#L1082) compiles `PIL._imaging` with `-msse4` only.
- AVX2 code is guarded by `#if defined(__AVX2__)` (AlphaComposite, ColorLUT, Convert, Resample, Filter-int), but the flag is never passed, so **shipped wheels never contain the AVX2 paths**.
- Building the whole module with `-mavx2` instead would crash on pre-Haswell CPUs.
- Fix: compile the hot kernels in separate translation units with per-file flags, then pick at import or first call.
  - Use `__builtin_cpu_supports("avx2")` or CPUID; for MSVC `__cpuid`.
  - Dispatch through function pointers, or GCC `target_clones` / `__attribute__((target("avx2")))`.
  - Keep SSE4 as the baseline and add the AVX2 tier.
- Same mechanism later hosts the NEON tier and any AVX-512 tier.
- Expected: AVX2 in the existing paths typically gives about 1.3–1.8x over the SSE4 versions on resample, filters and LUT. The commit history itself notes "near the same speed" for ColorLUT and "SSE4 still 1.4x faster than previous AVX2" for Convert, so gains are kernel-dependent and must be measured.

### 1.2 Wider x86 extensions

| Extension | Use here | Notes |
|---|---|---|
| **AVX-512 (BW/VL)** | Wider Resample, Reduce, Convert, ColorLUT; native masks remove tail loops; `vpermb`/`vpshufb` zmm; `_mm512_cvtps_epi32` etc. | Available on Intel Xeon/server, Core 11th gen mobile, AMD Zen 4/5. **Absent on this Core Ultra and on most recent Intel client parts.** On Skylake-X-era chips, license-based downclocking can erase gains. Zen 4 executes it double-pumped (256-bit units), Zen 5 has full width. Best used via the AVX-512VL 256-bit forms for the mask/permute features without the width risk. |
| **AVX-VNNI / AVX512-VNNI** (`vpdpbusd`, `vpdpwssd`) | Replaces `madd_epi16` + `add_epi32` pairs in Resample, Filter-int and ColorLUT with one instruction | Present on this CPU (AVX-VNNI). Small gain (fewer uops), cheap to adopt where 8x16-bit multiply-accumulate is used. |
| **GFNI** | Byte-level bit and LUT tricks (`vgf2p8affineqb`); could do per-channel point ops/inversions | Niche. Most `Image.point` work is already table-driven. |
| **BMI2 / `pext`/`pdep`** | Bit-packing for 1-bit, 16-bit mode conversions | Niche. |
| **AVX-IFMA, AMX** | AMX is tile matrix math for int8/bf16 | Not a fit. Pillow kernels are small convolutions and per-pixel ops, not large GEMMs. Setup overhead and tile state dominate. |
| **Non-temporal stores** (`_mm_stream_si128`) | Large copies/fills/pastes | Easy micro-win for images larger than L3; harmful for small ones. |

Recommendation: do 1.1, then AVX2-tier tuning, then a small AVX-512VL/VNNI tier on the three hottest kernels (Resample horizontal and vertical, Convert RGB→L) only if profiling on target servers shows a benefit.

### 1.3 Algorithmic changes that beat wider vectors
- Resample: the existing two-pass approach is already close to optimal. Wins left are cache blocking (process in strips that fit L2) and skipping vertical passes when the box-filter reduce path suffices.
- Reduce and BoxBlur are memory-bound at large sizes; vector width will not help past that point, but fusing passes (fewer trips through memory) will.
- Convert with `-mavx2` plus compiler auto-vectorisation on simple modes (L↔P, LA) can match hand-written code with less maintenance. Measure before adding more intrinsics.

## 2. Multithreading (CPU, no new ISA)

- Many operations already release the GIL via `ImagingSectionEnter/Leave` (Convert, Geometry, Paste, Resample, Filter, Reduce, BoxBlur, ColorLUT; see grep over `src/`), so callers can already parallelise **across images** with a thread pool.
- Parallelism **within one image** is not implemented. Resample, Reduce, BoxBlur, Filter, Convert and ColorLUT are row-independent (resample's vertical pass and horizontal pass each split cleanly by rows/strips).
- Options:
  - OpenMP (`#pragma omp parallel for` over rows) behind a compile flag. Lowest effort. Needs a runtime-configurable thread count (env var, e.g. `PILLOW_THREADS`) and a minimum-size threshold (e.g. > 1 MP) to avoid overhead on thumbnails.
  - A small pthread/Win32 thread pool for tighter control and no OpenMP runtime dependency (OpenMP is awkward in manylinux/musllinux/macOS wheels; macOS clang needs libomp).
- Caveats: services that already run one worker per core will see no benefit and possible oversubscription, so default to off or auto-detect conservatively. Results must be bit-identical regardless of thread count; row partitioning guarantees this.
- Expected: roughly 3–6x on a 16-thread CPU for large images in the memory-bandwidth-bound kernels, more for compute-bound ones.

## 3. Codec and library-level wins

Decode and encode often dominate real pipelines (open + resize + save), and resize speed-ups do nothing for that.

| Component | Option | Expected effect |
|---|---|---|
| JPEG decode/encode | **libjpeg-turbo** (SIMD for x86 and NEON; usually already what wheels bundle). Also `draft()` mode (DCT scaling) when shrinking by ≥2x, often the single biggest speed-up for thumbnails | 2–4x vs plain libjpeg; draft mode avoids decoding most of the data |
| PNG/zlib | **zlib-ng** or **libdeflate** | 2–4x deflate/inflate. Upstream wheels already moved toward zlib-ng. Check what this repo's `wheels-dependencies.sh` ships |
| WebP, AVIF | libwebp (SIMD, multithreaded encode), libavif/dav1d/rav1e/SVT-AV1 | Already SIMD and threaded in their libraries; ensure thread options are enabled |
| Allocator | jemalloc/mimalloc; Pillow's block allocator and `ImagingNew` cost matters for many small images | Small |
| Memory layout | Avoid RGBA↔RGB and mode round trips in pipelines | Free; documentation-level guidance |

Verify with `python -c "from PIL import features; features.pilinfo()"` which backends the target build actually uses.

## 4. Intel and AMD integrated GPUs, NPUs

Not recommended.

- **APIs:** oneAPI/SYCL (Intel), OpenCL, Vulkan compute, Level Zero, ROCm/HIP (AMD, mostly discrete cards), DirectML/DirectCompute (Windows).
- **Why the payoff is poor for Pillow:**
  - Integrated GPUs share system memory, which removes the PCIe cost, but each Pillow operation is a short, bandwidth-bound pass. Kernel launch, queue and synchronisation overhead is comparable to the CPU runtime for typical images (e.g. a 1–10 MP resize takes milliseconds on the CPU already).
  - The CPU SIMD path is close to memory bandwidth for most kernels. An iGPU shares that same memory bandwidth, so the ceiling is similar.
  - Driver, runtime and ABI variety (Mesa, Intel compute runtime, ROCm, Windows) greatly increases packaging and test cost.
- **NPUs** (Intel AI Boost, AMD XDNA): fixed-function int8/bf16 neural ops, no general image kernels and no stable low-level API for this use. No fit.
- Possible exception: video-frame pipelines on Intel (VA-API/QSV) that never leave GPU memory, which is outside Pillow's scope.

## 5. NVIDIA GPUs

Worth considering only as a **separate, opt-in backend**, not as a transparent replacement for `Image` operations.

### Where it pays off
- **Batches of large images** (thousands of photos, 10+ MP, server-side). Decode on GPU with **nvJPEG** (hardware engine on A100/H100), resize with **NPP** (`nppiResize`) or **DALI**, or **OpenCV CUDA**; keep data on the GPU through the whole chain.
- Pipelines that feed an ML model on the same GPU, where the data stays resident and no round trip is needed.
- Throughput numbers for nvJPEG + DALI are typically several thousand images/s on one data-centre GPU. A multi-core CPU with libjpeg-turbo and pillow-simd is in the hundreds per core per second for thumbnails. (Order-of-magnitude, unverified here.)

### Where it does not
- Single images or small ones: PCIe transfer (about 12–25 GB/s practical) plus launch latency is slower than the whole CPU operation.
- Per-call GPU use inside the existing `Image` API: each `resize()` would need upload, kernel, download. Pillow's mutable-image, mode-by-mode, pure-in-CPU-memory model (`ImagingMemoryInstance`) does not fit a device-resident image.
- Operations like `Image.filter`, `convert` and `point` are too cheap relative to transfer cost.

### Integration options
1. **Do not integrate; document interop.** Recommend DALI/nvJPEG/CuPy or `torchvision.io` + `torchvision.transforms.v2` for GPU pipelines, and keep pillow-simd for CPU. Lowest cost, and usually the right answer.
2. **Optional extension module** (e.g. `PIL._imaging_cuda`) exposing explicit batch APIs (`resize_batch(list_of_images, size, filter)`), compiled only when the CUDA toolkit is found.
   - Costs: build matrix per CUDA version and per compute capability, large wheels (CUDA runtime redistribution), driver version compatibility, and a test environment with a GPU.
   - Must keep CPU and GPU results consistent: Pillow's fixed-point resample coefficients (see `Resample.c`) would need a GPU kernel that reproduces them exactly, or the tolerance must be agreed.
3. **Pluggable device-backend interface** where `Image.resize` and similar can dispatch to a registered backend. This is a large API design change and is incompatible with pillow-simd's role as a drop-in replacement.

Recommendation: option 1 now. Reconsider option 2 only if there is a concrete batch-processing workload with a GPU already in the deployment.

## 6. Other ideas

- **Link-time/profile-guided optimisation** (`-O3`, PGO, LTO) on the wheels: free, typically 5–15% across scalar parts.
- **Compiler auto-vectorisation** for scalar remnants (`Point`, `Paste`, `Copy`, `Histo`, `Fill`, `Geometry` transforms): add `-O3 -ftree-vectorize` and check with `-fopt-info-vec`. Many kernels may vectorise without intrinsics, and that carries over to NEON for free.
- **Geometry** (`rotate`, `transform`, `affine`): `Geometry.c` has the most GIL-release sections (11) and is entirely scalar here. Bilinear and bicubic transforms are good SIMD and threading candidates (gather-heavy; AVX2 gathers are only so-so, so prefer row-wise precomputed coordinate vectors).
- **Histogram / Point / LUT**: `Histo.c` and `Point.c` are scalar; 8-bit LUT application with `vpshufb`-based nibble lookup or `vpgatherdd` is possible but gains are modest because the table is tiny and loops are already memory-bound.
- **SIMD for 16-bit / 32-bit / float modes (I;16, I, F)**: current SIMD work targets 8-bit modes only. Check usage of those modes before investing.

## 7. Suggested roadmap

1. Baseline benchmarks on target hardware using `benchmark.yml` (pillow-simd vs upstream Pillow, per operation and per size).
2. Runtime dispatch and AVX2 in wheels (section 1.1). This is also the prerequisite for NEON and AVX-512 tiers.
3. Verify codec stack (turbo, zlib-ng, `draft()` usage), as these often matter more for end-to-end time.
4. Optional intra-image threading with a size threshold and an env switch.
5. Auto-vectorisation flags and scalar kernels (Geometry, Point, Histo).
6. AVX-512VL/VNNI tier and GPU backend only if a concrete workload justifies it.

## Caveats

- No performance measurements were taken. Gains listed are ranges from general experience with these technologies, to be confirmed by benchmarking.
- Hardware facts (AVX-512 availability, Zen 4 double-pumping, NPU capabilities) reflect my knowledge and may be outdated for the newest CPU generations; verify against current vendor documentation.
- The local CPU only exposes AVX2/VNNI/GFNI, so AVX-512 code paths could not be tested here.
