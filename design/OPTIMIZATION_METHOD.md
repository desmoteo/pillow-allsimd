# How to find, make and verify the next optimization

Written 2026-11-04 from an assessment of the usual performance tooling
(sampling profilers, instruction counters, top-down analysis, compiler
remarks, static throughput analysers, SIMD-specific tools, measurement
practice) against what this fork already does. It keeps what applies, maps
it onto the fork's own scripts, and says what was left out and why. The
earlier profiles and gap analyses ([CALLGRIND_PROFILE.md](CALLGRIND_PROFILE.md),
[PROFILE_ASSESSMENT.md](PROFILE_ASSESSMENT.md),
[PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md)) are the results of this method
applied; this page is the method.

The order matters. Each stage answers one question, and the most common
waste in this work was answering a later question first: a SIMD rewrite of
a loop that turned out to be bound by memory, or by the Python layer
above it.

## 1. Where does the time go?

Three views, cheapest first:

- **Timings over the API**: `checks/simd_bench.py` (operations across
  builds, each in its own process) and the benchmark repository's 111
  cases against Pillow, OpenCV, libvips and ImageMagick. The ratio to
  OpenCV is the practical ceiling: where OpenCV is 3x faster on the same
  core, the kernel is not at the hardware's limit, whatever its counters
  say ([PERFORMANCE_GAPS.md](PERFORMANCE_GAPS.md)).
- **Exact instruction counts and call counts**: `checks/simd_callgrind.py`
  (callgrind, instrumentation switched on for one call after a warm-up),
  with `checks/simd_allocs.py` for heap allocations and
  `checks/simd_profile_report.py` to combine them with the timings into
  a report that ranks operations by the kind of work left in them.
  Callgrind is deterministic and 20 to 50 times slower, which makes it
  the tool for *algorithmic* questions: a function called once per pixel
  instead of once per row, a cost that scales with the wrong dimension, a
  temporary allocated per call. It is also the way to compare two
  versions of a function when the difference is too small for timing to
  resolve.
- **Samples on the real hardware**: `perf record` in the `pillow-simd-prof`
  image (the benchmark image plus Debian's `linux-perf` and `valgrind`)
  with `--cap-add PERFMON`, on both machines (`bench-results/prof-x86/prof.sh`
  and `bench-results/neon/prof/`). `perf annotate` crashes on the big
  transform functions; `hist.sh` maps the sampled addresses from
  `perf script` onto `objdump` instead and gives the same per-instruction
  view. Flame graphs and the GUIs (Hotspot, samply) are conveniences on
  top of the same data; nothing here needs them.

For the Python layer, `python3 -X perf` (Python 3.12 and later, the
benchmark image's Python) puts Python function names into the `perf`
stacks beside the native frames, so one profile shows where the time goes
across the boundary. The Python-overhead assessment
(`simd/python-to-c-assessment`) found no wholesale move to C worth making;
use this view when a small operation's time is suspected to be above the C.

## 2. Why is the hot loop slow?

Hardware counters decide which kind of fix can work. `perf stat` in the
`pillow-simd-prof` image, on the pinned core:

- x86 (Core Ultra, `cpu_core` PMU on CPUs 0 to 5): `cycles:u`,
  `instructions:u`, `cache-misses:u`, `branch-misses:u`, and
  `perf stat --topdown` for the top-down split (frontend-bound, bad
  speculation, backend-bound memory or core, retiring).
- arm64 (Cortex-A78AE): `cycles:u`, `inst_retired:u`, `stall_backend:u`,
  `stall_backend_mem:u`, `stall_frontend:u`. The A78 exposes no full
  top-down set, but the backend-memory stall share answers the main
  question.

The rule the split gives:

- **Memory-bound** (backend stalls on memory, instructions per cycle low
  and the bytes touched per second near the core's copy bandwidth): work
  on the data path, not the instructions. Fewer passes over the image,
  tiling so the working set fits L1 or L2, row-at-a-time instead of
  plane-at-a-time, no temporary the size of the image. SIMD will not
  help, and neither will instruction-level tuning. The fork has several
  of these on record: the PNG per-pixel unfilter and zlib-ng were dropped
  as slower, the box blur's vertical column strips gained nothing from
  wider vectors, and the nearest-neighbour rotate was a third faster in
  the SSE4 build than in the AVX2 build, which executed 7% more
  instructions and no 256-bit code, until the transforms were tiled,
  after which both took the same time: the order of the memory accesses,
  not the instructions
  ([CALLGRIND_PROFILE.md](CALLGRIND_PROFILE.md)).
- **Bad speculation** (branch misses): make the loop branchless (masks,
  selects, `min`/`max`) or partition the data so the branch is
  predictable.
- **Core-bound or retiring with a low instruction-per-cycle ceiling**:
  this is where vectorization and instruction-level work pay.

A cheap roofline, when the counters are not at hand: bytes read plus
written by the operation, divided by its time, against the one-core copy
bandwidth measured once on the same core with a `memcpy` loop of the
same size (not yet measured here; `tobytes` on a large image is a fair
stand-in). An operation at a third of that or more is memory-bound for
practical purposes.

When callgrind and the timing disagree, the disagreement is the finding:
instructions down 30% and time down 3% means the loop is waiting on
memory, and further instruction work, SIMD included, is wasted until the
access pattern changes.

Two lessons of this code base belong here, because they are counters
questions: on the A78, latency and the reorder buffer decide more than
instruction count does (the exact transforms are computed in pairs to
keep two chains in flight, `vdivq_f32` beats a Newton step, and the x86
tricks of splitting weights into 12 bits or correcting a reciprocal were
slower there), and a call per tile row instead of per row cost 14%, so a
kernel takes a whole row or image, never a fragment
([SIMD_FAST_KERNELS.md](SIMD_FAST_KERNELS.md),
[SIMD_DISPATCH_PLAN.md](SIMD_DISPATCH_PLAN.md)).

## 3. Help the compiler before replacing it

Before intrinsics, find out why the compiler did not vectorize the loop,
or vectorized it badly. GCC's remarks, on a build of the file in question:

```bash
# CFLAGS replaces Python's flags in setup.py, so give the whole set
CFLAGS="-O3 -g -fno-strict-overflow -fopt-info-vec-missed" \
  python3 -m pip install --no-deps --no-build-isolation -C simd=disable --target /tmp/b . 2>&1 | grep Convert.c
# much more detail on one loop
CFLAGS="-O3 -g -fno-strict-overflow -fdump-tree-vect-details" ...
```

The causes found here so far, with their fixes:

- **Signed index arithmetic under `-fno-strict-overflow`**, which every
  common Python passes: `in + x * 4` with an `int x` may wrap, so GCC
  vectorizes it at no optimization level. Use pointer increments, or
  fixed-size chunk loops with `restrict` (the `CONVERT_PIXELS` loops in
  `Convert.c`).
- **Trip counts and aliasing**: at `-O2` GCC vectorizes only fixed trip
  counts with `restrict` pointers (no remainder loops, no run-time alias
  checks). The published wheels are `-O3` (the `python:3.12` image),
  Debian, Ubuntu and conda Pythons are `-O2`, so a kernel that relies on
  auto-vectorization must be checked at both, and A/B timings must use
  `-O3` builds (the box blur's unrolling gains 1.46x at `-O3` and nothing
  at `-O2`).
- **Floating-point contraction**: GCC fuses `a * b + c` into `fmadd` on
  arm64 by default, so exact float results depend on the fusion pattern;
  kernels that must match a scalar double sum use `vfmaq_f64`
  explicitly, and `-ffp-contract=off` goes with `native` builds.

Checking the generated code: `objdump -d` of the built `_imaging*.so`
(what `hist.sh` does), or Compiler Explorer for one kernel across
compilers and flags. For choosing between two instruction sequences
without benchmarking noise, a static throughput analyser: `llvm-mca`
(from the `llvm` package; `-mcpu=cortex-a78` for the Orin, the laptop's
Arrow Lake cores by their LLVM name) predicts cycles per iteration and
the port pressure of a loop body, and uops.info has the per-instruction
latency and throughput tables it is built from. Neither is installed
here yet; the code-layout A/Bs so far used `variants-jetson.sh` and
accepted 5 to 10% drift between runs, which a static analysis would
settle.

**Profile-guided optimization and LTO** were listed in
[ACCELERATION_OPTIONS_ASSESSMENT.md](ACCELERATION_OPTIONS_ASSESSMENT.md)
as a free 5 to 15% on the scalar parts and never measured. A local
`-fprofile-generate` build trained on the benchmark, then
`-fprofile-use`, against the plain `-O3` build in `dbench.sh`, would
settle whether it is worth the complexity in the wheel build. The kernels
are hand-written and branch-light, so the gain, if any, is in the codecs,
drawing and the Python-facing C.

## 4. When to write SIMD, and how here

Write explicit SIMD when all three hold: the loop is hot (stage 1), it is
core-bound (stage 2), and the compiler still does not vectorize it well
after the fixes of stage 3. The cases that stay manual are shuffles and
permutations, gathers and table lookups, horizontal sums, saturating or
mixed-width integer arithmetic, and data-dependent control flow turned
into masks: most of this fork's kernels.

How a kernel is written here is settled and documented, and a new one
follows it:

- The SIMD code lives in the area's companion file (`<Area>SIMD.c`),
  compiled once per instruction-set tier into `SIMDKernels.c`, and the
  scalar file calls through the kernel table; no instruction-set guard in
  Pillow's own files ([SIMD_DISPATCH_PLAN.md](SIMD_DISPATCH_PLAN.md)).
- Every build gives the same pixels, and the exceptions are listed on the
  kernels page; a kernel that cannot be exact is a fast-mode kernel with
  a stated bound ([SIMD_FAST_KERNELS.md](SIMD_FAST_KERNELS.md)).
- Raw intrinsics per architecture, not a portable SIMD library (Highway,
  xsimd): the kernels were written and tuned per architecture, and the
  A78 and the x86 cores want different code (section 2). A portable
  layer is worth reconsidering only for a new architecture (SVE, RVV),
  where it would give a first version of every kernel at once.
- **Check the tails for over-reads before trusting the numbers.**
  Vector loops read past the end of a row in their last iteration unless
  the tail is handled; the `test-valgrind` workflow runs the suite under
  Memcheck for the SSE4 and AVX2 builds on every push, and
  `Tests/test_simd_consistency.py` runs the vector loops and their scalar
  tails on six image shapes. Valgrind 3.22 does not run AVX-512 code, so
  the `native512` build is not covered; and the NEON build has no
  Memcheck run in CI (an arm64 runner job, or
  `checks/simd_test_remote.sh` with valgrind in the container, would
  close that).
- Where a CPU at hand lacks a tier (AVX-VNNI, AVX-512), Intel SDE
  emulates it and, with `-mix`, reports the exact instruction mix of a
  run, which shows whether a loop is executing the vector width it was
  written for. Not installed here; the tiers are tested on the laptop
  (up to AVX-VNNI) and in CI.

## 5. Measure so the number means something

What the fork already does, and keeps doing:

- One pinned core (`taskset`, `--cpuset-cpus`), the performance governor
  and turbo off on the laptop; best of several interleaved rounds (`perf.py --rounds`), old and new builds in
  the same run, never two separate runs compared (they drift 5 to 10%).
- `-O3` builds in the benchmark image for every A/B
  (`bench-results/dbench.sh`, `round4/x86ab.sh`, `jetson-ab.sh`,
  `neon/variants-jetson.sh`).
- `rm -rf build` before rebuilding a kept source tree: setuptools checks
  only the including file's timestamp, so an edited kernel file is not
  rebuilt otherwise.
- Output hashes across builds with every timing (`checks/simd_hashes.py`
  and the HSV and filter checks in the A/B scripts), so a faster kernel
  that changed a pixel is caught in the same run.

Added on 2026-11-04, for a machine that is not idle: user-mode cycle and
instruction counts per run, read with `perf_event_open` around each call
(`bench-results/neon/prof/cyc.py`, run for two builds interleaved by
`bench-results/jetson-cycles.sh`, compared by `jetson-cycles-cmp.py`).
They count only while the process runs, so other load on the machine
reaches them only through cache and predictor pollution, and the minimum
over many runs removes most of that. Instruction counts are near
deterministic; cycles are the real cost. Check the machine first all the
same (`uptime`, `tegrastats` on the Orin): on 2026-11-04 the Orin was at
2% idle under other people's containers, and wall-clock A/Bs were not
worth running.

A candidate not yet built: an instruction-count regression gate in CI.
`checks/simd_callgrind.py` on a GitHub runner gives counts stable to a
fraction of a percent where timings there are noise; stored per
operation and compared on every push, it would catch a kernel that
silently fell back to scalar or gained a per-pixel call. The caveats are
the ones of stage 2: instructions are not time, a vector instruction
counts as one whatever its width, and the AVX-512 tier cannot run under
valgrind.

## Left out, and why

- Off-CPU analysis (eBPF, `offcputime`), timelines (Tracy, uftrace,
  Perfetto), causal profiling (Coz), hardware control-flow traces (Intel
  PT, CoreSight), continuous profilers: these answer questions about
  threads waiting, pipelines and latency spikes. The kernels here are
  single-threaded loops called from Python; the only waiting is the GIL,
  which the Python-overhead assessment measured directly (contention on
  small tiles).
- GPU and accelerator profiling (Nsight), energy counters: nothing here
  runs on the Orin's GPU, and the comparison basis is one CPU core.
- Portable SIMD layers and `sse2neon`-style shims: section 4.
- Google Benchmark, nanobench, hyperfine: the operations are timed from
  Python through `perf.py` and `simd_bench.py`, where the Python call is
  part of what users pay.

## The sequence, as a checklist

1. Time it across the API and against OpenCV; profile it with callgrind
   for call counts, allocations and scaling, and fix those first.
2. Read the counters on the hot loop. Memory-bound: change the data
   path. Branch-bound: remove the branch. Core-bound: go on.
3. Read the compiler's remarks; fix aliasing, index arithmetic,
   contraction; check `-O2` and `-O3`; look at the generated code.
4. Only then write the kernel, in the companion file, through the table,
   with its scalar definition; run the consistency tests and Memcheck.
5. Measure on `-O3` builds, pinned, interleaved with the old build, with
   the output hashes in the same run; under load, count cycles instead.
6. Record the result where it belongs: the kernels page's "Numbers"
   table comes from the benchmark repository (`make tables`), the design
   note from the A/B output in `bench-results/`.
