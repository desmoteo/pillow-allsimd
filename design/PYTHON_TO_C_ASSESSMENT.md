# Why so much of Pillow-AllSIMD is still Python, and what moving more of it to C would gain

As of `simd/pillow-13` at ac80889a4 (2026-11-02). Measured on the x86 laptop
(the build of the published runs: the `pillow-simd-perf` Docker image,
Python 3.12, GCC 14, -O3, `-C simd=avx2`) against the Pillow 13.0.0 wheel in
the same image. The scripts are in
[checks/python_overhead/](../checks/python_overhead/); the raw outputs are in
`bench-results/py2c/` (local).

## Summary

- **About 30% of the code is Python** (26,300 lines in 97 files, against
  about 60,700 lines of C). Python holds the format parsers (55% of the
  Python), the `Image` class and its method wrappers, drawing and text
  layout, and the small helpers in `ImageOps`, `ImageStat` and similar.
  Every pixel loop is already C. The fork added 16,000 lines of C and
  changed only 313 lines of Python.
- **For images of about a megapixel or more, Python costs almost nothing:**
  0.3–2% of the time for resize, convert, blur, transpose and copy, and 1–3%
  for decoding and encoding. Converting all the Python to C would gain under
  5% there, and that is the published benchmark.
- **For small images and per-object APIs, Python is most of the time:**
  - 55–90% of a crop, convert, copy or transpose at 64×64 to 256×256;
  - 50% of an ImageDraw rectangle and 73% of a `text()` call;
  - 90% of opening a file (the header);
  - nearly all of TIFF directory handling, `ImageStat` and `autocontrast`
    on small images, and `point()` with a list.
- **The SIMD work made this worse (Amdahl).** The C got 2–15x faster and the
  Python didn't, so Python's share grew:

  | Workload | Python share, Pillow 13 | Python share, AllSIMD |
  |---|---|---|
  | `text()` | 15% | 73% |
  | 64×64 tiles: crop, resize, convert | 37% | 69% |
  | ML loader (JPEG 256 with EXIF, orientation, resize to 224, numpy) | 20% | 40% |
  | 64×64 bilinear resize | 8% | 30% |

  (cProfile shares, except the resize row, which is a direct measurement.
  cProfile overstates Python by up to 1.3x; see the Method section.)
- **Python limits threads.** A thread pool running the 64×64 tile workload
  on four cores gets *0.47x* the one-thread throughput on AllSIMD, against
  1.87x on Pillow 13. With four threads, AllSIMD is slower than Pillow 13 on
  that workload. This is GIL contention over the Python part. The C calls
  releasing the GIL are not the cause: a build that never releases it
  scales no better.
- **What moving Python to C would gain:**
  - large images: under 5%;
  - single-threaded small-image pipelines: 1.3–2x at best;
  - single small calls: 2–5x;
  - TIFF directories and `text()`: up to about 3x.

  A complete rewrite of the `Image` class and the plugins would cost tens of
  thousands of lines of fork-only C. Every Pillow merge would conflict with
  it, and it would put C parsers in front of untrusted input. The fork
  avoids both of those today.
- **Recommendation:** don't convert the Python layer wholesale. Do three
  smaller things, in this order:
  1. A few Python fixes, each also suitable for upstream. Examples: TIFF
     parses each page's directory up to three times; `point()` rounds a list
     that is already integers; `open()` probes up to 38 formats in
     alphabetical order.
  2. A few small C entry points where one C call can replace a Python loop:
     `ImageStat`, `point()` tables, batch drawing, opening a WebP still.
  3. Two C kernels that the measurements exposed: the histogram and the
     table-lookup `point()`. Each takes 1.1–1.4 ms on a 1080p RGB image and
     sets the time of `autocontrast`, `equalize` and `ImageStat`.

  For threaded small-image workloads, the remedy is outside Pillow:
  processes, or free-threaded Python (untested here).

## 1. What is in Python

Non-blank, non-comment lines in `src/PIL/*.py` (26,303 in total):

| Group | Lines | Share | What it does |
|---|---|---|---|
| Format plugins (61 files) | 14,339 | 55% | `*ImagePlugin.py`, `PdfParser`, `TiffTags`, `ExifTags`, font and palette file readers. They parse headers and metadata, set up `tile` lists for the C decoders and write headers when saving. |
| `Image.py` | 3,492 | 13% | The `Image` class: argument checks, mode logic, `info` and palette bookkeeping, then a call into `ImagingCore`. |
| Image helpers (12 files) | 2,679 | 10% | `ImageOps`, `ImageFilter`, `ImageStat`, `ImageChops`, `ImageEnhance`, `ImageMath`, `ImageMorph`, `ImagePalette`, `ImageColor`, `ImageSequence`... Mostly compositions of C calls, plus small Python loops over histograms and tables. |
| Drawing and text (5 files) | 2,598 | 10% | `ImageDraw` (ink resolution, coordinates, compound shapes), `ImageFont` and `ImageText` (layout, anchors, multiline text, masks). |
| GUI and OS integration | 1,267 | 5% | Tk, Qt, Windows, screen grabbing, viewers, `features`. |
| `ImageCms` | 805 | 3% | Wrapper over `_imagingcms`. |
| `ImageFile.py` | 693 | 3% | The decode and encode loops around the C codecs, and `Parser`. |
| Other | 430 | 2% | `_util`, `_typing`, `_deprecate`, `PSDraw`, `GdImageFile`. |

All pixel work is already C: `libImaging` (85 files, 44,700 lines with
headers) and the bindings (`_imaging.c` 5,300 lines, `decode.c`, `encode.c`,
`_imagingft.c`, `_imagingcms.c`, `_webp.c`, `_avif.c`, ...). The codecs are
libjpeg-turbo, zlib, libtiff, libwebp, libavif and OpenJPEG.

The fork's own changes since 13.0.0 are almost all C:

| | Files | Lines added | Lines removed |
|---|---|---|---|
| `src/libImaging` | 47 | 15,986 | 925 |
| `_imaging.c`, `_webp.c` | 2 | 278 | 11 |
| Python (`Image`, `ImageDraw`, `ImageEnhance`, `ImageFilter`, `JpegImagePlugin`, `WebPImagePlugin`, `features`, `_version`) | 8 | 313 | 25 |

## 2. Why it is still Python

There are four reasons, and none of them is an oversight.

1. **Pillow's design.** Pillow keeps the C at the pixel level, and Pillow-SIMD
   followed it. Format logic (chunk walking, tag tables, metadata, the
   choice of decoder) is Python. The C sees a `tile` list: a decoder name,
   a region, an offset and arguments. That layer:
   - lets any Python code register a plugin (`Image.register_open`,
     `ImageFile.ImageFile` subclasses);
   - keeps metadata as Python objects (`info`, `getexif()`, `tag_v2`,
     `text`, `applist`);
   - lets the format code change without a rebuild.

   Much of it is public API. `ImageFile` and `TiffImagePlugin.ImageFileDirectory_v2`
   are subclassed and used directly, so moving them to C would also mean
   keeping their Python surface.
2. **Memory safety on untrusted input.** Parsing headers is the riskiest
   code in an image library. In Python, a malformed length raises an
   exception. In C, it can overflow a buffer. Pillow's C has had overflows
   of that kind (CVE-2024-28219 in `_imagingcms.c`, and earlier ones in the
   FLI, PCX, SGI and TIFF decoders). The Python parsers' problems have mostly
   been denial of service and `ImageMath` evaluation (CVE-2023-50447).
   Moving more parsing into C trades a Python exception for a possible
   overflow in the fork's own code, which only one person reviews.
3. **Merge cost.** The fork stays mergeable because its 16,000 lines live in
   new kernel files and behind dispatch points, and the Python is nearly
   Pillow's. Pillow changes its Python layer much more often than its
   kernels: almost every release changes several plugins and `Image.py`.
   Each converted plugin would be rewritten by hand at every merge.
4. **It doesn't matter where the fork is measured.** The fork targets pixel
   throughput, and the benchmark uses 1–2 megapixel images. There, Python
   is under 2% (section 4.1).

## 3. Method

- **Builds:**
  - AllSIMD: HEAD built in `pillow-simd-perf` with `dbench.sh build py2c-avx2
    avx2` (-O3).
  - Pillow 13: the 13.0.0 wheel in the same image.
  - Both run on CPU 2 (a P-core); the thread runs use CPUs 0, 1, 3 and 4.
- **Two ways of finding the Python share:**
  - **Direct:** time the `Image` method, then time the `ImagingCore` call it
    makes, with the same arguments. Each pair is interleaved three times and
    the best of each is kept. The difference is the Python layer exactly.
    It is used for single method calls (section 4.1).
  - **cProfile:** the share of time in Pillow's C calls, in other C (zlib,
    `BytesIO`) and in Python. It is used for anything that makes many calls.
    cProfile adds a cost to every Python call, so it overstates Python. The
    two methods were run on the same crops to calibrate: at 64×64, direct
    87% against cProfile 92%; at 256×256, 55% against 74%; at 1080p, 6%
    against 8%. Read a cProfile share of 40–70% as roughly 30–55% real.
- **Noise:** turbo was on and the power profile was "balanced" (no root
  access to change them). Absolute times jumped between two clock states
  about 2x apart: GIF `n_frames` is identical Python in both builds and gave
  162 µs and 356 µs. Each build was run three times, alternating, keeping
  the best time per case. **The shares are stable to about one point between
  runs; the absolute times and the AllSIMD-vs-Pillow-13 ratios are
  indicative only.** The conclusions rest on the shares and on the
  interleaved direct pairs.
- **Not measured:**
  - arm64: the Jetson was unreachable that day. Python is slower per call on
    the A78 and the NEON gains are smaller than AVX2's, so the shares should
    sit between this run's two columns.
  - Free-threaded Python.
  - Python 3.13 and 3.14, whose interpreter is faster per call than 3.12's.

## 4. Results

### 4.1 Per-call overhead of `Image` methods

Python layer, direct measurement, AllSIMD; the full times are in
`bench-results/py2c/merged.txt`:

| Method | 64×64 | Python | 256×256 | Python | 1920×1080 | Python |
|---|---|---|---|---|---|---|
| `resize` to half, bilinear | 4.5 µs | 30% | 30 µs | 5% | 760 µs | 1% |
| `convert("L")` | 1.9 µs | 57% | 12 µs | 9% | 336 µs | 1% |
| `crop` (a quarter) | 1.8 µs | 87% | 3.1 µs | 55% | 79 µs | 6% |
| `transpose` | 1.4 µs | 64% | 7.4 µs | 12% | 366 µs | 1% |
| `copy` | 1.2 µs | 77% | 6.1 µs | ~15% | 360 µs | 1% |
| `Image.new` | 1.0 µs | 64% | 4.6 µs | 12% | 186 µs | 1% |
| `GaussianBlur(2)` | 26 µs | ~1% | 133 µs | 2% | 2.4 ms | 0% |
| `point(list of 768)` | 31 µs | **89%** | 36 µs | **77%** | 388 µs | 8% |

So each `Image` method costs **0.6–1.7 µs of Python** whatever the image
size: argument checks, `load()`, `_new()`, `Image.__init__`, `info` copy and
palette handling. A method written in C with `METH_FASTCALL` would cost
about 0.1–0.3 µs. The gain is real only while the C work stays under about
10 µs per call: up to 64×64 for most operations, and up to 256×256 for crop
and copy.

`point()` is the exception, with 27 µs of Python. `Image.point` runs
`flatLut = [round(i) for i in flatLut]` over the whole table, even when every
entry is already an int. That is a Python fix (section 5, P1).

Pillow 13 has the same Python cost, but its C is slower, so its shares are
lower. At 64×64 the resize is 8% Python on Pillow 13 and 30% on AllSIMD;
the convert is 35% and 57%.

### 4.2 Helpers written in Python

| Case | Time | Python (cProfile) | Where the time goes |
|---|---|---|---|
| `ImageStat.Stat(im).stddev`, 64×64 RGB | 78 µs | 93% | Python loops over the 768-entry histogram, for each of `sum`, `sum2`, `mean`, `var`, `stddev` |
| same, 1920×1080 | 3.1 ms | 6% | `histogram()` in C: 1.34 ms |
| `ImageOps.autocontrast`, 64×64 | 80 µs | 96% | Python loops building the table |
| `autocontrast`/`equalize`, 1920×1080 | 2.5 ms | 9% | `histogram()` 1.4 ms and the table `point()` 1.1 ms, both C |
| `point(lambda)`, 1080p L | 88 µs | 46% | 256 Python calls, then the C table lookup |
| `getpixel` ×1000 | 249 µs | 85% | per-call Python; `PixelAccess[...]` is 5x faster (50 µs) |
| `np.asarray(im)`, 64×64 | 3.0 µs | 89% | building `__array_interface__` |
| `np.asarray(im)`, 1080p | 0.4–0.7 ms | 4% | one copy (`tobytes`); a zero-copy export would need contiguous image memory |
| `ImageOps.fit`, `thumbnail`, `ImageChops` at 1080p | 0.4–0.8 ms | 2–4% | C |

Two of these are C kernel items, not Python items. They turned up because
the Python share was small:

- `histogram()` on 1080p RGB takes 1.34 ms (0.2 ns per byte). It is scalar,
  with one table, and each increment waits for the previous one to the same
  bin. Four or eight sub-histograms, or SIMD, would make it 2–3x faster.
  `ImageStat`, `autocontrast` and `equalize` would follow, since it is half
  their time.
- The table-lookup `point()` takes about 1.1 ms on 1080p RGB for any table
  that isn't a straight line. That covers `autocontrast`'s clamped table and
  a random one. It is a gather; AVX2 `vpgatherdd` or a shuffle on 16-entry
  pieces could help. Tables that are a straight line, such as identity and
  inversion, already take a fast path (0.36 ms), and the `point(list)` row
  in section 4.1 used one.

### 4.3 Drawing and text

| Case (canvas 512×512) | Time | Python | Notes |
|---|---|---|---|
| 1,000 `rectangle(fill=(r, g, b))` | 279 µs | 49% (direct, against `draw_rectangle`) | 0.14 µs of Python per call: `_getink`, `draw_ink`, the `fill`/`outline` logic |
| 1,000 `rectangle(fill="red")` | 310 µs | 54% (direct) | plus `ImageColor.getcolor` (cached by `lru_cache`) |
| 1,000 `point` | 228 µs | 72% (cProfile) | |
| 1,000 `line(width=1)` | 977 µs | 40% (cProfile) | |
| 100 `text()`, default font | 412 µs | **73%** (cProfile) | 4.1 µs per call. FreeType `render` is about 1 µs thanks to the fork's glyph cache; the rest is `ImageDraw.text`, `draw_text`, `ImageText._split`, `getmask2` and the anchor and mode logic |

On Pillow 13 the same 100 `text()` calls take 4.3–9.5 ms and are 15% Python.
The glyph cache cut the C by about 20x, and Python went from 15% to 73%.
Here a C fast path would pay. It would cover the common case (one line, no
anchor, no stroke, no features, RGB or L ink), going straight from the
string to `font.render` and `draw_bitmap`, and could make `text()` about 2–3x
faster.

For rectangles, points and lines, Python is only 0.1–0.2 µs per call. A C
`Draw` would save at most half of a small number. The bigger gain is a batch
call taking many shapes at once, such as `draw.rectangles(boxes, fill)`.
That is new API, though, and belongs in Pillow first.

### 4.4 Opening, decoding and saving

| Case | Time | Python (cProfile) |
|---|---|---|
| open, header only: PNG / GIF / BMP / PPM | 4.4–6.2 µs | 93–97% |
| open, header only: JPEG | 10.6 µs (18 µs with EXIF) | 91% |
| open, header only: TIFF (one page) | 44 µs | 96% |
| open, header only: WebP | 30–70 µs | 86% |
| open + load JPEG 64×64 / 256×256 / 1080p | 63 µs / 272 µs / 4.3 ms | 58% / 17% / 2% |
| open + load + `exif_transpose`, JPEG 256×256 | 203 µs | 41% |
| open + load PNG 64×64 / 1080p | 115 µs / 24–50 ms | 37% / 1% |
| open + load TIFF 256×256 (uncompressed) | 89 µs | 89% |
| save JPEG 64×64 / 1080p | 16 µs / 3.6 ms | 60% / 1% |
| save TIFF 64×64 / 1080p | 43 µs / 1.1 ms | 98% / 26% |
| save GIF 64×64 | 50 µs | 79% |
| save BMP 64×64 | 5.7 µs | 89% |
| save PNG / WebP, any size | | under 14% (zlib and libwebp setup dominate) |

- **Probing** costs as much as parsing. After `Image.init()`, the
  plugin list `Image.ID` is in alphabetical order. `open()` checks the
  plugins in that order, and constructing `ImageFile` for each candidate
  costs time. Separate measurement (CPU 4):

  | Format | Position | `open()` | With `formats=[...]` |
  |---|---|---|---|
  | JPEG | 24 | 20.4 µs | 9.7 µs |
  | TIFF | 27 | 56.5 µs | 41.1 µs |
  | WebP | 38 | 72.1 µs | 54.5 µs |
  | PNG | 17 | 7.7 µs | 4.6 µs |

  Before `init()`, only the five preloaded plugins are tried, which is why
  short scripts don't notice. Trying the plugin whose `_accept` matches the
  first bytes before the others is a Python change, and also suits Pillow.
- **WebP:** with probing removed, opening still costs 54 µs. Most of it is
  C: `WebPAnimDecoderNew`, which allocates and clears a canvas the size of
  the image at `open()`. For a single-frame file, `WebPGetFeatures` gives
  the size and mode without that, and the fork already decodes stills with
  `decode_still`. The animation decoder could be created only for animated
  files. This is a C change in the fork's own code (section 5, C4).
- **JPEG headers are parsed twice:** once in Python (`JpegImagePlugin._open`
  walks the markers, 38 reads per file, for mode, size, `info`, EXIF, ICC,
  Adobe and MPO data) and again by libjpeg in the decoder. A C marker
  walker would cut the Python part (about 8 µs) to 1–2 µs. It would still
  have to build the same `info` and `applist` objects, and it would be fork
  C parsing untrusted data. For a 256×256 JPEG the gain is about 3%.
- **Small PNGs:** at 64×64, `load` is 115 µs, of which about 70 µs is the
  zlib decoder's setup and inflate in C. Python is about a third.
- **Large files: decoding is all C.** PNG IDAT chunks of 8 KiB, 64 KiB or
  1 MiB load at the same speed (checked separately: 23.4, 23.7 and 23.0 ms).
  `ImageFile.load`'s Python loop makes 47–374 `decode` calls, which costs
  nothing next to the inflate.

### 4.5 Multi-frame files

| Case | Time | Python (cProfile) |
|---|---|---|
| TIFF, 20 pages 256×256: open and load every page | 1.6 ms | **89%** |
| TIFF, save 20 pages | 1.5 ms | 87% |
| GIF 245×245, 42 frames: load every frame | 6.6 ms | 18% |
| GIF: `n_frames` | 160–330 µs | 73% |
| GIF: save 42 RGB frames | 215 ms | 6% (quantization is C) |

Multi-page TIFF is the worst case for Python. The profile shows three
problems; the first two are fixable in Python:

- **Each page's directory is parsed three times.** 2,000 page loads ran
  `ImageFileDirectory_v2.load` 6,000 times:
  - twice in `_seek`: in the loop that finds the frame's offset, then again
    for the page itself;
  - a third time from `Image.Exif.load_from_fp`, because
    `TiffImageFile.load` calls `ImageOps.exif_transpose`, which reads the
    EXIF data, and in a TIFF that data is the same directory.

  Each page then also gets a `tag`/`ifd` copy through
  `ImageFileDirectory_v1.from_v2`.
- **`logger.debug` is called 42 times per page** (84,300 calls for 2,000
  pages) with logging off. Each call is a Python call plus a level check.
- `_setitem`, `__getitem__`, `_unpack` and `TiffTags.lookup` handle every tag
  with several Python calls.

Caching the parsed directory for each frame offset, and guarding the debug
lines (one `isEnabledFor` per directory), should make page iteration about
2x faster. That estimate comes from the profile; no prototype was made. A C
directory parser would make it 5–10x faster still. But it has to rebuild
`tag_v2` as Python objects, and `ImageFileDirectory_v2` is public API used
for EXIF everywhere (`getexif()` reuses it). That is a large, fork-only C
parser of untrusted input.

### 4.6 Pipelines

| Pipeline | AllSIMD | Pillow 13 | Python, AllSIMD / Pillow 13 (cProfile) |
|---|---|---|---|
| ML loader: JPEG 256×256 with EXIF, `exif_transpose`, convert, resize to 224, `np.asarray` | 282 µs | 679 µs | 40% / 20% |
| Thumbnail: JPEG 1080p, `thumbnail(256)`, save JPEG | 4.5 ms | 6.1 ms | 4% / 3% |
| Thumbnail: JPEG 256×256, `thumbnail(64)`, save JPEG | 305 µs | 314 µs | 36% / 30% |
| 464 tiles of 64×64 from 1080p: crop, resize to 32, convert to L | 4.9 ms | 12.8 ms | 69% / 37% |

The upper bound for converting all the Python in these pipelines to C,
correcting cProfile's overstatement:

| Pipeline | Real Python share | Bound |
|---|---|---|
| ML loader | about 30% | 1.4x |
| Tiles | about 55% | 2.2x |
| Small thumbnail | about 28% | 1.4x |
| Large thumbnail | | 1.04x |

Converting only the `Image` methods and the JPEG header would capture about
half of each bound.

### 4.7 Threads

Thread pools on four P-cores, Python 3.12 with the GIL. Throughput relative
to one thread:

| Workload | AllSIMD 2 threads | AllSIMD 4 threads | Pillow 13 2 threads | Pillow 13 4 threads |
|---|---|---|---|---|
| ML loader | 1.38x | 2.25x | 1.54x | 2.76x |
| Thumbnail 1080p | 1.90x | 3.62x | 2.28x | 4.38x |
| Tiles 64×64 | **0.67x** | **0.47x** | 1.76x | 1.87x |

By tile size (`tilesize.py`, tiles per second, one thread against four):

| Tile | AllSIMD | Pillow 13 | AllSIMD, GIL never released |
|---|---|---|---|
| 32 | 0.62x | 0.80x | 0.43x |
| 64 | 0.59x | 2.00x | 0.46x |
| 128 | 1.05x | 3.70x | 0.61x |
| 256 | 2.54x | 3.87x | 0.48x |

The C sections release the GIL (`ImagingSectionEnter`). The Python between
them holds it. When the C part of each call is short, four threads spend
their time passing the GIL around. Throughput then falls below one thread's.

AllSIMD's faster kernels moved the point where threads help from about
32 px tiles to about 128 px. At 64 px and four threads, AllSIMD does 62,000
tiles/s against Pillow 13's 70,000.

The last column rules out a cheap fix. A build whose `ImagingSectionEnter`
keeps the GIL scales *worse*. Releasing the GIL around short C calls is not
what costs; the contention is over the Python. So threading small-image work
improves only if there is less Python per image, or no GIL:

- free-threaded CPython 3.13t or 3.14t, which Pillow supports;
- processes, which most ML data loaders already use.

## 5. Candidates

Ranked by gain for the cost, in four tiers:

- **P**: Python changes. They don't diverge from Pillow, and each could go
  upstream.
- **C**: small C entry points in the fork.
- **K**: C kernels.
- **R**: rewrites.

| # | Change | Gain (measured or estimated) | Cost and risk |
|---|---|---|---|
| P1 | `Image.point`: skip `[round(i) for i in lut]` when every entry is already an int, or let the C round | `point(list)` at 64×64 and 256×256 from 31–36 µs to about 5–10 µs (3–6x); 7% at 1080p | A few lines. Also for Pillow. |
| P2 | TIFF: cache the parsed directory per frame offset, and let the EXIF reload reuse it; one `isEnabledFor` per directory for the debug logging | Multi-page iteration and EXIF access about 1.5–2x (estimated from the profile) | Small, but TIFF seeking has subtle cases (`_frame_pos`, `_n_frames`, `seek(0)` after the end). Upstream. |
| P3 | `Image.open`: try the plugins whose `_accept` matches the first bytes before constructing the others | JPEG open 20→10 µs, TIFF 57→41 µs, WebP 72→55 µs after `init()` | Small. Changes the order in which a file matching two formats is tried, so ask Pillow first. |
| P4 | `ImageStat` and `ImageOps.autocontrast`/`equalize`: compute the per-band sums with `sum(map(...))` and precomputed squares instead of nested loops | Small images: 78 µs to about 20–30 µs | Small; numerically identical if done carefully. |
| C1 | A C `ImagingCore.stat()`: count, sum, sum of squares, extrema per band in one pass over the histogram, or over the pixels for F and I | Small images 78→about 5 µs; with K1, 1080p 3.1→about 0.7 ms | About 150 lines. A fork-only API under a Python wrapper that keeps the `Stat` class. |
| C2 | A `text()` fast path: one line, no anchor, stroke or features, RGB or L ink, straight to `render` and `draw_bitmap` | `text()` about 2–3x (4.1 µs to about 1.5–2 µs per call) | Medium: `ImageDraw.text` has many cases, and the fast path must give the same pixels (the tests compare them). The Python stays for the rest. |
| C3 | Batch drawing (`rectangles`, `points`, `lines` from a sequence) | 1,000 rectangles 279→about 150 µs; more for points | New API. Better proposed to Pillow than added only here. |
| C4 | WebP: create `WebPAnimDecoder` only for animated files; get a still's size and mode from `WebPGetFeatures` | WebP open 55→about 10 µs | Small, and in the fork's own WebP still path (`decode_still`). |
| K1 | SIMD or multi-table `histogram()` | 1080p RGB 1.34 ms to about 0.5 ms. `ImageStat`, `autocontrast`, `equalize` about 1.5–2x at 1080p | A normal fork kernel: matches scalar exactly and is easy to test. |
| K2 | Faster table `point()` for 8-bit images with general tables (gathers, or shuffles on 16-entry pieces) | 1080p RGB 1.1 ms, maybe to 0.5 ms; straight-line tables are already 0.36 ms | A fork kernel, as above. |
| R1 | The `Image` class methods in C (`crop`, `convert`, `copy`, `resize`, `transpose`, `_new`) | Small-image calls 2–5x; tile pipeline up to about 1.5x | Large. `Image.py` changes in every Pillow release, and subclasses override `load`, `_new` and others. Not recommended. |
| R2 | JPEG/PNG/TIFF header parsing in C | About 8 µs per JPEG open, 30–40 µs per TIFF page | Fork-only C parsers of untrusted input, which must produce the same Python metadata. Not recommended; P2 and P3 get most of it in Python. |
| R3 | `ImageFile.load`/`_save` loops in C | Under 2% except for tiny files | Not worth it. |

Taken together:

- **The P items and C4** are a few hundred lines of Python and C. They fix
  the worst cases (TIFF directories, `point(list)`, open probing, WebP open)
  without diverging from Pillow.
- **K1 and K2** are kernel work of the usual kind. They help large images,
  where the fork is measured.
- **C1 to C3** are optional. C2 is the one with a clear user-facing gain.
- **R1 to R3** wouldn't be worth what they cost.

## 6. If the Python layer were all C

Suppose, as a thought experiment, that every Python line between the user's
call and the C kernels cost nothing:

| Workload | Gain |
|---|---|
| Benchmark-sized images (1–2 MP): resize, convert, blur, decode, encode | 1.00–1.05x |
| Large thumbnails, ML loaders on full-size photos | 1.03–1.1x |
| ML loader on 256 px JPEGs | ≤ 1.4x |
| 64 px tiles, small thumbnails, sprite sheets | 1.4–2.2x |
| Text-heavy rendering | ≤ 3x |
| Multi-page TIFF, EXIF-heavy metadata work | ≤ 5–10x |
| Threaded small-image work (GIL build) | 2–4x more throughput, because the Python under the GIL is what doesn't scale |

The last row is the strongest case for C, but free-threaded Python and
process pools serve it without touching Pillow. The other large gains are in
narrow areas (TIFF, text) that targeted changes already cover. Converting the
rest would cost a large rewrite, a much harder merge for every Pillow
release, and more C on the untrusted-input path. All of that for gains that
don't show at the image sizes the fork is built for.

## 7. Reproducing

From the repository root, with a build on `PYTHONPATH`:

```bash
python3 checks/python_overhead/overhead.py out.json   # sections 4.1-4.6
python3 checks/python_overhead/threads.py             # 4.7, thread pools
python3 checks/python_overhead/tilesize.py            # 4.7, by tile size
python3 checks/python_overhead/merge.py DIR           # best of three runs per build
```

Run them pinned to one P-core (four for the thread scripts), with turbo off
if possible, and run each build several times, alternating. On this laptop
the clock alone moved absolute times 2x. The "GIL never released" column
came from a build with `ImagingSectionEnter` and `ImagingSectionLeave` in
`_imaging.c` made into no-ops. That build is local only
(`bench-results/dbuilds/py2c-nogil`).
