# Plan: the benchmark tables generated from the data

Status: done 2026-11-04, for finding 7 of
[REVIEW_2026-11.md](REVIEW_2026-11.md). What came out of it is at the end.

## The problem

The speed-up chart of the PyPI description is drawn by
`checks/simd_speedups_svg.py` from the result files of the benchmark
repository (pillow-allsimd-perf, `docs/data/*.json`), so it can't drift.
The tables and the figures in prose are typed by hand from the same data:

| Where | What | Figures |
|---|---|---|
| `PyPI.rst` | the 16-row benchmark table (Pillow 13's time, ours, the speed-up, two machines); the geometric means in the "Fast" and "Exact when you ask" bullets; the count of operations where OpenCV is slower | 18 + 7 |
| `README.md` | the fast-mode paragraph ("3.3x at radius 1") | 1 |
| `docs/reference/fast_mode.rst` | the speed-up column of the fast-kernel table (several cases per row) | 11 |
| `docs/allsimd-kernels.rst` | nine "Numbers" tables (resize, blur, filters, rank, conversions, compositing, geometry, drawing, codecs), Appendix A's geometric means, and figures in the prose of the sections | about 470 |

Every refresh of the benchmark (a new run after a kernel change) means
retyping them, and the kernels page was written by hand from the data
the same week it was published. The "What was tried" and "Where OpenCV
is faster" sections quote measurements that are not in the data files
(commit messages, one-off runs); those stay as prose.

## The design

One script, `checks/simd_tables.py`, next to the SVG script, sharing its
loader (`checks/simd_results.py`: the data files, the baseline, `best`
ratios, the geometric mean over the cases both variants have, as
`docs/app.js` computes it on the benchmark page). It rewrites marked
regions in place:

```
.. generated: pypi-benchmarks (checks/simd_tables.py, pillow-allsimd-perf@0d7d352)

<the table>

.. generated-end
```

in RST, and `<!-- generated: ... -->` / `<!-- generated-end -->` in
Markdown. Everything between the markers is the script's; the text
around them is hand-written. The marker carries the commit of the
benchmark repository the numbers come from, so the data is pinned and a
reader can find the run.

Each region is a named table in the script, defined as data: the case
ids and labels of its rows, the variants of its columns, the format of a
cell (a time, a speed-up, both). The nine "Numbers" tables of the kernels
page are the largest part of that data, about 120 rows; their row labels
move from the page into the script, the surrounding prose stays.

Figures in prose are handled in two ways. Where a sentence states a
headline number (the PyPI bullets' geometric means, the README's "3.3x at
radius 1"), the number becomes a substitution the script maintains in a
generated block at the top of the file (`.. |geomean-x86-fast| replace::
6.1` in RST; in Markdown the sentence itself sits inside a short
generated region). Where a section's prose quotes a figure that is also
in its table, the prose refers to the table instead ("a third of the
AVX2 time, as the table shows"), so the figure lives once.

`fast_mode.rst`: the hand-written list-table keeps its description
columns; its speed-up column goes, and a generated table of the fast
kernels' speed-ups (operation, x86-64, arm64) follows it. RST can't
generate one column of a hand-written table, and the operation list is
short.

### Modes

- `python3 checks/simd_tables.py --data ../pillow-allsimd-perf/docs/data
  --commit <sha>` rewrites every region (the `Makefile` gets a `tables`
  target).
- `--check` rewrites to memory and fails if any file would change. The
  lint workflow runs it with the data fetched from the benchmark
  repository at the commit each marker names
  (`raw.githubusercontent.com/desmoteo/pillow-allsimd-perf/<sha>/docs/data/`),
  so a table edited by hand, or a marker bumped without regenerating,
  fails CI. The files are 500 KB and public; they are not vendored into
  this repository (the sdist would carry them).
- The script also checks that every case id it names exists in both
  machines' data, and that the geometric means it writes are computed
  over the same case set the benchmark page uses (the cases both
  variants have), so Appendix A's figures equal the page's summary.

### Order

1. `simd_results.py` extracted from the SVG script; `simd_tables.py` with
   the PyPI table and bullets, the README figure, `fast_mode.rst`'s
   speed-up table and Appendix A. Regenerating must reproduce the
   committed figures (a run that changes a number is a finding to
   explain, not to accept).
2. The nine "Numbers" tables of the kernels page, one section at a time,
   each regenerated and diffed against the hand-written table; a
   difference is either a typo on the page (fixed by the generator) or a
   mapping error in the script (fixed there).
3. The lint step, the Makefile target, a note in `RELEASING.md` (refresh
   the data, regenerate, commit the tables with the data's commit in the
   markers) and in CHANGES.SIMD.rst.

## Done (2026-11-04)

- `checks/simd_results.py` (the loader: machines, speed-ups, geometric
  means over the cases both variants have, as the page's `app.js`; the
  fetch of a commit's files) and `checks/simd_tables.py` (the regions);
  the SVG script loads through the shared module and draws the same chart
  byte for byte.
- Regions: `pypi-figures` (a block of substitutions at the top of
  `PyPI.rst`: the geometric means, the case counts, the OpenCV counts, the
  ratio to Pillow-SIMD, the date; the bullets use them), `pypi-headline`
  (the bold first sentence, whole: a substitution can't sit inside inline
  markup, so it is a region of its own),
  `pypi-benchmarks` (the table; it regenerated byte for byte),
  `readme-gaussian-fast` (inline, in the README's fast-mode sentence),
  `fast-mode-speedups` (a table of the fast kernels' speed-ups over the
  exact ones from the data, on `fast_mode.rst`, in place of the
  hand-written table's speed-up column; the figures that weren't in the
  data, radius 0.5, 2 and 4 and the 1/16 downscale, went with it),
  `kernels-appendix-a` (byte for byte) and the nine `kernels-*` tables.
- The chart's alt text no longer quotes figures (an image option can't
  hold a substitution); it points at the table.
- Regenerating the nine "Numbers" tables changed about twenty cells by one
  in the last digit (speed-ups and milliseconds both), which is hand
  rounding from the page's displayed values; the generator's are the
  data's. Milliseconds are whole from 100 up in every table now (the
  rank, drawing and codec tables were, the others weren't).
- The README's "3.3x at radius 1" became "3.0x at radius 1 on x86-64,
  3.2x on arm64": the old figure was a one-off timing, the new one is the
  published run's.
- `make tables PERF=<checkout>` regenerates; the lint workflow runs
  `checks/simd_tables.py --check`, which fetches the two data files (500 KB)
  at the commit each marker names from raw.githubusercontent.com;
  `RELEASING.md` has the fork's steps.
- Not done: the sections' prose on the kernels page still quotes some
  figures that are also in its tables; they are rewritten to refer to the
  table as each section is next edited.
