"""Per case: the best of three overhead.py runs per build (files
py2c-avx2-N.json and upstream-N.json in DIR), AllSIMD vs Pillow 13.

    python3 merge.py DIR
"""

from __future__ import annotations

import json
import sys

D = sys.argv[1]


def load(b):
    runs = [json.load(open(f"{D}/{b}-{i}.json")) for i in (1, 2, 3)]
    out = {}
    for k in runs[0]:
        best = min((r[k] for r in runs if k in r), key=lambda r: r["us"])
        out[k] = best
    return out


a, u = load("py2c-avx2"), load("upstream")
json.dump({"allsimd": a, "upstream": u}, open(f"{D}/merged.json", "w"), indent=1)
for k in a:
    ra, ru = a[k], u.get(k)
    s = f"{k:55s} {ra['us']:10.1f} {ru['us']:10.1f} {ru['us']/ra['us']:5.2f}x"
    if "py_share" in ra:
        s += f"  py(direct) {100*ra['py_share']:5.1f}% / {100*ru['py_share']:5.1f}%"
    if "py" in ra:
        s += f"  py(cProfile) {100*ra['py']:5.1f}% / {100*ru['py']:5.1f}%"
    print(s)
