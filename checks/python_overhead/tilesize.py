from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

big = Image.effect_noise((1920, 1080), 30).convert("RGB")
for ts in (32, 64, 128, 256):
    boxes = [
        (x, y, x + ts, y + ts)
        for y in range(0, 1080 - ts, ts)
        for x in range(0, 1920 - ts, ts)
    ]

    def job(_):
        for b in boxes:
            big.crop(b).resize((ts // 2, ts // 2)).convert("L")

    res = []
    for th in (1, 4):
        best = 0
        for _ in range(3):
            with ThreadPoolExecutor(th) as p:
                t = time.perf_counter()
                list(p.map(job, range(16)))
                best = max(best, 16 / (time.perf_counter() - t))
        res.append(best)
    print(
        f"tile {ts:3d}: 1 thread {res[0] * len(boxes):9.0f} tiles/s, "
        f"4 threads {res[1] * len(boxes):9.0f} tiles/s, "
        f"scaling {res[1] / res[0]:4.2f}x"
    )
