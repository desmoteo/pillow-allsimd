from __future__ import annotations

import random

import pytest

from PIL import Image, ImageDraw

# Drawing paths that Pillow-SIMD implements differently from Pillow and
# that must give Pillow's pixels.


def _upstream_polygon_outline(
    im: Image.Image,
    xy: list[tuple[float, float]],
    outline: float | tuple[int, ...],
    width: int,
) -> None:
    # Pillow's ImageDraw.polygon() for width > 1: an image-sized mask
    # holding the polygon's fill, and wide lines drawn through it.
    draw = ImageDraw.Draw(im)
    ink = draw._getink(outline)[0]
    mask = Image.new("1", im.size)
    ImageDraw.Draw(mask).draw.draw_polygon(xy, draw._getink(1)[0], 1)
    draw.draw.draw_polygon(xy, ink, 0, width * 2 - 1, mask.im)


@pytest.mark.parametrize("mode", ("1", "L", "I;16", "I", "F", "RGB", "RGBA", "LA", "P"))
def test_polygon_outline_matches_full_mask(mode: str) -> None:
    rnd = random.Random(mode)
    for size in ((120, 80), (17, 300), (1, 1), (5, 90)):
        expected = Image.new(mode, size)
        actual = Image.new(mode, size)
        w, h = size
        for _ in range(30):
            n = rnd.choice((2, 3, 4, 6, 12))
            cx, cy = rnd.uniform(-0.5, 1.5) * w, rnd.uniform(-0.5, 1.5) * h
            spread = rnd.choice((0.3, 1.0, 3.0))
            dx, dy = spread * w, spread * h
            xy = [
                (cx + rnd.uniform(-1, 1) * dx, cy + rnd.uniform(-1, 1) * dy)
                for _ in range(n)
            ]
            ink: float | tuple[int, ...]
            if mode in ("RGB", "RGBA", "LA"):
                ink = tuple(rnd.randrange(256) for _ in range(len(mode)))
            elif mode == "F":
                ink = rnd.uniform(-5, 300)
            else:
                ink = rnd.randrange(256)
            width = rnd.choice((2, 3, 4, 7, 15))
            _upstream_polygon_outline(expected, xy, ink, width)
            ImageDraw.Draw(actual).polygon(xy, outline=ink, width=width)
        assert actual.tobytes() == expected.tobytes()
