#
# The Python Imaging Library.
# $Id$
#
# image enhancement classes
#
# For a background, see "Image Processing By Interpolation and
# Extrapolation", Paul Haeberli and Douglas Voorhies.  Available
# at http://www.graficaobscura.com/interp/index.html
#
# History:
# 1996-03-23 fl  Created
# 2009-06-16 fl  Fixed mean calculation
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from . import Image, ImageFilter, ImageStat

# Modes whose enhance() can be a lookup table on each band.
_LUT_MODES = ("L", "LA", "RGB", "RGBA")


class _Enhance:
    image: Image.Image
    degenerate: Image.Image

    def enhance(self, factor: float) -> Image.Image:
        """
        Returns an enhanced image.

        :param factor: A floating point value controlling the enhancement.
                       Factor 1.0 always returns a copy of the original image,
                       lower factors mean less color (brightness, contrast,
                       etc), and higher values more. There are no restrictions
                       on this value.
        """
        levels = self._levels()
        if levels is not None and self.image.mode in _LUT_MODES:
            return self.image.point(self._lut(levels, factor))
        return Image.blend(self.degenerate, self.image, factor)

    def _levels(self) -> tuple[int, ...] | None:
        """The degenerate image's value in each band if it is constant there,
        alpha excepted, which is the image's own; otherwise None."""
        return None

    def _lut(self, levels: tuple[int, ...], factor: float) -> list[int]:
        # Image.blend of a 0..255 gradient against the levels: each pixel of
        # the image gets the bytes that blending it would give.
        bands = self.image.getbands()
        gradient = Image.frombytes("L", (256, 1), bytes(range(256)))
        ramp = Image.merge(self.image.mode, [gradient] * len(bands))
        flat = [
            gradient if b == "A" else Image.new("L", (256, 1), level)
            for b, level in zip(bands, levels)
        ]
        out = Image.blend(Image.merge(self.image.mode, flat), ramp, factor)
        return [v for b in range(len(bands)) for v in out.getchannel(b).tobytes()]


class Color(_Enhance):
    """Adjust image color balance.

    This class can be used to adjust the colour balance of an image, in
    a manner similar to the controls on a colour TV set. An enhancement
    factor of 0.0 gives a black and white image. A factor of 1.0 gives
    the original image.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.intermediate_mode = "L"
        if "A" in image.getbands():
            self.intermediate_mode = "LA"

        if self.intermediate_mode != image.mode:
            image = image.convert(self.intermediate_mode).convert(image.mode)
        self.degenerate = image


class Contrast(_Enhance):
    """Adjust image contrast.

    This class can be used to control the contrast of an image, similar
    to the contrast control on a TV set. An enhancement factor of 0.0
    gives a solid gray image. A factor of 1.0 gives the original image.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        if image.mode != "L":
            image = image.convert("L")
        self._mean = int(ImageStat.Stat(image).mean[0] + 0.5)
        self._degenerate: Image.Image | None = None
        self._custom = False

    @property
    def degenerate(self) -> Image.Image:
        # Made when first needed: enhance() mostly uses a lookup table.
        if self._degenerate is None:
            self._degenerate = Image.new("L", self.image.size, self._mean)
            if self._degenerate.mode != self.image.mode:
                self._degenerate = self._degenerate.convert(self.image.mode)

            if "A" in self.image.getbands():
                self._degenerate.putalpha(self.image.getchannel("A"))
        return self._degenerate

    @degenerate.setter
    def degenerate(self, value: Image.Image) -> None:
        self._degenerate, self._custom = value, True

    def _levels(self) -> tuple[int, ...] | None:
        if self._custom:
            return None
        # The mean converted to the image's mode, as the degenerate image is.
        level = Image.new("L", (1, 1), self._mean).convert(self.image.mode)
        value = level.getpixel((0, 0))
        return value if isinstance(value, tuple) else (int(value or 0),)


class Brightness(_Enhance):
    """Adjust image brightness.

    This class can be used to control the brightness of an image.  An
    enhancement factor of 0.0 gives a black image. A factor of 1.0 gives the
    original image.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self._degenerate: Image.Image | None = None
        self._custom = False

    @property
    def degenerate(self) -> Image.Image:
        # Made when first needed: enhance() mostly uses a lookup table.
        if self._degenerate is None:
            self._degenerate = Image.new(self.image.mode, self.image.size, 0)

            if "A" in self.image.getbands():
                self._degenerate.putalpha(self.image.getchannel("A"))
        return self._degenerate

    @degenerate.setter
    def degenerate(self, value: Image.Image) -> None:
        self._degenerate, self._custom = value, True

    def _levels(self) -> tuple[int, ...] | None:
        return None if self._custom else (0,) * len(self.image.getbands())


class Sharpness(_Enhance):
    """Adjust image sharpness.

    This class can be used to adjust the sharpness of an image. An
    enhancement factor of 0.0 gives a blurred image, a factor of 1.0 gives the
    original image, and a factor of 2.0 gives a sharpened image.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.degenerate = image.filter(ImageFilter.SMOOTH)

        if "A" in image.getbands():
            self.degenerate.putalpha(image.getchannel("A"))
