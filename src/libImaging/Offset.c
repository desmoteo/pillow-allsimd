/*
 * The Python Imaging Library
 * $Id$
 *
 * offset an image in x and y directions
 *
 * history:
 * 96-07-22 fl: Created
 * 98-11-01 cgw@pgt.com: Fixed negative-array index bug
 *
 * Copyright (c) Fredrik Lundh 1996.
 * Copyright (c) Secret Labs AB 1997.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

Imaging
ImagingOffset(Imaging im, int xoffset, int yoffset) {
    int y;
    Imaging imOut;

    if (!im) {
        return (Imaging)ImagingError_ModeError();
    }

    imOut = ImagingNewDirty(im->mode, im->xsize, im->ysize);
    if (!imOut) {
        return NULL;
    }

    ImagingCopyPalette(imOut, im);

    /* make offsets positive to avoid negative coordinates */
    xoffset %= im->xsize;
    xoffset = im->xsize - xoffset;
    if (xoffset < 0) {
        xoffset += im->xsize;
    }

    yoffset %= im->ysize;
    yoffset = im->ysize - yoffset;
    if (yoffset < 0) {
        yoffset += im->ysize;
    }

    /* Each output row is input row (y + yoffset) % ysize rotated left by
       xoffset pixels: two copies per row. */
    ImagingSectionCookie cookie;
    ImagingSectionEnter(&cookie);
    // xoffset is in 0 .. 2 * xsize here; the per-pixel % did the rest.
    xoffset %= im->xsize;
    {
        size_t pixelsize = im->pixelsize;
        size_t head = (size_t)(im->xsize - xoffset) * pixelsize;
        size_t tail = (size_t)xoffset * pixelsize;
        for (y = 0; y < im->ysize; y++) {
            const char *in = im->image[(y + yoffset) % im->ysize];
            char *out = imOut->image[y];
            memcpy(out, in + tail, head);
            memcpy(out + head, in, tail);
        }
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}
