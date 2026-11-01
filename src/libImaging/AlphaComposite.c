/*
 * The Python Imaging Library
 * $Id$
 *
 * Alpha composite imSrc over imDst.
 * https://en.wikipedia.org/wiki/Alpha_compositing
 *
 * See the README file for details on usage and redistribution.
 */

#include "Imaging.h"
#include "AlphaComposite.h"

/* Composites one row of xsize pixels: out may be dst, for in place. */
static void
alpha_composite_row(
    rgba8 *out, const rgba8 *dst, const rgba8 *restrict src, int xsize
) {
    /* the active tier's kernel does the pixels it can (AlphaCompositeSIMD.c) */
    int x = IMAGING_SIMD_CALL(
        composite->row, (UINT8 *)out, (const UINT8 *)dst, (const UINT8 *)src, xsize
    );

    for (; x < xsize; x += 1) {
        if (src[x].a == 0) {
            // Copy 4 bytes at once.
            out[x] = dst[x];
        } else {
            // Integer implementation with increased precision.
            // Each variable has extra meaningful bits.
            // Divisions are rounded.

            UINT32 tmpr, tmpg, tmpb;
            UINT32 blend = dst[x].a * (255 - src[x].a);
            UINT32 outa255 = src[x].a * 255 + blend;
            // There we use 7 bits for precision.
            // We could use more, but we go beyond 32 bits.
            UINT32 coef1 = src[x].a * 255 * 255 * (1 << AC_PRECISION_BITS) / outa255;
            UINT32 coef2 = 255 * (1 << AC_PRECISION_BITS) - coef1;

            tmpr = src[x].r * coef1 + dst[x].r * coef2;
            tmpg = src[x].g * coef1 + dst[x].g * coef2;
            tmpb = src[x].b * coef1 + dst[x].b * coef2;
            out[x].r =
                SHIFTFORDIV255(tmpr + (0x80 << AC_PRECISION_BITS)) >> AC_PRECISION_BITS;
            out[x].g =
                SHIFTFORDIV255(tmpg + (0x80 << AC_PRECISION_BITS)) >> AC_PRECISION_BITS;
            out[x].b =
                SHIFTFORDIV255(tmpb + (0x80 << AC_PRECISION_BITS)) >> AC_PRECISION_BITS;
            out[x].a = SHIFTFORDIV255(outa255 + 0x80);
        }
    }
}

/**
 * Alpha-composite imSrc over imDst, returning a newly allocated result.
 *
 * Contract: imDst and imSrc are read-only and may alias each other.
 */
Imaging
ImagingAlphaComposite(Imaging imDst, Imaging imSrc) {
    Imaging imOut;
    ImagingSectionCookie cookie;

    /* Check arguments */
    if (!imDst || !imSrc ||
        (imDst->mode != IMAGING_MODE_RGBA && imDst->mode != IMAGING_MODE_LA)) {
        return ImagingError_ModeError();
    }

    if (imDst->mode != imSrc->mode || imDst->xsize != imSrc->xsize ||
        imDst->ysize != imSrc->ysize) {
        return ImagingError_Mismatch();
    }

    imOut = ImagingNewDirty(imDst->mode, imDst->xsize, imDst->ysize);
    if (!imOut) {
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    for (int y = 0; y < imDst->ysize; y++) {
        alpha_composite_row(
            (rgba8 *)imOut->image[y],
            (rgba8 *)imDst->image[y],
            (rgba8 *)imSrc->image[y],
            imDst->xsize
        );
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}

/* The lowest and the highest address of an image's rows. */
static void
image_span(Imaging im, const char **lo, const char **hi) {
    int y;
    *lo = *hi = NULL;
    for (y = 0; y < im->ysize; y++) {
        const char *row = im->image[y];
        if (!*lo || row < *lo) {
            *lo = row;
        }
        if (!*hi || row + im->linesize > *hi) {
            *hi = row + im->linesize;
        }
    }
}

/**
 * Alpha-composite the xsize x ysize pixels of imSrc at (sx, sy) over imDst
 * at (dx, dy), in place, as compositing them onto a copy of that area and
 * pasting it back would: each pixel's result depends only on the two
 * pixels. The area is clipped to imDst, and must lie within imSrc.
 *
 * Returns 0 if the images share memory, for the caller to composite
 * through copies; -1 on error.
 */
int
ImagingAlphaCompositeOnto(
    Imaging imDst, Imaging imSrc, int dx, int dy, int sx, int sy, int xsize, int ysize
) {
    ImagingSectionCookie cookie;
    const char *dlo, *dhi, *slo, *shi;
    int y;

    if (!imDst || !imSrc ||
        (imDst->mode != IMAGING_MODE_RGBA && imDst->mode != IMAGING_MODE_LA) ||
        imDst->mode != imSrc->mode) {
        (void)ImagingError_ModeError();
        return -1;
    }
    if (sx < 0 || sy < 0 || xsize < 0 || ysize < 0 || xsize > imSrc->xsize - sx ||
        ysize > imSrc->ysize - sy) {
        (void)ImagingError_ValueError("source area outside the image");
        return -1;
    }
    image_span(imDst, &dlo, &dhi);
    image_span(imSrc, &slo, &shi);
    if (dlo < shi && slo < dhi) {
        return 0;
    }

    /* Clip to imDst. */
    if (dx < 0) {
        sx -= dx, xsize += dx, dx = 0;
    }
    if (dy < 0) {
        sy -= dy, ysize += dy, dy = 0;
    }
    if (xsize > imDst->xsize - dx) {
        xsize = imDst->xsize - dx;
    }
    if (ysize > imDst->ysize - dy) {
        ysize = imDst->ysize - dy;
    }
    if (xsize <= 0 || ysize <= 0) {
        return 1;
    }

    ImagingSectionEnter(&cookie);
    for (y = 0; y < ysize; y++) {
        rgba8 *dst = (rgba8 *)imDst->image[dy + y] + dx;
        alpha_composite_row(dst, dst, (rgba8 *)imSrc->image[sy + y] + sx, xsize);
    }
    ImagingSectionLeave(&cookie);
    return 1;
}
