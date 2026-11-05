/*
 * The Python Imaging Library
 * $Id$
 *
 * interpolate between two existing images
 *
 * history:
 * 96-03-20 fl Created
 * 96-05-18 fl Simplified blend expression
 * 96-10-05 fl Fixed expression bug, special case for interpolation
 *
 * Copyright (c) Fredrik Lundh 1996.
 * Copyright (c) Secret Labs AB 1997.
 *
 * See the README file for details on usage and redistribution.
 */

#include "Imaging.h"
#include "PlainLoops.h"

/* Fast mode (design/SIMD_FAST_KERNELS.md): in1 + (in2 - in1) a for a = a15 / 2^15,
   rounded half up, (d a15 + 2^14) >> 15: what pmulhrsw and vqrdmulh compute,
   so every build gives the same bytes. The result lies between in1 and in2. */
static void
blend_fast_row(
    UINT8 *restrict out, const UINT8 *in1, const UINT8 *in2, int n, INT16 a15
) {
    /* the active tier's kernel does the bytes it can (BlendSIMD.c) */
    int x = IMAGING_SIMD_CALL(blend->fast_row, out, in1, in2, n, a15);
    for (; x < n; x++) {
        const int d = in2[x] - in1[x];
        out[x] = (UINT8)(in1[x] + ((d * a15 + 16384) >> 15));
    }
}

/**
 * Interpolate (or, for alpha outside [0, 1], extrapolate) between imIn1 and
 * imIn2 by `alpha`, returning a newly allocated result.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingBlend(Imaging imIn1, Imaging imIn2, float alpha) {
    Imaging imOut;
    int y;

    /* Check arguments */
    if (!imIn1 || !imIn2 || imIn1->type != IMAGING_TYPE_UINT8 || imIn1->palette ||
        imIn1->mode == IMAGING_MODE_1 || imIn2->palette ||
        imIn2->mode == IMAGING_MODE_1) {
        return ImagingError_ModeError();
    }

    if (imIn1->type != imIn2->type || imIn1->bands != imIn2->bands ||
        imIn1->xsize != imIn2->xsize || imIn1->ysize != imIn2->ysize) {
        return ImagingError_Mismatch();
    }

    /* Shortcuts */
    if (imIn1 == imIn2 || alpha == 0.0) {
        return ImagingCopy(imIn1);
    } else if (alpha == 1.0) {
        return ImagingCopy(imIn2);
    }

    imOut = ImagingNewDirty(imIn1->mode, imIn1->xsize, imIn1->ysize);
    if (!imOut) {
        return NULL;
    }

    // We know these won't change in the loops below,
    // so the compiler can optimize better if we pull them out.
    int ysize = imIn1->ysize;
    int linesize = imIn1->linesize;

    ImagingSectionCookie cookie;
    ImagingSectionEnter(&cookie);
    if (alpha > 0 && alpha < 1.0 && (ImagingGetFast() & IMAGING_FAST_ALL)) {
        /* alpha * 2^15 is exact, the rounding the only one */
        int a15 = (int)(alpha * 32768.0 + 0.5);
        if (a15 > 32767) {
            a15 = 32767;
        }
        for (y = 0; y < ysize; y++) {
            blend_fast_row(
                (UINT8 *)imOut->image[y],
                (UINT8 *)imIn1->image[y],
                (UINT8 *)imIn2->image[y],
                linesize,
                (INT16)a15
            );
        }
    } else if (alpha >= 0 && alpha <= 1.0) {
        /* Interpolate between bands */
        /* float arithmetic a byte at a time: the active tier's kernel is
           the loop of PlainLoops.h compiled for it (BlendSIMD.c) */
        for (y = 0; y < ysize; y++) {
            const UINT8 *in1 = (UINT8 *)imIn1->image[y];
            const UINT8 *in2 = (UINT8 *)imIn2->image[y];
            UINT8 *out = (UINT8 *)imOut->image[y];
            if (!IMAGING_SIMD_CALL(
                    blend->interpolate_row, out, in1, in2, linesize, alpha
                )) {
                plain_blend_interpolate_row(out, in1, in2, linesize, alpha);
            }
        }
    } else {
        /* Extrapolation; must make sure to clip resulting values */
        for (y = 0; y < ysize; y++) {
            const UINT8 *in1 = (UINT8 *)imIn1->image[y];
            const UINT8 *in2 = (UINT8 *)imIn2->image[y];
            UINT8 *out = (UINT8 *)imOut->image[y];
            if (!IMAGING_SIMD_CALL(
                    blend->extrapolate_row, out, in1, in2, linesize, alpha
                )) {
                plain_blend_extrapolate_row(out, in1, in2, linesize, alpha);
            }
        }
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}
