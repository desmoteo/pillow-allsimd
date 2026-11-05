/*
 * The Python Imaging Library
 * $Id$
 *
 * colour and luminance matrix transforms
 *
 * history:
 * 1996-05-18 fl:   created (brute force implementation)
 *
 * Copyright (c) Fredrik Lundh 1996.
 * Copyright (c) Secret Labs AB 1997.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"
#include "PlainLoops.h"

/**
 * Convert `im` to `mode` by applying the matrix `m`,
 * returning a newly allocated result.
 *
 * Contract: im is read-only.
 */
Imaging
ImagingConvertMatrix(Imaging im, const ModeID mode, const float m[12]) {
    Imaging imOut;
    ImagingSectionCookie cookie;

    /* Assume there's enough data in the buffer */
    if (!im || im->bands != 3) {
        return (Imaging)ImagingError_ModeError();
    }

    if (mode == IMAGING_MODE_L) {
        imOut = ImagingNewDirty(IMAGING_MODE_L, im->xsize, im->ysize);
        if (!imOut) {
            return NULL;
        }

        // Invariant over the loop.
        int xsize = im->xsize, ysize = im->ysize;

        ImagingSectionEnter(&cookie);
        for (int y = 0; y < ysize; y++) {
            /* float arithmetic a pixel at a time: the active tier's kernel
               is the loop of PlainLoops.h compiled for it (ConvertSIMD.c) */
            const UINT8 *in = (UINT8 *)im->image[y];
            UINT8 *out = (UINT8 *)imOut->image[y];
            if (!IMAGING_SIMD_CALL(convert->matrix_l_row, out, in, xsize, m)) {
                plain_matrix_l_row(out, in, xsize, m);
            }
        }
        ImagingSectionLeave(&cookie);
    } else if (mode == IMAGING_MODE_RGB) {
        imOut = ImagingNewDirty(mode, im->xsize, im->ysize);
        if (!imOut) {
            return NULL;
        }

        // Invariant over the loop.
        int xsize = im->xsize, ysize = im->ysize;

        for (int y = 0; y < ysize; y++) {
            const UINT8 *in = (UINT8 *)im->image[y];
            UINT8 *out = (UINT8 *)imOut->image[y];

            ImagingSectionEnter(&cookie);
            if (!IMAGING_SIMD_CALL(convert->matrix_rgb_row, out, in, xsize, m)) {
                plain_matrix_rgb_row(out, in, xsize, m);
            }
            ImagingSectionLeave(&cookie);
        }
    } else {
        return (Imaging)ImagingError_ModeError();
    }

    return imOut;
}
