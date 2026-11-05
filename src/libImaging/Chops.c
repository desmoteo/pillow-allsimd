/*
 * The Python Imaging Library
 * $Id$
 *
 * basic channel operations
 *
 * history:
 * 1996-03-28 fl   Created
 * 1996-08-13 fl   Added and/or/xor for "1" images
 * 1996-12-14 fl   Added add_modulo, sub_modulo
 * 2005-09-10 fl   Fixed output values from and/or/xor
 *
 * Copyright (c) 1996 by Fredrik Lundh.
 * Copyright (c) 1997 by Secret Labs AB.
 *
 * See the README file for details on usage and redistribution.
 */

#include "Imaging.h"
#include "PlainLoops.h"

// restrict safety: imIn1/imIn2 are read-only, and imOut is always freshly allocated
#define CHOP(operation)                                 \
    int x, y;                                           \
    Imaging imOut;                                      \
    imOut = create(imIn1, imIn2, IMAGING_MODE_UNKNOWN); \
    if (!imOut) {                                       \
        return NULL;                                    \
    }                                                   \
    int ysize = imOut->ysize;                           \
    ImagingSectionCookie cookie;                        \
    int linesize = imOut->linesize;                     \
    ImagingSectionEnter(&cookie);                       \
    for (y = 0; y < ysize; y++) {                       \
        UINT8 *restrict out = (UINT8 *)imOut->image[y]; \
        UINT8 *restrict in1 = (UINT8 *)imIn1->image[y]; \
        UINT8 *restrict in2 = (UINT8 *)imIn2->image[y]; \
        for (x = 0; x < linesize; x++) {                \
            int temp = operation;                       \
            if (temp <= 0) {                            \
                out[x] = 0;                             \
            } else if (temp >= 255) {                   \
                out[x] = 255;                           \
            } else {                                    \
                out[x] = temp;                          \
            }                                           \
        }                                               \
    }                                                   \
    ImagingSectionLeave(&cookie);                       \
    return imOut;

// restrict safety: imIn1/imIn2 are read-only, and imOut is always freshly allocated
#define CHOP2(operation, mode)                          \
    int x, y;                                           \
    Imaging imOut;                                      \
    imOut = create(imIn1, imIn2, mode);                 \
    if (!imOut) {                                       \
        return NULL;                                    \
    }                                                   \
    int ysize = imOut->ysize;                           \
    ImagingSectionCookie cookie;                        \
    int linesize = imOut->linesize;                     \
    ImagingSectionEnter(&cookie);                       \
    for (y = 0; y < ysize; y++) {                       \
        UINT8 *restrict out = (UINT8 *)imOut->image[y]; \
        UINT8 *restrict in1 = (UINT8 *)imIn1->image[y]; \
        UINT8 *restrict in2 = (UINT8 *)imIn2->image[y]; \
        for (x = 0; x < linesize; x++) {                \
            out[x] = operation;                         \
        }                                               \
    }                                                   \
    ImagingSectionLeave(&cookie);                       \
    return imOut;

static Imaging
create(Imaging im1, Imaging im2, const ModeID mode) {
    int xsize, ysize;

    if (!im1 || !im2 || im1->type != IMAGING_TYPE_UINT8 ||
        (mode != IMAGING_MODE_UNKNOWN &&
         (im1->mode != IMAGING_MODE_1 || im2->mode != IMAGING_MODE_1))) {
        return (Imaging)ImagingError_ModeError();
    }
    if (im1->type != im2->type || im1->bands != im2->bands) {
        return (Imaging)ImagingError_Mismatch();
    }

    xsize = (im1->xsize < im2->xsize) ? im1->xsize : im2->xsize;
    ysize = (im1->ysize < im2->ysize) ? im1->ysize : im2->ysize;

    return ImagingNewDirty(im1->mode, xsize, ysize);
}

/**
 * Return a newly allocated image containing the lighter pixels of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopLighter(Imaging imIn1, Imaging imIn2) {
    /* Never out of range: no clipping, which lets it vectorize. */
    CHOP2((in1[x] > in2[x]) ? in1[x] : in2[x], IMAGING_MODE_UNKNOWN);
}

/**
 * Return a newly allocated image containing the darker pixels of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopDarker(Imaging imIn1, Imaging imIn2) {
    CHOP2((in1[x] < in2[x]) ? in1[x] : in2[x], IMAGING_MODE_UNKNOWN);
}

/**
 * Return a newly allocated image containing the absolute per-pixel difference of the
 * two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopDifference(Imaging imIn1, Imaging imIn2) {
    CHOP2(
        ((int)in1[x] > (int)in2[x]) ? (in1[x] - in2[x]) : (in2[x] - in1[x]),
        IMAGING_MODE_UNKNOWN
    );
}

/* out = in1 * in2 / 255, rounded down. With invert = 255 every byte is
   inverted on the way in and out, which gives screen:
   255 - (255 - in1) * (255 - in2) / 255.
   For every product x = a * b of two bytes, x / 255 is exactly
   (x + 1 + (x >> 8)) >> 8, and no step leaves 16 bits. */
static void
chop_multiply_row(
    UINT8 *restrict out,
    const UINT8 *restrict in1,
    const UINT8 *restrict in2,
    int n,
    UINT8 invert
) {
    /* the active tier's kernel does the bytes it can (ChopsSIMD.c) */
    int x = IMAGING_SIMD_CALL(chops->multiply_row, out, in1, in2, n, invert);

    for (; x < n; x++) {
        unsigned int v = (unsigned int)(in1[x] ^ invert) * (in2[x] ^ invert);
        out[x] = (UINT8)(((v + 1 + (v >> 8)) >> 8) ^ invert);
    }
}

static Imaging
chop_multiply(Imaging imIn1, Imaging imIn2, UINT8 invert) {
    int y;
    Imaging imOut;
    ImagingSectionCookie cookie;

    imOut = create(imIn1, imIn2, IMAGING_MODE_UNKNOWN);
    if (!imOut) {
        return NULL;
    }
    ImagingSectionEnter(&cookie);
    for (y = 0; y < imOut->ysize; y++) {
        chop_multiply_row(
            (UINT8 *)imOut->image[y],
            (UINT8 *)imIn1->image[y],
            (UINT8 *)imIn2->image[y],
            imOut->linesize,
            invert
        );
    }
    ImagingSectionLeave(&cookie);
    return imOut;
}

/**
 * Return a newly allocated image containing the per-pixel product (scaled to 0-255) of
 * the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopMultiply(Imaging imIn1, Imaging imIn2) {
    return chop_multiply(imIn1, imIn2, 0);
}

/**
 * Return a newly allocated image containing the screen blend of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopScreen(Imaging imIn1, Imaging imIn2) {
    return chop_multiply(imIn1, imIn2, 255);
}

/**
 * Return a newly allocated image containing the per-pixel sum, divided by `scale` and
 * shifted by `offset`, clipped to 0-255.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
/* in1 + in2 (sign 1) or in1 - in2 (sign -1), clipped to 0-255: one
   saturating instruction per vector. Compilers don't find it themselves. */
static void
chop_saturate_row(
    UINT8 *restrict out,
    const UINT8 *restrict in1,
    const UINT8 *restrict in2,
    int n,
    int sign
) {
    /* the active tier's kernel does the bytes it can (ChopsSIMD.c) */
    int x = IMAGING_SIMD_CALL(chops->saturate_row, out, in1, in2, n, sign);

    for (; x < n; x++) {
        int v = sign > 0 ? in1[x] + in2[x] : in1[x] - in2[x];
        out[x] = v <= 0 ? 0 : v >= 255 ? 255 : v;
    }
}

/* (in1 + in2) / scale + offset for add (sign 1) and subtract (sign -1),
   clipped. With scale 1 that is integer arithmetic, which vectorizes, and
   with offset 0 as well, a saturating add or subtract; otherwise a table of
   the same float expression for every sum, instead of a division per byte. */
static Imaging
chop_add_subtract(Imaging imIn1, Imaging imIn2, float scale, int offset, int sign) {
    if (scale == 1.0 && offset == 0) {
        int y;
        ImagingSectionCookie cookie;
        Imaging imOut = create(imIn1, imIn2, IMAGING_MODE_UNKNOWN);
        if (!imOut) {
            return NULL;
        }
        ImagingSectionEnter(&cookie);
        for (y = 0; y < imOut->ysize; y++) {
            chop_saturate_row(
                (UINT8 *)imOut->image[y],
                (UINT8 *)imIn1->image[y],
                (UINT8 *)imIn2->image[y],
                imOut->linesize,
                sign
            );
        }
        ImagingSectionLeave(&cookie);
        return imOut;
    }
    if (scale == 1.0 && offset > -1024 && offset < 1024) {
        if (sign > 0) {
            CHOP((int)in1[x] + (int)in2[x] + offset);
        } else {
            CHOP((int)in1[x] - (int)in2[x] + offset);
        }
    } else {
        UINT8 table[511];
        int i;
        for (i = 0; i < 511; i++) {
            int temp = (i - 255 * (sign < 0)) / scale + offset;
            table[i] = temp <= 0 ? 0 : temp >= 255 ? 255 : temp;
        }
        if (sign > 0) {
            CHOP2(table[(int)in1[x] + (int)in2[x]], IMAGING_MODE_UNKNOWN);
        } else {
            CHOP2(table[(int)in1[x] - (int)in2[x] + 255], IMAGING_MODE_UNKNOWN);
        }
    }
}

Imaging
ImagingChopAdd(Imaging imIn1, Imaging imIn2, float scale, int offset) {
    return chop_add_subtract(imIn1, imIn2, scale, offset, 1);
}

/**
 * Return a newly allocated image containing the per-pixel difference, divided by
 * `scale` and shifted by `offset`, clipped to 0-255.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopSubtract(Imaging imIn1, Imaging imIn2, float scale, int offset) {
    return chop_add_subtract(imIn1, imIn2, scale, offset, -1);
}

/**
 * Return a newly allocated "1" image that is the logical AND of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopAnd(Imaging imIn1, Imaging imIn2) {
    CHOP2((in1[x] && in2[x]) ? 255 : 0, IMAGING_MODE_1);
}

/**
 * Return a newly allocated "1" image that is the logical OR of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopOr(Imaging imIn1, Imaging imIn2) {
    CHOP2((in1[x] || in2[x]) ? 255 : 0, IMAGING_MODE_1);
}

/**
 * Return a newly allocated "1" image that is the logical XOR of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopXor(Imaging imIn1, Imaging imIn2) {
    CHOP2(((in1[x] != 0) ^ (in2[x] != 0)) ? 255 : 0, IMAGING_MODE_1);
}

/**
 * Return a newly allocated image containing the per-pixel sum of the two images,
 * wrapping on overflow (no clipping).
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopAddModulo(Imaging imIn1, Imaging imIn2) {
    CHOP2(in1[x] + in2[x], IMAGING_MODE_UNKNOWN);
}

/**
 * Return a newly allocated image containing the per-pixel difference of the two images,
 * wrapping on underflow (no clipping).
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopSubtractModulo(Imaging imIn1, Imaging imIn2) {
    CHOP2(in1[x] - in2[x], IMAGING_MODE_UNKNOWN);
}

/* Soft light (kind 0), hard light (1) and overlay (2), a row at a time:
   int arithmetic with three divisions, which the active tier's kernel has
   compiled for its instruction set (PlainLoops.h, ChopsSIMD.c); the scalar
   table runs the same loop as compiled here. The output is a new image. */
static Imaging
chop_light(Imaging imIn1, Imaging imIn2, int kind) {
    int y;
    Imaging imOut = create(imIn1, imIn2, IMAGING_MODE_UNKNOWN);
    if (!imOut) {
        return NULL;
    }
    int ysize = imOut->ysize;
    int linesize = imOut->linesize;
    ImagingSectionCookie cookie;
    ImagingSectionEnter(&cookie);
    for (y = 0; y < ysize; y++) {
        UINT8 *out = (UINT8 *)imOut->image[y];
        const UINT8 *in1 = (UINT8 *)imIn1->image[y];
        const UINT8 *in2 = (UINT8 *)imIn2->image[y];
        if (kind == 0) {
            if (!IMAGING_SIMD_CALL(chops->soft_light_row, out, in1, in2, linesize)) {
                plain_soft_light_row(out, in1, in2, linesize);
            }
        } else {
            if (kind == 2) {
                /* overlay is hard light with the operands exchanged */
                const UINT8 *tmp = in1;
                in1 = in2;
                in2 = tmp;
            }
            if (!IMAGING_SIMD_CALL(chops->hard_light_row, out, in1, in2, linesize)) {
                plain_hard_light_row(out, in1, in2, linesize);
            }
        }
    }
    ImagingSectionLeave(&cookie);
    return imOut;
}

/**
 * Return a newly allocated image containing the soft-light blend of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopSoftLight(Imaging imIn1, Imaging imIn2) {
    return chop_light(imIn1, imIn2, 0);
}

/**
 * Return a newly allocated image containing the hard-light blend of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopHardLight(Imaging imIn1, Imaging imIn2) {
    return chop_light(imIn1, imIn2, 1);
}

/**
 * Return a newly allocated image containing the overlay blend of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingOverlay(Imaging imIn1, Imaging imIn2) {
    return chop_light(imIn1, imIn2, 2);
}
