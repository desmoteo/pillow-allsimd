/*
 * The Python Imaging Library
 * $Id$
 *
 * apply convolution kernel to image
 *
 * history:
 * 1995-11-26 fl   Created, supports 3x3 kernels
 * 1995-11-27 fl   Added 5x5 kernels, copy border
 * 1999-07-26 fl   Eliminated a few compiler warnings
 * 2002-06-09 fl   Moved kernel definitions to Python
 * 2002-06-11 fl   Support floating point kernels
 * 2003-09-15 fl   Added ImagingExpand helper
 *
 * Copyright (c) Secret Labs AB 1997-2002.  All rights reserved.
 * Copyright (c) Fredrik Lundh 1995.
 *
 * See the README file for information on usage and redistribution.
 */

/*
 * FIXME: Support RGB and RGBA/CMYK modes as well
 * FIXME: Expand image border (current version leaves border as is)
 * FIXME: Implement image processing gradient filters
 */

#include <stdint.h>
#include "Imaging.h"

#define INT32_MAX_F 2147483647.0F

static inline UINT8
clip8(float in) {
    // Branchless clamp to [0, 255].
    in = in < 0.0f ? 0.0f : in;
    in = in > 255.0f ? 255.0f : in;
    return (UINT8)in;
}

static inline int
clip16(float in) {
    // Branchless clamp to [0, 65535].
    in = in < 0.0f ? 0.0f : in;
    in = in > 65535.0f ? 65535.0f : in;
    return (int)in;
}

static inline INT32
clip32(float in) {
    // Clamp the low bound branchlessly (maxss).
    in = in < 0.0f ? 0.0f : in;
    // Must explicitly return INT32_MAX for the high bound
    // to avoid incorrect overflow to INT_MIN.
    return in >= INT32_MAX_F ? INT32_MAX : (INT32)in;
}

/* 5 is number of bits enought to account all kernel coefficients (1<<5 > 25).
   8 is number of bits in result.
   Any coefficients delta smaller than this precision will have no effect. */
#define PRECISION_BITS (8 + 5)

#if defined(IMAGING_SIMD_SSE4)
#include "FilterSIMD_3x3f_u8.c"
#include "FilterSIMD_3x3f_4u8.c"
#include "FilterSIMD_3x3i_4u8.c"
#include "FilterSIMD_5x5f_u8.c"
#include "FilterSIMD_5x5f_4u8.c"
#include "FilterSIMD_5x5i_4u8.c"
#endif
#if defined(IMAGING_SIMD_NEON)
#include <math.h>
#include "FilterSIMD_NxNf.c"
#endif
#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
#include "FilterSIMD_NxNi.c"
#define FILTER_SIMD
#endif

Imaging
ImagingExpand(Imaging imIn, int margin) {
    Imaging imOut;
    int x, y;
    ImagingSectionCookie cookie;

    if (margin < 0) {
        return (Imaging)ImagingError_ValueError("bad kernel size");
    }
    // Compute in int64_t via division, not squaring, so the check itself
    // can't overflow or divide by zero for any valid int margin.
    if (margin > 0 && (int64_t)margin > (int64_t)INT_MAX / ((int64_t)margin *
                                                            (int64_t)sizeof(FLOAT32))) {
        return (Imaging)ImagingError_ValueError("filter size too large");
    }

    int xsize = imIn->xsize + 2 * margin;
    int ysize = imIn->ysize + 2 * margin;
    if (imIn->xsize == 0 || imIn->ysize == 0) {
        imOut = ImagingNew(imIn->mode, xsize, ysize);
        if (!imOut) {
            return NULL;
        }
        ImagingCopyPalette(imOut, imIn);
        return imOut;
    }

    imOut = ImagingNewDirty(imIn->mode, xsize, ysize);
    if (!imOut) {
        return NULL;
    }

#define EXPAND_LINE(type, image, yin, yout)                       \
    {                                                             \
        for (x = 0; x < margin; x++) {                            \
            imOut->image[yout][x] = imIn->image[yin][0];          \
        }                                                         \
        for (x = 0; x < imIn->xsize; x++) {                       \
            imOut->image[yout][x + margin] = imIn->image[yin][x]; \
        }                                                         \
        for (x = 0; x < margin; x++) {                            \
            imOut->image[yout][margin + imIn->xsize + x] =        \
                imIn->image[yin][imIn->xsize - 1];                \
        }                                                         \
    }

#define EXPAND(type, image)                                                      \
    {                                                                            \
        for (y = 0; y < margin; y++) {                                           \
            EXPAND_LINE(type, image, 0, y);                                      \
        }                                                                        \
        for (y = 0; y < imIn->ysize; y++) {                                      \
            EXPAND_LINE(type, image, y, y + margin);                             \
        }                                                                        \
        for (y = 0; y < margin; y++) {                                           \
            EXPAND_LINE(type, image, imIn->ysize - 1, margin + imIn->ysize + y); \
        }                                                                        \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

    if (imIn->image8) {
        EXPAND(UINT8, image8);
    } else {
        EXPAND(INT32, image32);
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

    ImagingCopyPalette(imOut, imIn);

    return imOut;
}

float
kernel_i16(int size, UINT8 *in0, int x, const float *kernel, int bigendian) {
    int i;
    float result = 0;
    int half_size = (size - 1) / 2;
    for (i = 0; i < size; i++) {
        int x1 = x + i - half_size;
        result += (float)(in0[x1 * 2 + (bigendian ? 1 : 0)] +
                          (in0[x1 * 2 + (bigendian ? 0 : 1)] << 8)) *
                  kernel[i];
    }
    return result;
}

/**
 * Convolve `im` with a 3x3 kernel, writing the result to imOut.
 *
 * Contract: imOut and im *MUST* be distinct images.
 */
static void
ImagingFilter3x3(Imaging imOut, Imaging im, const float *kernel, float offset) {
#define KERNEL1x3(in0, x, kernel, d)                                     \
    ((float)(in0[x - d]) * (kernel)[0] + (float)(in0[x]) * (kernel)[1] + \
     (float)(in0[x + d]) * (kernel)[2])

    int x = 0, y = 0;
    // restrict safety: assert im and imOut to be separate,
    // so we can safely `restrict` the copy pointers below
    assert(im->image != imOut->image);
    // Hoist for loop optimization.
    int xsize = im->xsize, ysize = im->ysize;

    memcpy(imOut->image[0], im->image[0], im->linesize);
    if (im->bands == 1) {
        // Add one time for rounding
        offset += 0.5;
        if (im->type == IMAGING_TYPE_INT32) {
            for (y = 1; y < ysize - 1; y++) {
                INT32 *restrict in_1 = (INT32 *)im->image[y - 1];
                INT32 *restrict in0 = (INT32 *)im->image[y];
                INT32 *restrict in1 = (INT32 *)im->image[y + 1];
                INT32 *restrict out = (INT32 *)imOut->image[y];

                out[0] = in0[0];
                for (x = 1; x < xsize - 1; x++) {
                    float ss = offset;
                    ss += KERNEL1x3(in1, x, &kernel[0], 1);
                    ss += KERNEL1x3(in0, x, &kernel[3], 1);
                    ss += KERNEL1x3(in_1, x, &kernel[6], 1);
                    out[x] = clip32(ss);
                }
                out[x] = in0[x];
            }
        } else if (im->type == IMAGING_TYPE_I16) {
            int bigendian = 0;
            if (
                im->mode == IMAGING_MODE_I_16B
#ifdef WORDS_BIGENDIAN
                || im->mode == IMAGING_MODE_I_16N
#endif
            ) {
                bigendian = 1;
            }
            for (y = 1; y < ysize - 1; y++) {
                UINT8 *restrict in_1 = (UINT8 *)im->image[y - 1];
                UINT8 *restrict in0 = (UINT8 *)im->image[y];
                UINT8 *restrict in1 = (UINT8 *)im->image[y + 1];
                UINT8 *restrict out = (UINT8 *)imOut->image[y];

                out[0] = in0[0];
                out[1] = in0[1];
                for (x = 1; x < xsize - 1; x++) {
                    float ss = offset;
                    ss += kernel_i16(3, in1, x, &kernel[0], bigendian);
                    ss += kernel_i16(3, in0, x, &kernel[3], bigendian);
                    ss += kernel_i16(3, in_1, x, &kernel[6], bigendian);
                    // NOT rounding here because `offset` already has a +0.5 bias.
                    int ss_int = clip16(ss);
                    out[x * 2 + (bigendian ? 1 : 0)] = (UINT8)(ss_int & 0xff);
                    out[x * 2 + (bigendian ? 0 : 1)] = (UINT8)(ss_int >> 8);
                }
                out[x * 2] = in0[x * 2];
                out[x * 2 + 1] = in0[x * 2 + 1];
            }
        } else {
            for (y = 1; y < ysize - 1; y++) {
                UINT8 *restrict in_1 = (UINT8 *)im->image[y - 1];
                UINT8 *restrict in0 = (UINT8 *)im->image[y];
                UINT8 *restrict in1 = (UINT8 *)im->image[y + 1];
                UINT8 *restrict out = (UINT8 *)imOut->image[y];

                out[0] = in0[0];
                for (x = 1; x < xsize - 1; x++) {
                    float ss = offset;
                    ss += KERNEL1x3(in1, x, &kernel[0], 1);
                    ss += KERNEL1x3(in0, x, &kernel[3], 1);
                    ss += KERNEL1x3(in_1, x, &kernel[6], 1);
                    out[x] = clip8(ss);
                }
                out[x] = in0[x];
            }
        }
    } else {
        // Add one time for rounding
        offset += 0.5;
        for (y = 1; y < ysize - 1; y++) {
            UINT8 *restrict in_1 = (UINT8 *)im->image[y - 1];
            UINT8 *restrict in0 = (UINT8 *)im->image[y];
            UINT8 *restrict in1 = (UINT8 *)im->image[y + 1];
            UINT8 *restrict out = (UINT8 *)imOut->image[y];

            memcpy(out, in0, sizeof(UINT32));
            if (im->bands == 2) {
                for (x = 1; x < xsize - 1; x++) {
                    float ss0 = offset;
                    float ss3 = offset;
                    UINT32 v;
                    ss0 += KERNEL1x3(in1, x * 4 + 0, &kernel[0], 4);
                    ss3 += KERNEL1x3(in1, x * 4 + 3, &kernel[0], 4);
                    ss0 += KERNEL1x3(in0, x * 4 + 0, &kernel[3], 4);
                    ss3 += KERNEL1x3(in0, x * 4 + 3, &kernel[3], 4);
                    ss0 += KERNEL1x3(in_1, x * 4 + 0, &kernel[6], 4);
                    ss3 += KERNEL1x3(in_1, x * 4 + 3, &kernel[6], 4);
                    v = MAKE_UINT32(clip8(ss0), 0, 0, clip8(ss3));
                    memcpy(out + x * sizeof(v), &v, sizeof(v));
                }
            } else if (im->bands == 3) {
                for (x = 1; x < xsize - 1; x++) {
                    float ss0 = offset;
                    float ss1 = offset;
                    float ss2 = offset;
                    UINT32 v;
                    ss0 += KERNEL1x3(in1, x * 4 + 0, &kernel[0], 4);
                    ss1 += KERNEL1x3(in1, x * 4 + 1, &kernel[0], 4);
                    ss2 += KERNEL1x3(in1, x * 4 + 2, &kernel[0], 4);
                    ss0 += KERNEL1x3(in0, x * 4 + 0, &kernel[3], 4);
                    ss1 += KERNEL1x3(in0, x * 4 + 1, &kernel[3], 4);
                    ss2 += KERNEL1x3(in0, x * 4 + 2, &kernel[3], 4);
                    ss0 += KERNEL1x3(in_1, x * 4 + 0, &kernel[6], 4);
                    ss1 += KERNEL1x3(in_1, x * 4 + 1, &kernel[6], 4);
                    ss2 += KERNEL1x3(in_1, x * 4 + 2, &kernel[6], 4);
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), 0);
                    memcpy(out + x * sizeof(v), &v, sizeof(v));
                }
            } else if (im->bands == 4) {
                for (x = 1; x < xsize - 1; x++) {
                    float ss0 = offset;
                    float ss1 = offset;
                    float ss2 = offset;
                    float ss3 = offset;
                    UINT32 v;
                    ss0 += KERNEL1x3(in1, x * 4 + 0, &kernel[0], 4);
                    ss1 += KERNEL1x3(in1, x * 4 + 1, &kernel[0], 4);
                    ss2 += KERNEL1x3(in1, x * 4 + 2, &kernel[0], 4);
                    ss3 += KERNEL1x3(in1, x * 4 + 3, &kernel[0], 4);
                    ss0 += KERNEL1x3(in0, x * 4 + 0, &kernel[3], 4);
                    ss1 += KERNEL1x3(in0, x * 4 + 1, &kernel[3], 4);
                    ss2 += KERNEL1x3(in0, x * 4 + 2, &kernel[3], 4);
                    ss3 += KERNEL1x3(in0, x * 4 + 3, &kernel[3], 4);
                    ss0 += KERNEL1x3(in_1, x * 4 + 0, &kernel[6], 4);
                    ss1 += KERNEL1x3(in_1, x * 4 + 1, &kernel[6], 4);
                    ss2 += KERNEL1x3(in_1, x * 4 + 2, &kernel[6], 4);
                    ss3 += KERNEL1x3(in_1, x * 4 + 3, &kernel[6], 4);
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), clip8(ss3));
                    memcpy(out + x * sizeof(v), &v, sizeof(v));
                }
            }
            memcpy(out + x * sizeof(UINT32), in0 + x * sizeof(UINT32), sizeof(UINT32));
        }
    }
    memcpy(imOut->image[y], im->image[y], im->linesize);
}

/**
 * Convolve `im` with a 5x5 kernel, writing the result to imOut.
 *
 * Contract: imOut and im *MUST* be distinct images.
 */
static void
ImagingFilter5x5(Imaging imOut, Imaging im, const float *kernel, float offset) {
#define KERNEL1x5(in0, x, kernel, d)                                             \
    ((float)(in0[x - d - d]) * (kernel)[0] + (float)(in0[x - d]) * (kernel)[1] + \
     (float)(in0[x]) * (kernel)[2] + (float)(in0[x + d]) * (kernel)[3] +         \
     (float)(in0[x + d + d]) * (kernel)[4])

    int x = 0, y = 0;

    // restrict safety: assert im and imOut to be separate,
    // so we can safely `restrict` the copy pointers below
    assert(im->image != imOut->image);
    // Hoist for loop optimization.
    int xsize = im->xsize, ysize = im->ysize;

    memcpy(imOut->image[0], im->image[0], im->linesize);
    memcpy(imOut->image[1], im->image[1], im->linesize);
    if (im->bands == 1) {
        // Add one time for rounding
        offset += 0.5;
        if (im->type == IMAGING_TYPE_INT32) {
            for (y = 2; y < ysize - 2; y++) {
                INT32 *restrict in_2 = (INT32 *)im->image[y - 2];
                INT32 *restrict in_1 = (INT32 *)im->image[y - 1];
                INT32 *restrict in0 = (INT32 *)im->image[y];
                INT32 *restrict in1 = (INT32 *)im->image[y + 1];
                INT32 *restrict in2 = (INT32 *)im->image[y + 2];
                INT32 *restrict out = (INT32 *)imOut->image[y];

                out[0] = in0[0];
                out[1] = in0[1];
                for (x = 2; x < xsize - 2; x++) {
                    float ss = offset;
                    ss += KERNEL1x5(in2, x, &kernel[0], 1);
                    ss += KERNEL1x5(in1, x, &kernel[5], 1);
                    ss += KERNEL1x5(in0, x, &kernel[10], 1);
                    ss += KERNEL1x5(in_1, x, &kernel[15], 1);
                    ss += KERNEL1x5(in_2, x, &kernel[20], 1);
                    out[x] = clip32(ss);
                }
                out[x + 0] = in0[x + 0];
                out[x + 1] = in0[x + 1];
            }
        } else if (im->type == IMAGING_TYPE_I16) {
            int bigendian = 0;
            if (
                im->mode == IMAGING_MODE_I_16B
#ifdef WORDS_BIGENDIAN
                || im->mode == IMAGING_MODE_I_16N
#endif
            ) {
                bigendian = 1;
            }
            for (y = 2; y < ysize - 2; y++) {
                UINT8 *restrict in_2 = (UINT8 *)im->image[y - 2];
                UINT8 *restrict in_1 = (UINT8 *)im->image[y - 1];
                UINT8 *restrict in0 = (UINT8 *)im->image[y];
                UINT8 *restrict in1 = (UINT8 *)im->image[y + 1];
                UINT8 *restrict in2 = (UINT8 *)im->image[y + 2];
                UINT8 *restrict out = (UINT8 *)imOut->image[y];

                out[0] = in0[0];
                out[1] = in0[1];
                out[2] = in0[2];
                out[3] = in0[3];
                for (x = 2; x < xsize - 2; x++) {
                    float ss = offset;
                    ss += kernel_i16(5, in2, x, &kernel[0], bigendian);
                    ss += kernel_i16(5, in1, x, &kernel[5], bigendian);
                    ss += kernel_i16(5, in0, x, &kernel[10], bigendian);
                    ss += kernel_i16(5, in_1, x, &kernel[15], bigendian);
                    ss += kernel_i16(5, in_2, x, &kernel[20], bigendian);
                    // NOT rounding here because `offset` already has a +0.5 bias.
                    int ss_int = clip16(ss);
                    out[x * 2 + (bigendian ? 1 : 0)] = (UINT8)(ss_int & 0xff);
                    out[x * 2 + (bigendian ? 0 : 1)] = (UINT8)(ss_int >> 8);
                }
                out[x * 2 + 0] = in0[x * 2 + 0];
                out[x * 2 + 1] = in0[x * 2 + 1];
                out[x * 2 + 2] = in0[x * 2 + 2];
                out[x * 2 + 3] = in0[x * 2 + 3];
            }
        } else {
            for (y = 2; y < ysize - 2; y++) {
                UINT8 *restrict in_2 = (UINT8 *)im->image[y - 2];
                UINT8 *restrict in_1 = (UINT8 *)im->image[y - 1];
                UINT8 *restrict in0 = (UINT8 *)im->image[y];
                UINT8 *restrict in1 = (UINT8 *)im->image[y + 1];
                UINT8 *restrict in2 = (UINT8 *)im->image[y + 2];
                UINT8 *restrict out = (UINT8 *)imOut->image[y];

                out[0] = in0[0];
                out[1] = in0[1];
                for (x = 2; x < xsize - 2; x++) {
                    float ss = offset;
                    ss += KERNEL1x5(in2, x, &kernel[0], 1);
                    ss += KERNEL1x5(in1, x, &kernel[5], 1);
                    ss += KERNEL1x5(in0, x, &kernel[10], 1);
                    ss += KERNEL1x5(in_1, x, &kernel[15], 1);
                    ss += KERNEL1x5(in_2, x, &kernel[20], 1);
                    out[x] = clip8(ss);
                }
                out[x + 0] = in0[x + 0];
                out[x + 1] = in0[x + 1];
            }
        }
    } else {
        // Add one time for rounding
        offset += 0.5;
        for (y = 2; y < ysize - 2; y++) {
            UINT8 *restrict in_2 = (UINT8 *)im->image[y - 2];
            UINT8 *restrict in_1 = (UINT8 *)im->image[y - 1];
            UINT8 *restrict in0 = (UINT8 *)im->image[y];
            UINT8 *restrict in1 = (UINT8 *)im->image[y + 1];
            UINT8 *restrict in2 = (UINT8 *)im->image[y + 2];
            UINT8 *restrict out = (UINT8 *)imOut->image[y];

            memcpy(out, in0, sizeof(UINT32) * 2);
            if (im->bands == 2) {
                for (x = 2; x < xsize - 2; x++) {
                    float ss0 = offset;
                    float ss3 = offset;
                    UINT32 v;
                    ss0 += KERNEL1x5(in2, x * 4 + 0, &kernel[0], 4);
                    ss3 += KERNEL1x5(in2, x * 4 + 3, &kernel[0], 4);
                    ss0 += KERNEL1x5(in1, x * 4 + 0, &kernel[5], 4);
                    ss3 += KERNEL1x5(in1, x * 4 + 3, &kernel[5], 4);
                    ss0 += KERNEL1x5(in0, x * 4 + 0, &kernel[10], 4);
                    ss3 += KERNEL1x5(in0, x * 4 + 3, &kernel[10], 4);
                    ss0 += KERNEL1x5(in_1, x * 4 + 0, &kernel[15], 4);
                    ss3 += KERNEL1x5(in_1, x * 4 + 3, &kernel[15], 4);
                    ss0 += KERNEL1x5(in_2, x * 4 + 0, &kernel[20], 4);
                    ss3 += KERNEL1x5(in_2, x * 4 + 3, &kernel[20], 4);
                    v = MAKE_UINT32(clip8(ss0), 0, 0, clip8(ss3));
                    memcpy(out + x * sizeof(v), &v, sizeof(v));
                }
            } else if (im->bands == 3) {
                for (x = 2; x < xsize - 2; x++) {
                    float ss0 = offset;
                    float ss1 = offset;
                    float ss2 = offset;
                    UINT32 v;
                    ss0 += KERNEL1x5(in2, x * 4 + 0, &kernel[0], 4);
                    ss1 += KERNEL1x5(in2, x * 4 + 1, &kernel[0], 4);
                    ss2 += KERNEL1x5(in2, x * 4 + 2, &kernel[0], 4);
                    ss0 += KERNEL1x5(in1, x * 4 + 0, &kernel[5], 4);
                    ss1 += KERNEL1x5(in1, x * 4 + 1, &kernel[5], 4);
                    ss2 += KERNEL1x5(in1, x * 4 + 2, &kernel[5], 4);
                    ss0 += KERNEL1x5(in0, x * 4 + 0, &kernel[10], 4);
                    ss1 += KERNEL1x5(in0, x * 4 + 1, &kernel[10], 4);
                    ss2 += KERNEL1x5(in0, x * 4 + 2, &kernel[10], 4);
                    ss0 += KERNEL1x5(in_1, x * 4 + 0, &kernel[15], 4);
                    ss1 += KERNEL1x5(in_1, x * 4 + 1, &kernel[15], 4);
                    ss2 += KERNEL1x5(in_1, x * 4 + 2, &kernel[15], 4);
                    ss0 += KERNEL1x5(in_2, x * 4 + 0, &kernel[20], 4);
                    ss1 += KERNEL1x5(in_2, x * 4 + 1, &kernel[20], 4);
                    ss2 += KERNEL1x5(in_2, x * 4 + 2, &kernel[20], 4);
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), 0);
                    memcpy(out + x * sizeof(v), &v, sizeof(v));
                }
            } else if (im->bands == 4) {
                for (x = 2; x < xsize - 2; x++) {
                    float ss0 = offset;
                    float ss1 = offset;
                    float ss2 = offset;
                    float ss3 = offset;
                    UINT32 v;
                    ss0 += KERNEL1x5(in2, x * 4 + 0, &kernel[0], 4);
                    ss1 += KERNEL1x5(in2, x * 4 + 1, &kernel[0], 4);
                    ss2 += KERNEL1x5(in2, x * 4 + 2, &kernel[0], 4);
                    ss3 += KERNEL1x5(in2, x * 4 + 3, &kernel[0], 4);
                    ss0 += KERNEL1x5(in1, x * 4 + 0, &kernel[5], 4);
                    ss1 += KERNEL1x5(in1, x * 4 + 1, &kernel[5], 4);
                    ss2 += KERNEL1x5(in1, x * 4 + 2, &kernel[5], 4);
                    ss3 += KERNEL1x5(in1, x * 4 + 3, &kernel[5], 4);
                    ss0 += KERNEL1x5(in0, x * 4 + 0, &kernel[10], 4);
                    ss1 += KERNEL1x5(in0, x * 4 + 1, &kernel[10], 4);
                    ss2 += KERNEL1x5(in0, x * 4 + 2, &kernel[10], 4);
                    ss3 += KERNEL1x5(in0, x * 4 + 3, &kernel[10], 4);
                    ss0 += KERNEL1x5(in_1, x * 4 + 0, &kernel[15], 4);
                    ss1 += KERNEL1x5(in_1, x * 4 + 1, &kernel[15], 4);
                    ss2 += KERNEL1x5(in_1, x * 4 + 2, &kernel[15], 4);
                    ss3 += KERNEL1x5(in_1, x * 4 + 3, &kernel[15], 4);
                    ss0 += KERNEL1x5(in_2, x * 4 + 0, &kernel[20], 4);
                    ss1 += KERNEL1x5(in_2, x * 4 + 1, &kernel[20], 4);
                    ss2 += KERNEL1x5(in_2, x * 4 + 2, &kernel[20], 4);
                    ss3 += KERNEL1x5(in_2, x * 4 + 3, &kernel[20], 4);
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), clip8(ss3));
                    memcpy(out + x * sizeof(v), &v, sizeof(v));
                }
            }
            memcpy(
                out + x * sizeof(UINT32), in0 + x * sizeof(UINT32), sizeof(UINT32) * 2
            );
        }
    }
    memcpy(imOut->image[y], im->image[y], im->linesize);
    memcpy(imOut->image[y + 1], im->image[y + 1], im->linesize);
}

#if defined(FILTER_SIMD)
/* The n coefficients and the offset scaled by 1 << precision for the
   integer kernels. Returns 0 if a coefficient doesn't fit 16 bits, or with
   exact, if one of them or the offset isn't an integer once scaled. */
static int
filter_int_kernel(
    const FLOAT32 *kernel,
    int n,
    FLOAT32 offset,
    int precision,
    int exact,
    INT16 *norm_kernel,
    INT32 *norm_offset
) {
    int i;
    FLOAT32 tmp;
    for (i = 0; i < n; i++) {
        tmp = kernel[i] * (1 << precision);
        if (exact && (tmp >= 32768 || tmp <= -32769 || tmp != floorf(tmp))) {
            return 0;
        }
        tmp += (tmp >= 0) ? 0.5 : -0.5;
        if (tmp >= 32768 || tmp <= -32769) {
            return 0;
        }
        norm_kernel[i] = tmp;
    }
    tmp = offset * (1 << precision);
    if (exact && (tmp >= 1 << 30 || tmp <= -1 << 30 || tmp != floorf(tmp))) {
        return 0;
    }
    tmp += (tmp >= 0) ? 0.5 : -0.5;
    if (tmp >= 1 << 30) {
        *norm_offset = 1 << 30;
    } else if (tmp <= -1 << 30) {
        *norm_offset = -1 << 30;
    } else {
        *norm_offset = tmp;
    }
    return 1;
}

/* 8-bit images: integer kernels when the coefficients fit 16 bits,
   float kernels otherwise. Returns 0 if the image has to
   go to the generic code. */
static int
ImagingFilterSIMD(
    Imaging imOut,
    Imaging im,
    int xsize,
    int ysize,
    const FLOAT32 *kernel,
    FLOAT32 offset
) {
    int fast_path = 1;
    int precision = PRECISION_BITS;
    INT16 norm_kernel[25];
    INT32 norm_offset;

    /* The coefficients in 16 bits, scaled by 1 << PRECISION_BITS and
       rounded. If one doesn't fit, as coefficients of 4 and more don't
       (FIND_EDGES, CONTOUR, EDGE_ENHANCE), the largest precision at which
       all of them and the offset are exact: the sums are then exact, and
       rounded half up as by the code without SIMD, so the result is the
       same. */
    if (!filter_int_kernel(
            kernel, xsize * ysize, offset, precision, 0, norm_kernel, &norm_offset
        )) {
        for (precision = PRECISION_BITS - 1; precision >= 1; precision--) {
            if (filter_int_kernel(
                    kernel,
                    xsize * ysize,
                    offset,
                    precision,
                    1,
                    norm_kernel,
                    &norm_offset
                )) {
                break;
            }
        }
        fast_path = precision >= 1;
    }

#if defined(IMAGING_SIMD_SSE4)
    if (xsize == 3) {
        if (im->image8 && fast_path) {
            ImagingFilter3x3i_u8_simd(imOut, im, norm_kernel, norm_offset, precision);
        } else if (im->image8) {
            ImagingFilter3x3f_u8(imOut, im, kernel, offset);
        } else if (fast_path) {
            ImagingFilter3x3i_4u8(imOut, im, norm_kernel, norm_offset, precision);
        } else {
            ImagingFilter3x3f_4u8(imOut, im, kernel, offset);
        }
    } else {
        if (im->image8 && fast_path) {
            ImagingFilter5x5i_u8_simd(imOut, im, norm_kernel, norm_offset, precision);
        } else if (im->image8) {
            ImagingFilter5x5f_u8(imOut, im, kernel, offset);
        } else if (fast_path) {
            ImagingFilter5x5i_4u8(imOut, im, norm_kernel, norm_offset, precision);
        } else {
            ImagingFilter5x5f_4u8(imOut, im, kernel, offset);
        }
    }
#else
    if (!fast_path) {
        /* 1-byte images keep the generic code, which GCC vectorises well:
           on the Jetson Orin it was 1.5x faster than the NEON float kernel
           (CONTOUR on L: 15.5 vs 22.9 ms). 4-byte images gain 1.7x. */
        if (im->image8) {
            return 0;
        }
        ImagingFilterNxNf_neon(imOut, im, kernel, offset, xsize, 4);
        return 1;
    }
    if (xsize == 3) {
        if (im->image8) {
            ImagingFilter3x3i_u8_simd(imOut, im, norm_kernel, norm_offset, precision);
        } else {
            ImagingFilter3x3i_4u8_simd(imOut, im, norm_kernel, norm_offset, precision);
        }
    } else {
        if (im->image8) {
            ImagingFilter5x5i_u8_simd(imOut, im, norm_kernel, norm_offset, precision);
        } else {
            ImagingFilter5x5i_4u8_simd(imOut, im, norm_kernel, norm_offset, precision);
        }
    }
#endif
    return 1;
}
#endif

Imaging
ImagingFilter(Imaging im, int xsize, int ysize, const FLOAT32 *kernel, FLOAT32 offset) {
    Imaging imOut;
    ImagingSectionCookie cookie;

    if (im->type == IMAGING_TYPE_FLOAT32) {
        return (Imaging)ImagingError_ModeError();
    }

    if (im->xsize < xsize || im->ysize < ysize) {
        return ImagingCopy(im);
    }

    if ((xsize != 3 && xsize != 5) || xsize != ysize) {
        return (Imaging)ImagingError_ValueError("bad kernel size");
    }

    imOut = ImagingNewDirty(im->mode, im->xsize, im->ysize);
    if (!imOut) {
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&im->mutex);

    int done = 0;
#if defined(FILTER_SIMD)
    if (im->type == IMAGING_TYPE_UINT8) {
        done = ImagingFilterSIMD(imOut, im, xsize, ysize, kernel, offset);
    }
#endif
    if (!done) {
        if (xsize == 3) {
            ImagingFilter3x3(imOut, im, kernel, offset);
        } else {
            ImagingFilter5x5(imOut, im, kernel, offset);
        }
    }
    MUTEX_UNLOCK(&im->mutex);
    ImagingSectionLeave(&cookie);
    return imOut;
}
