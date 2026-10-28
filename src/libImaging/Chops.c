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

Imaging
ImagingChopLighter(Imaging imIn1, Imaging imIn2) {
    CHOP((in1[x] > in2[x]) ? in1[x] : in2[x]);
}

Imaging
ImagingChopDarker(Imaging imIn1, Imaging imIn2) {
    CHOP((in1[x] < in2[x]) ? in1[x] : in2[x]);
}

Imaging
ImagingChopDifference(Imaging imIn1, Imaging imIn2) {
    CHOP(abs((int)in1[x] - (int)in2[x]));
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
    int x = 0;

#if defined(IMAGING_SIMD_AVX2)
    {
        __m256i mm_inv = _mm256_set1_epi8((char)invert);
        __m256i mm_one = _mm256_set1_epi16(1);
        __m256i mm_zero = _mm256_setzero_si256();
        for (; x < n - 31; x += 32) {
            __m256i a = _mm256_xor_si256(
                _mm256_loadu_si256((const __m256i *)(in1 + x)), mm_inv
            );
            __m256i b = _mm256_xor_si256(
                _mm256_loadu_si256((const __m256i *)(in2 + x)), mm_inv
            );
            // Unpack and pack both work within 128-bit lanes,
            // so the bytes come back in their original order.
            __m256i lo = _mm256_mullo_epi16(
                _mm256_unpacklo_epi8(a, mm_zero), _mm256_unpacklo_epi8(b, mm_zero)
            );
            __m256i hi = _mm256_mullo_epi16(
                _mm256_unpackhi_epi8(a, mm_zero), _mm256_unpackhi_epi8(b, mm_zero)
            );
            lo = _mm256_srli_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(lo, mm_one), _mm256_srli_epi16(lo, 8)
                ),
                8
            );
            hi = _mm256_srli_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(hi, mm_one), _mm256_srli_epi16(hi, 8)
                ),
                8
            );
            _mm256_storeu_si256(
                (__m256i *)(out + x),
                _mm256_xor_si256(_mm256_packus_epi16(lo, hi), mm_inv)
            );
        }
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    {
        __m128i mm_inv = _mm_set1_epi8((char)invert);
        __m128i mm_one = _mm_set1_epi16(1);
        __m128i mm_zero = _mm_setzero_si128();
        for (; x < n - 15; x += 16) {
            __m128i a =
                _mm_xor_si128(_mm_loadu_si128((const __m128i *)(in1 + x)), mm_inv);
            __m128i b =
                _mm_xor_si128(_mm_loadu_si128((const __m128i *)(in2 + x)), mm_inv);
            __m128i lo = _mm_mullo_epi16(
                _mm_unpacklo_epi8(a, mm_zero), _mm_unpacklo_epi8(b, mm_zero)
            );
            __m128i hi = _mm_mullo_epi16(
                _mm_unpackhi_epi8(a, mm_zero), _mm_unpackhi_epi8(b, mm_zero)
            );
            lo = _mm_srli_epi16(
                _mm_add_epi16(_mm_add_epi16(lo, mm_one), _mm_srli_epi16(lo, 8)), 8
            );
            hi = _mm_srli_epi16(
                _mm_add_epi16(_mm_add_epi16(hi, mm_one), _mm_srli_epi16(hi, 8)), 8
            );
            _mm_storeu_si128(
                (__m128i *)(out + x), _mm_xor_si128(_mm_packus_epi16(lo, hi), mm_inv)
            );
        }
    }
#endif

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

Imaging
ImagingChopMultiply(Imaging imIn1, Imaging imIn2) {
    return chop_multiply(imIn1, imIn2, 0);
}

Imaging
ImagingChopScreen(Imaging imIn1, Imaging imIn2) {
    return chop_multiply(imIn1, imIn2, 255);
}

Imaging
ImagingChopAdd(Imaging imIn1, Imaging imIn2, float scale, int offset) {
    CHOP(((int)in1[x] + (int)in2[x]) / scale + offset);
}

Imaging
ImagingChopSubtract(Imaging imIn1, Imaging imIn2, float scale, int offset) {
    CHOP(((int)in1[x] - (int)in2[x]) / scale + offset);
}

Imaging
ImagingChopAnd(Imaging imIn1, Imaging imIn2) {
    CHOP2((in1[x] && in2[x]) ? 255 : 0, IMAGING_MODE_1);
}

Imaging
ImagingChopOr(Imaging imIn1, Imaging imIn2) {
    CHOP2((in1[x] || in2[x]) ? 255 : 0, IMAGING_MODE_1);
}

Imaging
ImagingChopXor(Imaging imIn1, Imaging imIn2) {
    CHOP2(((in1[x] != 0) ^ (in2[x] != 0)) ? 255 : 0, IMAGING_MODE_1);
}

Imaging
ImagingChopAddModulo(Imaging imIn1, Imaging imIn2) {
    CHOP2(in1[x] + in2[x], IMAGING_MODE_UNKNOWN);
}

Imaging
ImagingChopSubtractModulo(Imaging imIn1, Imaging imIn2) {
    CHOP2(in1[x] - in2[x], IMAGING_MODE_UNKNOWN);
}

Imaging
ImagingChopSoftLight(Imaging imIn1, Imaging imIn2) {
    CHOP2(
        (((255 - in1[x]) * (in1[x] * in2[x])) / 65536) +
            (in1[x] * (255 - ((255 - in1[x]) * (255 - in2[x]) / 255))) / 255,
        IMAGING_MODE_UNKNOWN
    );
}

Imaging
ImagingChopHardLight(Imaging imIn1, Imaging imIn2) {
    CHOP2(
        (in2[x] < 128) ? ((in1[x] * in2[x]) / 127)
                       : 255 - (((255 - in2[x]) * (255 - in1[x])) / 127),
        IMAGING_MODE_UNKNOWN
    );
}

Imaging
ImagingOverlay(Imaging imIn1, Imaging imIn2) {
    CHOP2(
        (in1[x] < 128) ? ((in1[x] * in2[x]) / 127)
                       : 255 - (((255 - in1[x]) * (255 - in2[x])) / 127),
        IMAGING_MODE_UNKNOWN
    );
}
