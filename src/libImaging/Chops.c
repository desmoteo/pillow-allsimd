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

/**
 * Return a newly allocated image containing the lighter pixels of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopLighter(Imaging imIn1, Imaging imIn2) {
    CHOP((in1[x] > in2[x]) ? in1[x] : in2[x]);
}

/**
 * Return a newly allocated image containing the darker pixels of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopDarker(Imaging imIn1, Imaging imIn2) {
    CHOP((in1[x] < in2[x]) ? in1[x] : in2[x]);
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
#elif defined(IMAGING_SIMD_NEON)
    {
        uint8x16_t inv = vdupq_n_u8(invert);
        uint16x8_t one = vdupq_n_u16(1);
        for (; x < n - 15; x += 16) {
            uint8x16_t a = veorq_u8(vld1q_u8(in1 + x), inv);
            uint8x16_t b = veorq_u8(vld1q_u8(in2 + x), inv);
            uint16x8_t lo = vmull_u8(vget_low_u8(a), vget_low_u8(b));
            uint16x8_t hi = vmull_high_u8(a, b);
            // (x + (1 + (x >> 8))) >> 8, narrowed to 8 bits
            uint8x8_t out_lo = vaddhn_u16(lo, vsraq_n_u16(one, lo, 8));
            uint8x8_t out_hi = vaddhn_u16(hi, vsraq_n_u16(one, hi, 8));
            vst1q_u8(out + x, veorq_u8(vcombine_u8(out_lo, out_hi), inv));
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
Imaging
ImagingChopAdd(Imaging imIn1, Imaging imIn2, float scale, int offset) {
    CHOP(((int)in1[x] + (int)in2[x]) / scale + offset);
}

/**
 * Return a newly allocated image containing the per-pixel difference, divided by
 * `scale` and shifted by `offset`, clipped to 0-255.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopSubtract(Imaging imIn1, Imaging imIn2, float scale, int offset) {
    CHOP(((int)in1[x] - (int)in2[x]) / scale + offset);
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

/**
 * Return a newly allocated image containing the soft-light blend of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopSoftLight(Imaging imIn1, Imaging imIn2) {
    CHOP2(
        (((255 - in1[x]) * (in1[x] * in2[x])) / 65536) +
            (in1[x] * (255 - ((255 - in1[x]) * (255 - in2[x]) / 255))) / 255,
        IMAGING_MODE_UNKNOWN
    );
}

/**
 * Return a newly allocated image containing the hard-light blend of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingChopHardLight(Imaging imIn1, Imaging imIn2) {
    CHOP2(
        (in2[x] < 128) ? ((in1[x] * in2[x]) / 127)
                       : 255 - (((255 - in2[x]) * (255 - in1[x])) / 127),
        IMAGING_MODE_UNKNOWN
    );
}

/**
 * Return a newly allocated image containing the overlay blend of the two images.
 *
 * Contract: imIn1 and imIn2 are read-only and may alias each other.
 */
Imaging
ImagingOverlay(Imaging imIn1, Imaging imIn2) {
    CHOP2(
        (in1[x] < 128) ? ((in1[x] * in2[x]) / 127)
                       : 255 - (((255 - in1[x]) * (255 - in2[x])) / 127),
        IMAGING_MODE_UNKNOWN
    );
}
