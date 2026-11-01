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

/* Fast mode (design/SIMD_FAST_KERNELS.md): in1 + (in2 - in1) a for a = a15 / 2^15,
   rounded half up, (d a15 + 2^14) >> 15: what pmulhrsw and vqrdmulh compute,
   so every build gives the same bytes. The result lies between in1 and in2. */
static void
blend_fast_row(
    UINT8 *restrict out, const UINT8 *in1, const UINT8 *in2, int n, INT16 a15
) {
    int x = 0;
#if defined(IMAGING_SIMD_AVX2)
    const __m256i va = _mm256_set1_epi16(a15), z = _mm256_setzero_si256();
    for (; x + 32 <= n; x += 32) {
        const __m256i p = _mm256_loadu_si256((const __m256i *)(in1 + x));
        const __m256i q = _mm256_loadu_si256((const __m256i *)(in2 + x));
        __m256i p0 = _mm256_unpacklo_epi8(p, z), p1 = _mm256_unpackhi_epi8(p, z);
        const __m256i d0 = _mm256_sub_epi16(_mm256_unpacklo_epi8(q, z), p0);
        const __m256i d1 = _mm256_sub_epi16(_mm256_unpackhi_epi8(q, z), p1);
        p0 = _mm256_add_epi16(p0, _mm256_mulhrs_epi16(d0, va));
        p1 = _mm256_add_epi16(p1, _mm256_mulhrs_epi16(d1, va));
        _mm256_storeu_si256((__m256i *)(out + x), _mm256_packus_epi16(p0, p1));
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    {
        const __m128i va = _mm_set1_epi16(a15), z = _mm_setzero_si128();
        for (; x + 16 <= n; x += 16) {
            const __m128i p = _mm_loadu_si128((const __m128i *)(in1 + x));
            const __m128i q = _mm_loadu_si128((const __m128i *)(in2 + x));
            __m128i p0 = _mm_unpacklo_epi8(p, z), p1 = _mm_unpackhi_epi8(p, z);
            const __m128i d0 = _mm_sub_epi16(_mm_unpacklo_epi8(q, z), p0);
            const __m128i d1 = _mm_sub_epi16(_mm_unpackhi_epi8(q, z), p1);
            p0 = _mm_add_epi16(p0, _mm_mulhrs_epi16(d0, va));
            p1 = _mm_add_epi16(p1, _mm_mulhrs_epi16(d1, va));
            _mm_storeu_si128((__m128i *)(out + x), _mm_packus_epi16(p0, p1));
        }
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; x + 16 <= n; x += 16) {
        const uint8x16_t p = vld1q_u8(in1 + x), q = vld1q_u8(in2 + x);
        const int16x8_t d0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(q), vget_low_u8(p)));
        const int16x8_t d1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(q), vget_high_u8(p)));
        const int16x8_t r0 = vaddq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p))), vqrdmulhq_n_s16(d0, a15)
        );
        const int16x8_t r1 = vaddq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p))), vqrdmulhq_n_s16(d1, a15)
        );
        vst1q_u8(out + x, vcombine_u8(vqmovun_s16(r0), vqmovun_s16(r1)));
    }
#endif
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
    int x, y;

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
        for (y = 0; y < ysize; y++) {
            // restrict safe: imIn1 and imIn2 are read-only; imOut is a new allocation
            UINT8 *restrict in1 = (UINT8 *)imIn1->image[y];
            UINT8 *restrict in2 = (UINT8 *)imIn2->image[y];
            UINT8 *restrict out = (UINT8 *)imOut->image[y];
            for (x = 0; x < linesize; x++) {
                out[x] = (UINT8)((int)in1[x] + alpha * ((int)in2[x] - (int)in1[x]));
            }
        }
    } else {
        /* Extrapolation; must make sure to clip resulting values */
        for (y = 0; y < ysize; y++) {
            // restrict safe: imIn1 and imIn2 are read-only; imOut is a new allocation
            UINT8 *restrict in1 = (UINT8 *)imIn1->image[y];
            UINT8 *restrict in2 = (UINT8 *)imIn2->image[y];
            UINT8 *restrict out = (UINT8 *)imOut->image[y];
            for (x = 0; x < linesize; x++) {
                float temp = (float)((int)in1[x] + alpha * ((int)in2[x] - (int)in1[x]));
                if (temp <= 0.0) {
                    out[x] = 0;
                } else if (temp >= 255.0) {
                    out[x] = 255;
                } else {
                    out[x] = (UINT8)temp;
                }
            }
        }
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}
