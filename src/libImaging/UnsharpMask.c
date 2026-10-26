/* PILusm, a gaussian blur and unsharp masking library for PIL
   By Kevin Cazabon, copyright 2003
   kevin_cazabon@hotmail.com
   kevin@cazabon.com */

/* Originally released under LGPL.  Graciously donated to PIL
   for distribution under the standard PIL license in 2009." */

#include "Imaging.h"

static inline UINT8
clip8(int in) {
    if (in >= 255) {
        return 255;
    }
    if (in <= 0) {
        return 0;
    }
    return (UINT8)in;
}

/* Sharpens one row of bytes in place: out holds the blurred row. */
static void
ImagingUnsharpMaskRow(
    UINT8 *out, const UINT8 *in, int size, int percent, int threshold
) {
    int x = 0, diff;

    /* arm64 uses the loop below, which compilers vectorize with a
       multiply-high for the division; that beat NEON float division. */
#if defined(IMAGING_SIMD_SSE4)
    /* diff * percent / 100 via float division: while |diff * percent| is
       below 2^24 the quotient truncates to the integer division result. */
    if (percent >= -65535 && percent <= 65535) {
        __m128i pct = _mm_set1_epi32(percent);
        __m128i thr = _mm_set1_epi32(threshold);
        __m128 hundred = _mm_set1_ps(100.0f);
        for (; x < size - 15; x += 16) {
            __m128i src = _mm_loadu_si128((__m128i *)&in[x]);
            __m128i blur = _mm_loadu_si128((__m128i *)&out[x]);
            __m128i res[4];
#define UNSHARP_4(i)                                                                  \
    {                                                                                 \
        __m128i v = _mm_cvtepu8_epi32(_mm_srli_si128(src, i * 4));                    \
        __m128i d = _mm_sub_epi32(v, _mm_cvtepu8_epi32(_mm_srli_si128(blur, i * 4))); \
        __m128i q = _mm_cvttps_epi32(                                                 \
            _mm_div_ps(_mm_cvtepi32_ps(_mm_mullo_epi32(d, pct)), hundred)             \
        );                                                                            \
        __m128i mask = _mm_cmpgt_epi32(_mm_abs_epi32(d), thr);                        \
        res[i] = _mm_blendv_epi8(v, _mm_add_epi32(v, q), mask);                       \
    }
            UNSHARP_4(0);
            UNSHARP_4(1);
            UNSHARP_4(2);
            UNSHARP_4(3);
#undef UNSHARP_4
            __m128i lo = _mm_packs_epi32(res[0], res[1]);
            __m128i hi = _mm_packs_epi32(res[2], res[3]);
            _mm_storeu_si128((__m128i *)&out[x], _mm_packus_epi16(lo, hi));
        }
    }
#endif

    for (; x < size; x++) {
        /* compare in/out pixels, apply sharpening */
        diff = in[x] - out[x];
        if (abs(diff) > threshold) {
            /* add the diff*percent to the original pixel */
            out[x] = clip8(in[x] + diff * percent / 100);
        } else {
            /* new pixel is the same as imIn */
            out[x] = in[x];
        }
    }
}

Imaging
ImagingUnsharpMask(
    Imaging imOut, Imaging imIn, float radius, int percent, int threshold
) {
    ImagingSectionCookie cookie;
    Imaging result;

    int y;

    /* First, do a gaussian blur on the image, putting results in imOut
       temporarily. All format checks are in gaussian blur. */
    result = ImagingGaussianBlur(imOut, imIn, radius, radius, 3);
    if (!result) {
        return NULL;
    }

    /* Now, go through each pixel, compare "normal" pixel to blurred
       pixel. If the difference is more than threshold values, apply
       the OPPOSITE correction to the amount of blur, multiplied by
       percent. */

    ImagingSectionEnter(&cookie);

    for (y = 0; y < imIn->ysize; y++) {
        ImagingUnsharpMaskRow(
            (UINT8 *)imOut->image[y],
            (UINT8 *)imIn->image[y],
            imIn->linesize,
            percent,
            threshold
        );
    }

    ImagingSectionLeave(&cookie);

    return imOut;
}
