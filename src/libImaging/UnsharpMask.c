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
ImagingUnsharpMaskRow(UINT8 *out, const UINT8 *in, int size, int percent, int threshold) {
    int x = 0, diff;

#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
    /* diff * percent / 100 via float division: while |diff * percent| is
       below 2^24 the quotient truncates to the integer division result. */
    if (percent >= -65535 && percent <= 65535) {
#if defined(IMAGING_SIMD_SSE4)
        __m128i pct = _mm_set1_epi32(percent);
        __m128i thr = _mm_set1_epi32(threshold);
        __m128 hundred = _mm_set1_ps(100.0f);
        for (; x < size - 15; x += 16) {
            __m128i src = _mm_loadu_si128((__m128i *)&in[x]);
            __m128i blur = _mm_loadu_si128((__m128i *)&out[x]);
            __m128i res[4];
#define UNSHARP_4(i)                                                              \
    {                                                                             \
        __m128i v = _mm_cvtepu8_epi32(_mm_srli_si128(src, i * 4));                \
        __m128i d = _mm_sub_epi32(v, _mm_cvtepu8_epi32(_mm_srli_si128(blur, i * 4))); \
        __m128i q = _mm_cvttps_epi32(                                             \
            _mm_div_ps(_mm_cvtepi32_ps(_mm_mullo_epi32(d, pct)), hundred));       \
        __m128i mask = _mm_cmpgt_epi32(_mm_abs_epi32(d), thr);                    \
        res[i] = _mm_blendv_epi8(v, _mm_add_epi32(v, q), mask);                   \
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
#else
        int32x4_t pct = vdupq_n_s32(percent);
        int32x4_t thr = vdupq_n_s32(threshold);
        float32x4_t hundred = vdupq_n_f32(100.0f);
        for (; x < size - 15; x += 16) {
            uint8x16_t src = vld1q_u8(&in[x]);
            uint8x16_t blur = vld1q_u8(&out[x]);
            int16x8_t v16[2] = {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src)))};
            int16x8_t b16[2] = {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(blur))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(blur)))};
            int16x4_t res[4];
            for (int i = 0; i < 4; i++) {
                int16x4_t vs = i & 1 ? vget_high_s16(v16[i / 2]) : vget_low_s16(v16[i / 2]);
                int16x4_t bs = i & 1 ? vget_high_s16(b16[i / 2]) : vget_low_s16(b16[i / 2]);
                int32x4_t v = vmovl_s16(vs);
                int32x4_t d = vsubl_s16(vs, bs);
                int32x4_t q = vcvtq_s32_f32(
                    vdivq_f32(vcvtq_f32_s32(vmulq_s32(d, pct)), hundred));
                uint32x4_t mask = vcgtq_s32(vabsq_s32(d), thr);
                res[i] = vqmovn_s32(vbslq_s32(mask, vaddq_s32(v, q), v));
            }
            vst1q_u8(&out[x], vcombine_u8(
                vqmovun_s16(vcombine_s16(res[0], res[1])),
                vqmovun_s16(vcombine_s16(res[2], res[3]))));
        }
#endif
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
            (UINT8 *)imOut->image[y], (UINT8 *)imIn->image[y], imIn->linesize,
            percent, threshold
        );
    }

    ImagingSectionLeave(&cookie);

    return imOut;
}
