/*
 * The Python Imaging Library
 * $Id$
 *
 * convert images
 *
 * history:
 * 1995-06-15 fl   created
 * 1995-11-28 fl   added some "RGBA" and "CMYK" conversions
 * 1996-04-22 fl   added "1" conversions (same as "L")
 * 1996-05-05 fl   added palette conversions (hack)
 * 1996-07-23 fl   fixed "1" conversions to zero/non-zero convention
 * 1996-11-01 fl   fixed "P" to "L" and "RGB" to "1" conversions
 * 1996-12-29 fl   set alpha byte in RGB converters
 * 1997-05-12 fl   added ImagingConvert2
 * 1997-05-30 fl   added floating point support
 * 1997-08-27 fl   added "P" to "1" and "P" to "F" conversions
 * 1998-01-11 fl   added integer support
 * 1998-07-01 fl   added "YCbCr" support
 * 1998-07-02 fl   added "RGBX" conversions (sort of)
 * 1998-07-04 fl   added floyd-steinberg dithering
 * 1998-07-12 fl   changed "YCrCb" to "YCbCr" (!)
 * 1998-12-29 fl   added basic "I;16" and "I;16B" conversions
 * 1999-02-03 fl   added "RGBa", and "BGR" conversions (experimental)
 * 2003-09-26 fl   added "LA" and "PA" conversions (experimental)
 * 2005-05-05 fl   fixed "P" to "1" threshold
 * 2005-12-08 fl   fixed palette memory leak in topalette
 *
 * Copyright (c) 1997-2005 by Secret Labs AB.
 * Copyright (c) 1995-1997 by Fredrik Lundh.
 *
 * See the README file for details on usage and redistribution.
 */

#include "Imaging.h"

/* ITU-R Recommendation 601-2 (assuming nonlinear RGB) */
#define L(rgb) ((INT32)(rgb)[0] * 299 + (INT32)(rgb)[1] * 587 + (INT32)(rgb)[2] * 114)
#define L24(rgb) ((rgb)[0] * 19595 + (rgb)[1] * 38470 + (rgb)[2] * 7471 + 0x8000)

/* Runs pixel(out, in) over a row in chunks of a fixed length. With restrict
   pointers, GCC vectorizes the chunks at -O2 too, which adds no aliasing
   checks or remainder loops. The addresses are pointers, not "in + x * 4"
   for an int x: under -fno-strict-overflow, which Python's CFLAGS have, that
   may wrap, and GCC then vectorizes it at no level. */
#define CONVERT_CHUNK 16
#define CONVERT_PIXELS(pixel, out, in, xsize, in_bytes, out_bytes)       \
    {                                                                    \
        int n_ = (xsize);                                                \
        for (; n_ >= CONVERT_CHUNK; n_ -= CONVERT_CHUNK) {               \
            for (int i_ = 0; i_ < CONVERT_CHUNK; i_++) {                 \
                pixel((out) + i_ * (out_bytes), (in) + i_ * (in_bytes)); \
            }                                                            \
            (in) += CONVERT_CHUNK * (in_bytes);                          \
            (out) += CONVERT_CHUNK * (out_bytes);                        \
        }                                                                \
        for (; n_ > 0; n_--, (in) += (in_bytes), (out) += (out_bytes)) { \
            pixel((out), (in));                                          \
        }                                                                \
    }

/* ------------------- */
/* 1 (bit) conversions */
/* ------------------- */

static void
bit2l(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++) *out++ = (*in++ != 0) ? 255 : 0;
}

static void
bit2rgb(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++) {
        UINT8 v = (*in++ != 0) ? 255 : 0;
        *out++ = v;
        *out++ = v;
        *out++ = v;
        *out++ = 255;
    }
}

static void
bit2cmyk(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++) {
        *out++ = 0;
        *out++ = 0;
        *out++ = 0;
        *out++ = (*in++ != 0) ? 0 : 255;
    }
}

static void
bit2ycbcr(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++) {
        *out++ = (*in++ != 0) ? 255 : 0;
        *out++ = 128;
        *out++ = 128;
        *out++ = 255;
    }
}

static void
bit2hsv(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, out += 4) {
        UINT8 v = (*in++ != 0) ? 255 : 0;
        out[0] = 0;
        out[1] = 0;
        out[2] = v;
        out[3] = 255;
    }
}

/* ----------------- */
/* RGB/L conversions */
/* ----------------- */

static void
l2bit(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++) {
        *out++ = (*in++ >= 128) ? 255 : 0;
    }
}

static void
lA2la(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    unsigned int alpha, pixel, tmp;
    for (x = 0; x < xsize; x++, in += 4) {
        alpha = in[3];
        pixel = MULDIV255(in[0], alpha, tmp);
        *out++ = (UINT8)pixel;
        *out++ = (UINT8)pixel;
        *out++ = (UINT8)pixel;
        *out++ = (UINT8)alpha;
    }
}

/* RGBa -> RGBA conversion to remove premultiplication
   Needed for correct transforms/resizing on RGBA images */
static void
la2lA(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    unsigned int alpha, pixel;
    for (x = 0; x < xsize; x++, in += 4) {
        alpha = in[3];
        if (alpha == 255 || alpha == 0) {
            pixel = in[0];
        } else {
            pixel = CLIP8((255 * in[0]) / alpha);
        }
        *out++ = (UINT8)pixel;
        *out++ = (UINT8)pixel;
        *out++ = (UINT8)pixel;
        *out++ = (UINT8)alpha;
    }
}

static inline void
l2rgb_pixel(UINT8 *restrict out, const UINT8 *restrict in) {
    UINT32 v = MAKE_UINT32(in[0], in[0], in[0], 255);
    memcpy(out, &v, sizeof(v));
}

static void
l2rgb(UINT8 *restrict out, const UINT8 *restrict in, int xsize) {
#if defined(IMAGING_SIMD_NEON)
    /* Stores of four planes: on arm64 GCC makes those of byte loops at -O3
       only, and builds the whole pixels with widening multiplies. */
    for (; xsize >= 16; xsize -= 16, in += 16, out += 64) {
        uint8x16_t v = vld1q_u8(in);
        uint8x16x4_t px = {{v, v, v, vdupq_n_u8(255)}};
        vst4q_u8(out, px);
    }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    /* The byte loop, which GCC turns into st4 at -O3 where the whole pixels
       stay scalar: those were 0.79x in the scalar build on the Jetson Orin. */
    for (; xsize > 0; xsize--) {
        UINT8 v = *in++;
        *out++ = v;
        *out++ = v;
        *out++ = v;
        *out++ = 255;
    }
#else
    CONVERT_PIXELS(l2rgb_pixel, out, in, xsize, 1, 4);
#endif
}

static void
l2hsv(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, out += 4) {
        UINT8 v = *in++;
        out[0] = 0;
        out[1] = 0;
        out[2] = v;
        out[3] = 255;
    }
}

static void
la2l(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = in[0];
    }
}

static void
la2rgb(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        UINT8 v = in[0];
        *out++ = v;
        *out++ = v;
        *out++ = v;
        *out++ = in[3];
    }
}

static void
la2hsv(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4, out += 4) {
        UINT8 v = in[0];
        out[0] = 0;
        out[1] = 0;
        out[2] = v;
        out[3] = in[3];
    }
}

static void
rgb2bit(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        /* ITU-R Recommendation 601-2 (assuming nonlinear RGB) */
        *out++ = (L(in) >= 128000) ? 255 : 0;
    }
}

#if defined(IMAGING_SIMD_SSE4)
#define CONVERT_VECTOR

/* ITU-R Recommendation 601-2 luma of four RGBX pixels as 32-bit lanes,
   with 15-bit coefficients. */
static inline __m128i
rgb2l_4px(__m128i source) {
    __m128i coeff = _mm_set_epi16(0, 3735, 19235, 9798, 0, 3735, 19235, 9798);
    __m128i pix0 = _mm_unpacklo_epi8(source, _mm_setzero_si128());
    __m128i pix1 = _mm_unpackhi_epi8(source, _mm_setzero_si128());
    pix0 = _mm_madd_epi16(pix0, coeff);
    pix1 = _mm_madd_epi16(pix1, coeff);
    pix0 = _mm_hadd_epi32(pix0, pix1);
    pix0 = _mm_add_epi32(pix0, _mm_set1_epi32(0x4000));
    return _mm_srli_epi32(pix0, 15);
}

static inline void
rgb2l_4(UINT8 *out, const UINT8 *in) {
    __m128i pix = rgb2l_4px(_mm_loadu_si128((__m128i *)in));
    pix = _mm_packus_epi32(pix, pix);
    UINT32 v = _mm_cvtsi128_si32(_mm_packus_epi16(pix, pix));
    memcpy(out, &v, sizeof(v));
}

static inline void
rgb2la_4(UINT8 *out, const UINT8 *in) {
    __m128i pix = rgb2l_4px(_mm_loadu_si128((__m128i *)in));
    pix = _mm_shuffle_epi8(
        pix, _mm_set_epi8(-1, 12, 12, 12, -1, 8, 8, 8, -1, 4, 4, 4, -1, 0, 0, 0)
    );
    pix = _mm_or_si128(pix, _mm_set1_epi32(0xff000000));
    _mm_storeu_si128((__m128i *)out, pix);
}

static inline void
rgb2i_4(UINT8 *out, const UINT8 *in) {
    _mm_storeu_si128((__m128i *)out, rgb2l_4px(_mm_loadu_si128((__m128i *)in)));
}

static inline void
rgba2la_4(UINT8 *out, const UINT8 *in) {
    __m128i source = _mm_loadu_si128((__m128i *)in);
    __m128i alpha = _mm_and_si128(source, _mm_set1_epi32(0xff000000));
    __m128i pix = rgb2l_4px(source);
    pix = _mm_shuffle_epi8(
        pix, _mm_set_epi8(-1, 12, 12, 12, -1, 8, 8, 8, -1, 4, 4, 4, -1, 0, 0, 0)
    );
    _mm_storeu_si128((__m128i *)out, _mm_or_si128(pix, alpha));
}

/* Removes premultiplication with the scalar code's result: 255 * c / alpha
   truncated and clipped to 255, and c unchanged where alpha is 0 or 255.
   c is first clipped to alpha, which gives 255 as the clip would. The
   quotient from an approximate reciprocal is within 1 of the exact one,
   and one step with the remainder 255 * c - q * alpha corrects it; every
   value fits 16 bits. The macro works on 128- or 256-bit vectors of whole
   pixels: P is _mm or _mm256, V the vector type, SI si128 or si256, and
   the last three the byte shuffles for alpha and the two factor halves. */
#define UNPREMULTIPLY(P, V, SI, source, out, ALPHA_BYTES, FACTOR_LO, FACTOR_HI)       \
    {                                                                                 \
        V zero = P##_setzero_##SI();                                                  \
        V alpha = P##_shuffle_epi8(source, ALPHA_BYTES);                              \
        V c = P##_min_epu8(source, alpha);                                            \
        /* 255.5 * 256 / alpha as 16-bit factors, one per channel */                  \
        V f = P##_cvtps_epi32(                                                        \
            P##_mul_ps(                                                               \
                P##_set1_ps(255.5 * 256),                                             \
                P##_rcp_ps(P##_cvtepi32_ps(P##_srli_epi32(source, 24)))               \
            )                                                                         \
        );                                                                            \
        V f0 = P##_shuffle_epi8(f, FACTOR_LO);                                        \
        V f1 = P##_shuffle_epi8(f, FACTOR_HI);                                        \
        V c0 = P##_unpacklo_epi8(c, zero), c1 = P##_unpackhi_epi8(c, zero);           \
        V a0 = P##_unpacklo_epi8(alpha, zero);                                        \
        V a1 = P##_unpackhi_epi8(alpha, zero);                                        \
        V q0 = P##_mulhi_epu16(P##_slli_epi16(c0, 8), f0);                            \
        V q1 = P##_mulhi_epu16(P##_slli_epi16(c1, 8), f1);                            \
        /* remainders, modulo 2^16; their true values are within +-510 */             \
        V r0 = P##_sub_epi16(                                                         \
            P##_sub_epi16(P##_slli_epi16(c0, 8), c0), P##_mullo_epi16(q0, a0)         \
        );                                                                            \
        V r1 = P##_sub_epi16(                                                         \
            P##_sub_epi16(P##_slli_epi16(c1, 8), c1), P##_mullo_epi16(q1, a1)         \
        );                                                                            \
        V one = P##_set1_epi16(1);                                                    \
        q0 = P##_sub_epi16(q0, P##_cmpgt_epi16(r0, P##_sub_epi16(a0, one)));          \
        q1 = P##_sub_epi16(q1, P##_cmpgt_epi16(r1, P##_sub_epi16(a1, one)));          \
        q0 = P##_add_epi16(q0, P##_cmpgt_epi16(zero, r0));                            \
        q1 = P##_add_epi16(q1, P##_cmpgt_epi16(zero, r1));                            \
        V res = P##_packus_epi16(q0, q1);                                             \
        /* the source bytes for alpha itself and where alpha is 0 or 255 */           \
        V keep = P##_or_##SI(                                                         \
            P##_or_##SI(                                                              \
                P##_cmpeq_epi8(alpha, zero), P##_cmpeq_epi8(alpha, P##_set1_epi8(-1)) \
            ),                                                                        \
            P##_set1_epi32((int)0xff000000)                                           \
        );                                                                            \
        out = P##_blendv_epi8(res, source, keep);                                     \
    }

static inline void
rgba2rgbA_4(UINT8 *out, const UINT8 *in) {
    __m128i source = _mm_loadu_si128((__m128i *)in), result;
    UNPREMULTIPLY(
        _mm,
        __m128i,
        si128,
        source,
        result,
        _mm_set_epi8(15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3),
        _mm_set_epi8(5, 4, 5, 4, 5, 4, 5, 4, 1, 0, 1, 0, 1, 0, 1, 0),
        _mm_set_epi8(13, 12, 13, 12, 13, 12, 13, 12, 9, 8, 9, 8, 9, 8, 9, 8)
    );
    _mm_storeu_si128((__m128i *)out, result);
}

#if defined(IMAGING_SIMD_AVX2)
/* Eight pixels: the shuffles use the same byte indices in both lanes. */
static inline void
rgba2rgbA_8(UINT8 *out, const UINT8 *in) {
    __m256i source = _mm256_loadu_si256((__m256i *)in), result;
    UNPREMULTIPLY(
        _mm256,
        __m256i,
        si256,
        source,
        result,
        _mm256_broadcastsi128_si256(
            _mm_set_epi8(15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3)
        ),
        _mm256_broadcastsi128_si256(
            _mm_set_epi8(5, 4, 5, 4, 5, 4, 5, 4, 1, 0, 1, 0, 1, 0, 1, 0)
        ),
        _mm256_broadcastsi128_si256(
            _mm_set_epi8(13, 12, 13, 12, 13, 12, 13, 12, 9, 8, 9, 8, 9, 8, 9, 8)
        )
    );
    _mm256_storeu_si256((__m256i *)out, result);
}
#endif
#undef UNPREMULTIPLY
#endif

#if defined(IMAGING_SIMD_NEON)
#define CONVERT_VECTOR

/* ITU-R Recommendation 601-2 luma of 8 pixels with the same 15-bit
   coefficients as the SSE4 code, so results are identical. Each
   coefficient is split into bytes (9798 = 38 * 256 + 70, 19235 = 75 * 256
   + 35, 3735 = 14 * 256 + 151) so all products fit 16-bit lanes:
   (hi * 256 + lo + 0x4000) >> 15 == (hi + (lo >> 8) + 64) >> 7. */
static inline uint8x8_t
rgb2l_8px(uint8x8_t r, uint8x8_t g, uint8x8_t b) {
    uint16x8_t hi = vmull_u8(r, vdup_n_u8(38));
    hi = vmlal_u8(hi, g, vdup_n_u8(75));
    hi = vmlal_u8(hi, b, vdup_n_u8(14));
    uint16x8_t lo = vmull_u8(r, vdup_n_u8(70));
    lo = vmlal_u8(lo, g, vdup_n_u8(35));
    lo = vmlal_u8(lo, b, vdup_n_u8(151));
    return vrshrn_n_u16(vsraq_n_u16(hi, lo, 8), 7);
}

static inline uint8x16_t
rgb2l_16px(uint8x16x4_t px) {
    return vcombine_u8(
        rgb2l_8px(
            vget_low_u8(px.val[0]), vget_low_u8(px.val[1]), vget_low_u8(px.val[2])
        ),
        rgb2l_8px(
            vget_high_u8(px.val[0]), vget_high_u8(px.val[1]), vget_high_u8(px.val[2])
        )
    );
}

static inline void
rgb2l_16(UINT8 *out, const UINT8 *in) {
    vst1q_u8(out, rgb2l_16px(vld4q_u8(in)));
}

static inline void
rgb2la_16(UINT8 *out, const UINT8 *in) {
    uint8x16_t l = rgb2l_16px(vld4q_u8(in));
    uint8x16x4_t res = {{l, l, l, vdupq_n_u8(255)}};
    vst4q_u8(out, res);
}

static inline void
rgb2i_16(UINT8 *out, const UINT8 *in) {
    uint8x16_t l = rgb2l_16px(vld4q_u8(in));
    uint16x8_t lo = vmovl_u8(vget_low_u8(l));
    uint16x8_t hi = vmovl_u8(vget_high_u8(l));
    vst1q_u8(out, vreinterpretq_u8_u32(vmovl_u16(vget_low_u16(lo))));
    vst1q_u8(out + 16, vreinterpretq_u8_u32(vmovl_u16(vget_high_u16(lo))));
    vst1q_u8(out + 32, vreinterpretq_u8_u32(vmovl_u16(vget_low_u16(hi))));
    vst1q_u8(out + 48, vreinterpretq_u8_u32(vmovl_u16(vget_high_u16(hi))));
}

static inline void
rgba2la_16(UINT8 *out, const UINT8 *in) {
    uint8x16x4_t px = vld4q_u8(in);
    uint8x16_t l = rgb2l_16px(px);
    uint8x16x4_t res = {{l, l, l, px.val[3]}};
    vst4q_u8(out, res);
}

/* MULDIV255(v, a) for 16 pixels, exact. */
static inline uint8x16_t
neon_muldiv255(uint8x16_t v, uint8x16_t a) {
    uint16x8_t lo = vmull_u8(vget_low_u8(v), vget_low_u8(a));
    uint16x8_t hi = vmull_u8(vget_high_u8(v), vget_high_u8(a));
    return vcombine_u8(
        vraddhn_u16(lo, vrshrq_n_u16(lo, 8)), vraddhn_u16(hi, vrshrq_n_u16(hi, 8))
    );
}

/* CLIP8(255 * v / a), truncated, for four lanes. The float quotient is
   exact enough that truncating it equals the integer division. */
static inline uint16x4_t
neon_div_alpha(uint16x4_t v, float32x4_t a) {
    float32x4_t q = vdivq_f32(vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(v)), 255.0f), a);
    return vqmovn_u32(vcvtq_u32_f32(q));
}

static inline uint8x16_t
neon_unpremultiply(uint8x16_t v, uint8x16_t a) {
    uint16x8_t a_lo = vmovl_u8(vget_low_u8(a));
    uint16x8_t a_hi = vmovl_u8(vget_high_u8(a));
    uint16x8_t v_lo = vmovl_u8(vget_low_u8(v));
    uint16x8_t v_hi = vmovl_u8(vget_high_u8(v));
    float32x4_t a0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_lo)));
    float32x4_t a1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_lo)));
    float32x4_t a2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_hi)));
    float32x4_t a3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_hi)));
    uint16x8_t r_lo = vcombine_u16(
        neon_div_alpha(vget_low_u16(v_lo), a0), neon_div_alpha(vget_high_u16(v_lo), a1)
    );
    uint16x8_t r_hi = vcombine_u16(
        neon_div_alpha(vget_low_u16(v_hi), a2), neon_div_alpha(vget_high_u16(v_hi), a3)
    );
    uint8x16_t r = vcombine_u8(vqmovn_u16(r_lo), vqmovn_u16(r_hi));
    // Alpha 0 keeps the colour.
    return vbslq_u8(vceqzq_u8(a), v, r);
}
#endif

#if defined(CONVERT_VECTOR)
/* Runs a kernel for `step` pixels over a row. The last xsize % step
   pixels go through the same kernel via zero-padded buffers, so the
   result for a pixel doesn't depend on its position. */
#define CONVERT_ROW(kernel, step, out, in, xsize, in_bytes, out_bytes) \
    {                                                                  \
        int x_ = 0;                                                    \
        for (; x_ < (xsize) - ((step) - 1); x_ += (step)) {            \
            kernel((out) + x_ * (out_bytes), (in) + x_ * (in_bytes));  \
        }                                                              \
        if (x_ < (xsize)) {                                            \
            UINT8 tin_[(step) * (in_bytes)];                           \
            UINT8 tout_[(step) * (out_bytes)];                         \
            int n_ = (xsize) - x_;                                     \
            memset(tin_, 0, sizeof(tin_));                             \
            memcpy(tin_, (in) + x_ * (in_bytes), n_ * (in_bytes));     \
            kernel(tout_, tin_);                                       \
            memcpy((out) + x_ * (out_bytes), tout_, n_ * (out_bytes)); \
        }                                                              \
    }
#endif

#if defined(IMAGING_SIMD_SSE4)
#define CONVERT_STEP 4
#define CONVERT_KERNEL(name) name##_4
#elif defined(IMAGING_SIMD_NEON)
#define CONVERT_STEP 16
#define CONVERT_KERNEL(name) name##_16
#endif

static void
rgb2l(UINT8 *out, const UINT8 *in, int xsize) {
#if defined(CONVERT_VECTOR)
    CONVERT_ROW(CONVERT_KERNEL(rgb2l), CONVERT_STEP, out, in, xsize, 4, 1);
#else
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        /* ITU-R Recommendation 601-2 (assuming nonlinear RGB) */
        *out++ = L24(in) >> 16;
    }
#endif
}

static void
rgb2la(UINT8 *out, const UINT8 *in, int xsize) {
#if defined(CONVERT_VECTOR)
    CONVERT_ROW(CONVERT_KERNEL(rgb2la), CONVERT_STEP, out, in, xsize, 4, 4);
#else
    int x;
    for (x = 0; x < xsize; x++, in += 4, out += 4) {
        /* ITU-R Recommendation 601-2 (assuming nonlinear RGB) */
        out[0] = out[1] = out[2] = L24(in) >> 16;
        out[3] = 255;
    }
#endif
}

static void
rgb2i(UINT8 *out_, const UINT8 *in, int xsize) {
#if defined(CONVERT_VECTOR)
    CONVERT_ROW(CONVERT_KERNEL(rgb2i), CONVERT_STEP, out_, in, xsize, 4, 4);
#else
    int x;
    for (x = 0; x < xsize; x++, in += 4, out_ += 4) {
        INT32 v = L24(in) >> 16;
        memcpy(out_, &v, sizeof(v));
    }
#endif
}

static void
rgb2i16l(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        *out_++ = L24(in) >> 16;
        *out_++ = 0;
    }
}

static void
rgb2i16b(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        *out_++ = 0;
        *out_++ = L24(in) >> 16;
    }
}

static void
rgb2f(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4, out_ += 4) {
        FLOAT32 v = (float)L(in) / 1000.0F;
        memcpy(out_, &v, sizeof(v));
    }
}

static void
rgb2hsv_row(UINT8 *out, const UINT8 *in) {  // following colorsys.py
    float h, s, rc, gc, bc, cr;
    UINT8 maxc, minc;
    UINT8 r, g, b;
    UINT8 uh, us, uv;

    r = in[0];
    g = in[1];
    b = in[2];
    maxc = MAX(r, MAX(g, b));
    minc = MIN(r, MIN(g, b));
    uv = maxc;
    if (minc == maxc) {
        uh = 0;
        us = 0;
    } else {
        cr = (float)(maxc - minc);
        s = cr / (float)maxc;
        rc = ((float)(maxc - r)) / cr;
        gc = ((float)(maxc - g)) / cr;
        bc = ((float)(maxc - b)) / cr;
        if (r == maxc) {
            h = bc - gc;
        } else if (g == maxc) {
            h = 2.0 + rc - bc;
        } else {
            h = 4.0 + gc - rc;
        }
        // incorrect hue happens if h/6 is negative.
        // h is in [-1, 5], so h / 6.0 + 1.0 is in [5/6, 11/6] and fmod(.., 1.0)
        // subtracts 1 exactly when it is at least 1.
        {
            double hd = h / 6.0 + 1.0;
            h = hd >= 1.0 ? hd - 1.0 : hd;
        }

        uh = (UINT8)CLIP8((int)(h * 255.0));
        us = (UINT8)CLIP8((int)(s * 255.0));
    }
    out[0] = uh;
    out[1] = us;
    out[2] = uv;
}

#if defined(IMAGING_SIMD_SSE4)
/* (int)(v * 255.0) for four values given as two pairs of doubles. */
static inline __m128i
rgb2hsv_scale(__m128d lo, __m128d hi) {
    __m128d k = _mm_set1_pd(255.0);
    return _mm_unpacklo_epi64(
        _mm_cvttpd_epi32(_mm_mul_pd(lo, k)), _mm_cvttpd_epi32(_mm_mul_pd(hi, k))
    );
}

/* rgb2hsv_row for four RGBX pixels: the same float and double operations,
   with the three hue cases computed and selected in the same priority. */
static inline void
rgb2hsv_4(UINT8 *out, const UINT8 *in) {
    __m128i pix = _mm_loadu_si128((const __m128i *)in);
    __m128i mask = _mm_set1_epi32(0xff);
    __m128i r = _mm_and_si128(pix, mask);
    __m128i g = _mm_and_si128(_mm_srli_epi32(pix, 8), mask);
    __m128i b = _mm_and_si128(_mm_srli_epi32(pix, 16), mask);
    __m128i maxc = _mm_max_epi32(r, _mm_max_epi32(g, b));
    __m128i minc = _mm_min_epi32(r, _mm_min_epi32(g, b));
    __m128i flat = _mm_cmpeq_epi32(maxc, minc);
    __m128 cr = _mm_cvtepi32_ps(_mm_sub_epi32(maxc, minc));
    __m128 s = _mm_div_ps(cr, _mm_cvtepi32_ps(maxc));
    __m128 rc = _mm_div_ps(_mm_cvtepi32_ps(_mm_sub_epi32(maxc, r)), cr);
    __m128 gc = _mm_div_ps(_mm_cvtepi32_ps(_mm_sub_epi32(maxc, g)), cr);
    __m128 bc = _mm_div_ps(_mm_cvtepi32_ps(_mm_sub_epi32(maxc, b)), cr);
    /* 2.0 + rc - bc and 4.0 + gc - rc are double, rounded back to float. */
    __m128d rc0 = _mm_cvtps_pd(rc), rc1 = _mm_cvtps_pd(_mm_movehl_ps(rc, rc));
    __m128d gc0 = _mm_cvtps_pd(gc), gc1 = _mm_cvtps_pd(_mm_movehl_ps(gc, gc));
    __m128d bc0 = _mm_cvtps_pd(bc), bc1 = _mm_cvtps_pd(_mm_movehl_ps(bc, bc));
    __m128d two = _mm_set1_pd(2.0), four = _mm_set1_pd(4.0);
    __m128 hg = _mm_movelh_ps(
        _mm_cvtpd_ps(_mm_sub_pd(_mm_add_pd(two, rc0), bc0)),
        _mm_cvtpd_ps(_mm_sub_pd(_mm_add_pd(two, rc1), bc1))
    );
    __m128 hb = _mm_movelh_ps(
        _mm_cvtpd_ps(_mm_sub_pd(_mm_add_pd(four, gc0), rc0)),
        _mm_cvtpd_ps(_mm_sub_pd(_mm_add_pd(four, gc1), rc1))
    );
    __m128 h = _mm_blendv_ps(
        _mm_blendv_ps(hb, hg, _mm_castsi128_ps(_mm_cmpeq_epi32(g, maxc))),
        _mm_sub_ps(bc, gc),
        _mm_castsi128_ps(_mm_cmpeq_epi32(r, maxc))
    );
    /* fmod(h / 6.0 + 1.0, 1.0), as in rgb2hsv_row. */
    __m128d one = _mm_set1_pd(1.0), six = _mm_set1_pd(6.0);
    __m128d h0 = _mm_add_pd(_mm_div_pd(_mm_cvtps_pd(h), six), one);
    __m128d h1 = _mm_add_pd(_mm_div_pd(_mm_cvtps_pd(_mm_movehl_ps(h, h)), six), one);
    h0 = _mm_blendv_pd(h0, _mm_sub_pd(h0, one), _mm_cmpge_pd(h0, one));
    h1 = _mm_blendv_pd(h1, _mm_sub_pd(h1, one), _mm_cmpge_pd(h1, one));
    /* stored in the float h before h * 255.0 */
    h0 = _mm_cvtps_pd(_mm_cvtpd_ps(h0));
    h1 = _mm_cvtps_pd(_mm_cvtpd_ps(h1));
    /* h and s are in [0, 1), so CLIP8 changes nothing. */
    __m128i uh = _mm_andnot_si128(flat, rgb2hsv_scale(h0, h1));
    __m128i us = _mm_andnot_si128(
        flat, rgb2hsv_scale(_mm_cvtps_pd(s), _mm_cvtps_pd(_mm_movehl_ps(s, s)))
    );
    __m128i hsv =
        _mm_or_si128(_mm_or_si128(uh, _mm_slli_epi32(us, 8)), _mm_slli_epi32(maxc, 16));
    hsv = _mm_or_si128(hsv, _mm_andnot_si128(_mm_set1_epi32(0xffffff), pix));
    _mm_storeu_si128((__m128i *)out, hsv);
}
#endif

#if defined(IMAGING_SIMD_AVX2)
/* rgb2hsv_4 for eight pixels: the same operations per lane. */
static inline __m256i
rgb2hsv_scale8(__m256d lo, __m256d hi) {
    __m256d k = _mm256_set1_pd(255.0);
    return _mm256_set_m128i(
        _mm256_cvttpd_epi32(_mm256_mul_pd(hi, k)),
        _mm256_cvttpd_epi32(_mm256_mul_pd(lo, k))
    );
}

static inline void
rgb2hsv_8(UINT8 *out, const UINT8 *in) {
/* The low and high four floats as doubles, and two sets of four doubles
   back to floats. */
#define HSV_LO(v) _mm256_cvtps_pd(_mm256_castps256_ps128(v))
#define HSV_HI(v) _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1))
#define HSV_PS(lo, hi) _mm256_set_m128(_mm256_cvtpd_ps(hi), _mm256_cvtpd_ps(lo))
    __m256i pix = _mm256_loadu_si256((const __m256i *)in);
    __m256i mask = _mm256_set1_epi32(0xff);
    __m256i r = _mm256_and_si256(pix, mask);
    __m256i g = _mm256_and_si256(_mm256_srli_epi32(pix, 8), mask);
    __m256i b = _mm256_and_si256(_mm256_srli_epi32(pix, 16), mask);
    __m256i maxc = _mm256_max_epi32(r, _mm256_max_epi32(g, b));
    __m256i minc = _mm256_min_epi32(r, _mm256_min_epi32(g, b));
    __m256i flat = _mm256_cmpeq_epi32(maxc, minc);
    __m256 cr = _mm256_cvtepi32_ps(_mm256_sub_epi32(maxc, minc));
    __m256 s = _mm256_div_ps(cr, _mm256_cvtepi32_ps(maxc));
    __m256 rc = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(maxc, r)), cr);
    __m256 gc = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(maxc, g)), cr);
    __m256 bc = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(maxc, b)), cr);
    __m256d two = _mm256_set1_pd(2.0), four = _mm256_set1_pd(4.0);
    __m256 hg = HSV_PS(
        _mm256_sub_pd(_mm256_add_pd(two, HSV_LO(rc)), HSV_LO(bc)),
        _mm256_sub_pd(_mm256_add_pd(two, HSV_HI(rc)), HSV_HI(bc))
    );
    __m256 hb = HSV_PS(
        _mm256_sub_pd(_mm256_add_pd(four, HSV_LO(gc)), HSV_LO(rc)),
        _mm256_sub_pd(_mm256_add_pd(four, HSV_HI(gc)), HSV_HI(rc))
    );
    __m256 h = _mm256_blendv_ps(
        _mm256_blendv_ps(hb, hg, _mm256_castsi256_ps(_mm256_cmpeq_epi32(g, maxc))),
        _mm256_sub_ps(bc, gc),
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(r, maxc))
    );
    __m256d one = _mm256_set1_pd(1.0), six = _mm256_set1_pd(6.0);
    __m256d h0 = _mm256_add_pd(_mm256_div_pd(HSV_LO(h), six), one);
    __m256d h1 = _mm256_add_pd(_mm256_div_pd(HSV_HI(h), six), one);
    h0 = _mm256_blendv_pd(
        h0, _mm256_sub_pd(h0, one), _mm256_cmp_pd(h0, one, _CMP_GE_OS)
    );
    h1 = _mm256_blendv_pd(
        h1, _mm256_sub_pd(h1, one), _mm256_cmp_pd(h1, one, _CMP_GE_OS)
    );
    h0 = _mm256_cvtps_pd(_mm256_cvtpd_ps(h0));
    h1 = _mm256_cvtps_pd(_mm256_cvtpd_ps(h1));
    __m256i uh = _mm256_andnot_si256(flat, rgb2hsv_scale8(h0, h1));
    __m256i us = _mm256_andnot_si256(flat, rgb2hsv_scale8(HSV_LO(s), HSV_HI(s)));
    __m256i hsv = _mm256_or_si256(
        _mm256_or_si256(uh, _mm256_slli_epi32(us, 8)), _mm256_slli_epi32(maxc, 16)
    );
    hsv = _mm256_or_si256(hsv, _mm256_andnot_si256(_mm256_set1_epi32(0xffffff), pix));
    _mm256_storeu_si256((__m256i *)out, hsv);
#undef HSV_LO
#undef HSV_HI
#undef HSV_PS
}
#endif

#if defined(IMAGING_SIMD_NEON)
/* (int)(v * 255.0) for four floats, through doubles. */
static inline int32x4_t
rgb2hsv_scale_neon(float32x4_t v) {
    float64x2_t k = vdupq_n_f64(255.0);
    int64x2_t lo = vcvtq_s64_f64(vmulq_f64(vcvt_f64_f32(vget_low_f32(v)), k));
    int64x2_t hi = vcvtq_s64_f64(vmulq_f64(vcvt_high_f64_f32(v), k));
    return vcombine_s32(vmovn_s64(lo), vmovn_s64(hi));
}

/* rgb2hsv_4 for NEON: the same float and double operations as
   rgb2hsv_row. None is a multiply followed by an add, so none can become
   a fused multiply-add, in this code or in rgb2hsv_row. */
static inline void
rgb2hsv_neon4(UINT8 *out, const UINT8 *in) {
    uint32x4_t pix = vreinterpretq_u32_u8(vld1q_u8(in));
    uint32x4_t mask = vdupq_n_u32(0xff);
    int32x4_t r = vreinterpretq_s32_u32(vandq_u32(pix, mask));
    int32x4_t g = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(pix, 8), mask));
    int32x4_t b = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(pix, 16), mask));
    int32x4_t maxc = vmaxq_s32(r, vmaxq_s32(g, b));
    int32x4_t minc = vminq_s32(r, vminq_s32(g, b));
    uint32x4_t flat = vceqq_s32(maxc, minc);
    float32x4_t cr = vcvtq_f32_s32(vsubq_s32(maxc, minc));
    float32x4_t s = vdivq_f32(cr, vcvtq_f32_s32(maxc));
    float32x4_t rc = vdivq_f32(vcvtq_f32_s32(vsubq_s32(maxc, r)), cr);
    float32x4_t gc = vdivq_f32(vcvtq_f32_s32(vsubq_s32(maxc, g)), cr);
    float32x4_t bc = vdivq_f32(vcvtq_f32_s32(vsubq_s32(maxc, b)), cr);
    /* 2.0 + rc - bc and 4.0 + gc - rc are double, rounded back to float. */
    float64x2_t two = vdupq_n_f64(2.0), four = vdupq_n_f64(4.0);
    float64x2_t rc0 = vcvt_f64_f32(vget_low_f32(rc)), rc1 = vcvt_high_f64_f32(rc);
    float64x2_t gc0 = vcvt_f64_f32(vget_low_f32(gc)), gc1 = vcvt_high_f64_f32(gc);
    float64x2_t bc0 = vcvt_f64_f32(vget_low_f32(bc)), bc1 = vcvt_high_f64_f32(bc);
    float32x4_t hg = vcvt_high_f32_f64(
        vcvt_f32_f64(vsubq_f64(vaddq_f64(two, rc0), bc0)),
        vsubq_f64(vaddq_f64(two, rc1), bc1)
    );
    float32x4_t hb = vcvt_high_f32_f64(
        vcvt_f32_f64(vsubq_f64(vaddq_f64(four, gc0), rc0)),
        vsubq_f64(vaddq_f64(four, gc1), rc1)
    );
    float32x4_t h = vbslq_f32(
        vceqq_s32(r, maxc), vsubq_f32(bc, gc), vbslq_f32(vceqq_s32(g, maxc), hg, hb)
    );
    /* fmod(h / 6.0 + 1.0, 1.0), as in rgb2hsv_row */
    float64x2_t one = vdupq_n_f64(1.0), six = vdupq_n_f64(6.0);
    float64x2_t h0 = vaddq_f64(vdivq_f64(vcvt_f64_f32(vget_low_f32(h)), six), one);
    float64x2_t h1 = vaddq_f64(vdivq_f64(vcvt_high_f64_f32(h), six), one);
    h0 = vbslq_f64(vcgeq_f64(h0, one), vsubq_f64(h0, one), h0);
    h1 = vbslq_f64(vcgeq_f64(h1, one), vsubq_f64(h1, one), h1);
    /* stored in the float h before h * 255.0; h and s are in [0, 1), so
       CLIP8 changes nothing */
    int32x4_t uh = rgb2hsv_scale_neon(vcvt_high_f32_f64(vcvt_f32_f64(h0), h1));
    int32x4_t us = rgb2hsv_scale_neon(s);
    uint32x4_t hsv = vorrq_u32(
        vbicq_u32(vreinterpretq_u32_s32(uh), flat),
        vshlq_n_u32(vbicq_u32(vreinterpretq_u32_s32(us), flat), 8)
    );
    hsv = vorrq_u32(hsv, vshlq_n_u32(vreinterpretq_u32_s32(maxc), 16));
    hsv = vorrq_u32(hsv, vandq_u32(pix, vdupq_n_u32(0xff000000)));
    vst1q_u8(out, vreinterpretq_u8_u32(hsv));
}
#endif

static void
rgb2hsv(UINT8 *out, const UINT8 *in, int xsize) {
#if defined(IMAGING_SIMD_AVX2)
    CONVERT_ROW(rgb2hsv_8, 8, out, in, xsize, 4, 4);
#elif defined(IMAGING_SIMD_SSE4)
    CONVERT_ROW(rgb2hsv_4, 4, out, in, xsize, 4, 4);
#elif defined(IMAGING_SIMD_NEON)
    CONVERT_ROW(rgb2hsv_neon4, 4, out, in, xsize, 4, 4);
#else
    int x;
    for (x = 0; x < xsize; x++, in += 4, out += 4) {
        rgb2hsv_row(out, in);
        out[3] = in[3];
    }
#endif
}

static void
hsv2rgb(UINT8 *out, const UINT8 *in, int xsize) {  // following colorsys.py

    int p, q, t;
    UINT8 up, uq, ut;
    int i, x;
    float f, fs;
    UINT8 h, s, v;

    for (x = 0; x < xsize; x++, in += 4) {
        h = in[0];
        s = in[1];
        v = in[2];

        if (s == 0) {
            *out++ = v;
            *out++ = v;
            *out++ = v;
        } else {
            i = floor((float)h * 6.0 / 255.0);      // 0 - 6
            f = (float)h * 6.0 / 255.0 - (float)i;  // 0-1 : remainder.
            fs = ((float)s) / 255.0;

            p = round((float)v * (1.0 - fs));
            q = round((float)v * (1.0 - fs * f));
            t = round((float)v * (1.0 - fs * (1.0 - f)));
            up = (UINT8)CLIP8(p);
            uq = (UINT8)CLIP8(q);
            ut = (UINT8)CLIP8(t);

            switch (i % 6) {
                case 0:
                    *out++ = v;
                    *out++ = ut;
                    *out++ = up;
                    break;
                case 1:
                    *out++ = uq;
                    *out++ = v;
                    *out++ = up;
                    break;
                case 2:
                    *out++ = up;
                    *out++ = v;
                    *out++ = ut;
                    break;
                case 3:
                    *out++ = up;
                    *out++ = uq;
                    *out++ = v;
                    break;
                case 4:
                    *out++ = ut;
                    *out++ = up;
                    *out++ = v;
                    break;
                case 5:
                    *out++ = v;
                    *out++ = up;
                    *out++ = uq;
                    break;
            }
        }
        *out++ = in[3];
    }
}

/* ---------------- */
/* RGBA conversions */
/* ---------------- */

static inline void
rgb2rgba_pixel(UINT8 *restrict out, const UINT8 *restrict in) {
    UINT32 v;
    memcpy(&v, in, sizeof(v));
    v |= MAKE_UINT32(0, 0, 0, 255);
    memcpy(out, &v, sizeof(v));
}

#if defined(IMAGING_SIMD_NEON)
/* 16 pixels at a time on arm64, where GCC's code for the loops below
   depends more on the optimization level: ((v or ~v) & and) | or. */
static inline int
convert_mask16(
    UINT8 *restrict out,
    const UINT8 *restrict in,
    int xsize,
    int invert,
    UINT32 and,
    UINT32 or
) {
    int x = 0, k;
    uint32x4_t a = vdupq_n_u32(and), o = vdupq_n_u32(or);
    for (; x < xsize - 15; x += 16, in += 64, out += 64) {
        for (k = 0; k < 4; k++) {
            uint32x4_t v = vreinterpretq_u32_u8(vld1q_u8(in + k * 16));
            v = vorrq_u32(vandq_u32(invert ? vmvnq_u32(v) : v, a), o);
            vst1q_u8(out + k * 16, vreinterpretq_u8_u32(v));
        }
    }
    return x;
}
#endif

static void
rgb2rgba(UINT8 *restrict out, const UINT8 *restrict in, int xsize) {
#if defined(IMAGING_SIMD_NEON)
    int x = convert_mask16(out, in, xsize, 0, ~0u, MAKE_UINT32(0, 0, 0, 255));
    out += x * 4, in += x * 4, xsize -= x;
#endif
    CONVERT_PIXELS(rgb2rgba_pixel, out, in, xsize, 4, 4);
}

static void
rgba2la(UINT8 *out, const UINT8 *in, int xsize) {
#if defined(CONVERT_VECTOR)
    CONVERT_ROW(CONVERT_KERNEL(rgba2la), CONVERT_STEP, out, in, xsize, 4, 4);
#else
    int x;
    for (x = 0; x < xsize; x++, in += 4, out += 4) {
        /* ITU-R Recommendation 601-2 (assuming nonlinear RGB) */
        out[0] = out[1] = out[2] = L24(in) >> 16;
        out[3] = in[3];
    }
#endif
}

static void
rgbA2rgba(UINT8 *out, const UINT8 *in, int xsize) {
    unsigned int tmp;
    unsigned char alpha;
    int x = 0;

#if defined(IMAGING_SIMD_AVX2)

    __m256i zero = _mm256_setzero_si256();
    __m256i half = _mm256_set1_epi16(128);
    __m256i maxalpha = _mm256_set_epi32(
        0xff000000,
        0xff000000,
        0xff000000,
        0xff000000,
        0xff000000,
        0xff000000,
        0xff000000,
        0xff000000
    );
    __m256i factormask = _mm256_set_epi8(
        15,
        15,
        15,
        15,
        11,
        11,
        11,
        11,
        7,
        7,
        7,
        7,
        3,
        3,
        3,
        3,
        15,
        15,
        15,
        15,
        11,
        11,
        11,
        11,
        7,
        7,
        7,
        7,
        3,
        3,
        3,
        3
    );
    __m256i factorsource, source, pix1, pix2, factors;

    for (; x < xsize - 7; x += 8) {
        source = _mm256_loadu_si256((__m256i *)&in[x * 4]);
        factorsource = _mm256_shuffle_epi8(source, factormask);
        factorsource = _mm256_or_si256(factorsource, maxalpha);

        pix1 = _mm256_unpacklo_epi8(source, zero);
        factors = _mm256_unpacklo_epi8(factorsource, zero);
        pix1 = _mm256_add_epi16(_mm256_mullo_epi16(pix1, factors), half);
        pix1 = _mm256_add_epi16(pix1, _mm256_srli_epi16(pix1, 8));
        pix1 = _mm256_srli_epi16(pix1, 8);

        pix2 = _mm256_unpackhi_epi8(source, zero);
        factors = _mm256_unpackhi_epi8(factorsource, zero);
        pix2 = _mm256_add_epi16(_mm256_mullo_epi16(pix2, factors), half);
        pix2 = _mm256_add_epi16(pix2, _mm256_srli_epi16(pix2, 8));
        pix2 = _mm256_srli_epi16(pix2, 8);

        source = _mm256_packus_epi16(pix1, pix2);
        _mm256_storeu_si256((__m256i *)&out[x * 4], source);
    }

#elif defined(IMAGING_SIMD_SSE4)

    __m128i zero = _mm_setzero_si128();
    __m128i half = _mm_set1_epi16(128);
    __m128i maxalpha = _mm_set1_epi32(0xff000000);
    __m128i factormask =
        _mm_set_epi8(15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3);
    __m128i factorsource, source, pix1, pix2, factors;

    for (; x < xsize - 3; x += 4) {
        source = _mm_loadu_si128((__m128i *)&in[x * 4]);
        factorsource = _mm_shuffle_epi8(source, factormask);
        factorsource = _mm_or_si128(factorsource, maxalpha);

        pix1 = _mm_unpacklo_epi8(source, zero);
        factors = _mm_unpacklo_epi8(factorsource, zero);
        pix1 = _mm_add_epi16(_mm_mullo_epi16(pix1, factors), half);
        pix1 = _mm_add_epi16(pix1, _mm_srli_epi16(pix1, 8));
        pix1 = _mm_srli_epi16(pix1, 8);

        pix2 = _mm_unpackhi_epi8(source, zero);
        factors = _mm_unpackhi_epi8(factorsource, zero);
        pix2 = _mm_add_epi16(_mm_mullo_epi16(pix2, factors), half);
        pix2 = _mm_add_epi16(pix2, _mm_srli_epi16(pix2, 8));
        pix2 = _mm_srli_epi16(pix2, 8);

        source = _mm_packus_epi16(pix1, pix2);
        _mm_storeu_si128((__m128i *)&out[x * 4], source);
    }

#elif defined(IMAGING_SIMD_NEON)

    for (; x < xsize - 15; x += 16) {
        uint8x16x4_t px = vld4q_u8(&in[x * 4]);
        px.val[0] = neon_muldiv255(px.val[0], px.val[3]);
        px.val[1] = neon_muldiv255(px.val[1], px.val[3]);
        px.val[2] = neon_muldiv255(px.val[2], px.val[3]);
        vst4q_u8(&out[x * 4], px);
    }

#endif

    for (; x < xsize; x++) {
        alpha = in[x * 4 + 3];
        out[x * 4 + 0] = MULDIV255(in[x * 4 + 0], alpha, tmp);
        out[x * 4 + 1] = MULDIV255(in[x * 4 + 1], alpha, tmp);
        out[x * 4 + 2] = MULDIV255(in[x * 4 + 2], alpha, tmp);
        out[x * 4 + 3] = alpha;
    }
}

/* RGBa -> RGBA conversion to remove premultiplication
   Needed for correct transforms/resizing on RGBA images */
static void
rgba2rgbA(UINT8 *out, const UINT8 *in, int xsize) {
#if defined(CONVERT_VECTOR)
    int x = 0;

#if defined(IMAGING_SIMD_AVX2)
    for (; x < xsize - 7; x += 8) {
        rgba2rgbA_8(out + x * 4, in + x * 4);
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    CONVERT_ROW(rgba2rgbA_4, 4, out + x * 4, in + x * 4, xsize - x, 4, 4);
#else
    // The NEON kernel is exact, so the scalar code below can finish the row.
    for (; x < xsize - 15; x += 16) {
        uint8x16x4_t px = vld4q_u8(in + x * 4);
        px.val[0] = neon_unpremultiply(px.val[0], px.val[3]);
        px.val[1] = neon_unpremultiply(px.val[1], px.val[3]);
        px.val[2] = neon_unpremultiply(px.val[2], px.val[3]);
        vst4q_u8(out + x * 4, px);
    }
    in += x * 4;
    out += x * 4;
    for (; x < xsize; x++, in += 4) {
        unsigned int alpha = in[3];
        if (alpha == 255 || alpha == 0) {
            *out++ = in[0];
            *out++ = in[1];
            *out++ = in[2];
        } else {
            *out++ = CLIP8((255 * in[0]) / alpha);
            *out++ = CLIP8((255 * in[1]) / alpha);
            *out++ = CLIP8((255 * in[2]) / alpha);
        }
        *out++ = in[3];
    }
#endif
#else
    int x;
    unsigned int alpha;
    for (x = 0; x < xsize; x++, in += 4) {
        alpha = in[3];
        if (alpha == 255 || alpha == 0) {
            *out++ = in[0];
            *out++ = in[1];
            *out++ = in[2];
        } else {
            *out++ = CLIP8((255 * in[0]) / alpha);
            *out++ = CLIP8((255 * in[1]) / alpha);
            *out++ = CLIP8((255 * in[2]) / alpha);
        }
        *out++ = in[3];
    }
#endif
}

static void
rgba2rgb_(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    unsigned int alpha;
    for (x = 0; x < xsize; x++, in += 4) {
        alpha = in[3];
        if (alpha == 255 || alpha == 0) {
            *out++ = in[0];
            *out++ = in[1];
            *out++ = in[2];
        } else {
            *out++ = CLIP8((255 * in[0]) / alpha);
            *out++ = CLIP8((255 * in[1]) / alpha);
            *out++ = CLIP8((255 * in[2]) / alpha);
        }
        *out++ = 255;
    }
}

/*
 * Conversion of RGB + single transparent color either to
 * RGBA or LA, where any pixel matching the color will have the alpha channel set to 0,
 * or RGBa or La, where any pixel matching the color will have all channels set to 0
 */

static void
rgbT2a(UINT8 *out, UINT8 *in, int xsize, int r, int g, int b, int premultiplied) {
#ifdef WORDS_BIGENDIAN
    UINT32 trns = ((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8) | 0xff;
    UINT32 repl = premultiplied ? 0 : (trns & 0xffffff00);
#else
    UINT32 trns = (0xffU << 24) | ((b & 0xff) << 16) | ((g & 0xff) << 8) | (r & 0xff);
    UINT32 repl = premultiplied ? 0 : (trns & 0x00ffffff);
#endif

    int i;

    UINT8 *ref = in != NULL ? in : out;
    for (i = 0; i < xsize; i++, ref += sizeof(trns), out += sizeof(trns)) {
        UINT32 v;
        memcpy(&v, ref, sizeof(v));
        if (v == trns) {
            memcpy(out, &repl, sizeof(repl));
        }
    }
}

/* ---------------- */
/* CMYK conversions */
/* ---------------- */

static void
l2cmyk(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++) {
        *out++ = 0;
        *out++ = 0;
        *out++ = 0;
        *out++ = ~(*in++);
    }
}

static void
la2cmyk(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = 0;
        *out++ = 0;
        *out++ = 0;
        *out++ = ~(in[0]);
    }
}

static inline void
rgb2cmyk_pixel(UINT8 *restrict out, const UINT8 *restrict in) {
    UINT32 v;
    memcpy(&v, in, sizeof(v));
    v = ~v & MAKE_UINT32(255, 255, 255, 0);
    memcpy(out, &v, sizeof(v));
}

static void
rgb2cmyk(UINT8 *restrict out, const UINT8 *restrict in, int xsize) {
    /* Note: no undercolour removal. A pixel at a time, which the compiler
       vectorizes the same way for every target; byte by byte, its AVX2
       code was 15% slower than its SSE4 code. */
#if defined(IMAGING_SIMD_NEON)
    int x = convert_mask16(out, in, xsize, 1, MAKE_UINT32(255, 255, 255, 0), 0);
    out += x * 4, in += x * 4, xsize -= x;
#endif
    CONVERT_PIXELS(rgb2cmyk_pixel, out, in, xsize, 4, 4);
}

void
cmyk2rgb(UINT8 *out, const UINT8 *in, int xsize) {
    int x, nk, tmp;
    for (x = 0; x < xsize; x++) {
        nk = 255 - in[3];
        out[0] = CLIP8(nk - MULDIV255(in[0], nk, tmp));
        out[1] = CLIP8(nk - MULDIV255(in[1], nk, tmp));
        out[2] = CLIP8(nk - MULDIV255(in[2], nk, tmp));
        out[3] = 255;
        out += 4;
        in += 4;
    }
}

static void
cmyk2hsv(UINT8 *out, const UINT8 *in, int xsize) {
    int x, nk, tmp;
    for (x = 0; x < xsize; x++) {
        nk = 255 - in[3];
        out[0] = CLIP8(nk - MULDIV255(in[0], nk, tmp));
        out[1] = CLIP8(nk - MULDIV255(in[1], nk, tmp));
        out[2] = CLIP8(nk - MULDIV255(in[2], nk, tmp));
        rgb2hsv_row(out, out);
        out[3] = 255;
        out += 4;
        in += 4;
    }
}

/* ------------- */
/* I conversions */
/* ------------- */

static void
bit2i(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, out_ += 4) {
        INT32 v = (*in++ != 0) ? 255 : 0;
        memcpy(out_, &v, sizeof(v));
    }
}

static void
l2i(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, out_ += 4) {
        INT32 v = *in++;
        memcpy(out_, &v, sizeof(v));
    }
}

static void
i2l(UINT8 *out, const UINT8 *in_, int xsize) {
    for (int x = 0; x < xsize; x++, out++, in_ += 4) {
        INT32 v;
        memcpy(&v, in_, sizeof(v));
        // Branchless saturation
        *out = (UINT8)(v <= 0 ? 0 : (v >= 255 ? 255 : v));
    }
}

static void
i2f(UINT8 *out_, const UINT8 *in_, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in_ += 4, out_ += 4) {
        INT32 i;
        FLOAT32 f;
        memcpy(&i, in_, sizeof(i));
        f = i;
        memcpy(out_, &f, sizeof(f));
    }
}

static void
i2rgb(UINT8 *out, const UINT8 *in_, int xsize) {
    int x;
    INT32 *in = (INT32 *)in_;
    for (x = 0; x < xsize; x++, in++, out += 4) {
        if (*in <= 0) {
            out[0] = out[1] = out[2] = 0;
        } else if (*in >= 255) {
            out[0] = out[1] = out[2] = 255;
        } else {
            out[0] = out[1] = out[2] = (UINT8)*in;
        }
        out[3] = 255;
    }
}

static void
i2hsv(UINT8 *out, const UINT8 *in_, int xsize) {
    int x;
    INT32 *in = (INT32 *)in_;
    for (x = 0; x < xsize; x++, in++, out += 4) {
        out[0] = 0;
        out[1] = 0;
        if (*in <= 0) {
            out[2] = 0;
        } else if (*in >= 255) {
            out[2] = 255;
        } else {
            out[2] = (UINT8)*in;
        }
        out[3] = 255;
    }
}

/* ------------- */
/* F conversions */
/* ------------- */

static void
bit2f(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, out_ += 4) {
        FLOAT32 f = (*in++ != 0) ? 255.0F : 0.0F;
        memcpy(out_, &f, sizeof(f));
    }
}

static void
l2f(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, out_ += 4) {
        FLOAT32 f = (FLOAT32)*in++;
        memcpy(out_, &f, sizeof(f));
    }
}

static void
f2l(UINT8 *out, const UINT8 *in_, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, out++, in_ += 4) {
        FLOAT32 v;
        memcpy(&v, in_, sizeof(v));
        if (v <= 0.0) {
            *out = 0;
        } else if (v >= 255.0) {
            *out = 255;
        } else {
            *out = (UINT8)v;
        }
    }
}

static void
f2i(UINT8 *out_, const UINT8 *in_, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in_ += 4, out_ += 4) {
        FLOAT32 f;
        INT32 i;
        memcpy(&f, in_, sizeof(f));
        i = f;
        memcpy(out_, &i, sizeof(i));
    }
}

/* ----------------- */
/* YCbCr conversions */
/* ----------------- */

/* See ConvertYCbCr.c for RGB/YCbCr tables */

static void
l2ycbcr(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++) {
        *out++ = *in++;
        *out++ = 128;
        *out++ = 128;
        *out++ = 255;
    }
}

static void
la2ycbcr(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = in[0];
        *out++ = 128;
        *out++ = 128;
        *out++ = 255;
    }
}

static void
ycbcr2l(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = in[0];
    }
}

static void
ycbcr2la(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 4, out += 4) {
        out[0] = out[1] = out[2] = in[0];
        out[3] = 255;
    }
}

/* ------------------------- */
/* I;16 (16-bit) conversions */
/* ------------------------- */

static void
I_I16L(UINT8 *out, const UINT8 *in_, int xsize) {
    int x, v;
    for (x = 0; x < xsize; x++, in_ += 4) {
        INT32 i;
        memcpy(&i, in_, sizeof(i));
        v = CLIP16(i);
        *out++ = (UINT8)v;
        *out++ = (UINT8)(v >> 8);
    }
}

static void
I_I16B(UINT8 *out, const UINT8 *in_, int xsize) {
    int x, v;
    for (x = 0; x < xsize; x++, in_ += 4) {
        INT32 i;
        memcpy(&i, in_, sizeof(i));
        v = CLIP16(i);
        *out++ = (UINT8)(v >> 8);
        *out++ = (UINT8)v;
    }
}

static void
I16L_I(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 2, out_ += 4) {
        INT32 v = in[0] + ((int)in[1] << 8);
        memcpy(out_, &v, sizeof(v));
    }
}

static void
I16B_I(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 2, out_ += 4) {
        INT32 v = in[1] + ((int)in[0] << 8);
        memcpy(out_, &v, sizeof(v));
    }
}

static void
I16L_F(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 2, out_ += 4) {
        FLOAT32 v = in[0] + ((int)in[1] << 8);
        memcpy(out_, &v, sizeof(v));
    }
}

static void
I16B_F(UINT8 *out_, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 2, out_ += 4) {
        FLOAT32 v = in[1] + ((int)in[0] << 8);
        memcpy(out_, &v, sizeof(v));
    }
}

static void
L_I16L(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in++) {
        *out++ = *in;
        *out++ = 0;
    }
}

static void
L_I16B(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in++) {
        *out++ = 0;
        *out++ = *in;
    }
}

static void
I16L_L(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 2) {
        if (in[1] != 0) {
            *out++ = 255;
        } else {
            *out++ = in[0];
        }
    }
}

static void
I16B_L(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 2) {
        if (in[0] != 0) {
            *out++ = 255;
        } else {
            *out++ = in[1];
        }
    }
}

static void
I16_RGB(UINT8 *out, const UINT8 *in, int xsize) {
    int x;
    for (x = 0; x < xsize; x++, in += 2) {
        UINT8 v = in[1] == 0 ? in[0] : 255;
        *out++ = v;
        *out++ = v;
        *out++ = v;
        *out++ = 255;
    }
}

/* ------------------- */
/* Palette conversions */
/* ------------------- */

/* FIXME: translate indexed versions to pointer versions below this line */

static void
p2bit(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    /* FIXME: precalculate grayscale palette? */
    for (x = 0; x < xsize; x++) {
        *out++ = (L(&palette->palette[in[x] * 4]) >= 128000) ? 255 : 0;
    }
}

static void
pa2bit(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    /* FIXME: precalculate grayscale palette? */
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = (L(&palette->palette[in[0] * 4]) >= 128000) ? 255 : 0;
    }
}

static void
p2l(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    /* FIXME: precalculate grayscale palette? */
    for (x = 0; x < xsize; x++) {
        *out++ = L24(&palette->palette[in[x] * 4]) >> 16;
    }
}

static void
pa2l(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    /* FIXME: precalculate grayscale palette? */
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = L24(&palette->palette[in[0] * 4]) >> 16;
    }
}

static void
pa2p(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = in[0];
    }
}

static void
p2pa(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    const int rgb = palette->mode == IMAGING_MODE_RGB;
    for (x = 0; x < xsize; x++, in++) {
        const UINT8 *rgba = &palette->palette[in[0] * 4];
        *out++ = in[0];
        *out++ = in[0];
        *out++ = in[0];
        *out++ = rgb ? 255 : rgba[3];
    }
}

static void
p2la(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    /* FIXME: precalculate grayscale palette? */
    for (x = 0; x < xsize; x++, out += 4) {
        const UINT8 *rgba = &palette->palette[*in++ * 4];
        out[0] = out[1] = out[2] = L24(rgba) >> 16;
        out[3] = rgba[3];
    }
}

static void
pa2la(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    /* FIXME: precalculate grayscale palette? */
    for (x = 0; x < xsize; x++, in += 4, out += 4) {
        out[0] = out[1] = out[2] = L24(&palette->palette[in[0] * 4]) >> 16;
        out[3] = in[3];
    }
}

static void
p2i(UINT8 *out_, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    for (x = 0; x < xsize; x++, out_ += 4) {
        INT32 v = L24(&palette->palette[in[x] * 4]) >> 16;
        memcpy(out_, &v, sizeof(v));
    }
}

static void
pa2i(UINT8 *out_, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    INT32 *out = (INT32 *)out_;
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = L24(&palette->palette[in[0] * 4]) >> 16;
    }
}

static void
p2f(UINT8 *out_, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    for (x = 0; x < xsize; x++, out_ += 4) {
        FLOAT32 v = L(&palette->palette[in[x] * 4]) / 1000.0F;
        memcpy(out_, &v, sizeof(v));
    }
}

static void
pa2f(UINT8 *out_, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    FLOAT32 *out = (FLOAT32 *)out_;
    for (x = 0; x < xsize; x++, in += 4) {
        *out++ = (float)L(&palette->palette[in[0] * 4]) / 1000.0F;
    }
}

// Set the alpha channel of the UINT32 `v` in-place to the given value.
#ifdef WORDS_BIGENDIAN
#define SET_ALPHA_32(v, alpha) v = ((v & 0xFFFFFF00u) | (alpha))
#else
#define SET_ALPHA_32(v, alpha) v = ((v & 0x00FFFFFFu) | ((UINT32)(alpha) << 24))
#endif

static void
p2rgb(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    for (int x = 0; x < xsize; x++, in++, out += 4) {
        UINT32 v;
        memcpy(&v, &palette->palette[in[0] * 4], sizeof(v));
        SET_ALPHA_32(v, 0xFF);
        memcpy(out, &v, sizeof(v));
    }
}

static void
pa2rgb(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    for (int x = 0; x < xsize; x++, in += 4, out += 4) {
        UINT32 v;
        memcpy(&v, &palette->palette[in[0] * 4], sizeof(v));
        SET_ALPHA_32(v, 0xFF);
        memcpy(out, &v, sizeof(v));
    }
}

static void
p2hsv(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    for (x = 0; x < xsize; x++, out += 4) {
        const UINT8 *rgb = &palette->palette[*in++ * 4];
        rgb2hsv_row(out, rgb);
        out[3] = 255;
    }
}

static void
pa2hsv(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    int x;
    for (x = 0; x < xsize; x++, in += 4, out += 4) {
        const UINT8 *rgb = &palette->palette[in[0] * 4];
        rgb2hsv_row(out, rgb);
        out[3] = 255;
    }
}

static void
p2rgba(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    for (int x = 0; x < xsize; x++, in++, out += 4) {
        memcpy(out, &palette->palette[in[0] * 4], 4);
    }
}

static void
pa2rgba(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    for (int x = 0; x < xsize; x++, in += 4, out += 4) {
        UINT32 v;
        memcpy(&v, &palette->palette[in[0] * 4], sizeof(v));
        SET_ALPHA_32(v, in[3]);
        memcpy(out, &v, sizeof(v));
    }
}

static void
p2cmyk(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    p2rgb(out, in, xsize, palette);
    rgb2cmyk(out, out, xsize);
}

static void
pa2cmyk(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    pa2rgb(out, in, xsize, palette);
    rgb2cmyk(out, out, xsize);
}

static void
p2ycbcr(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    p2rgb(out, in, xsize, palette);
    ImagingConvertRGB2YCbCr(out, out, xsize);
}

static void
pa2ycbcr(UINT8 *out, const UINT8 *in, int xsize, ImagingPalette palette) {
    pa2rgb(out, in, xsize, palette);
    ImagingConvertRGB2YCbCr(out, out, xsize);
}

static Imaging
frompalette(Imaging imOut, Imaging imIn, const ModeID mode) {
    ImagingSectionCookie cookie;
    int alpha;
    int y;
    void (*convert)(UINT8 *, const UINT8 *, int, ImagingPalette);

    /* Map palette image to L, RGB, RGBA, or CMYK */

    if (!imIn->palette) {
        return (Imaging)ImagingError_ValueError("no palette");
    }

    alpha = imIn->mode == IMAGING_MODE_PA;

    if (mode == IMAGING_MODE_1) {
        convert = alpha ? pa2bit : p2bit;
    } else if (mode == IMAGING_MODE_L) {
        convert = alpha ? pa2l : p2l;
    } else if (mode == IMAGING_MODE_LA) {
        convert = alpha ? pa2la : p2la;
    } else if (mode == IMAGING_MODE_P) {
        convert = pa2p;
    } else if (mode == IMAGING_MODE_PA) {
        convert = p2pa;
    } else if (mode == IMAGING_MODE_I) {
        convert = alpha ? pa2i : p2i;
    } else if (mode == IMAGING_MODE_F) {
        convert = alpha ? pa2f : p2f;
    } else if (mode == IMAGING_MODE_RGB) {
        convert = alpha ? pa2rgb : p2rgb;
    } else if (mode == IMAGING_MODE_RGBA || mode == IMAGING_MODE_RGBX) {
        convert = alpha ? pa2rgba : p2rgba;
    } else if (mode == IMAGING_MODE_CMYK) {
        convert = alpha ? pa2cmyk : p2cmyk;
    } else if (mode == IMAGING_MODE_YCbCr) {
        convert = alpha ? pa2ycbcr : p2ycbcr;
    } else if (mode == IMAGING_MODE_HSV) {
        convert = alpha ? pa2hsv : p2hsv;
    } else {
        return (Imaging)ImagingError_ValueError("conversion not supported");
    }

    imOut = ImagingNew2Dirty(mode, imOut, imIn);
    if (!imOut) {
        return NULL;
    }
    if (mode == IMAGING_MODE_P || mode == IMAGING_MODE_PA) {
        ImagingPaletteDelete(imOut->palette);
        imOut->palette = ImagingPaletteDuplicate(imIn->palette);
    }

    ImagingSectionEnter(&cookie);
    for (y = 0; y < imIn->ysize; y++) {
        (*convert)(
            (UINT8 *)imOut->image[y],
            (UINT8 *)imIn->image[y],
            imIn->xsize,
            imIn->palette
        );
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}

#if defined(_MSC_VER)
#pragma optimize("", off)
#endif
static Imaging
topalette(
    Imaging imOut, Imaging imIn, const ModeID mode, ImagingPalette inpalette, int dither
) {
    ImagingSectionCookie cookie;
    int alpha;
    int x, y;
    ImagingPalette palette = inpalette;

    /* Map L or RGB/RGBX/RGBA/RGBa to palette image */
    if (imIn->mode != IMAGING_MODE_L && imIn->mode != IMAGING_MODE_RGB &&
        imIn->mode != IMAGING_MODE_RGBX && imIn->mode != IMAGING_MODE_RGBA &&
        imIn->mode != IMAGING_MODE_RGBa) {
        return (Imaging)ImagingError_ValueError("conversion not supported");
    }

    alpha = mode == IMAGING_MODE_PA;

    if (palette != NULL) {
        if (palette->size == 0) {
            return (Imaging)ImagingError_ValueError("empty palette");
        }
    } else {
        /* FIXME: make user configurable */
        if (imIn->bands == 1) {
            palette = ImagingPaletteNew(IMAGING_MODE_RGB);

            palette->size = 256;
            int i;
            for (i = 0; i < 256; i++) {
                palette->palette[i * 4] = palette->palette[i * 4 + 1] =
                    palette->palette[i * 4 + 2] = (UINT8)i;
            }
        } else {
            palette = ImagingPaletteNewBrowser(); /* Standard colour cube */
        }
    }

    if (!palette) {
        return (Imaging)ImagingError_ValueError("no palette");
    }

    imOut = ImagingNew2Dirty(mode, imOut, imIn);
    if (!imOut) {
        if (palette != inpalette) {
            ImagingPaletteDelete(palette);
        }
        return NULL;
    }

    ImagingPaletteDelete(imOut->palette);
    imOut->palette = ImagingPaletteDuplicate(palette);

    if (imIn->bands == 1) {
        /* grayscale image */

        /* Grayscale palette: copy data as is */
        ImagingSectionEnter(&cookie);
        for (y = 0; y < imIn->ysize; y++) {
            if (alpha) {
                l2rgb((UINT8 *)imOut->image[y], (UINT8 *)imIn->image[y], imIn->xsize);
            } else {
                memcpy(imOut->image[y], imIn->image[y], imIn->linesize);
            }
        }
        ImagingSectionLeave(&cookie);

    } else {
        /* colour image */

        /* Create mapping cache */
        if (ImagingPaletteCachePrepare(palette) < 0) {
            ImagingDelete(imOut);
            if (palette != inpalette) {
                ImagingPaletteDelete(palette);
            }
            return NULL;
        }

        if (dither) {
            /* floyd-steinberg dither */

            int *errors;
            errors = calloc(imIn->xsize + 1, sizeof(int) * 3);
            if (!errors) {
                ImagingDelete(imOut);
                return ImagingError_MemoryError();
            }

            /* Map each pixel to the nearest palette entry */
            ImagingSectionEnter(&cookie);
            for (y = 0; y < imIn->ysize; y++) {
                int r, r0, r1, r2;
                int g, g0, g1, g2;
                int b, b0, b1, b2;
                UINT8 *in = (UINT8 *)imIn->image[y];
                UINT8 *out = alpha ? (UINT8 *)imOut->image32[y] : imOut->image8[y];
                int *e = errors;

                r = r0 = r1 = 0;
                g = g0 = g1 = 0;
                b = b0 = b1 = b2 = 0;

                for (x = 0; x < imIn->xsize; x++, in += 4) {
                    int d2;
                    INT16 *cache;

                    r = CLIP8(in[0] + (r + e[3 + 0]) / 16);
                    g = CLIP8(in[1] + (g + e[3 + 1]) / 16);
                    b = CLIP8(in[2] + (b + e[3 + 2]) / 16);

                    /* get closest colour */
                    cache = &ImagingPaletteCache(palette, r, g, b);
                    if (cache[0] == 0x100) {
                        ImagingPaletteCacheUpdate(palette, r, g, b);
                    }
                    if (alpha) {
                        out[x * 4] = out[x * 4 + 1] = out[x * 4 + 2] = (UINT8)cache[0];
                        out[x * 4 + 3] = 255;
                    } else {
                        out[x] = (UINT8)cache[0];
                    }

                    r -= (int)palette->palette[cache[0] * 4];
                    g -= (int)palette->palette[cache[0] * 4 + 1];
                    b -= (int)palette->palette[cache[0] * 4 + 2];

                    /* propagate errors (don't ask ;-) */
                    r2 = r;
                    d2 = r + r;
                    r += d2;
                    e[0] = r + r0;
                    r += d2;
                    r0 = r + r1;
                    r1 = r2;
                    r += d2;
                    g2 = g;
                    d2 = g + g;
                    g += d2;
                    e[1] = g + g0;
                    g += d2;
                    g0 = g + g1;
                    g1 = g2;
                    g += d2;
                    b2 = b;
                    d2 = b + b;
                    b += d2;
                    e[2] = b + b0;
                    b += d2;
                    b0 = b + b1;
                    b1 = b2;
                    b += d2;

                    e += 3;
                }

                e[0] = b0;
                e[1] = b1;
                e[2] = b2;
            }
            ImagingSectionLeave(&cookie);
            free(errors);

        } else {
            /* closest colour */
            ImagingSectionEnter(&cookie);
            for (y = 0; y < imIn->ysize; y++) {
                int r, g, b;
                UINT8 *in = (UINT8 *)imIn->image[y];
                UINT8 *out = alpha ? (UINT8 *)imOut->image32[y] : imOut->image8[y];

                for (x = 0; x < imIn->xsize; x++, in += 4) {
                    INT16 *cache;

                    r = in[0];
                    g = in[1];
                    b = in[2];

                    /* get closest colour */
                    cache = &ImagingPaletteCache(palette, r, g, b);
                    if (cache[0] == 0x100) {
                        ImagingPaletteCacheUpdate(palette, r, g, b);
                    }
                    if (alpha) {
                        out[x * 4] = out[x * 4 + 1] = out[x * 4 + 2] = (UINT8)cache[0];
                        out[x * 4 + 3] = 255;
                    } else {
                        out[x] = (UINT8)cache[0];
                    }
                }
            }
            ImagingSectionLeave(&cookie);
        }
        if (inpalette != palette) {
            ImagingPaletteCacheDelete(palette);
        }
    }

    if (inpalette != palette) {
        ImagingPaletteDelete(palette);
    }

    return imOut;
}

static Imaging
tobilevel(Imaging imOut, Imaging imIn) {
    ImagingSectionCookie cookie;
    int x, y;
    int *errors;

    /* Map L or RGB to dithered 1 image */
    if (imIn->mode != IMAGING_MODE_L && imIn->mode != IMAGING_MODE_RGB) {
        return (Imaging)ImagingError_ValueError("conversion not supported");
    }

    imOut = ImagingNew2Dirty(IMAGING_MODE_1, imOut, imIn);
    if (!imOut) {
        return NULL;
    }

    errors = calloc(imIn->xsize + 1, sizeof(int));
    if (!errors) {
        ImagingDelete(imOut);
        return ImagingError_MemoryError();
    }

    if (imIn->bands == 1) {
        /* map each pixel to black or white, using error diffusion */
        ImagingSectionEnter(&cookie);
        for (y = 0; y < imIn->ysize; y++) {
            int l, l0, l1, l2, d2;
            UINT8 *in = (UINT8 *)imIn->image[y];
            UINT8 *out = imOut->image8[y];

            l = l0 = l1 = 0;

            for (x = 0; x < imIn->xsize; x++) {
                /* pick closest colour */
                l = CLIP8(in[x] + (l + errors[x + 1]) / 16);
                out[x] = (l > 128) ? 255 : 0;

                /* propagate errors */
                l -= (int)out[x];
                l2 = l;
                d2 = l + l;
                l += d2;
                errors[x] = l + l0;
                l += d2;
                l0 = l + l1;
                l1 = l2;
                l += d2;
            }

            errors[x] = l0;
        }
        ImagingSectionLeave(&cookie);

    } else {
        /* map each pixel to black or white, using error diffusion */
        ImagingSectionEnter(&cookie);
        for (y = 0; y < imIn->ysize; y++) {
            int l, l0, l1, l2, d2;
            UINT8 *in = (UINT8 *)imIn->image[y];
            UINT8 *out = imOut->image8[y];

            l = l0 = l1 = 0;

            for (x = 0; x < imIn->xsize; x++, in += 4) {
                /* pick closest colour */
                l = CLIP8(L(in) / 1000 + (l + errors[x + 1]) / 16);
                out[x] = (l > 128) ? 255 : 0;

                /* propagate errors */
                l -= (int)out[x];
                l2 = l;
                d2 = l + l;
                l += d2;
                errors[x] = l + l0;
                l += d2;
                l0 = l + l1;
                l1 = l2;
                l += d2;
            }

            errors[x] = l0;
        }
        ImagingSectionLeave(&cookie);
    }

    free(errors);

    return imOut;
}
#if defined(_MSC_VER)
#pragma optimize("", on)
#endif

/* ------------------- */
/* Conversion handlers */
/* ------------------- */

static struct {
    const ModeID from;
    const ModeID to;
    ImagingShuffler convert;
} converters[] = {
    {IMAGING_MODE_1, IMAGING_MODE_L, bit2l},
    {IMAGING_MODE_1, IMAGING_MODE_I, bit2i},
    {IMAGING_MODE_1, IMAGING_MODE_F, bit2f},
    {IMAGING_MODE_1, IMAGING_MODE_LA, bit2rgb},
    {IMAGING_MODE_1, IMAGING_MODE_RGB, bit2rgb},
    {IMAGING_MODE_1, IMAGING_MODE_RGBA, bit2rgb},
    {IMAGING_MODE_1, IMAGING_MODE_RGBX, bit2rgb},
    {IMAGING_MODE_1, IMAGING_MODE_CMYK, bit2cmyk},
    {IMAGING_MODE_1, IMAGING_MODE_YCbCr, bit2ycbcr},
    {IMAGING_MODE_1, IMAGING_MODE_HSV, bit2hsv},

    {IMAGING_MODE_L, IMAGING_MODE_1, l2bit},
    {IMAGING_MODE_L, IMAGING_MODE_LA, l2rgb},
    {IMAGING_MODE_L, IMAGING_MODE_I, l2i},
    {IMAGING_MODE_L, IMAGING_MODE_F, l2f},
    {IMAGING_MODE_L, IMAGING_MODE_RGB, l2rgb},
    {IMAGING_MODE_L, IMAGING_MODE_RGBA, l2rgb},
    {IMAGING_MODE_L, IMAGING_MODE_RGBX, l2rgb},
    {IMAGING_MODE_L, IMAGING_MODE_CMYK, l2cmyk},
    {IMAGING_MODE_L, IMAGING_MODE_YCbCr, l2ycbcr},
    {IMAGING_MODE_L, IMAGING_MODE_HSV, l2hsv},

    {IMAGING_MODE_LA, IMAGING_MODE_L, la2l},
    {IMAGING_MODE_LA, IMAGING_MODE_La, lA2la},
    {IMAGING_MODE_LA, IMAGING_MODE_RGB, la2rgb},
    {IMAGING_MODE_LA, IMAGING_MODE_RGBA, la2rgb},
    {IMAGING_MODE_LA, IMAGING_MODE_RGBX, la2rgb},
    {IMAGING_MODE_LA, IMAGING_MODE_CMYK, la2cmyk},
    {IMAGING_MODE_LA, IMAGING_MODE_YCbCr, la2ycbcr},
    {IMAGING_MODE_LA, IMAGING_MODE_HSV, la2hsv},

    {IMAGING_MODE_La, IMAGING_MODE_LA, la2lA},

    {IMAGING_MODE_I, IMAGING_MODE_L, i2l},
    {IMAGING_MODE_I, IMAGING_MODE_F, i2f},
    {IMAGING_MODE_I, IMAGING_MODE_RGB, i2rgb},
    {IMAGING_MODE_I, IMAGING_MODE_RGBA, i2rgb},
    {IMAGING_MODE_I, IMAGING_MODE_RGBX, i2rgb},
    {IMAGING_MODE_I, IMAGING_MODE_HSV, i2hsv},

    {IMAGING_MODE_F, IMAGING_MODE_L, f2l},
    {IMAGING_MODE_F, IMAGING_MODE_I, f2i},

    {IMAGING_MODE_RGB, IMAGING_MODE_1, rgb2bit},
    {IMAGING_MODE_RGB, IMAGING_MODE_L, rgb2l},
    {IMAGING_MODE_RGB, IMAGING_MODE_LA, rgb2la},
    {IMAGING_MODE_RGB, IMAGING_MODE_La, rgb2la},
    {IMAGING_MODE_RGB, IMAGING_MODE_I, rgb2i},
    {IMAGING_MODE_RGB, IMAGING_MODE_I_16, rgb2i16l},
    {IMAGING_MODE_RGB, IMAGING_MODE_I_16L, rgb2i16l},
    {IMAGING_MODE_RGB, IMAGING_MODE_I_16B, rgb2i16b},
#ifdef WORDS_BIGENDIAN
    {IMAGING_MODE_RGB, IMAGING_MODE_I_16N, rgb2i16b},
#else
    {IMAGING_MODE_RGB, IMAGING_MODE_I_16N, rgb2i16l},
#endif
    {IMAGING_MODE_RGB, IMAGING_MODE_F, rgb2f},
    {IMAGING_MODE_RGB, IMAGING_MODE_RGBA, rgb2rgba},
    {IMAGING_MODE_RGB, IMAGING_MODE_RGBa, rgb2rgba},
    {IMAGING_MODE_RGB, IMAGING_MODE_RGBX, rgb2rgba},
    {IMAGING_MODE_RGB, IMAGING_MODE_CMYK, rgb2cmyk},
    {IMAGING_MODE_RGB, IMAGING_MODE_YCbCr, ImagingConvertRGB2YCbCr},
    {IMAGING_MODE_RGB, IMAGING_MODE_HSV, rgb2hsv},

    {IMAGING_MODE_RGBA, IMAGING_MODE_1, rgb2bit},
    {IMAGING_MODE_RGBA, IMAGING_MODE_L, rgb2l},
    {IMAGING_MODE_RGBA, IMAGING_MODE_LA, rgba2la},
    {IMAGING_MODE_RGBA, IMAGING_MODE_I, rgb2i},
    {IMAGING_MODE_RGBA, IMAGING_MODE_F, rgb2f},
    {IMAGING_MODE_RGBA, IMAGING_MODE_RGB, rgb2rgba},
    {IMAGING_MODE_RGBA, IMAGING_MODE_RGBa, rgbA2rgba},
    {IMAGING_MODE_RGBA, IMAGING_MODE_RGBX, rgb2rgba},
    {IMAGING_MODE_RGBA, IMAGING_MODE_CMYK, rgb2cmyk},
    {IMAGING_MODE_RGBA, IMAGING_MODE_YCbCr, ImagingConvertRGB2YCbCr},
    {IMAGING_MODE_RGBA, IMAGING_MODE_HSV, rgb2hsv},

    {IMAGING_MODE_RGBa, IMAGING_MODE_RGBA, rgba2rgbA},
    {IMAGING_MODE_RGBa, IMAGING_MODE_RGB, rgba2rgb_},

    {IMAGING_MODE_RGBX, IMAGING_MODE_1, rgb2bit},
    {IMAGING_MODE_RGBX, IMAGING_MODE_L, rgb2l},
    {IMAGING_MODE_RGBX, IMAGING_MODE_LA, rgb2la},
    {IMAGING_MODE_RGBX, IMAGING_MODE_I, rgb2i},
    {IMAGING_MODE_RGBX, IMAGING_MODE_F, rgb2f},
    {IMAGING_MODE_RGBX, IMAGING_MODE_RGB, rgb2rgba},
    {IMAGING_MODE_RGBX, IMAGING_MODE_CMYK, rgb2cmyk},
    {IMAGING_MODE_RGBX, IMAGING_MODE_YCbCr, ImagingConvertRGB2YCbCr},
    {IMAGING_MODE_RGBX, IMAGING_MODE_HSV, rgb2hsv},

    {IMAGING_MODE_CMYK, IMAGING_MODE_RGB, cmyk2rgb},
    {IMAGING_MODE_CMYK, IMAGING_MODE_RGBA, cmyk2rgb},
    {IMAGING_MODE_CMYK, IMAGING_MODE_RGBX, cmyk2rgb},
    {IMAGING_MODE_CMYK, IMAGING_MODE_HSV, cmyk2hsv},

    {IMAGING_MODE_YCbCr, IMAGING_MODE_L, ycbcr2l},
    {IMAGING_MODE_YCbCr, IMAGING_MODE_LA, ycbcr2la},
    {IMAGING_MODE_YCbCr, IMAGING_MODE_RGB, ImagingConvertYCbCr2RGB},

    {IMAGING_MODE_HSV, IMAGING_MODE_RGB, hsv2rgb},

    {IMAGING_MODE_I, IMAGING_MODE_I_16, I_I16L},
    {IMAGING_MODE_I_16, IMAGING_MODE_I, I16L_I},
    {IMAGING_MODE_I_16, IMAGING_MODE_RGB, I16_RGB},
    {IMAGING_MODE_L, IMAGING_MODE_I_16, L_I16L},
    {IMAGING_MODE_I_16, IMAGING_MODE_L, I16L_L},

    {IMAGING_MODE_I, IMAGING_MODE_I_16L, I_I16L},
    {IMAGING_MODE_I_16L, IMAGING_MODE_I, I16L_I},
    {IMAGING_MODE_I, IMAGING_MODE_I_16B, I_I16B},
    {IMAGING_MODE_I_16B, IMAGING_MODE_I, I16B_I},
#ifdef WORDS_BIGENDIAN
    {IMAGING_MODE_I, IMAGING_MODE_I_16N, I_I16B},
    {IMAGING_MODE_I_16N, IMAGING_MODE_I, I16B_I},
#else
    {IMAGING_MODE_I, IMAGING_MODE_I_16N, I_I16L},
    {IMAGING_MODE_I_16N, IMAGING_MODE_I, I16L_I},
#endif

    {IMAGING_MODE_L, IMAGING_MODE_I_16L, L_I16L},
    {IMAGING_MODE_I_16L, IMAGING_MODE_L, I16L_L},
    {IMAGING_MODE_L, IMAGING_MODE_I_16B, L_I16B},
    {IMAGING_MODE_I_16B, IMAGING_MODE_L, I16B_L},
#ifdef WORDS_BIGENDIAN
    {IMAGING_MODE_L, IMAGING_MODE_I_16N, L_I16B},
    {IMAGING_MODE_I_16N, IMAGING_MODE_L, I16B_L},
#else
    {IMAGING_MODE_L, IMAGING_MODE_I_16N, L_I16L},
    {IMAGING_MODE_I_16N, IMAGING_MODE_L, I16L_L},
#endif

    {IMAGING_MODE_I_16, IMAGING_MODE_F, I16L_F},
    {IMAGING_MODE_I_16L, IMAGING_MODE_F, I16L_F},
    {IMAGING_MODE_I_16B, IMAGING_MODE_F, I16B_F},
#ifdef WORDS_BIGENDIAN
    {IMAGING_MODE_I_16N, IMAGING_MODE_F, I16B_F}
#else
    {IMAGING_MODE_I_16N, IMAGING_MODE_F, I16L_F}
#endif
};

/**
 * Convert imIn to `mode`.
 * If imIn is already in `mode`, this performs a copy into imOut
 * (or a newly allocated image if imOut is NULL).
 *
 * @param imOut   Existing image to write into
 *                (must already be in `mode` and the same size as imIn),
 *                or NULL to allocate a new image for the result.
 * @param imIn    Source image to convert.
 * @param mode    Target mode.
 * @param palette Target palette for conversions to "P" or "PA";
 *                NULL to use a default palette.
 * @param dither  Nonzero to dither when converting to "P", "PA" or "1".
 * @return        The resulting Imaging object,
 *                or NULL with a Python exception set on failure.
 */
Imaging
ImagingConvert(
    Imaging imOut, Imaging imIn, ModeID mode, ImagingPalette palette, int dither
) {
    ImagingSectionCookie cookie;
    ImagingShuffler convert;

    if (!imIn) {
        return (Imaging)ImagingError_ModeError();
    }

    if (mode == IMAGING_MODE_UNKNOWN) {
        /* Map palette image to full depth */
        if (!imIn->palette) {
            return (Imaging)ImagingError_ModeError();
        }
        mode = imIn->palette->mode;
    } else {
        /* Same mode? */
        if (imIn->mode == mode) {
            return ImagingCopy2(imOut, imIn);
        }
    }

    /* test for special conversions */

    if (imIn->mode == IMAGING_MODE_P || imIn->mode == IMAGING_MODE_PA) {
        return frompalette(imOut, imIn, mode);
    }

    if (mode == IMAGING_MODE_P || mode == IMAGING_MODE_PA) {
        return topalette(imOut, imIn, mode, palette, dither);
    }

    if (dither && mode == IMAGING_MODE_1) {
        return tobilevel(imOut, imIn);
    }

    /* standard conversion machinery */

    convert = NULL;
    for (size_t i = 0; i < sizeof(converters) / sizeof(*converters); i++) {
        if (imIn->mode == converters[i].from && mode == converters[i].to) {
            convert = converters[i].convert;
            break;
        }
    }

    if (!convert) {
        return (Imaging)PyErr_Format(
            PyExc_ValueError,
            "conversion from %.10s to %.10s not supported",
            getModeData(imIn->mode)->name,
            getModeData(mode)->name
        );
    }

    imOut = ImagingNew2Dirty(mode, imOut, imIn);
    if (!imOut) {
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);
    for (int y = 0; y < imIn->ysize; y++) {
        (*convert)((UINT8 *)imOut->image[y], (UINT8 *)imIn->image[y], imIn->xsize);
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

    return imOut;
}

Imaging
ImagingConvertTransparent(Imaging imIn, const ModeID mode, int r, int g, int b) {
    ImagingSectionCookie cookie;
    ImagingShuffler convert;
    Imaging imOut = NULL;
    int premultiplied = 0;
    // If the transparency matches pixels in the source image, not the converted image
    UINT8 *source;
    int source_transparency = 0;
    int y;

    if (!imIn) {
        return (Imaging)ImagingError_ModeError();
    }

    if (imIn->mode == IMAGING_MODE_RGB &&
        (mode == IMAGING_MODE_RGBA || mode == IMAGING_MODE_RGBa)) {
        convert = rgb2rgba;
        if (mode == IMAGING_MODE_RGBa) {
            premultiplied = 1;
        }
    } else if (
        imIn->mode == IMAGING_MODE_RGB &&
        (mode == IMAGING_MODE_LA || mode == IMAGING_MODE_La)
    ) {
        convert = rgb2la;
        source_transparency = 1;
        if (mode == IMAGING_MODE_La) {
            premultiplied = 1;
        }
    } else if (
        (imIn->mode == IMAGING_MODE_1 || imIn->mode == IMAGING_MODE_I ||
         imIn->mode == IMAGING_MODE_I_16 || imIn->mode == IMAGING_MODE_L) &&
        (mode == IMAGING_MODE_RGBA || mode == IMAGING_MODE_LA)
    ) {
        if (imIn->mode == IMAGING_MODE_1) {
            convert = bit2rgb;
        } else if (imIn->mode == IMAGING_MODE_I) {
            convert = i2rgb;
        } else if (imIn->mode == IMAGING_MODE_I_16) {
            convert = I16_RGB;
        } else {
            convert = l2rgb;
        }
        g = b = r;
    } else {
        return (Imaging)PyErr_Format(
            PyExc_ValueError,
            "conversion from %.10s to %.10s not supported in convert_transparent",
            getModeData(imIn->mode)->name,
            getModeData(mode)->name
        );
    }

    imOut = ImagingNew2Dirty(mode, imOut, imIn);
    if (!imOut) {
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);
    for (y = 0; y < imIn->ysize; y++) {
        (*convert)((UINT8 *)imOut->image[y], (UINT8 *)imIn->image[y], imIn->xsize);

        source = source_transparency ? (UINT8 *)imIn->image[y] : NULL;
        rgbT2a((UINT8 *)imOut->image[y], source, imIn->xsize, r, g, b, premultiplied);
    }
    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

    return imOut;
}
