/* The SIMD kernels of the mode conversions (Convert.c), compiled into
   SIMDKernels.c. Each takes a row, does the pixels its vector loop can and
   returns how many, and Convert.c finishes the row; the kernels built on
   CONVERT_ROW do the whole row (the last pixels through zero-padded
   buffers, so a pixel's result doesn't depend on its position). */

#include <stdint.h>
#include <string.h>

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

/* Removes premultiplication from 16 pixels: CLIP8(255 * c / alpha),
   truncated, and c where alpha is 0, with one division per pixel. c is
   clipped to alpha (giving 255, as the clip would), then multiplied by
   (255 + 2^-12) / alpha in floats: the two roundings stay within 2^-22 of
   the product, so it is above 255 c / alpha when that is an integer and
   below the next integer otherwise (at least 1 / alpha away). Checked for
   every c and alpha. */
static inline void
neon_unpremultiply_16(uint8x16x4_t *px) {
    const uint8x16_t a = px->val[3];
    const uint16x8_t alo = vmovl_u8(vget_low_u8(a)), ahi = vmovl_u8(vget_high_u8(a));
    const float32x4_t k = vdupq_n_f32(255.0f + 1.0f / 4096);
    const float32x4_t r[4] = {
        vdivq_f32(k, vcvtq_f32_u32(vmovl_u16(vget_low_u16(alo)))),
        vdivq_f32(k, vcvtq_f32_u32(vmovl_u16(vget_high_u16(alo)))),
        vdivq_f32(k, vcvtq_f32_u32(vmovl_u16(vget_low_u16(ahi)))),
        vdivq_f32(k, vcvtq_f32_u32(vmovl_u16(vget_high_u16(ahi))))
    };
    int b;
    for (b = 0; b < 3; b++) {
        const uint8x16_t c = vminq_u8(px->val[b], a);
        const uint16x8_t lo = vmovl_u8(vget_low_u8(c)), hi = vmovl_u8(vget_high_u8(c));
        /* alpha 0: 0 times infinity, NaN, converts to 0 and is replaced */
        const uint32x4_t q0 =
            vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), r[0]));
        const uint32x4_t q1 =
            vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), r[1]));
        const uint32x4_t q2 =
            vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), r[2]));
        const uint32x4_t q3 =
            vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), r[3]));
        const uint8x16_t res = vcombine_u8(
            vmovn_u16(vcombine_u16(vmovn_u32(q0), vmovn_u32(q1))),
            vmovn_u16(vcombine_u16(vmovn_u32(q2), vmovn_u32(q3)))
        );
        px->val[b] = vbslq_u8(vceqzq_u8(a), px->val[b], res);
    }
}
#endif

#if defined(IMAGING_SIMD_NEON) && \
    (!defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define UNPREMULTIPLY_TABLE
/* The scalar code's results for every alpha (rows) and c: removing
   premultiplication by table lookups, two pixels per 64-bit word, is 1.7
   times as fast on the Cortex-A78AE as neon_unpremultiply_16, which
   converts to floats (on x86 it's slower than the vector kernels). Built on
   first use; while another thread builds it, the vector kernel runs. */
static UINT8 unpremultiply_table[256 * 256];
static long unpremultiply_state = 0; /* 0 missing, 1 being built, 2 built */

static void
unpremultiply_build(void) {
    unsigned int alpha, c;
    for (alpha = 0; alpha < 256; alpha++) {
        for (c = 0; c < 256; c++) {
            unpremultiply_table[alpha * 256 + c] =
                alpha == 0 || alpha == 255 ? c : CLIP8((255 * c) / alpha);
        }
    }
}

static const UINT8 *
unpremultiply_lookup(void) {
#if defined(__GNUC__) || defined(__clang__)
    long expected = 0;
    if (__atomic_load_n(&unpremultiply_state, __ATOMIC_ACQUIRE) == 2) {
        return unpremultiply_table;
    }
    if (__atomic_compare_exchange_n(
            &unpremultiply_state, &expected, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        )) {
        unpremultiply_build();
        __atomic_store_n(&unpremultiply_state, 2, __ATOMIC_RELEASE);
        return unpremultiply_table;
    }
#elif defined(_MSC_VER)
    if (_InterlockedOr(&unpremultiply_state, 0) == 2) {
        return unpremultiply_table;
    }
    if (_InterlockedCompareExchange(&unpremultiply_state, 1, 0) == 0) {
        unpremultiply_build();
        _InterlockedExchange(&unpremultiply_state, 2);
        return unpremultiply_table;
    }
#endif
    return NULL;
}

/* Two little-endian RGBa pixels */
static inline uint64_t
unpremultiply_2px(const UINT8 *table, uint64_t p) {
    const UINT8 *t0 = table + (((p >> 24) & 255) << 8);
    const UINT8 *t1 = table + ((p >> 56) << 8);
    return (uint64_t)t0[p & 255] | (uint64_t)t0[(p >> 8) & 255] << 8 |
           (uint64_t)t0[(p >> 16) & 255] << 16 | (p & 0xff000000u) |
           (uint64_t)t1[(p >> 32) & 255] << 32 | (uint64_t)t1[(p >> 40) & 255] << 40 |
           (uint64_t)t1[(p >> 48) & 255] << 48 | (p & 0xff00000000000000ull);
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

/* ITU-R Recommendation 601-2 luma: L, LA, I and LA from RGBA, whole rows */
static int
convert_rgb2l_simd(UINT8 *out, const UINT8 *in, int xsize) {
    CONVERT_ROW(CONVERT_KERNEL(rgb2l), CONVERT_STEP, out, in, xsize, 4, 1);
    return xsize;
}

static int
convert_rgb2la_simd(UINT8 *out, const UINT8 *in, int xsize) {
    CONVERT_ROW(CONVERT_KERNEL(rgb2la), CONVERT_STEP, out, in, xsize, 4, 4);
    return xsize;
}

static int
convert_rgb2i_simd(UINT8 *out, const UINT8 *in, int xsize) {
    CONVERT_ROW(CONVERT_KERNEL(rgb2i), CONVERT_STEP, out, in, xsize, 4, 4);
    return xsize;
}

static int
convert_rgba2la_simd(UINT8 *out, const UINT8 *in, int xsize) {
    CONVERT_ROW(CONVERT_KERNEL(rgba2la), CONVERT_STEP, out, in, xsize, 4, 4);
    return xsize;
}

/* RGB to HSV, exact mode: whole rows */
static int
convert_rgb2hsv_simd(UINT8 *out, const UINT8 *in, int xsize) {
#if defined(IMAGING_SIMD_AVX2)
    CONVERT_ROW(rgb2hsv_8, 8, out, in, xsize, 4, 4);
#elif defined(IMAGING_SIMD_SSE4)
    CONVERT_ROW(rgb2hsv_4, 4, out, in, xsize, 4, 4);
#elif defined(IMAGING_SIMD_NEON)
    CONVERT_ROW(rgb2hsv_neon4, 4, out, in, xsize, 4, 4);
#endif
    return xsize;
}

/* RGB to HSV, fast mode (rgb2hsv_fast_pixel in Convert.c defines it) */
static int
convert_rgb2hsv_fast_simd(UINT8 *out, const UINT8 *in, int xsize) {
    int x = 0;
#if defined(IMAGING_SIMD_AVX2)
    {
        const __m256i m8 = _mm256_set1_epi32(0xff), zero = _mm256_setzero_si256();
        const __m256i alpha = _mm256_set1_epi32((int)0xff000000);
        for (; x + 8 <= xsize; x += 8) {
            __m256i px = _mm256_loadu_si256((const __m256i *)(in + x * 4));
            __m256i r = _mm256_and_si256(px, m8);
            __m256i g = _mm256_and_si256(_mm256_srli_epi32(px, 8), m8);
            __m256i b = _mm256_and_si256(_mm256_srli_epi32(px, 16), m8);
            __m256i v = _mm256_max_epi32(r, _mm256_max_epi32(g, b));
            __m256i d =
                _mm256_sub_epi32(v, _mm256_min_epi32(r, _mm256_min_epi32(g, b)));
            __m256i rmax = _mm256_cmpeq_epi32(r, v);
            __m256i gmax = _mm256_andnot_si256(rmax, _mm256_cmpeq_epi32(g, v));
            __m256i d2 = _mm256_add_epi32(d, d), d4 = _mm256_add_epi32(d2, d2);
            __m256i d6 = _mm256_add_epi32(d4, d2);
            __m256i n = _mm256_add_epi32(d4, _mm256_sub_epi32(r, g));
            n = _mm256_blendv_epi8(
                n, _mm256_add_epi32(d2, _mm256_sub_epi32(b, r)), gmax
            );
            n = _mm256_blendv_epi8(n, _mm256_sub_epi32(g, b), rmax);
            n = _mm256_add_epi32(n, _mm256_and_si256(_mm256_cmpgt_epi32(zero, n), d6));
            __m256i flat = _mm256_cmpeq_epi32(d, zero);
            __m256i h = _mm256_cvttps_epi32(_mm256_div_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_slli_epi32(n, 8), n)),
                _mm256_cvtepi32_ps(d6)
            ));
            __m256i sat = _mm256_cvttps_epi32(_mm256_div_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_slli_epi32(d, 8), d)),
                _mm256_cvtepi32_ps(v)
            ));
            h = _mm256_andnot_si256(flat, h);
            sat = _mm256_andnot_si256(flat, sat);
            _mm256_storeu_si256(
                (__m256i *)(out + x * 4),
                _mm256_or_si256(
                    _mm256_or_si256(h, _mm256_slli_epi32(sat, 8)),
                    _mm256_or_si256(
                        _mm256_slli_epi32(v, 16), _mm256_and_si256(px, alpha)
                    )
                )
            );
        }
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    {
        const __m128i m8 = _mm_set1_epi32(0xff), zero = _mm_setzero_si128();
        const __m128i alpha = _mm_set1_epi32((int)0xff000000);
        for (; x + 4 <= xsize; x += 4) {
            __m128i px = _mm_loadu_si128((const __m128i *)(in + x * 4));
            __m128i r = _mm_and_si128(px, m8);
            __m128i g = _mm_and_si128(_mm_srli_epi32(px, 8), m8);
            __m128i b = _mm_and_si128(_mm_srli_epi32(px, 16), m8);
            __m128i v = _mm_max_epi32(r, _mm_max_epi32(g, b));
            __m128i d = _mm_sub_epi32(v, _mm_min_epi32(r, _mm_min_epi32(g, b)));
            __m128i rmax = _mm_cmpeq_epi32(r, v);
            __m128i gmax = _mm_andnot_si128(rmax, _mm_cmpeq_epi32(g, v));
            __m128i d2 = _mm_add_epi32(d, d), d4 = _mm_add_epi32(d2, d2);
            __m128i d6 = _mm_add_epi32(d4, d2);
            __m128i n = _mm_add_epi32(d4, _mm_sub_epi32(r, g));
            n = _mm_blendv_epi8(n, _mm_add_epi32(d2, _mm_sub_epi32(b, r)), gmax);
            n = _mm_blendv_epi8(n, _mm_sub_epi32(g, b), rmax);
            n = _mm_add_epi32(n, _mm_and_si128(_mm_cmpgt_epi32(zero, n), d6));
            __m128i flat = _mm_cmpeq_epi32(d, zero);
            __m128i h = _mm_cvttps_epi32(_mm_div_ps(
                _mm_cvtepi32_ps(_mm_sub_epi32(_mm_slli_epi32(n, 8), n)),
                _mm_cvtepi32_ps(d6)
            ));
            __m128i sat = _mm_cvttps_epi32(_mm_div_ps(
                _mm_cvtepi32_ps(_mm_sub_epi32(_mm_slli_epi32(d, 8), d)),
                _mm_cvtepi32_ps(v)
            ));
            h = _mm_andnot_si128(flat, h);
            sat = _mm_andnot_si128(flat, sat);
            _mm_storeu_si128(
                (__m128i *)(out + x * 4),
                _mm_or_si128(
                    _mm_or_si128(h, _mm_slli_epi32(sat, 8)),
                    _mm_or_si128(_mm_slli_epi32(v, 16), _mm_and_si128(px, alpha))
                )
            );
        }
    }
#elif defined(IMAGING_SIMD_NEON)
    /* The hue's numerator and denominator in 16-bit lanes, then one float
       division for 4 lanes; the saturation floor(255 d / v) is the
       unpremultiply table's entry for v and d (d <= v), or a division while
       another thread builds the table. 1.75 times as fast on the
       Cortex-A78AE as all in 32-bit lanes with two divisions (a reciprocal
       estimate with Newton steps was slower than the division). */
#if defined(UNPREMULTIPLY_TABLE)
    const UINT8 *table = unpremultiply_lookup();
#else
    const UINT8 *table = NULL;
#endif
    for (; x + 16 <= xsize; x += 16) {
        uint8x16x4_t px = vld4q_u8(in + x * 4);
        uint8x16_t v8 = vmaxq_u8(px.val[0], vmaxq_u8(px.val[1], px.val[2]));
        uint8x16_t d8 =
            vsubq_u8(v8, vminq_u8(px.val[0], vminq_u8(px.val[1], px.val[2])));
        uint16x8_t hq[2], sq[2];
        int half, k;
        for (half = 0; half < 2; half++) {
#define HSV_H(plane) \
    vreinterpretq_s16_u16(half ? vmovl_high_u8(plane) : vmovl_u8(vget_low_u8(plane)))
            int16x8_t r = HSV_H(px.val[0]), g = HSV_H(px.val[1]), b = HSV_H(px.val[2]);
            int16x8_t v = HSV_H(v8), d = HSV_H(d8);
#undef HSV_H
            uint16x8_t rmax = vceqq_s16(r, v), gmax = vbicq_u16(vceqq_s16(g, v), rmax);
            int16x8_t d2 = vaddq_s16(d, d), d4 = vaddq_s16(d2, d2);
            int16x8_t d6 = vaddq_s16(d4, d2);
            int16x8_t n = vbslq_s16(
                rmax,
                vsubq_s16(g, b),
                vbslq_s16(
                    gmax, vaddq_s16(d2, vsubq_s16(b, r)), vaddq_s16(d4, vsubq_s16(r, g))
                )
            );
            uint32x4_t q[2], sat[2];
            n = vaddq_s16(n, vandq_s16(vreinterpretq_s16_u16(vcltzq_s16(n)), d6));
            for (k = 0; k < 2; k++) {
                /* d = 0 divides 0 by 0: NaN converts to 0, cleared below */
                float32x4_t nf =
                    vcvtq_f32_s32(vmovl_s16(k ? vget_high_s16(n) : vget_low_s16(n)));
                float32x4_t df =
                    vcvtq_f32_s32(vmovl_s16(k ? vget_high_s16(d6) : vget_low_s16(d6)));
                q[k] = vcvtq_u32_f32(vdivq_f32(vmulq_n_f32(nf, 255.0f), df));
                if (!table) {
                    float32x4_t sd = vcvtq_f32_s32(
                        vmovl_s16(k ? vget_high_s16(d) : vget_low_s16(d))
                    );
                    float32x4_t sv = vcvtq_f32_s32(
                        vmovl_s16(k ? vget_high_s16(v) : vget_low_s16(v))
                    );
                    sat[k] = vcvtq_u32_f32(vdivq_f32(vmulq_n_f32(sd, 255.0f), sv));
                }
            }
            hq[half] = vcombine_u16(vmovn_u32(q[0]), vmovn_u32(q[1]));
            if (!table) {
                sq[half] = vcombine_u16(vmovn_u32(sat[0]), vmovn_u32(sat[1]));
            }
        }
        uint8x16_t flat = vceqzq_u8(d8);
        px.val[0] = vbicq_u8(vcombine_u8(vmovn_u16(hq[0]), vmovn_u16(hq[1])), flat);
        if (table) {
            UINT8 vv[16], dd[16], ss[16];
            vst1q_u8(vv, v8);
            vst1q_u8(dd, d8);
            for (k = 0; k < 16; k++) {
                ss[k] = table[vv[k] * 256 + dd[k]];
            }
            px.val[1] = vld1q_u8(ss);
        } else {
            px.val[1] = vbicq_u8(vcombine_u8(vmovn_u16(sq[0]), vmovn_u16(sq[1])), flat);
        }
        px.val[2] = v8;
        vst4q_u8(out + x * 4, px);
    }
#endif
    return x;
}

#if defined(IMAGING_SIMD_NEON)
static int
convert_l2rgb_neon(UINT8 *restrict out, const UINT8 *restrict in, int xsize) {
    int x = 0;
    /* Stores of four planes: on arm64 GCC makes those of byte loops at -O3
       only, and builds the whole pixels with widening multiplies. */
    for (; x + 16 <= xsize; x += 16, in += 16, out += 64) {
        uint8x16_t v = vld1q_u8(in);
        uint8x16x4_t px = {{v, v, v, vdupq_n_u8(255)}};
        vst4q_u8(out, px);
    }
    return x;
}

static int
convert_rgb2rgba_neon(UINT8 *restrict out, const UINT8 *restrict in, int xsize) {
    return convert_mask16(out, in, xsize, 0, ~0u, MAKE_UINT32(0, 0, 0, 255));
}

static int
convert_rgb2cmyk_neon(UINT8 *restrict out, const UINT8 *restrict in, int xsize) {
    /* no undercolour removal */
    return convert_mask16(out, in, xsize, 1, MAKE_UINT32(255, 255, 255, 0), 0);
}
#endif

/* RGBA to RGBa: MULDIV255 of each channel by alpha */
static int
convert_premultiply_simd(UINT8 *out, const UINT8 *in, int xsize) {
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
    return x;
}

/* RGBa to RGBA: CLIP8(255 * c / alpha), c where alpha is 0 or 255. The x86
   kernels do the whole row; the NEON kernel and the table are exact, so the
   scalar code can finish the row. */
static int
convert_unpremultiply_simd(UINT8 *out, const UINT8 *in, int xsize) {
#if defined(IMAGING_SIMD_SSE4)
    int x = 0;
#if defined(IMAGING_SIMD_AVX2)
    for (; x < xsize - 7; x += 8) {
        rgba2rgbA_8(out + x * 4, in + x * 4);
    }
#endif
    CONVERT_ROW(rgba2rgbA_4, 4, out + x * 4, in + x * 4, xsize - x, 4, 4);
    return xsize;
#else
    int x = 0;
#if defined(UNPREMULTIPLY_TABLE)
    const UINT8 *table = unpremultiply_lookup();
    if (table) {
        for (; x < xsize - 3; x += 4) {
            uint64_t p[2];
            memcpy(p, in + x * 4, sizeof(p));
            p[0] = unpremultiply_2px(table, p[0]);
            p[1] = unpremultiply_2px(table, p[1]);
            memcpy(out + x * 4, p, sizeof(p));
        }
    }
#endif
    for (; x < xsize - 15; x += 16) {
        uint8x16x4_t px = vld4q_u8(in + x * 4);
        neon_unpremultiply_16(&px);
        vst4q_u8(out + x * 4, px);
    }
    return x;
#endif
}

#include "ConvertYCbCrSIMD.c"

/* L to I and back, and the colour matrix's rows (Matrix.c): the loops of
   PlainLoops.h as this tier's compiler makes them. */
static int
convert_l2i_simd(UINT8 *out, const UINT8 *in, int xsize) {
    plain_l2i_row(out, in, xsize);
    return xsize;
}

static int
convert_i2l_simd(UINT8 *out, const UINT8 *in, int xsize) {
    plain_i2l_row(out, in, xsize);
    return xsize;
}

static int
convert_matrix_l_row_simd(UINT8 *out, const UINT8 *in, int n, const float *m) {
    plain_matrix_l_row(out, in, n, m);
    return n;
}

static int
convert_matrix_rgb_row_simd(UINT8 *out, const UINT8 *in, int n, const float *m) {
    plain_matrix_rgb_row(out, in, n, m);
    return n;
}

static const ImagingSIMDConvertKernels convert_simd_kernels = {
#if defined(IMAGING_SIMD_NEON)
    .l2rgb = convert_l2rgb_neon,
    .rgb2rgba = convert_rgb2rgba_neon,
    .rgb2cmyk = convert_rgb2cmyk_neon,
#endif
    .rgb2l = convert_rgb2l_simd,
    .rgb2la = convert_rgb2la_simd,
    .rgb2i = convert_rgb2i_simd,
    .rgba2la = convert_rgba2la_simd,
    .rgb2hsv = convert_rgb2hsv_simd,
    .rgb2hsv_fast = convert_rgb2hsv_fast_simd,
    .premultiply = convert_premultiply_simd,
    .unpremultiply = convert_unpremultiply_simd,
    .rgb2ycbcr = ycc_rgb2ycbcr_simd,
    .ycbcr2rgb = ycc_ycbcr2rgb_simd,
    .rgb2ycbcr_fast = ycc_rgb2ycbcr_fast_simd,
    .ycbcr2rgb_fast = ycc_ycbcr2rgb_fast_simd,
    .l2i = convert_l2i_simd,
    .i2l = convert_i2l_simd,
    .matrix_l_row = convert_matrix_l_row_simd,
    .matrix_rgb_row = convert_matrix_rgb_row_simd,
};
