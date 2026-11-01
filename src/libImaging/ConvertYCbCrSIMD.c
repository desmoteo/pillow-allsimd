/* The SIMD kernels of the YCbCr conversions (ConvertYCbCr.c), compiled into
   SIMDKernels.c. Each does the pixels its vector loop can and returns how
   many; ConvertYCbCr.c finishes the row. */

#include "ConvertYCbCr.h"

#if defined(IMAGING_SIMD_SSE4)
static inline __m128i
ycc_term(__m128 v, float k) {
    return _mm_cvttps_epi32(
        _mm_add_ps(_mm_mul_ps(v, _mm_set1_ps(k)), _mm_set1_ps(0.5f))
    );
}

/* Bytes c (0 to 3) of four 4-byte pixels as 32-bit lanes. */
static inline __m128i
ycc_channel(__m128i px, int c) {
    return _mm_and_si128(_mm_srli_epi32(px, 8 * c), _mm_set1_epi32(0xff));
}
#if defined(IMAGING_SIMD_AVX2)
/* The same for eight pixels. */
static inline __m256i
ycc_term8(__m256 v, float k) {
    return _mm256_cvttps_epi32(
        _mm256_add_ps(_mm256_mul_ps(v, _mm256_set1_ps(k)), _mm256_set1_ps(0.5f))
    );
}

static inline __m256i
ycc_channel8(__m256i px, int c) {
    return _mm256_and_si256(_mm256_srli_epi32(px, 8 * c), _mm256_set1_epi32(0xff));
}
#endif
#elif defined(IMAGING_SIMD_NEON)
static inline int32x4_t
ycc_term(float32x4_t v, float k) {
    return vcvtq_s32_f32(vaddq_f32(vmulq_n_f32(v, k), vdupq_n_f32(0.5f)));
}

/* Lanes 4 * q to 4 * q + 3 of a plane of 16 bytes, minus sub, as floats. */
static inline float32x4_t
ycc_quarter(uint8x16_t plane, int q, int sub) {
    uint16x8_t h = q < 2 ? vmovl_u8(vget_low_u8(plane)) : vmovl_high_u8(plane);
    uint32x4_t w = q % 2 == 0 ? vmovl_u16(vget_low_u16(h)) : vmovl_high_u16(h);
    return vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(w), vdupq_n_s32(sub)));
}

/* Four groups of 32-bit lanes to 16 bytes: wrapped, as (UINT8) does, or
   clipped to 0..255. */
static inline uint8x16_t
ycc_wrap16(int32x4_t v0, int32x4_t v1, int32x4_t v2, int32x4_t v3) {
    return vcombine_u8(
        vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(vmovn_s32(v0), vmovn_s32(v1)))),
        vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(vmovn_s32(v2), vmovn_s32(v3))))
    );
}

static inline uint8x16_t
ycc_clip16(int32x4_t v0, int32x4_t v1, int32x4_t v2, int32x4_t v3) {
    return vcombine_u8(
        vqmovun_s16(vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1))),
        vqmovun_s16(vcombine_s16(vqmovn_s32(v2), vqmovn_s32(v3)))
    );
}
#endif

static int
ycc_rgb2ycbcr_simd(UINT8 *out, const UINT8 *in, int pixels) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
#if defined(IMAGING_SIMD_AVX2)
    for (; x < pixels - 7; x += 8, in += 32, out += 32) {
        __m256i px = _mm256_loadu_si256((const __m256i *)in);
        __m256 fr = _mm256_cvtepi32_ps(ycc_channel8(px, 0));
        __m256 fg = _mm256_cvtepi32_ps(ycc_channel8(px, 1));
        __m256 fb = _mm256_cvtepi32_ps(ycc_channel8(px, 2));
        __m256i vy = _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(ycc_term8(fr, K_Y_R), ycc_term8(fg, K_Y_G)),
                ycc_term8(fb, K_Y_B)
            ),
            SCALE
        );
        __m256i vcb = _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(ycc_term8(fr, K_Cb_R), ycc_term8(fg, K_Cb_G)),
                ycc_term8(fb, K_Cb_B)
            ),
            SCALE
        );
        __m256i vcr = _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(ycc_term8(fr, K_Cb_B), ycc_term8(fg, K_Cr_G)),
                ycc_term8(fb, K_Cr_B)
            ),
            SCALE
        );
        __m256i m = _mm256_set1_epi32(0xff), c128 = _mm256_set1_epi32(128);
        __m256i bytes = _mm256_or_si256(
            _mm256_and_si256(vy, m),
            _mm256_or_si256(
                _mm256_slli_epi32(_mm256_and_si256(_mm256_add_epi32(vcb, c128), m), 8),
                _mm256_slli_epi32(_mm256_and_si256(_mm256_add_epi32(vcr, c128), m), 16)
            )
        );
        bytes = _mm256_or_si256(
            bytes, _mm256_and_si256(px, _mm256_set1_epi32((int)0xff000000))
        );
        _mm256_storeu_si256((__m256i *)out, bytes);
    }
#endif
    for (; x < pixels - 3; x += 4, in += 16, out += 16) {
        __m128i px = _mm_loadu_si128((const __m128i *)in);
        __m128 fr = _mm_cvtepi32_ps(ycc_channel(px, 0));
        __m128 fg = _mm_cvtepi32_ps(ycc_channel(px, 1));
        __m128 fb = _mm_cvtepi32_ps(ycc_channel(px, 2));
        __m128i vy = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_add_epi32(ycc_term(fr, K_Y_R), ycc_term(fg, K_Y_G)),
                ycc_term(fb, K_Y_B)
            ),
            SCALE
        );
        __m128i vcb = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_add_epi32(ycc_term(fr, K_Cb_R), ycc_term(fg, K_Cb_G)),
                ycc_term(fb, K_Cb_B)
            ),
            SCALE
        );
        __m128i vcr = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_add_epi32(ycc_term(fr, K_Cb_B), ycc_term(fg, K_Cr_G)),
                ycc_term(fb, K_Cr_B)
            ),
            SCALE
        );
        // (UINT8) of y, cb + 128 and cr + 128, with the alpha byte kept
        __m128i bytes = _mm_or_si128(
            _mm_and_si128(vy, _mm_set1_epi32(0xff)),
            _mm_or_si128(
                _mm_slli_epi32(
                    _mm_and_si128(
                        _mm_add_epi32(vcb, _mm_set1_epi32(128)), _mm_set1_epi32(0xff)
                    ),
                    8
                ),
                _mm_slli_epi32(
                    _mm_and_si128(
                        _mm_add_epi32(vcr, _mm_set1_epi32(128)), _mm_set1_epi32(0xff)
                    ),
                    16
                )
            )
        );
        bytes = _mm_or_si128(bytes, _mm_and_si128(px, _mm_set1_epi32((int)0xff000000)));
        _mm_storeu_si128((__m128i *)out, bytes);
    }
#elif defined(IMAGING_SIMD_NEON)
    /* 16 pixels as planes of r, g, b and alpha */
    for (; x < pixels - 15; x += 16, in += 64, out += 64) {
        uint8x16x4_t px = vld4q_u8(in);
        int32x4_t vy[4], vcb[4], vcr[4];
        int q;
        for (q = 0; q < 4; q++) {
            float32x4_t fr = ycc_quarter(px.val[0], q, 0);
            float32x4_t fg = ycc_quarter(px.val[1], q, 0);
            float32x4_t fb = ycc_quarter(px.val[2], q, 0);
            vy[q] = vshrq_n_s32(
                vaddq_s32(
                    vaddq_s32(ycc_term(fr, K_Y_R), ycc_term(fg, K_Y_G)),
                    ycc_term(fb, K_Y_B)
                ),
                SCALE
            );
            vcb[q] = vaddq_s32(
                vshrq_n_s32(
                    vaddq_s32(
                        vaddq_s32(ycc_term(fr, K_Cb_R), ycc_term(fg, K_Cb_G)),
                        ycc_term(fb, K_Cb_B)
                    ),
                    SCALE
                ),
                vdupq_n_s32(128)
            );
            vcr[q] = vaddq_s32(
                vshrq_n_s32(
                    vaddq_s32(
                        vaddq_s32(ycc_term(fr, K_Cb_B), ycc_term(fg, K_Cr_G)),
                        ycc_term(fb, K_Cr_B)
                    ),
                    SCALE
                ),
                vdupq_n_s32(128)
            );
        }
        // (UINT8) of y, cb + 128 and cr + 128, with the alpha byte kept
        px.val[0] = ycc_wrap16(vy[0], vy[1], vy[2], vy[3]);
        px.val[1] = ycc_wrap16(vcb[0], vcb[1], vcb[2], vcb[3]);
        px.val[2] = ycc_wrap16(vcr[0], vcr[1], vcr[2], vcr[3]);
        vst4q_u8(out, px);
    }
#endif
    return x;
}

static int
ycc_ycbcr2rgb_simd(UINT8 *out, const UINT8 *in, int pixels) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
#if defined(IMAGING_SIMD_AVX2)
    for (; x < pixels - 7; x += 8, in += 32, out += 32) {
        __m256i px = _mm256_loadu_si256((const __m256i *)in);
        __m256i vy = ycc_channel8(px, 0), c128 = _mm256_set1_epi32(128);
        __m256 fcb = _mm256_cvtepi32_ps(_mm256_sub_epi32(ycc_channel8(px, 1), c128));
        __m256 fcr = _mm256_cvtepi32_ps(_mm256_sub_epi32(ycc_channel8(px, 2), c128));
        __m256i vr =
            _mm256_add_epi32(vy, _mm256_srai_epi32(ycc_term8(fcr, K_R_Cr), SCALE));
        __m256i vg = _mm256_add_epi32(
            vy,
            _mm256_srai_epi32(
                _mm256_add_epi32(ycc_term8(fcb, K_G_Cb), ycc_term8(fcr, K_G_Cr)), SCALE
            )
        );
        __m256i vb =
            _mm256_add_epi32(vy, _mm256_srai_epi32(ycc_term8(fcb, K_B_Cb), SCALE));
        // as below, in each 128-bit lane: four pixels each
        __m256i rg = _mm256_packs_epi32(vr, vg),
                ba = _mm256_packs_epi32(vb, ycc_channel8(px, 3));
        __m256i p = _mm256_packus_epi16(rg, ba);
        p = _mm256_shuffle_epi8(
            p,
            _mm256_set_epi8(
                15,
                11,
                7,
                3,
                14,
                10,
                6,
                2,
                13,
                9,
                5,
                1,
                12,
                8,
                4,
                0,
                15,
                11,
                7,
                3,
                14,
                10,
                6,
                2,
                13,
                9,
                5,
                1,
                12,
                8,
                4,
                0
            )
        );
        _mm256_storeu_si256((__m256i *)out, p);
    }
#endif
    for (; x < pixels - 3; x += 4, in += 16, out += 16) {
        __m128i px = _mm_loadu_si128((const __m128i *)in);
        __m128i vy = ycc_channel(px, 0);
        __m128 fcb =
            _mm_cvtepi32_ps(_mm_sub_epi32(ycc_channel(px, 1), _mm_set1_epi32(128)));
        __m128 fcr =
            _mm_cvtepi32_ps(_mm_sub_epi32(ycc_channel(px, 2), _mm_set1_epi32(128)));
        __m128i vr = _mm_add_epi32(vy, _mm_srai_epi32(ycc_term(fcr, K_R_Cr), SCALE));
        __m128i vg = _mm_add_epi32(
            vy,
            _mm_srai_epi32(
                _mm_add_epi32(ycc_term(fcb, K_G_Cb), ycc_term(fcr, K_G_Cr)), SCALE
            )
        );
        __m128i vb = _mm_add_epi32(vy, _mm_srai_epi32(ycc_term(fcb, K_B_Cb), SCALE));
        // clipped to 0..255 by the saturating packs; the alpha lane is kept
        __m128i rg = _mm_packs_epi32(vr, vg),
                ba = _mm_packs_epi32(vb, ycc_channel(px, 3));
        __m128i p = _mm_packus_epi16(rg, ba);
        // r0..r3 g0..g3 b0..b3 a0..a3 to pixels
        p = _mm_shuffle_epi8(
            p, _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0)
        );
        _mm_storeu_si128((__m128i *)out, p);
    }
#elif defined(IMAGING_SIMD_NEON)
    /* 16 pixels as planes of y, cb, cr and alpha */
    for (; x < pixels - 15; x += 16, in += 64, out += 64) {
        uint8x16x4_t px = vld4q_u8(in);
        int32x4_t vr[4], vg[4], vb[4];
        int q;
        for (q = 0; q < 4; q++) {
            int32x4_t vy = vcvtq_s32_f32(ycc_quarter(px.val[0], q, 0));
            float32x4_t fcb = ycc_quarter(px.val[1], q, 128);
            float32x4_t fcr = ycc_quarter(px.val[2], q, 128);
            vr[q] = vaddq_s32(vy, vshrq_n_s32(ycc_term(fcr, K_R_Cr), SCALE));
            vg[q] = vaddq_s32(
                vy,
                vshrq_n_s32(
                    vaddq_s32(ycc_term(fcb, K_G_Cb), ycc_term(fcr, K_G_Cr)), SCALE
                )
            );
            vb[q] = vaddq_s32(vy, vshrq_n_s32(ycc_term(fcb, K_B_Cb), SCALE));
        }
        // clipped to 0..255; the alpha plane is kept
        px.val[0] = ycc_clip16(vr[0], vr[1], vr[2], vr[3]);
        px.val[1] = ycc_clip16(vg[0], vg[1], vg[2], vg[3]);
        px.val[2] = ycc_clip16(vb[0], vb[1], vb[2], vb[3]);
        vst4q_u8(out, px);
    }
#endif
    return x;
}

static int
ycc_rgb2ycbcr_fast_simd(UINT8 *out, const UINT8 *in, int pixels) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
    const __m128i ky = _mm_setr_epi16(
        YCC_K_YR, YCC_K_YG, YCC_K_YB, 0, YCC_K_YR, YCC_K_YG, YCC_K_YB, 0
    );
    const __m128i kcb = _mm_setr_epi16(
        YCC_K_CBR, YCC_K_CBG, YCC_K_HALF, 0, YCC_K_CBR, YCC_K_CBG, YCC_K_HALF, 0
    );
    const __m128i kcr = _mm_setr_epi16(
        YCC_K_HALF, YCC_K_CRG, YCC_K_CRB, 0, YCC_K_HALF, YCC_K_CRG, YCC_K_CRB, 0
    );
    const __m128i half = _mm_set1_epi32(1 << 13), biasc = _mm_set1_epi32(YCC_BIAS_C);
    const __m128i top = _mm_set1_epi32(255), alpha = _mm_set1_epi32((int)0xff000000);
#if defined(IMAGING_SIMD_AVX2)
    {
        /* 8 pixels; hadd works within 128-bit lanes, so the sums come out
           as pixels 0 1 4 5 | 2 3 6 7 and a permute puts them in order. */
        const __m256i ky8 = _mm256_broadcastsi128_si256(ky);
        const __m256i kcb8 = _mm256_broadcastsi128_si256(kcb);
        const __m256i kcr8 = _mm256_broadcastsi128_si256(kcr);
        const __m256i half8 = _mm256_set1_epi32(1 << 13);
        const __m256i biasc8 = _mm256_set1_epi32(YCC_BIAS_C);
        const __m256i top8 = _mm256_set1_epi32(255);
        const __m256i alpha8 = _mm256_set1_epi32((int)0xff000000);
        for (; x + 8 <= pixels; x += 8) {
            __m256i px = _mm256_loadu_si256((const __m256i *)(in + x * 4));
            __m256i lo = mm256_cvtepu8_epi16(in + x * 4);
            __m256i hi = mm256_cvtepu8_epi16(in + x * 4 + 16);
            __m256i y = _mm256_permute4x64_epi64(
                _mm256_hadd_epi32(
                    _mm256_madd_epi16(lo, ky8), _mm256_madd_epi16(hi, ky8)
                ),
                0xd8
            );
            __m256i cb = _mm256_permute4x64_epi64(
                _mm256_hadd_epi32(
                    _mm256_madd_epi16(lo, kcb8), _mm256_madd_epi16(hi, kcb8)
                ),
                0xd8
            );
            __m256i cr = _mm256_permute4x64_epi64(
                _mm256_hadd_epi32(
                    _mm256_madd_epi16(lo, kcr8), _mm256_madd_epi16(hi, kcr8)
                ),
                0xd8
            );
            y = _mm256_srli_epi32(_mm256_add_epi32(y, half8), 14);
            cb = _mm256_min_epi32(
                _mm256_srli_epi32(_mm256_add_epi32(cb, biasc8), 14), top8
            );
            cr = _mm256_min_epi32(
                _mm256_srli_epi32(_mm256_add_epi32(cr, biasc8), 14), top8
            );
            __m256i res = _mm256_or_si256(
                _mm256_or_si256(y, _mm256_slli_epi32(cb, 8)),
                _mm256_or_si256(_mm256_slli_epi32(cr, 16), _mm256_and_si256(px, alpha8))
            );
            _mm256_storeu_si256((__m256i *)(out + x * 4), res);
        }
    }
#endif
    /* 4 pixels: two halves as 16-bit words, R G B X per pixel; madd gives
       (R kR + G kG, B kB) per pixel, hadd the sum. */
    for (; x + 4 <= pixels; x += 4) {
        __m128i px = _mm_loadu_si128((const __m128i *)(in + x * 4));
        __m128i lo = _mm_cvtepu8_epi16(px),
                hi = _mm_cvtepu8_epi16(_mm_srli_si128(px, 8));
        __m128i y = _mm_hadd_epi32(_mm_madd_epi16(lo, ky), _mm_madd_epi16(hi, ky));
        __m128i cb = _mm_hadd_epi32(_mm_madd_epi16(lo, kcb), _mm_madd_epi16(hi, kcb));
        __m128i cr = _mm_hadd_epi32(_mm_madd_epi16(lo, kcr), _mm_madd_epi16(hi, kcr));
        y = _mm_srli_epi32(_mm_add_epi32(y, half), 14);
        cb = _mm_min_epi32(_mm_srli_epi32(_mm_add_epi32(cb, biasc), 14), top);
        cr = _mm_min_epi32(_mm_srli_epi32(_mm_add_epi32(cr, biasc), 14), top);
        __m128i res = _mm_or_si128(
            _mm_or_si128(y, _mm_slli_epi32(cb, 8)),
            _mm_or_si128(_mm_slli_epi32(cr, 16), _mm_and_si128(px, alpha))
        );
        _mm_storeu_si128((__m128i *)(out + x * 4), res);
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; x + 16 <= pixels; x += 16) {
        uint8x16x4_t px = vld4q_u8(in + x * 4);
        int16x8_t r[2], g[2], b[2];
        int16x4_t vy[4], vcb[4], vcr[4];
        int h, q;
        r[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(px.val[0])));
        r[1] = vreinterpretq_s16_u16(vmovl_high_u8(px.val[0]));
        g[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(px.val[1])));
        g[1] = vreinterpretq_s16_u16(vmovl_high_u8(px.val[1]));
        b[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(px.val[2])));
        b[1] = vreinterpretq_s16_u16(vmovl_high_u8(px.val[2]));
        for (q = 0; q < 4; q++) {
            int16x4_t rq, gq, bq;
            int32x4_t s;
            h = q / 2;
            rq = q % 2 ? vget_high_s16(r[h]) : vget_low_s16(r[h]);
            gq = q % 2 ? vget_high_s16(g[h]) : vget_low_s16(g[h]);
            bq = q % 2 ? vget_high_s16(b[h]) : vget_low_s16(b[h]);
            s = vmlal_n_s16(
                vmlal_n_s16(vmull_n_s16(rq, YCC_K_YR), gq, YCC_K_YG), bq, YCC_K_YB
            );
            vy[q] = vshrn_n_s32(vaddq_s32(s, vdupq_n_s32(1 << 13)), 14);
            s = vmlal_n_s16(
                vmlal_n_s16(vmull_n_s16(rq, YCC_K_CBR), gq, YCC_K_CBG), bq, YCC_K_HALF
            );
            vcb[q] = vshrn_n_s32(vaddq_s32(s, vdupq_n_s32(YCC_BIAS_C)), 14);
            s = vmlal_n_s16(
                vmlal_n_s16(vmull_n_s16(rq, YCC_K_HALF), gq, YCC_K_CRG), bq, YCC_K_CRB
            );
            vcr[q] = vshrn_n_s32(vaddq_s32(s, vdupq_n_s32(YCC_BIAS_C)), 14);
        }
        /* saturation clips 256 to 255 */
        px.val[0] = vcombine_u8(
            vqmovun_s16(vcombine_s16(vy[0], vy[1])),
            vqmovun_s16(vcombine_s16(vy[2], vy[3]))
        );
        px.val[1] = vcombine_u8(
            vqmovun_s16(vcombine_s16(vcb[0], vcb[1])),
            vqmovun_s16(vcombine_s16(vcb[2], vcb[3]))
        );
        px.val[2] = vcombine_u8(
            vqmovun_s16(vcombine_s16(vcr[0], vcr[1])),
            vqmovun_s16(vcombine_s16(vcr[2], vcr[3]))
        );
        vst4q_u8(out + x * 4, px);
    }
#endif
    return x;
}

static int
ycc_ycbcr2rgb_fast_simd(UINT8 *out, const UINT8 *in, int pixels) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
    /* 4 pixels: (Cb - 128, Cr - 128) as a pair of words per pixel, so one
       madd per channel gives each pixel's sum. */
    const __m128i pick =
        _mm_setr_epi8(1, -1, 2, -1, 5, -1, 6, -1, 9, -1, 10, -1, 13, -1, 14, -1);
    const __m128i c128 = _mm_set1_epi16(128);
    const __m128i kr = _mm_set1_epi32((int)((UINT32)YCC_K_RCR << 16));
    const __m128i kg =
        _mm_set1_epi32((int)(((UINT32)(UINT16)YCC_K_GCR << 16) | (UINT16)YCC_K_GCB));
    const __m128i kb = _mm_set1_epi32(YCC_K_BCB);
    const __m128i half = _mm_set1_epi32(1 << 13), zero = _mm_setzero_si128();
    const __m128i top = _mm_set1_epi32(255), lowbyte = _mm_set1_epi32(0xff);
    const __m128i alpha = _mm_set1_epi32((int)0xff000000);
#if defined(IMAGING_SIMD_AVX2)
    {
        /* 8 pixels, every step within a 32-bit lane */
        const __m256i pick8 = _mm256_broadcastsi128_si256(pick);
        const __m256i c1288 = _mm256_set1_epi16(128);
        const __m256i kr8 = _mm256_broadcastsi128_si256(kr);
        const __m256i kg8 = _mm256_broadcastsi128_si256(kg);
        const __m256i kb8 = _mm256_broadcastsi128_si256(kb);
        const __m256i half8 = _mm256_set1_epi32(1 << 13);
        const __m256i zero8 = _mm256_setzero_si256();
        const __m256i top8 = _mm256_set1_epi32(255);
        const __m256i lowbyte8 = _mm256_set1_epi32(0xff);
        const __m256i alpha8 = _mm256_set1_epi32((int)0xff000000);
        for (; x + 8 <= pixels; x += 8) {
            __m256i px = _mm256_loadu_si256((const __m256i *)(in + x * 4));
            __m256i c = _mm256_sub_epi16(_mm256_shuffle_epi8(px, pick8), c1288);
            __m256i y = _mm256_and_si256(px, lowbyte8);
            __m256i r = _mm256_srai_epi32(
                _mm256_add_epi32(_mm256_madd_epi16(c, kr8), half8), 14
            );
            __m256i g = _mm256_srai_epi32(
                _mm256_add_epi32(_mm256_madd_epi16(c, kg8), half8), 14
            );
            __m256i b = _mm256_srai_epi32(
                _mm256_add_epi32(_mm256_madd_epi16(c, kb8), half8), 14
            );
            r = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(y, r), top8), zero8);
            g = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(y, g), top8), zero8);
            b = _mm256_max_epi32(_mm256_min_epi32(_mm256_add_epi32(y, b), top8), zero8);
            __m256i res = _mm256_or_si256(
                _mm256_or_si256(r, _mm256_slli_epi32(g, 8)),
                _mm256_or_si256(_mm256_slli_epi32(b, 16), _mm256_and_si256(px, alpha8))
            );
            _mm256_storeu_si256((__m256i *)(out + x * 4), res);
        }
    }
#endif
    for (; x + 4 <= pixels; x += 4) {
        __m128i px = _mm_loadu_si128((const __m128i *)(in + x * 4));
        __m128i c = _mm_sub_epi16(_mm_shuffle_epi8(px, pick), c128);
        __m128i y = _mm_and_si128(px, lowbyte);
        __m128i r = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(c, kr), half), 14);
        __m128i g = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(c, kg), half), 14);
        __m128i b = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(c, kb), half), 14);
        r = _mm_max_epi32(_mm_min_epi32(_mm_add_epi32(y, r), top), zero);
        g = _mm_max_epi32(_mm_min_epi32(_mm_add_epi32(y, g), top), zero);
        b = _mm_max_epi32(_mm_min_epi32(_mm_add_epi32(y, b), top), zero);
        __m128i res = _mm_or_si128(
            _mm_or_si128(r, _mm_slli_epi32(g, 8)),
            _mm_or_si128(_mm_slli_epi32(b, 16), _mm_and_si128(px, alpha))
        );
        _mm_storeu_si128((__m128i *)(out + x * 4), res);
    }
#elif defined(IMAGING_SIMD_NEON)
    /* R and B have one product each: vqrdmulh of 2 (C - 128) and K is
       (4 (C - 128) K + 2^15) >> 16 = ((C - 128) K + 2^13) >> 14, the same
       rounding in 16-bit lanes. G sums two products before rounding, so it
       stays in 32 bits, with vrshrn adding the 2^13. 1.5 times as fast on the
       Cortex-A78AE as all three in 32 bits. */
    for (; x + 16 <= pixels; x += 16) {
        uint8x16x4_t px = vld4q_u8(in + x * 4);
        int16x8_t res[3][2];
        int h, c;
        for (h = 0; h < 2; h++) {
            int16x8_t y, cb, cr, r, g, b;
#define YCC_HALF(plane) \
    vreinterpretq_s16_u16(h ? vmovl_high_u8(plane) : vmovl_u8(vget_low_u8(plane)))
            y = YCC_HALF(px.val[0]);
            cb = vsubq_s16(YCC_HALF(px.val[1]), vdupq_n_s16(128));
            cr = vsubq_s16(YCC_HALF(px.val[2]), vdupq_n_s16(128));
#undef YCC_HALF
            r = vaddq_s16(y, vqrdmulhq_n_s16(vaddq_s16(cr, cr), YCC_K_RCR));
            b = vaddq_s16(y, vqrdmulhq_n_s16(vaddq_s16(cb, cb), YCC_K_BCB));
            g = vaddq_s16(
                y,
                vcombine_s16(
                    vrshrn_n_s32(
                        vmlal_n_s16(
                            vmull_n_s16(vget_low_s16(cb), YCC_K_GCB),
                            vget_low_s16(cr),
                            YCC_K_GCR
                        ),
                        14
                    ),
                    vrshrn_n_s32(
                        vmlal_high_n_s16(
                            vmull_high_n_s16(cb, YCC_K_GCB), cr, YCC_K_GCR
                        ),
                        14
                    )
                )
            );
            res[0][h] = r;
            res[1][h] = g;
            res[2][h] = b;
        }
        for (c = 0; c < 3; c++) {
            px.val[c] = vcombine_u8(vqmovun_s16(res[c][0]), vqmovun_s16(res[c][1]));
        }
        vst4q_u8(out + x * 4, px);
    }
#endif
    return x;
}
