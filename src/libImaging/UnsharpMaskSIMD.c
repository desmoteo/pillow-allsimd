/* The unsharp mask's row kernel (UnsharpMask.c), compiled into SIMDKernels.c:
   sharpens the bytes it can of a row in place, from its blurred copy in out,
   and returns how many. */

static int
unsharp_row_simd(UINT8 *out, const UINT8 *in, int size, int percent, int threshold) {
    int x = 0;

#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
    /* diff * percent / 100 via float division: while |diff * percent| is
       below 2^24 the quotient truncates to the integer division result.
       On arm64 NEON beats the compiler's multiply-high loop with GCC 12 and
       14 (unsharp RGB 1.31x on Cortex-A78AE), not with GCC 11. */
    if (percent >= -65535 && percent <= 65535) {
#if defined(IMAGING_SIMD_SSE4)
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
#else
        int32x4_t pct = vdupq_n_s32(percent);
        int32x4_t thr = vdupq_n_s32(threshold);
        float32x4_t hundred = vdupq_n_f32(100.0f);
        for (; x < size - 15; x += 16) {
            uint8x16_t src = vld1q_u8(&in[x]);
            uint8x16_t blur = vld1q_u8(&out[x]);
            int16x8_t v16[2] = {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src)))
            };
            int16x8_t b16[2] = {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(blur))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(blur)))
            };
            int16x4_t res[4];
            for (int i = 0; i < 4; i++) {
                int16x4_t vs =
                    i & 1 ? vget_high_s16(v16[i / 2]) : vget_low_s16(v16[i / 2]);
                int16x4_t bs =
                    i & 1 ? vget_high_s16(b16[i / 2]) : vget_low_s16(b16[i / 2]);
                int32x4_t v = vmovl_s16(vs);
                int32x4_t d = vsubl_s16(vs, bs);
                int32x4_t q =
                    vcvtq_s32_f32(vdivq_f32(vcvtq_f32_s32(vmulq_s32(d, pct)), hundred));
                uint32x4_t mask = vcgtq_s32(vabsq_s32(d), thr);
                res[i] = vqmovn_s32(vbslq_s32(mask, vaddq_s32(v, q), v));
            }
            vst1q_u8(
                &out[x],
                vcombine_u8(
                    vqmovun_s16(vcombine_s16(res[0], res[1])),
                    vqmovun_s16(vcombine_s16(res[2], res[3]))
                )
            );
        }
#endif
    }
#endif
    return x;
}

static const ImagingSIMDUnsharpKernels unsharp_simd_kernels = {
    .row = unsharp_row_simd,
};
