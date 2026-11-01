/* Fast mode's blend (Blend.c), compiled into SIMDKernels.c: in1 + (in2 -
   in1) a for a = a15 / 2^15, rounded half up, for the bytes the vector loop
   can; returns how many. */

static int
blend_fast_row_simd(
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
    return x;
}

static const ImagingSIMDBlendKernels blend_simd_kernels = {
    .fast_row = blend_fast_row_simd,
};
