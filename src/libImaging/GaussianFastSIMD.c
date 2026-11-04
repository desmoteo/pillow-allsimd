/* Fast mode's Gaussian blur (GaussianFast.c), compiled into SIMDKernels.c:
   the taps of radius r with weights wt (wt[0] at the centre) over the 2r + 1
   sources, for the bytes the vector loop can; returns how many. */

static int
gauss_taps_simd(
    UINT8 *restrict out, const UINT8 **src, int r, const UINT16 *wt, int n
) {
    int i = 0, j;

#if defined(IMAGING_SIMD_AVX2)
    for (; i + 32 <= n; i += 32) {
        __m256i c0 = mm256_cvtepu8_epi16(src[r] + i);
        __m256i c1 = mm256_cvtepu8_epi16(src[r] + i + 16);
        __m256i w = _mm256_set1_epi16((short)wt[0]);
        __m256i a0 =
            _mm256_add_epi16(_mm256_set1_epi16(128), _mm256_mullo_epi16(c0, w));
        __m256i a1 =
            _mm256_add_epi16(_mm256_set1_epi16(128), _mm256_mullo_epi16(c1, w));
        for (j = 1; j <= r; j++) {
            __m256i p0 = _mm256_add_epi16(
                mm256_cvtepu8_epi16(src[r - j] + i), mm256_cvtepu8_epi16(src[r + j] + i)
            );
            __m256i p1 = _mm256_add_epi16(
                mm256_cvtepu8_epi16(src[r - j] + i + 16),
                mm256_cvtepu8_epi16(src[r + j] + i + 16)
            );
            w = _mm256_set1_epi16((short)wt[j]);
            a0 = _mm256_add_epi16(a0, _mm256_mullo_epi16(p0, w));
            a1 = _mm256_add_epi16(a1, _mm256_mullo_epi16(p1, w));
        }
        __m256i packed =
            _mm256_packus_epi16(_mm256_srli_epi16(a0, 8), _mm256_srli_epi16(a1, 8));
        _mm256_storeu_si256(
            (__m256i *)(out + i), _mm256_permute4x64_epi64(packed, 0xd8)
        );
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    for (; i + 16 <= n; i += 16) {
        __m128i zero = _mm_setzero_si128();
        __m128i c = _mm_loadu_si128((const __m128i *)(src[r] + i));
        __m128i w = _mm_set1_epi16((short)wt[0]);
        __m128i a0 = _mm_add_epi16(
            _mm_set1_epi16(128), _mm_mullo_epi16(_mm_unpacklo_epi8(c, zero), w)
        );
        __m128i a1 = _mm_add_epi16(
            _mm_set1_epi16(128), _mm_mullo_epi16(_mm_unpackhi_epi8(c, zero), w)
        );
        for (j = 1; j <= r; j++) {
            __m128i lo = _mm_loadu_si128((const __m128i *)(src[r - j] + i));
            __m128i hi = _mm_loadu_si128((const __m128i *)(src[r + j] + i));
            w = _mm_set1_epi16((short)wt[j]);
            a0 = _mm_add_epi16(
                a0,
                _mm_mullo_epi16(
                    _mm_add_epi16(
                        _mm_unpacklo_epi8(lo, zero), _mm_unpacklo_epi8(hi, zero)
                    ),
                    w
                )
            );
            a1 = _mm_add_epi16(
                a1,
                _mm_mullo_epi16(
                    _mm_add_epi16(
                        _mm_unpackhi_epi8(lo, zero), _mm_unpackhi_epi8(hi, zero)
                    ),
                    w
                )
            );
        }
        _mm_storeu_si128(
            (__m128i *)(out + i),
            _mm_packus_epi16(_mm_srli_epi16(a0, 8), _mm_srli_epi16(a1, 8))
        );
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; i + 16 <= n; i += 16) {
        uint8x16_t c = vld1q_u8(src[r] + i);
        uint16x8_t a0 = vmlaq_n_u16(vdupq_n_u16(128), vmovl_u8(vget_low_u8(c)), wt[0]);
        uint16x8_t a1 = vmlaq_n_u16(vdupq_n_u16(128), vmovl_high_u8(c), wt[0]);
        for (j = 1; j <= r; j++) {
            uint8x16_t lo = vld1q_u8(src[r - j] + i);
            uint8x16_t hi = vld1q_u8(src[r + j] + i);
            a0 = vmlaq_n_u16(a0, vaddl_u8(vget_low_u8(lo), vget_low_u8(hi)), wt[j]);
            a1 = vmlaq_n_u16(a1, vaddl_high_u8(lo, hi), wt[j]);
        }
        vst1q_u8(out + i, vcombine_u8(vshrn_n_u16(a0, 8), vshrn_n_u16(a1, 8)));
    }
#endif
    return i;
}

static const ImagingSIMDGaussKernels gauss_simd_kernels = {
    .taps = gauss_taps_simd,
};
