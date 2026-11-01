/* Fast mode's resize (ResampleFast.c): the vertical pass of one output row
   in 16-bit lanes, included by ResampleSIMD.c. Does the bytes the vector
   loop can and returns how many; rf_vrow() finishes the row. */

static int
resample_fast_vrow_simd(
    UINT8 *restrict out, INT16 **rows, const INT16 *w, int taps, int n
) {
    int x = 0, k;
#if defined(IMAGING_SIMD_AVX2)
    const __m256i half8 = _mm256_set1_epi16(1 << (RF_FRAC - 1));
    for (; x + 32 <= n; x += 32) {
        __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
        for (k = 0; k < taps; k++) {
            __m256i wk = _mm256_set1_epi16(w[k]);
            s0 = _mm256_add_epi16(
                s0,
                _mm256_mulhrs_epi16(
                    _mm256_loadu_si256((const __m256i *)(rows[k] + x)), wk
                )
            );
            s1 = _mm256_add_epi16(
                s1,
                _mm256_mulhrs_epi16(
                    _mm256_loadu_si256((const __m256i *)(rows[k] + x + 16)), wk
                )
            );
        }
        s0 = _mm256_srai_epi16(_mm256_add_epi16(s0, half8), RF_FRAC);
        s1 = _mm256_srai_epi16(_mm256_add_epi16(s1, half8), RF_FRAC);
        /* packus works within 128-bit lanes: 64-bit groups 0 2 1 3 */
        _mm256_storeu_si256(
            (__m256i *)(out + x),
            _mm256_permute4x64_epi64(_mm256_packus_epi16(s0, s1), 0xd8)
        );
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    const __m128i half = _mm_set1_epi16(1 << (RF_FRAC - 1));
    for (; x + 16 <= n; x += 16) {
        __m128i s0 = _mm_setzero_si128(), s1 = _mm_setzero_si128();
        for (k = 0; k < taps; k++) {
            __m128i wk = _mm_set1_epi16(w[k]);
            s0 = _mm_add_epi16(
                s0,
                _mm_mulhrs_epi16(_mm_loadu_si128((const __m128i *)(rows[k] + x)), wk)
            );
            s1 = _mm_add_epi16(
                s1,
                _mm_mulhrs_epi16(
                    _mm_loadu_si128((const __m128i *)(rows[k] + x + 8)), wk
                )
            );
        }
        s0 = _mm_srai_epi16(_mm_add_epi16(s0, half), RF_FRAC);
        s1 = _mm_srai_epi16(_mm_add_epi16(s1, half), RF_FRAC);
        _mm_storeu_si128((__m128i *)(out + x), _mm_packus_epi16(s0, s1));
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; x + 16 <= n; x += 16) {
        int16x8_t s0 = vdupq_n_s16(0), s1 = vdupq_n_s16(0);
        for (k = 0; k < taps; k++) {
            s0 = vaddq_s16(s0, vqrdmulhq_n_s16(vld1q_s16(rows[k] + x), w[k]));
            s1 = vaddq_s16(s1, vqrdmulhq_n_s16(vld1q_s16(rows[k] + x + 8), w[k]));
        }
        /* (s + 2^5) >> 6, saturated to 0 .. 255 */
        vst1q_u8(
            out + x,
            vcombine_u8(vqrshrun_n_s16(s0, RF_FRAC), vqrshrun_n_s16(s1, RF_FRAC))
        );
    }
#endif
    return x;
}
