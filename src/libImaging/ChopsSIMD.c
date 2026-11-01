/* ImageChops' row kernels (Chops.c), compiled into SIMDKernels.c: multiply
   (and screen, with invert = 255) without a division, and the saturating add
   and subtract; each does the bytes its vector loop can and returns how
   many. */

static int
chop_multiply_row_simd(
    UINT8 *restrict out,
    const UINT8 *restrict in1,
    const UINT8 *restrict in2,
    int n,
    UINT8 invert
) {
    int x = 0;

#if defined(IMAGING_SIMD_AVX2)
    {
        __m256i mm_inv = _mm256_set1_epi8((char)invert);
        __m256i mm_one = _mm256_set1_epi16(1);
        __m256i mm_zero = _mm256_setzero_si256();
        for (; x < n - 31; x += 32) {
            __m256i a = _mm256_xor_si256(
                _mm256_loadu_si256((const __m256i *)(in1 + x)), mm_inv
            );
            __m256i b = _mm256_xor_si256(
                _mm256_loadu_si256((const __m256i *)(in2 + x)), mm_inv
            );
            // Unpack and pack both work within 128-bit lanes,
            // so the bytes come back in their original order.
            __m256i lo = _mm256_mullo_epi16(
                _mm256_unpacklo_epi8(a, mm_zero), _mm256_unpacklo_epi8(b, mm_zero)
            );
            __m256i hi = _mm256_mullo_epi16(
                _mm256_unpackhi_epi8(a, mm_zero), _mm256_unpackhi_epi8(b, mm_zero)
            );
            lo = _mm256_srli_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(lo, mm_one), _mm256_srli_epi16(lo, 8)
                ),
                8
            );
            hi = _mm256_srli_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(hi, mm_one), _mm256_srli_epi16(hi, 8)
                ),
                8
            );
            _mm256_storeu_si256(
                (__m256i *)(out + x),
                _mm256_xor_si256(_mm256_packus_epi16(lo, hi), mm_inv)
            );
        }
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    {
        __m128i mm_inv = _mm_set1_epi8((char)invert);
        __m128i mm_one = _mm_set1_epi16(1);
        __m128i mm_zero = _mm_setzero_si128();
        for (; x < n - 15; x += 16) {
            __m128i a =
                _mm_xor_si128(_mm_loadu_si128((const __m128i *)(in1 + x)), mm_inv);
            __m128i b =
                _mm_xor_si128(_mm_loadu_si128((const __m128i *)(in2 + x)), mm_inv);
            __m128i lo = _mm_mullo_epi16(
                _mm_unpacklo_epi8(a, mm_zero), _mm_unpacklo_epi8(b, mm_zero)
            );
            __m128i hi = _mm_mullo_epi16(
                _mm_unpackhi_epi8(a, mm_zero), _mm_unpackhi_epi8(b, mm_zero)
            );
            lo = _mm_srli_epi16(
                _mm_add_epi16(_mm_add_epi16(lo, mm_one), _mm_srli_epi16(lo, 8)), 8
            );
            hi = _mm_srli_epi16(
                _mm_add_epi16(_mm_add_epi16(hi, mm_one), _mm_srli_epi16(hi, 8)), 8
            );
            _mm_storeu_si128(
                (__m128i *)(out + x), _mm_xor_si128(_mm_packus_epi16(lo, hi), mm_inv)
            );
        }
    }
#elif defined(IMAGING_SIMD_NEON)
    {
        uint8x16_t inv = vdupq_n_u8(invert);
        uint16x8_t one = vdupq_n_u16(1);
        for (; x < n - 15; x += 16) {
            uint8x16_t a = veorq_u8(vld1q_u8(in1 + x), inv);
            uint8x16_t b = veorq_u8(vld1q_u8(in2 + x), inv);
            uint16x8_t lo = vmull_u8(vget_low_u8(a), vget_low_u8(b));
            uint16x8_t hi = vmull_high_u8(a, b);
            // (x + (1 + (x >> 8))) >> 8, narrowed to 8 bits
            uint8x8_t out_lo = vaddhn_u16(lo, vsraq_n_u16(one, lo, 8));
            uint8x8_t out_hi = vaddhn_u16(hi, vsraq_n_u16(one, hi, 8));
            vst1q_u8(out + x, veorq_u8(vcombine_u8(out_lo, out_hi), inv));
        }
    }
#endif
    return x;
}

static int
chop_saturate_row_simd(
    UINT8 *restrict out,
    const UINT8 *restrict in1,
    const UINT8 *restrict in2,
    int n,
    int sign
) {
    int x = 0;

#if defined(IMAGING_SIMD_AVX2)
    for (; x < n - 31; x += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + x));
        __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + x));
        _mm256_storeu_si256(
            (__m256i *)(out + x),
            sign > 0 ? _mm256_adds_epu8(a, b) : _mm256_subs_epu8(a, b)
        );
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    for (; x < n - 15; x += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(in1 + x));
        __m128i b = _mm_loadu_si128((const __m128i *)(in2 + x));
        _mm_storeu_si128(
            (__m128i *)(out + x), sign > 0 ? _mm_adds_epu8(a, b) : _mm_subs_epu8(a, b)
        );
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; x < n - 15; x += 16) {
        uint8x16_t a = vld1q_u8(in1 + x);
        uint8x16_t b = vld1q_u8(in2 + x);
        vst1q_u8(out + x, sign > 0 ? vqaddq_u8(a, b) : vqsubq_u8(a, b));
    }
#endif
    return x;
}

static const ImagingSIMDChopsKernels chops_simd_kernels = {
    .multiply_row = chop_multiply_row_simd,
    .saturate_row = chop_saturate_row_simd,
};
