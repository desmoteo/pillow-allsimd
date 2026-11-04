/* The masked fills of pasting and drawing text (Paste.c), compiled into
   SIMDKernels.c: ink blended into pixels through an L mask, for the pixels
   the vector loop can; returns how many. */

/* BLEND(m, out, ink) = DIV255(out * (255 - m) + ink * m) for vectors of
   bytes: every term fits 16 bits, so the result is the scalar one. */
#if defined(IMAGING_SIMD_SSE4)
static inline __m128i
fill_blend16(__m128i out, __m128i m, __m128i ink) {
    __m128i zero = _mm_setzero_si128(), c255 = _mm_set1_epi16(255);
    __m128i r128 = _mm_set1_epi16(128), res[2];
    int h;
    for (h = 0; h < 2; h++) {
        __m128i o = h ? _mm_unpackhi_epi8(out, zero) : _mm_unpacklo_epi8(out, zero);
        __m128i k = h ? _mm_unpackhi_epi8(m, zero) : _mm_unpacklo_epi8(m, zero);
        __m128i c = h ? _mm_unpackhi_epi8(ink, zero) : _mm_unpacklo_epi8(ink, zero);
        __m128i t = _mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(o, _mm_sub_epi16(c255, k)), _mm_mullo_epi16(c, k)
            ),
            r128
        );
        res[h] = _mm_srli_epi16(_mm_add_epi16(t, _mm_srli_epi16(t, 8)), 8);
    }
    return _mm_packus_epi16(res[0], res[1]);
}
#elif defined(IMAGING_SIMD_NEON)
static inline uint8x16_t
fill_blend16(uint8x16_t out, uint8x16_t m, uint8x16_t ink) {
    uint8x16_t nm = vsubq_u8(vdupq_n_u8(255), m);
    uint16x8_t lo = vmlal_u8(
        vmull_u8(vget_low_u8(out), vget_low_u8(nm)), vget_low_u8(ink), vget_low_u8(m)
    );
    uint16x8_t hi = vmlal_high_u8(vmull_high_u8(out, nm), ink, m);
    lo = vaddq_u16(lo, vdupq_n_u16(128));
    hi = vaddq_u16(hi, vdupq_n_u16(128));
    // (t + (t >> 8)) >> 8
    return vcombine_u8(
        vshrn_n_u16(vsraq_n_u16(lo, lo, 8), 8), vshrn_n_u16(vsraq_n_u16(hi, hi, 8), 8)
    );
}
#endif

/* Blends ink into n pixels of out through mask, as fill_mask_L's loops
   below: 1-byte pixels, or 4-byte ones, with alpha_channel as there.
   Returns the number of pixels done; the caller does the rest. */
static int
fill_mask_L_simd(
    UINT8 *out, const UINT8 *mask, const UINT8 *ink, int n, int pixelsize, int alpha
) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
    if (pixelsize == 1) {
        __m128i k = _mm_set1_epi8((char)ink[0]);
        for (; x < n - 15; x += 16) {
            __m128i o = _mm_loadu_si128((__m128i *)(out + x));
            __m128i m = _mm_loadu_si128((const __m128i *)(mask + x));
            _mm_storeu_si128((__m128i *)(out + x), fill_blend16(o, m, k));
        }
    } else if (pixelsize == 4) {
        INT32 ink4;
        memcpy(&ink4, ink, 4);
        __m128i k = _mm_set1_epi32(ink4);
        __m128i spread = _mm_set_epi8(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
        __m128i alpha_bytes =
            _mm_set_epi8(15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3);
        __m128i colour = _mm_set1_epi32(0x00ffffff), zero = _mm_setzero_si128();
        for (; x < n - 3; x += 4) {
            __m128i o = _mm_loadu_si128((__m128i *)(out + x * 4));
            __m128i m =
                _mm_shuffle_epi8(_mm_cvtsi32_si128(mm_load_si32(mask + x)), spread);
            if (alpha) {
                // colour channels of transparent pixels take a mask of 255
                __m128i transparent =
                    _mm_cmpeq_epi8(_mm_shuffle_epi8(o, alpha_bytes), zero);
                __m128i set = _mm_andnot_si128(
                    _mm_cmpeq_epi8(m, zero), _mm_and_si128(transparent, colour)
                );
                m = _mm_or_si128(m, set);
            }
            _mm_storeu_si128((__m128i *)(out + x * 4), fill_blend16(o, m, k));
        }
    }
#elif defined(IMAGING_SIMD_NEON)
    if (pixelsize == 1) {
        uint8x16_t k = vdupq_n_u8(ink[0]);
        for (; x < n - 15; x += 16) {
            vst1q_u8(out + x, fill_blend16(vld1q_u8(out + x), vld1q_u8(mask + x), k));
        }
    } else if (pixelsize == 4) {
        uint32_t ink4;
        memcpy(&ink4, ink, 4);
        uint8x16_t k = vreinterpretq_u8_u32(vdupq_n_u32(ink4));
        const uint8_t spread_idx[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        const uint8_t alpha_idx[16] = {
            3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15
        };
        uint8x16_t spread = vld1q_u8(spread_idx), alpha_bytes = vld1q_u8(alpha_idx);
        uint8x16_t colour = vreinterpretq_u8_u32(vdupq_n_u32(0x00ffffff));
        for (; x < n - 3; x += 4) {
            uint32_t m4;
            memcpy(&m4, mask + x, 4);
            uint8x16_t o = vld1q_u8(out + x * 4);
            uint8x16_t m = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(m4)), spread);
            if (alpha) {
                uint8x16_t transparent = vceqzq_u8(vqtbl1q_u8(o, alpha_bytes));
                m = vorrq_u8(
                    m, vandq_u8(vandq_u8(transparent, colour), vtstq_u8(m, m))
                );
            }
            vst1q_u8(out + x * 4, fill_blend16(o, m, k));
        }
    }
#endif
    return x;
}

static const ImagingSIMDPasteKernels paste_simd_kernels = {
    .fill_mask_L = fill_mask_L_simd,
};
