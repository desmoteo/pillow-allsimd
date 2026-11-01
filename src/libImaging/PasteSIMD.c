/* The masked fills of pasting and drawing text and the paste through an L
   mask (Paste.c), compiled into SIMDKernels.c: ink, or another image's
   pixels, blended into pixels through the mask, for the pixels the vector
   loop can; returns how many. */

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
#if defined(IMAGING_SIMD_AVX2)
/* The same on 32 bytes: the unpacks and the pack work within each half. */
static inline __m256i
fill_blend32(__m256i out, __m256i m, __m256i ink) {
    __m256i zero = _mm256_setzero_si256(), c255 = _mm256_set1_epi16(255);
    __m256i r128 = _mm256_set1_epi16(128), res[2];
    int h;
    for (h = 0; h < 2; h++) {
        __m256i o =
            h ? _mm256_unpackhi_epi8(out, zero) : _mm256_unpacklo_epi8(out, zero);
        __m256i k = h ? _mm256_unpackhi_epi8(m, zero) : _mm256_unpacklo_epi8(m, zero);
        __m256i c =
            h ? _mm256_unpackhi_epi8(ink, zero) : _mm256_unpacklo_epi8(ink, zero);
        __m256i t = _mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_mullo_epi16(o, _mm256_sub_epi16(c255, k)),
                _mm256_mullo_epi16(c, k)
            ),
            r128
        );
        res[h] = _mm256_srli_epi16(_mm256_add_epi16(t, _mm256_srli_epi16(t, 8)), 8);
    }
    return _mm256_packus_epi16(res[0], res[1]);
}
#endif
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

/* Blends n pixels of in into out through mask, as paste_mask_L's loops:
   1-byte pixels, or 4-byte ones, all four bytes as there. The rows do not
   overlap (the caller checks). Returns the number of pixels done; the
   caller does the rest. The portable build's compiler vectorizes the plain
   loop when it is given AVX2, but for the baseline of a dispatch build it
   makes code half as fast, which is why the loop is a kernel. */
static int
paste_mask_L_simd(
    UINT8 *out, const UINT8 *in, const UINT8 *mask, int n, int pixelsize
) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
    if (pixelsize == 1) {
#if defined(IMAGING_SIMD_AVX2)
        for (; x < n - 31; x += 32) {
            __m256i o = _mm256_loadu_si256((__m256i *)(out + x));
            __m256i m = _mm256_loadu_si256((const __m256i *)(mask + x));
            __m256i c = _mm256_loadu_si256((const __m256i *)(in + x));
            _mm256_storeu_si256((__m256i *)(out + x), fill_blend32(o, m, c));
        }
#endif
        for (; x < n - 15; x += 16) {
            __m128i o = _mm_loadu_si128((__m128i *)(out + x));
            __m128i m = _mm_loadu_si128((const __m128i *)(mask + x));
            __m128i c = _mm_loadu_si128((const __m128i *)(in + x));
            _mm_storeu_si128((__m128i *)(out + x), fill_blend16(o, m, c));
        }
    } else if (pixelsize == 4) {
        __m128i spread = _mm_set_epi8(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
#if defined(IMAGING_SIMD_AVX2)
        /* eight mask bytes, one in each 32-bit lane, then in its four bytes */
        __m256i spread8 = _mm256_setr_epi8(
            0,
            0,
            0,
            0,
            4,
            4,
            4,
            4,
            8,
            8,
            8,
            8,
            12,
            12,
            12,
            12,
            0,
            0,
            0,
            0,
            4,
            4,
            4,
            4,
            8,
            8,
            8,
            8,
            12,
            12,
            12,
            12
        );
        for (; x < n - 7; x += 8) {
            __m256i o = _mm256_loadu_si256((__m256i *)(out + x * 4));
            __m256i c = _mm256_loadu_si256((const __m256i *)(in + x * 4));
            __m256i m = _mm256_shuffle_epi8(
                _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(mask + x))),
                spread8
            );
            _mm256_storeu_si256((__m256i *)(out + x * 4), fill_blend32(o, m, c));
        }
#endif
        for (; x < n - 3; x += 4) {
            __m128i o = _mm_loadu_si128((__m128i *)(out + x * 4));
            __m128i c = _mm_loadu_si128((const __m128i *)(in + x * 4));
            __m128i m =
                _mm_shuffle_epi8(_mm_cvtsi32_si128(mm_load_si32(mask + x)), spread);
            _mm_storeu_si128((__m128i *)(out + x * 4), fill_blend16(o, m, c));
        }
    }
#elif defined(IMAGING_SIMD_NEON)
    if (pixelsize == 1) {
        for (; x < n - 15; x += 16) {
            vst1q_u8(
                out + x,
                fill_blend16(vld1q_u8(out + x), vld1q_u8(mask + x), vld1q_u8(in + x))
            );
        }
    } else if (pixelsize == 4) {
        const uint8_t spread_idx[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
        uint8x16_t spread = vld1q_u8(spread_idx);
        for (; x < n - 3; x += 4) {
            uint32_t m4;
            memcpy(&m4, mask + x, 4);
            uint8x16_t m = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(m4)), spread);
            vst1q_u8(
                out + x * 4,
                fill_blend16(vld1q_u8(out + x * 4), m, vld1q_u8(in + x * 4))
            );
        }
    }
#endif
    return x;
}

/* The same with the mask in the alpha bytes of n 4-byte pixels (an RGBA
   image pasted through itself), as paste_mask_RGBA's loop. */
static int
paste_mask_RGBA_simd(UINT8 *out, const UINT8 *in, const UINT8 *mask, int n) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
    __m128i alpha_bytes =
        _mm_set_epi8(15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3);
#if defined(IMAGING_SIMD_AVX2)
    __m256i alpha_bytes8 = _mm256_broadcastsi128_si256(alpha_bytes);
    for (; x < n - 7; x += 8) {
        __m256i o = _mm256_loadu_si256((__m256i *)(out + x * 4));
        __m256i c = _mm256_loadu_si256((const __m256i *)(in + x * 4));
        __m256i m = _mm256_shuffle_epi8(
            _mm256_loadu_si256((const __m256i *)(mask + x * 4)), alpha_bytes8
        );
        _mm256_storeu_si256((__m256i *)(out + x * 4), fill_blend32(o, m, c));
    }
#endif
    for (; x < n - 3; x += 4) {
        __m128i o = _mm_loadu_si128((__m128i *)(out + x * 4));
        __m128i c = _mm_loadu_si128((const __m128i *)(in + x * 4));
        __m128i m = _mm_shuffle_epi8(
            _mm_loadu_si128((const __m128i *)(mask + x * 4)), alpha_bytes
        );
        _mm_storeu_si128((__m128i *)(out + x * 4), fill_blend16(o, m, c));
    }
#elif defined(IMAGING_SIMD_NEON)
    const uint8_t alpha_idx[16] = {
        3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15
    };
    uint8x16_t alpha_bytes = vld1q_u8(alpha_idx);
    for (; x < n - 3; x += 4) {
        uint8x16_t m = vqtbl1q_u8(vld1q_u8(mask + x * 4), alpha_bytes);
        vst1q_u8(
            out + x * 4, fill_blend16(vld1q_u8(out + x * 4), m, vld1q_u8(in + x * 4))
        );
    }
#endif
    return x;
}

static const ImagingSIMDPasteKernels paste_simd_kernels = {
    .fill_mask_L = fill_mask_L_simd,
    .paste_mask_L = paste_mask_L_simd,
    .paste_mask_RGBA = paste_mask_RGBA_simd,
};
