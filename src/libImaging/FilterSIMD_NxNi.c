/* SSE4 and NEON 3x3 and 5x5 integer kernels for 8-bit images,
   included from Filter.c.

   Same arithmetic as the SSE4 integer kernels: 16-bit coefficients scaled
   by 1 << PRECISION_BITS, 32-bit sums, an arithmetic shift and saturation
   to 0..255. The pixels left over at the end of a row use the same integer
   formula in C, so results don't depend on position.

   Kernel row 0 applies to image row y + R, the last kernel row to y - R,
   and the R border rows and columns are copied, as in the generic code. */

static inline UINT8
filter_clip(INT32 ss) {
    ss >>= PRECISION_BITS;
    return ss < 0 ? 0 : (ss > 255 ? 255 : ss);
}

typedef __m128i filter_acc;
#define FILTER_ACC_INIT(v) _mm_set1_epi32(v)

/* Shifts and saturates 16 sums to bytes. */
static inline __m128i
filter_pack16(__m128i s0, __m128i s1, __m128i s2, __m128i s3) {
    __m128i lo = _mm_packs_epi32(
        _mm_srai_epi32(s0, PRECISION_BITS), _mm_srai_epi32(s1, PRECISION_BITS)
    );
    __m128i hi = _mm_packs_epi32(
        _mm_srai_epi32(s2, PRECISION_BITS), _mm_srai_epi32(s3, PRECISION_BITS)
    );
    return _mm_packus_epi16(lo, hi);
}
#define FILTER_STORE16(ptr, v) _mm_storeu_si128((__m128i *)(ptr), v)

/* Adds a[i] * k0 + b[i] * k1 for 16 bytes a and b to four 32-bit sums. */
#define FILTER_MLA16x2(s0, s1, s2, s3, pa, pb, k0, k1)                             \
    {                                                                              \
        __m128i a_ = _mm_loadu_si128((__m128i *)(pa));                             \
        __m128i b_ = _mm_loadu_si128((__m128i *)(pb));                             \
        __m128i kk_ = _mm_set1_epi32(((UINT32)(UINT16)(k1) << 16) | (UINT16)(k0)); \
        __m128i alo_ = _mm_cvtepu8_epi16(a_);                                      \
        __m128i ahi_ = _mm_cvtepu8_epi16(_mm_srli_si128(a_, 8));                   \
        __m128i blo_ = _mm_cvtepu8_epi16(b_);                                      \
        __m128i bhi_ = _mm_cvtepu8_epi16(_mm_srli_si128(b_, 8));                   \
        s0 = mm_dpwssd(s0, _mm_unpacklo_epi16(alo_, blo_), kk_);                   \
        s1 = mm_dpwssd(s1, _mm_unpackhi_epi16(alo_, blo_), kk_);                   \
        s2 = mm_dpwssd(s2, _mm_unpacklo_epi16(ahi_, bhi_), kk_);                   \
        s3 = mm_dpwssd(s3, _mm_unpackhi_epi16(ahi_, bhi_), kk_);                   \
    }
#define FILTER_MLA16x1(s0, s1, s2, s3, p, k) FILTER_MLA16x2(s0, s1, s2, s3, p, p, k, 0)

#if defined(IMAGING_SIMD_AVX2)
/* The same for 32 bytes. The 16-bit pairs are interleaved within each
   128-bit lane: s0 holds pixels 0-3 and 8-11, s1 4-7 and 12-15, s2 16-19
   and 24-27, s3 20-23 and 28-31. */
#define FILTER_MLA32x2(s0, s1, s2, s3, pa, pb, k0, k1)                                \
    {                                                                                 \
        __m256i kk_ = _mm256_set1_epi32(((UINT32)(UINT16)(k1) << 16) | (UINT16)(k0)); \
        __m256i alo_ = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pa)));        \
        __m256i ahi_ = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)((pa) + 16))); \
        __m256i blo_ = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pb)));        \
        __m256i bhi_ = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)((pb) + 16))); \
        s0 = mm256_dpwssd(s0, _mm256_unpacklo_epi16(alo_, blo_), kk_);                \
        s1 = mm256_dpwssd(s1, _mm256_unpackhi_epi16(alo_, blo_), kk_);                \
        s2 = mm256_dpwssd(s2, _mm256_unpacklo_epi16(ahi_, bhi_), kk_);                \
        s3 = mm256_dpwssd(s3, _mm256_unpackhi_epi16(ahi_, bhi_), kk_);                \
    }

/* Shifts and saturates 32 sums to bytes, back in pixel order. */
static inline __m256i
filter_pack32(__m256i s0, __m256i s1, __m256i s2, __m256i s3) {
    __m256i lo = _mm256_packs_epi32(
        _mm256_srai_epi32(s0, PRECISION_BITS), _mm256_srai_epi32(s1, PRECISION_BITS)
    );
    __m256i hi = _mm256_packs_epi32(
        _mm256_srai_epi32(s2, PRECISION_BITS), _mm256_srai_epi32(s3, PRECISION_BITS)
    );
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xd8);
}
#endif

/* size is 3 or 5; bpp is 1 or 4 bytes per pixel. */
static inline void
ImagingFilterNxN_simd(
    Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int size, int bpp
) {
    int x, y, r, d, c;
    int R = size / 2;
    int xsize = im->xsize, ysize = im->ysize;
    /* Each step makes 16 bytes: 16 1-byte pixels or 4 4-byte pixels. */
    int step = 16 / bpp;
    const UINT8 *rows[5];

    offset += 1 << (PRECISION_BITS - 1);

    for (y = 0; y < R; y++) {
        memcpy(imOut->image[y], im->image[y], im->linesize);
    }
    for (y = R; y < ysize - R; y++) {
        UINT8 *out = (UINT8 *)imOut->image[y];
        const UINT8 *in0 = (UINT8 *)im->image[y];
        for (r = 0; r < size; r++) {
            rows[r] = (UINT8 *)im->image[y + R - r];
        }

        memcpy(out, in0, R * bpp);
        x = R;
#if defined(IMAGING_SIMD_AVX2)
        if (bpp == 1) {
            for (; x < xsize - R - 31; x += 32) {
                __m256i s0 = _mm256_set1_epi32(offset), s1 = s0, s2 = s0, s3 = s0;
                for (r = 0; r < size; r++) {
                    const INT16 *k = &kernel[r * size];
                    const UINT8 *p = rows[r] + x - R;
                    for (d = 0; d < size - 1; d += 2) {
                        FILTER_MLA32x2(
                            s0, s1, s2, s3, p + d, p + d + 1, k[d], k[d + 1]
                        );
                    }
                    FILTER_MLA32x2(s0, s1, s2, s3, p + d, p + d, k[d], 0);
                }
                _mm256_storeu_si256(
                    (__m256i *)(out + x), filter_pack32(s0, s1, s2, s3)
                );
            }
        }
#endif
        for (; x < xsize - R - (step - 1); x += step) {
            filter_acc s0 = FILTER_ACC_INIT(offset), s1 = s0, s2 = s0, s3 = s0;
            for (r = 0; r < size; r++) {
                const INT16 *k = &kernel[r * size];
                const UINT8 *p = rows[r] + (x - R) * bpp;
                /* Taps in pairs, then the odd last one. */
                for (d = 0; d < size - 1; d += 2) {
                    FILTER_MLA16x2(
                        s0, s1, s2, s3, p + d * bpp, p + (d + 1) * bpp, k[d], k[d + 1]
                    );
                }
                FILTER_MLA16x1(s0, s1, s2, s3, p + d * bpp, k[d]);
            }
            FILTER_STORE16(out + x * bpp, filter_pack16(s0, s1, s2, s3));
        }
        for (; x < xsize - R; x++) {
            for (c = 0; c < bpp; c++) {
                INT32 ss = offset;
                for (r = 0; r < size; r++) {
                    for (d = 0; d < size; d++) {
                        ss += rows[r][(x + d - R) * bpp + c] * kernel[r * size + d];
                    }
                }
                out[x * bpp + c] = filter_clip(ss);
            }
        }
        memcpy(out + x * bpp, in0 + x * bpp, R * bpp);
    }
    for (; y < ysize; y++) {
        memcpy(imOut->image[y], im->image[y], im->linesize);
    }
}

#undef FILTER_MLA16x2
#undef FILTER_MLA16x1
#undef FILTER_MLA32x2
#undef FILTER_STORE16

/* Separate entry points so the compiler unrolls each size fully. */
static void
ImagingFilter3x3i_u8_simd(
    Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset
) {
    ImagingFilterNxN_simd(imOut, im, kernel, offset, 3, 1);
}

static void
ImagingFilter5x5i_u8_simd(
    Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset
) {
    ImagingFilterNxN_simd(imOut, im, kernel, offset, 5, 1);
}
