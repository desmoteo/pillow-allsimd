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

#if defined(IMAGING_SIMD_SSE4)
typedef __m128i filter_acc;
#define FILTER_ACC_INIT(v) _mm_set1_epi32(v)

/* Shifts and saturates 16 sums to bytes. */
static inline __m128i
filter_pack16(__m128i s0, __m128i s1, __m128i s2, __m128i s3) {
    __m128i lo = _mm_packs_epi32(
        _mm_srai_epi32(s0, PRECISION_BITS), _mm_srai_epi32(s1, PRECISION_BITS));
    __m128i hi = _mm_packs_epi32(
        _mm_srai_epi32(s2, PRECISION_BITS), _mm_srai_epi32(s3, PRECISION_BITS));
    return _mm_packus_epi16(lo, hi);
}
#define FILTER_STORE16(ptr, v) _mm_storeu_si128((__m128i *)(ptr), v)

/* Adds a[i] * k0 + b[i] * k1 for 16 bytes a and b to four 32-bit sums. */
#define FILTER_MLA16x2(s0, s1, s2, s3, pa, pb, k0, k1)                        \
    {                                                                         \
        __m128i a_ = _mm_loadu_si128((__m128i *)(pa));                        \
        __m128i b_ = _mm_loadu_si128((__m128i *)(pb));                        \
        __m128i kk_ = _mm_set1_epi32(((UINT32)(UINT16)(k1) << 16) | (UINT16)(k0)); \
        __m128i alo_ = _mm_cvtepu8_epi16(a_);                                 \
        __m128i ahi_ = _mm_cvtepu8_epi16(_mm_srli_si128(a_, 8));              \
        __m128i blo_ = _mm_cvtepu8_epi16(b_);                                 \
        __m128i bhi_ = _mm_cvtepu8_epi16(_mm_srli_si128(b_, 8));              \
        s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_unpacklo_epi16(alo_, blo_), kk_)); \
        s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_unpackhi_epi16(alo_, blo_), kk_)); \
        s2 = _mm_add_epi32(s2, _mm_madd_epi16(_mm_unpacklo_epi16(ahi_, bhi_), kk_)); \
        s3 = _mm_add_epi32(s3, _mm_madd_epi16(_mm_unpackhi_epi16(ahi_, bhi_), kk_)); \
    }
#define FILTER_MLA16x1(s0, s1, s2, s3, p, k) FILTER_MLA16x2(s0, s1, s2, s3, p, p, k, 0)

#else
typedef int32x4_t filter_acc;
#define FILTER_ACC_INIT(v) vdupq_n_s32(v)

/* Shifts and saturates 16 sums to bytes. */
static inline uint8x16_t
filter_pack16(int32x4_t s0, int32x4_t s1, int32x4_t s2, int32x4_t s3) {
    int32x4_t shift = vdupq_n_s32(-PRECISION_BITS);
    int16x8_t lo = vcombine_s16(
        vqmovn_s32(vshlq_s32(s0, shift)), vqmovn_s32(vshlq_s32(s1, shift)));
    int16x8_t hi = vcombine_s16(
        vqmovn_s32(vshlq_s32(s2, shift)), vqmovn_s32(vshlq_s32(s3, shift)));
    return vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi));
}
#define FILTER_STORE16(ptr, v) vst1q_u8(ptr, v)

/* Adds 16 bytes times k to four 32-bit sums. */
#define FILTER_MLA16(s0, s1, s2, s3, ptr, k)                                  \
    {                                                                         \
        uint8x16_t src_ = vld1q_u8(ptr);                                      \
        int16x8_t lo_ = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_)));   \
        int16x8_t hi_ = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src_)));  \
        s0 = vmlal_n_s16(s0, vget_low_s16(lo_), k);                           \
        s1 = vmlal_n_s16(s1, vget_high_s16(lo_), k);                          \
        s2 = vmlal_n_s16(s2, vget_low_s16(hi_), k);                           \
        s3 = vmlal_n_s16(s3, vget_high_s16(hi_), k);                          \
    }

/* Same interface as the SSE4 version, which pairs taps for pmaddwd. */
#define FILTER_MLA16x2(s0, s1, s2, s3, pa, pb, k0, k1)                        \
    {                                                                         \
        FILTER_MLA16(s0, s1, s2, s3, pa, k0);                                 \
        FILTER_MLA16(s0, s1, s2, s3, pb, k1);                                 \
    }
#define FILTER_MLA16x1(s0, s1, s2, s3, p, k) FILTER_MLA16(s0, s1, s2, s3, p, k)
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
#undef FILTER_STORE16

/* Separate entry points so the compiler unrolls each size fully. */
static void
ImagingFilter3x3i_u8_simd(Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset) {
    ImagingFilterNxN_simd(imOut, im, kernel, offset, 3, 1);
}

static void
ImagingFilter5x5i_u8_simd(Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset) {
    ImagingFilterNxN_simd(imOut, im, kernel, offset, 5, 1);
}

#if defined(IMAGING_SIMD_NEON)
/* x86 has dedicated SSE4/AVX2 kernels for 4-byte images. */
static void
ImagingFilter3x3i_4u8_simd(Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset) {
    ImagingFilterNxN_simd(imOut, im, kernel, offset, 3, 4);
}

static void
ImagingFilter5x5i_4u8_simd(Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset) {
    ImagingFilterNxN_simd(imOut, im, kernel, offset, 5, 4);
}
#endif
