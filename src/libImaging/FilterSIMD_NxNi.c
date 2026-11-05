/* SSE4 and NEON 3x3 and 5x5 integer kernels for 8-bit images,
   included from Filter.c.

   Same arithmetic as the SSE4 integer kernels: 16-bit coefficients scaled
   by 1 << precision, 32-bit sums, an arithmetic shift and saturation
   to 0..255. The pixels left over at the end of a row use the same integer
   formula in C, so results don't depend on position.

   Kernel row 0 applies to image row y + R, the last kernel row to y - R,
   and the R border rows and columns are copied, as in the generic code. */

static inline UINT8
filter_clip(INT32 ss, int precision) {
    ss >>= precision;
    return ss < 0 ? 0 : (ss > 255 ? 255 : ss);
}

#if defined(IMAGING_SIMD_SSE4)
typedef __m128i filter_acc;
#define FILTER_ACC_INIT(v) _mm_set1_epi32(v)

/* Shifts and saturates 16 sums to bytes. */
static inline __m128i
filter_pack16(__m128i s0, __m128i s1, __m128i s2, __m128i s3, int precision) {
    __m128i n = _mm_cvtsi32_si128(precision);
    __m128i lo = _mm_packs_epi32(_mm_sra_epi32(s0, n), _mm_sra_epi32(s1, n));
    __m128i hi = _mm_packs_epi32(_mm_sra_epi32(s2, n), _mm_sra_epi32(s3, n));
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
filter_pack32(__m256i s0, __m256i s1, __m256i s2, __m256i s3, int precision) {
    __m128i n = _mm_cvtsi32_si128(precision);
    __m256i lo = _mm256_packs_epi32(_mm256_sra_epi32(s0, n), _mm256_sra_epi32(s1, n));
    __m256i hi = _mm256_packs_epi32(_mm256_sra_epi32(s2, n), _mm256_sra_epi32(s3, n));
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xd8);
}
#endif

#else
typedef int32x4_t filter_acc;
#define FILTER_ACC_INIT(v) vdupq_n_s32(v)

/* Shifts and saturates 16 sums to bytes. */
static inline uint8x16_t
filter_pack16(int32x4_t s0, int32x4_t s1, int32x4_t s2, int32x4_t s3, int precision) {
    int32x4_t shift = vdupq_n_s32(-precision);
    int16x8_t lo = vcombine_s16(
        vqmovn_s32(vshlq_s32(s0, shift)), vqmovn_s32(vshlq_s32(s1, shift))
    );
    int16x8_t hi = vcombine_s16(
        vqmovn_s32(vshlq_s32(s2, shift)), vqmovn_s32(vshlq_s32(s3, shift))
    );
    return vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi));
}
#define FILTER_STORE16(ptr, v) vst1q_u8(ptr, v)

/* Adds 16 bytes times k to four 32-bit sums. */
#define FILTER_MLA16(s0, s1, s2, s3, ptr, k)                                 \
    {                                                                        \
        uint8x16_t src_ = vld1q_u8(ptr);                                     \
        int16x8_t lo_ = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_)));  \
        int16x8_t hi_ = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src_))); \
        s0 = vmlal_n_s16(s0, vget_low_s16(lo_), k);                          \
        s1 = vmlal_n_s16(s1, vget_high_s16(lo_), k);                         \
        s2 = vmlal_n_s16(s2, vget_low_s16(hi_), k);                          \
        s3 = vmlal_n_s16(s3, vget_high_s16(hi_), k);                         \
    }

/* Same interface as the SSE4 version, which pairs taps for pmaddwd. */
#define FILTER_MLA16x2(s0, s1, s2, s3, pa, pb, k0, k1) \
    {                                                  \
        FILTER_MLA16(s0, s1, s2, s3, pa, k0);          \
        FILTER_MLA16(s0, s1, s2, s3, pb, k1);          \
    }
#define FILTER_MLA16x1(s0, s1, s2, s3, p, k) FILTER_MLA16(s0, s1, s2, s3, p, k)
#endif

/* size is 3 or 5; bpp is 1 or 4 bytes per pixel. */
SIMD_FORCE_INLINE void
ImagingFilterNxN_simd(
    Imaging imOut,
    Imaging im,
    const INT16 *kernel_in,
    INT32 offset,
    int precision,
    int size,
    int bpp
) {
    int x, y, r, d, c;
    int R = size / 2;
    int xsize = im->xsize, ysize = im->ysize;
    /* Each step makes 16 bytes: 16 1-byte pixels or 4 4-byte pixels. */
    int step = 16 / bpp;
    const UINT8 *rows[5];
    /* For 3x3 on 1-byte images, a copy the byte stores to the output
       can't alias, so the nine coefficients stay in registers across the
       row: L SHARPEN was 11% faster on NEON, 8% with AVX2. Not for 5x5
       (x86 spills the 25) nor for 4-byte pixels (NEON was 1.3x slower). */
    INT16 kernel3[9];
    const INT16 *kernel = kernel_in;
    if (size == 3 && bpp == 1) {
        memcpy(kernel3, kernel_in, sizeof(kernel3));
        kernel = kernel3;
    }

    offset += 1 << (precision - 1);

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
                    (__m256i *)(out + x), filter_pack32(s0, s1, s2, s3, precision)
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
            FILTER_STORE16(out + x * bpp, filter_pack16(s0, s1, s2, s3, precision));
        }
        for (; x < xsize - R; x++) {
            for (c = 0; c < bpp; c++) {
                INT32 ss = offset;
                for (r = 0; r < size; r++) {
                    for (d = 0; d < size; d++) {
                        ss += rows[r][(x + d - R) * bpp + c] * kernel[r * size + d];
                    }
                }
                out[x * bpp + c] = filter_clip(ss, precision);
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

/* Separate entry points so the compiler unrolls each size fully, and
   with IMAGING_FILTER_PRECISION_BITS as a constant where it is the precision: NEON
   then shifts and narrows in one instruction (6% faster on the Jetson). */
#define FILTER_NxN(size, bpp)                                                   \
    if (precision == IMAGING_FILTER_PRECISION_BITS) {                           \
        ImagingFilterNxN_simd(                                                  \
            imOut, im, kernel, offset, IMAGING_FILTER_PRECISION_BITS, size, bpp \
        );                                                                      \
    } else {                                                                    \
        ImagingFilterNxN_simd(imOut, im, kernel, offset, precision, size, bpp); \
    }

static void
ImagingFilter3x3i_u8_simd(
    Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int precision
) {
    FILTER_NxN(3, 1);
}

static void
ImagingFilter5x5i_u8_simd(
    Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int precision
) {
    FILTER_NxN(5, 1);
}

#if defined(IMAGING_SIMD_NEON)
/* x86 has dedicated SSE4/AVX2 kernels for 4-byte images. */
static void
ImagingFilter3x3i_4u8_simd(
    Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int precision
) {
    FILTER_NxN(3, 4);
}

static void
ImagingFilter5x5i_4u8_simd(
    Imaging imOut, Imaging im, const INT16 *kernel, INT32 offset, int precision
) {
    FILTER_NxN(5, 4);
}

/* Kernels whose coefficients are small once the power of two that they,
   the offset and the rounding share is divided out (filter_small_kernel()
   in Filter.c): each at most 255 in size and every sum within 16 bits, as
   for SHARPEN, FIND_EDGES, EDGE_ENHANCE, EMBOSS, CONTOUR and BLUR. The
   sums go in 16-bit lanes, eight per operation where the kernel above has
   four, a coefficient of 1 or -1 as a widening add or subtract and any
   other as one widening multiply-accumulate of bytes: the Cortex-A78 has
   one multiplier pipe, which the four multiplies per tap of the general
   kernel keep busy for 9 cycles per pixel. The sums are the general
   kernel's divided by the shared power of two, exactly, so the shift and
   the saturation give the same bytes. Any pixel size: bytes are filtered
   with the byte bpp positions away. */
static void
ImagingFilterSmall_simd(
    Imaging imOut,
    Imaging im,
    const INT16 *kernel,
    INT32 offset,
    int precision,
    int size
) {
    int x, y, r, d;
    int R = size / 2, bpp = im->image8 ? 1 : 4;
    int rowbytes = im->xsize * bpp, edge = R * bpp;
    int ysize = im->ysize;
    const UINT8 *rows[5];
    int16x8_t shift = vdupq_n_s16((int16_t)-precision);

    /* precision 0: nothing is shifted and nothing rounded */
    if (precision) {
        offset += 1 << (precision - 1);
    }

    for (y = 0; y < R && y < ysize; y++) {
        memcpy(imOut->image[y], im->image[y], im->linesize);
    }
    for (y = R; y < ysize - R; y++) {
        UINT8 *out = (UINT8 *)imOut->image[y];
        const UINT8 *in0 = (UINT8 *)im->image[y];
        for (r = 0; r < size; r++) {
            rows[r] = (UINT8 *)im->image[y + R - r];
        }
        memcpy(out, in0, edge < rowbytes ? edge : rowbytes);
        x = edge;
        for (; x + 16 <= rowbytes - edge; x += 16) {
            uint16x8_t a0 = vdupq_n_u16((uint16_t)offset), a1 = a0;
            for (r = 0; r < size; r++) {
                for (d = 0; d < size; d++) {
                    int k = kernel[r * size + d];
                    uint8x16_t src;
                    if (!k) {
                        continue;
                    }
                    src = vld1q_u8(rows[r] + x + (d - R) * bpp);
                    if (k == 1) {
                        a0 = vaddw_u8(a0, vget_low_u8(src));
                        a1 = vaddw_high_u8(a1, src);
                    } else if (k == -1) {
                        a0 = vsubw_u8(a0, vget_low_u8(src));
                        a1 = vsubw_high_u8(a1, src);
                    } else if (k > 0) {
                        uint8x16_t kv = vdupq_n_u8((uint8_t)k);
                        a0 = vmlal_u8(a0, vget_low_u8(src), vget_low_u8(kv));
                        a1 = vmlal_high_u8(a1, src, kv);
                    } else {
                        uint8x16_t kv = vdupq_n_u8((uint8_t)-k);
                        a0 = vmlsl_u8(a0, vget_low_u8(src), vget_low_u8(kv));
                        a1 = vmlsl_high_u8(a1, src, kv);
                    }
                }
            }
            vst1q_u8(
                out + x,
                vcombine_u8(
                    vqmovun_s16(vshlq_s16(vreinterpretq_s16_u16(a0), shift)),
                    vqmovun_s16(vshlq_s16(vreinterpretq_s16_u16(a1), shift))
                )
            );
        }
        for (; x < rowbytes - edge; x++) {
            INT32 ss = offset;
            for (r = 0; r < size; r++) {
                for (d = 0; d < size; d++) {
                    ss += rows[r][x + (d - R) * bpp] * kernel[r * size + d];
                }
            }
            out[x] = filter_clip(ss, precision);
        }
        if (x < rowbytes) {
            memcpy(out + x, in0 + x, rowbytes - x);
        }
    }
    for (; y < ysize; y++) {
        memcpy(imOut->image[y], im->image[y], im->linesize);
    }
}
#endif

#undef FILTER_NxN
