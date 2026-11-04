/* The box blur's SIMD kernels (BoxBlur.c), compiled into SIMDKernels.c:
   the running-sum row blur in SIMD lanes (and two rows at once with AVX2),
   the direct passes for small radii, the interleaving of four 1-byte rows
   into one 4-byte row, and the vertical pass's running sums. The partial
   kernels do the bytes their vector loops can and return how many. */

#include "BoxBlur.h"

#if defined(IMAGING_SIMD_AVX2)
/* Two rows at once, one in each 128-bit half: the same arithmetic per
   channel with half the instructions. */
static inline __m256i
px8_load(const UINT32 *a, const UINT32 *b) {
    return _mm256_cvtepu8_epi32(
        _mm_unpacklo_epi32(_mm_cvtsi32_si128(*a), _mm_cvtsi32_si128(*b))
    );
}

static void
box_line_blur32x2(
    UINT32 *lineOut0,
    UINT32 *lineOut1,
    UINT32 *lineIn0,
    UINT32 *lineIn1,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
) {
    const __m256i b23 = _mm256_set1_epi32(1 << 23);
    const __m256i bytes = _mm256_setr_epi8(
        0,
        4,
        8,
        12,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        4,
        8,
        12,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1
    );
#define V __m256i
#define V_LOAD(i) px8_load(&lineIn0[i], &lineIn1[i])
#define V_SET1 _mm256_set1_epi32
#define V_ADD _mm256_add_epi32
#define V_SUB _mm256_sub_epi32
#define V_MUL _mm256_mullo_epi32
#define V_SAVE(x, bulk)                                                   \
    {                                                                     \
        __m256i v_ = _mm256_shuffle_epi8(                                 \
            _mm256_srli_epi32(_mm256_add_epi32(bulk, b23), 24), bytes     \
        );                                                                \
        lineOut0[x] = _mm_cvtsi128_si32(_mm256_castsi256_si128(v_));      \
        lineOut1[x] = _mm_cvtsi128_si32(_mm256_extracti128_si256(v_, 1)); \
    }
    BOX_LINE_BLUR
#undef V
#undef V_LOAD
#undef V_SET1
#undef V_ADD
#undef V_SUB
#undef V_MUL
#undef V_SAVE
}
#endif

/* The box around each byte directly, from d->count sources, with count a
   constant per radius so the sums unroll (BoxBlur.c, box_direct). */
SIMD_FORCE_INLINE int
box_direct_simd_body(
    UINT8 *restrict out, const UINT8 **src, int n, const BoxDirect *d, int count
) {
    int i = 0, j;
    UINT32 ww = d->ww, fw = d->fw;
#if defined(IMAGING_SIMD_SSE4)
/* c, the sum of the box at offset o */
#define BOX_DIRECT_SUM(P, load, o)                \
    c = load(src[1] + (o));                       \
    for (j = 2; j < count - 1; j++) {             \
        c = P##_add_epi16(c, load(src[j] + (o))); \
    }
    if (d->fast) {
#define BOX_DIRECT_FAST(P, c) \
    P##_srl_epi16(P##_mulhi_epu16(P##_add_epi16(P##_sll_epi16(c, s), k), m), t)
        __m128i s = _mm_cvtsi32_si128(d->s), t = _mm_cvtsi32_si128(d->t);
#if defined(IMAGING_SIMD_AVX2)
        {
            __m256i k = _mm256_set1_epi16((short)d->k);
            __m256i m = _mm256_set1_epi16((short)d->m);
            for (; i <= n - 32; i += 32) {
                __m256i r[2], c;
                int h;
                for (h = 0; h < 2; h++) {
                    BOX_DIRECT_SUM(_mm256, mm256_cvtepu8_epi16, i + h * 16)
                    r[h] = BOX_DIRECT_FAST(_mm256, c);
                }
                /* 64-bit groups 0-7, 16-23, 8-15, 24-31 */
                _mm256_storeu_si256(
                    (__m256i *)(out + i),
                    _mm256_permute4x64_epi64(_mm256_packus_epi16(r[0], r[1]), 0xd8)
                );
            }
        }
#endif
        {
            __m128i k = _mm_set1_epi16((short)d->k), m = _mm_set1_epi16((short)d->m);
            for (; i <= n - 16; i += 16) {
                __m128i r[2], c;
                int h;
                for (h = 0; h < 2; h++) {
                    BOX_DIRECT_SUM(_mm, mm_cvtepu8_epi16, i + h * 8)
                    r[h] = BOX_DIRECT_FAST(_mm, c);
                }
                _mm_storeu_si128((__m128i *)(out + i), _mm_packus_epi16(r[0], r[1]));
            }
        }
#undef BOX_DIRECT_FAST
    } else {
        /* The products exactly, with the weights split in 12-bit halves:
           pmaddwd makes two 16-bit products and their sum in one
           instruction, where pmulld takes two for one product.
           ww * c + fw * e = ((wh * c + fh * e) << 12) + (wl * c + fl * e),
           with no step above 2^32: the weights add up to at most 2^24, and
           c, at most 255 * 3 here, fits in 16 bits. */
        const int wh = (int)((fw >> 12) << 16 | (ww >> 12));
        const int wl = (int)((fw & 0xfff) << 16 | (ww & 0xfff));
#define BOX_DIRECT_PAIRS(P, v, h, l, b23)                                      \
    P##_srli_epi32(                                                            \
        P##_add_epi32(                                                         \
            P##_add_epi32(                                                     \
                P##_slli_epi32(P##_madd_epi16(v, h), 12), P##_madd_epi16(v, l) \
            ),                                                                 \
            b23                                                                \
        ),                                                                     \
        24                                                                     \
    )
/* e, the sum of the two far sources, at offset o */
#define BOX_DIRECT_FAR(P, load, o) \
    e = P##_add_epi16(load(src[0] + (o)), load(src[count - 1] + (o)));
#if defined(IMAGING_SIMD_AVX2)
        {
            __m256i h = _mm256_set1_epi32(wh), l = _mm256_set1_epi32(wl);
            __m256i b23 = _mm256_set1_epi32(1 << 23);
            for (; i <= n - 32; i += 32) {
                __m256i r[2], c, e;
                int k;
                for (k = 0; k < 2; k++) {
                    /* bytes 0-7 in the low 128-bit lane, 8-15 in the high one */
                    BOX_DIRECT_SUM(_mm256, mm256_cvtepu8_epi16, i + k * 16)
                    BOX_DIRECT_FAR(_mm256, mm256_cvtepu8_epi16, i + k * 16)
                    r[k] = _mm256_packus_epi32(
                        BOX_DIRECT_PAIRS(
                            _mm256, _mm256_unpacklo_epi16(c, e), h, l, b23
                        ),
                        BOX_DIRECT_PAIRS(_mm256, _mm256_unpackhi_epi16(c, e), h, l, b23)
                    );
                }
                _mm256_storeu_si256(
                    (__m256i *)(out + i),
                    _mm256_permute4x64_epi64(_mm256_packus_epi16(r[0], r[1]), 0xd8)
                );
            }
        }
#endif
        {
            __m128i h = _mm_set1_epi32(wh), l = _mm_set1_epi32(wl);
            __m128i b23 = _mm_set1_epi32(1 << 23);
            for (; i <= n - 16; i += 16) {
                __m128i r[2], c, e;
                int k;
                for (k = 0; k < 2; k++) {
                    BOX_DIRECT_SUM(_mm, mm_cvtepu8_epi16, i + k * 8)
                    BOX_DIRECT_FAR(_mm, mm_cvtepu8_epi16, i + k * 8)
                    r[k] = _mm_packus_epi32(
                        BOX_DIRECT_PAIRS(_mm, _mm_unpacklo_epi16(c, e), h, l, b23),
                        BOX_DIRECT_PAIRS(_mm, _mm_unpackhi_epi16(c, e), h, l, b23)
                    );
                }
                _mm_storeu_si128((__m128i *)(out + i), _mm_packus_epi16(r[0], r[1]));
            }
        }
#undef BOX_DIRECT_FAR
#undef BOX_DIRECT_PAIRS
    }
#undef BOX_DIRECT_SUM
#elif defined(IMAGING_SIMD_NEON)
    /* The same with 16 bytes at a time: c, the sum of the box, in two
       halves of 8 16-bit lanes */
#define BOX_DIRECT_SUM(o)                                       \
    {                                                           \
        uint8x16_t v_ = vld1q_u8(src[1] + (o));                 \
        c0 = vmovl_u8(vget_low_u8(v_)), c1 = vmovl_high_u8(v_); \
        for (j = 2; j < count - 1; j++) {                       \
            v_ = vld1q_u8(src[j] + (o));                        \
            c0 = vaddw_u8(c0, vget_low_u8(v_));                 \
            c1 = vaddw_high_u8(c1, v_);                         \
        }                                                       \
    }
    if (d->fast) {
        int16x8_t s = vdupq_n_s16(d->s), t = vdupq_n_s16(-d->t);
        uint16x8_t k = vdupq_n_u16(d->k);
        uint16x4_t m = vdup_n_u16(d->m);
        for (; i <= n - 16; i += 16) {
            uint16x8_t c0, c1, r[2];
            int h;
            BOX_DIRECT_SUM(i)
            for (h = 0; h < 2; h++) {
                uint16x8_t v = vaddq_u16(vshlq_u16(h ? c1 : c0, s), k);
                /* the high halves of the 32-bit products: mulhi */
                uint16x8_t hi = vuzp2q_u16(
                    vreinterpretq_u16_u32(vmull_u16(vget_low_u16(v), m)),
                    vreinterpretq_u16_u32(vmull_high_u16(v, vcombine_u16(m, m)))
                );
                r[h] = vshlq_u16(hi, t);
            }
            vst1q_u8(out + i, vcombine_u8(vmovn_u16(r[0]), vmovn_u16(r[1])));
        }
    } else {
        /* The products exactly, in 32 bits: one instruction each on arm64,
           where splitting the weights as for x86 was slower. vaddhn gives
           (total + 2^23) >> 16. */
        uint32x4_t vww = vdupq_n_u32(ww), vfw = vdupq_n_u32(fw);
        uint32x4_t b23 = vdupq_n_u32(1 << 23);
        for (; i <= n - 16; i += 16) {
            uint16x8_t c0, c1;
            uint8x8_t r[2];
            uint8x16_t e0 = vld1q_u8(src[0] + i), e1 = vld1q_u8(src[count - 1] + i);
            uint16x8_t e[2] = {
                vaddl_u8(vget_low_u8(e0), vget_low_u8(e1)), vaddl_high_u8(e0, e1)
            };
            int h;
            BOX_DIRECT_SUM(i)
            for (h = 0; h < 2; h++) {
                uint16x8_t c = h ? c1 : c0;
                uint32x4_t lo = vmlaq_u32(
                    vmulq_u32(vmovl_u16(vget_low_u16(c)), vww),
                    vmovl_u16(vget_low_u16(e[h])),
                    vfw
                );
                uint32x4_t hi = vmlaq_u32(
                    vmulq_u32(vmovl_high_u16(c), vww), vmovl_high_u16(e[h]), vfw
                );
                r[h] = vshrn_n_u16(
                    vcombine_u16(vaddhn_u32(lo, b23), vaddhn_u32(hi, b23)), 8
                );
            }
            vst1q_u8(out + i, vcombine_u8(r[0], r[1]));
        }
    }
#undef BOX_DIRECT_SUM
#endif
    return i;
}

static int
box_direct_simd(
    UINT8 *restrict out, const UINT8 **src, int n, const BoxDirect *d, int count
) {
    switch (count) {
        case 3:
            return box_direct_simd_body(out, src, n, d, 3);
        case 5:
            return box_direct_simd_body(out, src, n, d, 5);
        case 7:
            return box_direct_simd_body(out, src, n, d, 7);
        case 9:
            return box_direct_simd_body(out, src, n, d, 9);
        case 11:
            return box_direct_simd_body(out, src, n, d, 11);
        default:
            return box_direct_simd_body(out, src, n, d, count);
    }
}

#if defined(IMAGING_SIMD_SSE4)
/* 1-byte images are blurred four rows at a time: byte x of rows 0..3 become
   the four channels of pixel x of a 4-byte row, which the 4-byte line blur
   processes with the same arithmetic per byte. Not on arm64: on the Jetson
   Orin (GCC 11) this was 1-5% slower than the compiler's 1-byte code. */
static void
box_interleave4(
    UINT32 *out, UINT8 *in0, UINT8 *in1, UINT8 *in2, UINT8 *in3, int xsize
) {
    int x = 0;
    for (; x < xsize - 15; x += 16) {
        __m128i a = _mm_loadu_si128((__m128i *)&in0[x]);
        __m128i b = _mm_loadu_si128((__m128i *)&in1[x]);
        __m128i c = _mm_loadu_si128((__m128i *)&in2[x]);
        __m128i d = _mm_loadu_si128((__m128i *)&in3[x]);
        __m128i ab_lo = _mm_unpacklo_epi8(a, b), ab_hi = _mm_unpackhi_epi8(a, b);
        __m128i cd_lo = _mm_unpacklo_epi8(c, d), cd_hi = _mm_unpackhi_epi8(c, d);
        _mm_storeu_si128((__m128i *)&out[x], _mm_unpacklo_epi16(ab_lo, cd_lo));
        _mm_storeu_si128((__m128i *)&out[x + 4], _mm_unpackhi_epi16(ab_lo, cd_lo));
        _mm_storeu_si128((__m128i *)&out[x + 8], _mm_unpacklo_epi16(ab_hi, cd_hi));
        _mm_storeu_si128((__m128i *)&out[x + 12], _mm_unpackhi_epi16(ab_hi, cd_hi));
    }
    for (; x < xsize; x++) {
        out[x] = MAKE_UINT32(in0[x], in1[x], in2[x], in3[x]);
    }
}

static void
box_deinterleave4(
    UINT8 *out0, UINT8 *out1, UINT8 *out2, UINT8 *out3, UINT32 *in, int xsize
) {
    int x = 0;
    __m128i by_row =
        _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    for (; x < xsize - 15; x += 16) {
        // [32] 4 pixels of row 3, row 2, row 1, row 0
        __m128i v0 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)&in[x]), by_row);
        __m128i v1 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)&in[x + 4]), by_row);
        __m128i v2 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)&in[x + 8]), by_row);
        __m128i v3 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)&in[x + 12]), by_row);
        __m128i t0 = _mm_unpacklo_epi32(v0, v1), t1 = _mm_unpacklo_epi32(v2, v3);
        __m128i t2 = _mm_unpackhi_epi32(v0, v1), t3 = _mm_unpackhi_epi32(v2, v3);
        _mm_storeu_si128((__m128i *)&out0[x], _mm_unpacklo_epi64(t0, t1));
        _mm_storeu_si128((__m128i *)&out1[x], _mm_unpackhi_epi64(t0, t1));
        _mm_storeu_si128((__m128i *)&out2[x], _mm_unpacklo_epi64(t2, t3));
        _mm_storeu_si128((__m128i *)&out3[x], _mm_unpackhi_epi64(t2, t3));
    }
    for (; x < xsize; x++) {
        UINT8 *p = (UINT8 *)&in[x];
        out0[x] = p[0];
        out1[x] = p[1];
        out2[x] = p[2];
        out3[x] = p[3];
    }
}
#endif

#if defined(IMAGING_SIMD_SSE4)
/* The vertical pass's running sums (BoxBlur.c, ImagingBoxBlurColumns). */
static int
box_columns_simd(
    UINT8 *out,
    UINT32 *acc,
    const UINT8 *add,
    const UINT8 *sub,
    const UINT8 *far_row,
    int size,
    UINT32 ww,
    UINT32 fw
) {
    int b = 0;
#if defined(IMAGING_SIMD_AVX2)
    {
        __m256i wws = _mm256_set1_epi32(ww);
        __m256i fws = _mm256_set1_epi32(fw);
        __m256i b23 = _mm256_set1_epi32(1 << 23);
        /* packs and packus work within 128-bit lanes: this puts the
           four groups of 8 results back in order. */
        __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (; b < size - 31; b += 32) {
            __m256i r[4];
#define COLUMNS_8(k)                                                \
    {                                                               \
        __m256i a = mm256_cvtepu8_epi32(&add[b + k * 8]);           \
        __m256i s = mm256_cvtepu8_epi32(&sub[b + k * 8]);           \
        __m256i f = mm256_cvtepu8_epi32(&far_row[b + k * 8]);       \
        __m256i c = _mm256_loadu_si256((__m256i *)&acc[b + k * 8]); \
        c = _mm256_sub_epi32(_mm256_add_epi32(c, a), s);            \
        _mm256_storeu_si256((__m256i *)&acc[b + k * 8], c);         \
        c = _mm256_add_epi32(                                       \
            _mm256_mullo_epi32(c, wws),                             \
            _mm256_mullo_epi32(_mm256_add_epi32(s, f), fws)         \
        );                                                          \
        r[k] = _mm256_srli_epi32(_mm256_add_epi32(c, b23), 24);     \
    }
            COLUMNS_8(0);
            COLUMNS_8(1);
            COLUMNS_8(2);
            COLUMNS_8(3);
#undef COLUMNS_8
            __m256i v = _mm256_packus_epi16(
                _mm256_packs_epi32(r[0], r[1]), _mm256_packs_epi32(r[2], r[3])
            );
            _mm256_storeu_si256(
                (__m256i *)&out[b], _mm256_permutevar8x32_epi32(v, order)
            );
        }
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    __m128i wws = _mm_set1_epi32(ww);
    __m128i fws = _mm_set1_epi32(fw);
    __m128i b23 = _mm_set1_epi32(1 << 23);
    for (; b < size - 15; b += 16) {
        __m128i add8 = _mm_loadu_si128((__m128i *)&add[b]);
        __m128i sub8 = _mm_loadu_si128((__m128i *)&sub[b]);
        __m128i far8 = _mm_loadu_si128((__m128i *)&far_row[b]);
        __m128i r[4];
#define COLUMNS_4(k)                                                           \
    {                                                                          \
        __m128i a = _mm_cvtepu8_epi32(_mm_srli_si128(add8, k * 4));            \
        __m128i s = _mm_cvtepu8_epi32(_mm_srli_si128(sub8, k * 4));            \
        __m128i f = _mm_cvtepu8_epi32(_mm_srli_si128(far8, k * 4));            \
        __m128i c = _mm_loadu_si128((__m128i *)&acc[b + k * 4]);               \
        c = _mm_sub_epi32(_mm_add_epi32(c, a), s);                             \
        _mm_storeu_si128((__m128i *)&acc[b + k * 4], c);                       \
        c = _mm_add_epi32(                                                     \
            _mm_mullo_epi32(c, wws), _mm_mullo_epi32(_mm_add_epi32(s, f), fws) \
        );                                                                     \
        r[k] = _mm_srli_epi32(_mm_add_epi32(c, b23), 24);                      \
    }
        COLUMNS_4(0);
        COLUMNS_4(1);
        COLUMNS_4(2);
        COLUMNS_4(3);
#undef COLUMNS_4
        _mm_storeu_si128(
            (__m128i *)&out[b],
            _mm_packus_epi16(_mm_packs_epi32(r[0], r[1]), _mm_packs_epi32(r[2], r[3]))
        );
    }
#endif
    return b;
}
#endif

static const ImagingSIMDBlurKernels blur_simd_kernels = {
    .line32 = box_line_blur32,
#if defined(IMAGING_SIMD_AVX2)
    .line32x2 = box_line_blur32x2,
#endif
    .direct = box_direct_simd,
#if defined(IMAGING_SIMD_SSE4)
    .interleave4 = box_interleave4,
    .deinterleave4 = box_deinterleave4,
    .columns = box_columns_simd,
#endif
};
