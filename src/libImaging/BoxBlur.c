#include "Imaging.h"
#include "ImagingPx4.h"

typedef UINT8 pixel[4];

/* The running-sum box blur of a row of 4-byte pixels, for any vector
   type: the caller defines V (the type), V_LOAD(i) (pixel i as four
   32-bit channels), V_SET1, V_ADD, V_SUB, V_MUL and V_SAVE(x, bulk). */
#define BOX_LINE_BLUR                                                                 \
    {                                                                                 \
        int x;                                                                        \
        V acc, bulk;                                                                  \
        V wws = V_SET1(ww);                                                           \
        V fws = V_SET1(fw);                                                           \
        /* Compute acc for -1 pixel (outside of image): from "-radius-1" to */        \
        /* "-1" get first pixel, then from "0" to "radius-1". */                      \
        acc = V_MUL(V_LOAD(0), V_SET1(radius + 1));                                   \
        /* As radius can be bigger than xsize, iterate to edgeA -1. */                \
        for (x = 0; x < edgeA - 1; x++) {                                             \
            acc = V_ADD(acc, V_LOAD(x));                                              \
        }                                                                             \
        /* Then multiply remainder to last x. */                                      \
        acc = V_ADD(acc, V_MUL(V_LOAD(x), V_SET1(radius - edgeA + 1)));               \
        if (edgeA <= edgeB) {                                                         \
            /* Subtract pixel from left ("0"). Add pixels from radius. */             \
            for (x = 0; x < edgeA; x++) {                                             \
                BOX_STEP(0, x + radius, 0, x + radius + 1);                           \
            }                                                                         \
            /* Subtract previous pixel from "-radius". Add from radius. */            \
            for (x = edgeA; x < edgeB; x++) {                                         \
                BOX_STEP(x - radius - 1, x + radius, x - radius - 1, x + radius + 1); \
            }                                                                         \
            /* Subtract previous pixel from "-radius". Add last pixel. */             \
            for (x = edgeB; x <= lastx; x++) {                                        \
                BOX_STEP(x - radius - 1, lastx, x - radius - 1, lastx);               \
            }                                                                         \
        } else {                                                                      \
            for (x = 0; x < edgeB; x++) {                                             \
                BOX_STEP(0, x + radius, 0, x + radius + 1);                           \
            }                                                                         \
            for (x = edgeB; x < edgeA; x++) {                                         \
                BOX_STEP(0, lastx, 0, lastx);                                         \
            }                                                                         \
            for (x = edgeA; x <= lastx; x++) {                                        \
                BOX_STEP(x - radius - 1, lastx, x - radius - 1, lastx);               \
            }                                                                         \
        }                                                                             \
    }

/* acc[0] += lineIn[add][0] - lineIn[subtract][0];
   bulk[0] = (acc[0] * ww) + (lineIn[left][0] + lineIn[right][0]) * fw;
   lineOut[x][0] = (UINT8)((bulk[0] + (1 << 23)) >> 24); */
#define BOX_STEP(subtract, add, left, right)                                       \
    acc = V_SUB(V_ADD(acc, V_LOAD(add)), V_LOAD(subtract));                        \
    bulk = V_ADD(V_MUL(V_ADD(V_LOAD(left), V_LOAD(right)), fws), V_MUL(acc, wws)); \
    V_SAVE(x, bulk);

void static inline ImagingLineBoxBlur32(
    UINT32 *lineOut,
    UINT32 *lineIn,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
) {
#define V px4
#define V_LOAD(i) px4_load(&lineIn[i])
#define V_SET1 px4_set1
#define V_ADD px4_add
#define V_SUB px4_sub
#define V_MUL px4_mul
#define V_SAVE(x, bulk) lineOut[x] = px4_round24(bulk)
    BOX_LINE_BLUR
#undef V
#undef V_LOAD
#undef V_SET1
#undef V_ADD
#undef V_SUB
#undef V_MUL
#undef V_SAVE
}

#if defined(IMAGING_SIMD_AVX2)
/* Two rows at once, one in each 128-bit half: the same arithmetic per
   channel with half the instructions. */
static inline __m256i
px8_load(const UINT32 *a, const UINT32 *b) {
    return _mm256_cvtepu8_epi32(
        _mm_unpacklo_epi32(_mm_cvtsi32_si128(*a), _mm_cvtsi32_si128(*b))
    );
}

void static inline ImagingLineBoxBlur32x2(
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

/* Up to this radius a pass may use ImagingBoxBlurDirect instead of running
   sums: always for radius 0, and with SIMD for radius 1, and to
   BOX_DIRECT_MAX where the fast form below applies. */
#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
#define BOX_DIRECT_SIMD
#define BOX_DIRECT_MAX 4
#else
#define BOX_DIRECT_MAX 0
#endif

/* How ImagingBoxBlurDirect computes a pass: from count = 2 * radius + 3
   sources with the weights ww and fw, or, if fast, from the sum c of the
   box alone, out = (((c << s) + k) * m >> 16) >> t in 16 bits. That needs
   fw = 0, which integer radii from 1 to 4 have, and the result equals the
   one of the full arithmetic for every sum (checked in BoxDirectInit). */
typedef struct {
    int direct, count;
    UINT32 ww, fw;
    int fast, s, t, m, k;
} BoxDirect;

static void
BoxDirectInit(BoxDirect *d, int radius, UINT32 ww, UINT32 fw) {
#if defined(BOX_DIRECT_SIMD)
    /* s, t, m and k for the integer radii 1 to 4, found by a search
       around m = ww * 2^(t - s - 8) and k = 2^(15 + t) / m. */
    static const unsigned short fast[4][4] = {
        {6, 0, 341, 112}, {5, 1, 819, 80}, {5, 1, 585, 112}, {4, 0, 455, 73}
    };
#endif
    d->count = 2 * radius + 3;
    d->ww = ww;
    d->fw = fw;
    d->fast = 0;
    d->direct = radius == 0;
#if defined(BOX_DIRECT_SIMD)
    if (radius >= 1 && radius <= BOX_DIRECT_MAX && fw == 0) {
        const unsigned short *f = fast[radius - 1];
        int c, cmax = 255 * (2 * radius + 1);
        for (c = 0; c <= cmax; c++) {
            UINT32 v = ((((UINT32)c << f[0]) + f[3]) * f[2]) >> (16 + f[1]);
            if (v != (c * ww + (1 << 23)) >> 24) {
                break;
            }
        }
        if (c > cmax && (cmax << f[0]) + f[3] <= 0xffff) {
            d->fast = 1;
            d->s = f[0], d->t = f[1], d->m = f[2], d->k = f[3];
        }
    }
    d->direct = radius <= 1 || d->fast;
#endif
}

/* Radius below 1 without SIMD: in chunks of a fixed length, with restrict
   pointers, which GCC vectorizes at -O2 too. */
#define BOX_ZERO_CHUNK 32
static inline void
ImagingBoxBlurZeroChunk(
    UINT8 *restrict out,
    const UINT8 *restrict in,
    const UINT8 *restrict before,
    const UINT8 *restrict after,
    UINT32 ww,
    UINT32 fw
) {
    for (int i = 0; i < BOX_ZERO_CHUNK; i++) {
        UINT32 bulk = in[i] * ww + (before[i] + after[i]) * fw;
        out[i] = (UINT8)((bulk + (1 << 23)) >> 24);
    }
}

/* The box around each byte directly, from d->count sources:
   out[i] = (src[1][i] + ... + src[count - 2][i]) * ww
            + (src[0][i] + src[count - 1][i]) * fw, rounded,
   the same arithmetic as the running sums. The sources are rows (vertical
   passes) or one row shifted by whole pixels (horizontal passes). With no
   running sum, it vectorizes across the row. */
#if defined(__GNUC__) || defined(__clang__)
#define BOX_DIRECT_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define BOX_DIRECT_INLINE static __forceinline
#else
#define BOX_DIRECT_INLINE static inline
#endif
BOX_DIRECT_INLINE void
box_direct(
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
    if (count == 3) {
        for (; i <= n - BOX_ZERO_CHUNK; i += BOX_ZERO_CHUNK) {
            ImagingBoxBlurZeroChunk(
                out + i, src[1] + i, src[0] + i, src[2] + i, ww, fw
            );
        }
    }
    for (; i < n; i++) {
        UINT32 c = 0;
        for (j = 1; j < count - 1; j++) {
            c += src[j][i];
        }
        UINT32 bulk = c * ww + (src[0][i] + src[count - 1][i]) * fw;
        out[i] = (UINT8)((bulk + (1 << 23)) >> 24);
    }
}

/* With the count of sources a constant for each radius, so the compiler
   unrolls the sums and keeps the source pointers in registers: 1.46 times
   as fast at radius 2 on x86 (at -O3), where the loop over the sources
   reloaded each pointer for every vector. */
static void
ImagingBoxBlurDirect(
    UINT8 *restrict out, const UINT8 **src, int n, const BoxDirect *d
) {
    switch (d->count) {
        case 3:
            box_direct(out, src, n, d, 3);
            break;
        case 5:
            box_direct(out, src, n, d, 5);
            break;
#if BOX_DIRECT_MAX >= 2
        case 7:
            box_direct(out, src, n, d, 7);
            break;
        case 9:
            box_direct(out, src, n, d, 9);
            break;
        case 11:
            box_direct(out, src, n, d, 11);
            break;
#endif
        default:
            box_direct(out, src, n, d, d->count);
    }
}
#undef BOX_DIRECT_INLINE

/* A row of n bytes, step bytes per pixel, edge pixels repeated. The middle
   reads the row itself, the first and last radius + 1 pixels a copy with
   the edge pixel repeated, in pad, which has room for the row and radius +
   1 more pixels on either side (or the whole row, if it is short). The
   running sums give the same result: they treat the edges as repeated
   pixels too, for any radius. */
static void
ImagingLineBoxBlurDirect(
    UINT8 *restrict out,
    const UINT8 *restrict in,
    int n,
    int step,
    const BoxDirect *d,
    UINT8 *restrict pad
) {
    const UINT8 *src[2 * BOX_DIRECT_MAX + 3];
    int edge = (d->count - 1) / 2 * step, j;
    if (n < 2 * edge) {
        for (j = 0; j < edge; j += step) {
            memcpy(pad + j, in, step);
            memcpy(pad + edge + n + j, in + n - step, step);
        }
        memcpy(pad + edge, in, n);
        for (j = 0; j < d->count; j++) {
            src[j] = pad + j * step;
        }
        ImagingBoxBlurDirect(out, src, n, d);
        return;
    }
    /* first edge bytes: pad holds edge repeated bytes, then 2 * edge */
    for (j = 0; j < edge; j += step) {
        memcpy(pad + j, in, step);
    }
    memcpy(pad + edge, in, 2 * edge);
    for (j = 0; j < d->count; j++) {
        src[j] = pad + j * step;
    }
    ImagingBoxBlurDirect(out, src, edge, d);
    /* the middle */
    for (j = 0; j < d->count; j++) {
        src[j] = in + j * step;
    }
    ImagingBoxBlurDirect(out + edge, src, n - 2 * edge, d);
    /* last edge bytes: 2 * edge, then edge repeated bytes */
    memcpy(pad, in + n - 2 * edge, 2 * edge);
    for (j = 0; j < edge; j += step) {
        memcpy(pad + 2 * edge + j, in + n - step, step);
    }
    for (j = 0; j < d->count; j++) {
        src[j] = pad + j * step;
    }
    ImagingBoxBlurDirect(out + n - edge, src, edge, d);
}

void static inline ImagingLineBoxBlur8(
    UINT8 *lineOut,
    UINT8 *lineIn,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
) {
    int x;
    UINT32 acc;
    UINT32 bulk;

#define MOVE_ACC(acc, subtract, add) acc += lineIn[add] - lineIn[subtract];

#define ADD_FAR(bulk, acc, left, right) \
    bulk = (acc * ww) + (lineIn[left] + lineIn[right]) * fw;

#define SAVE(x, bulk) lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24)

    acc = lineIn[0] * (radius + 1);
    for (x = 0; x < edgeA - 1; x++) {
        acc += lineIn[x];
    }
    acc += lineIn[lastx] * (radius - edgeA + 1);

    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}

#if defined(IMAGING_SIMD_SSE4)
/* 1-byte images are blurred four rows at a time: byte x of rows 0..3 become
   the four channels of pixel x of a 4-byte row, which the 4-byte line blur
   processes with the same arithmetic per byte. Not on arm64: on the Jetson
   Orin (GCC 11) this was 1-5% slower than the compiler's 1-byte code. */
static void
interleave4(UINT32 *out, UINT8 *in0, UINT8 *in1, UINT8 *in2, UINT8 *in3, int xsize) {
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
deinterleave4(
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

/* passes horizontal box blurs. All passes run on one row before the next
   row, so the row stays in cache: the same arithmetic as blurring the
   whole image once per pass, with one read and one write of the image.
   imOut may be imIn. */
static Imaging
ImagingHorizontalBoxBlur(Imaging imOut, Imaging imIn, float floatRadius, int passes) {
    ImagingSectionCookie cookie;

    int y, p;

    int radius = (int)floatRadius;
    UINT32 ww = (UINT32)(1 << 24) / (floatRadius * 2 + 1);
    UINT32 fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2;

    int lastx = imIn->xsize - 1;
    int edgeA = MIN(radius + 1, imIn->xsize);
    int edgeB = MAX(imIn->xsize - radius - 1, 0);

    BoxDirect direct;
    BoxDirectInit(&direct, radius, ww, fw);

    /* Two row buffers the passes alternate between, and one for
       ImagingLineBoxBlurDirect. */
    UINT32 *buf[2];
    UINT8 *pad;
    buf[0] = calloc(imIn->xsize, sizeof(UINT32));
    buf[1] = calloc(imIn->xsize, sizeof(UINT32));
    pad = malloc((imIn->xsize + 2 * (BOX_DIRECT_MAX + 1)) * sizeof(UINT32));
    if (buf[0] == NULL || buf[1] == NULL || pad == NULL) {
        free(buf[0]);
        free(buf[1]);
        free(pad);
        return ImagingError_MemoryError();
    }

/* Runs the passes from row in: each one reads src and writes the other
   buffer, except the last one, which writes out unless it reads it. */
#define BLUR_PASSES(type, in, out, blur)                               \
    {                                                                  \
        type *src_ = (type *)(in);                                     \
        for (p = 0; p < passes; p++) {                                 \
            type *dst_ = (p == passes - 1 && src_ != (type *)(out))    \
                             ? (type *)(out)                           \
                             : (type *)buf[p % 2];                     \
            if (direct.direct) {                                       \
                ImagingLineBoxBlurDirect(                              \
                    (UINT8 *)dst_,                                     \
                    (UINT8 *)src_,                                     \
                    imIn->xsize * (int)sizeof(type),                   \
                    sizeof(type),                                      \
                    &direct,                                           \
                    pad                                                \
                );                                                     \
            } else {                                                   \
                blur(dst_, src_, lastx, radius, edgeA, edgeB, ww, fw); \
            }                                                          \
            src_ = dst_;                                               \
        }                                                              \
        if (src_ != (type *)(out)) {                                   \
            memcpy((out), src_, imIn->xsize * sizeof(type));           \
        }                                                              \
    }

#if defined(IMAGING_SIMD_AVX2)
    /* The same for two rows of 4-byte pixels at once. */
    UINT32 *pair[2][2] = {{NULL, NULL}, {NULL, NULL}};
#define BLUR_PASSES_2(in0, in1, out0, out1)                                 \
    {                                                                       \
        UINT32 *s0_ = (in0), *s1_ = (in1);                                  \
        for (p = 0; p < passes; p++) {                                      \
            int last_ = p == passes - 1;                                    \
            UINT32 *d0_ = last_ && s0_ != (out0) ? (out0) : pair[p % 2][0]; \
            UINT32 *d1_ = last_ && s1_ != (out1) ? (out1) : pair[p % 2][1]; \
            if (direct.direct) {                                            \
                int n_ = imIn->xsize * 4;                                   \
                ImagingLineBoxBlurDirect(                                   \
                    (UINT8 *)d0_, (UINT8 *)s0_, n_, 4, &direct, pad         \
                );                                                          \
                ImagingLineBoxBlurDirect(                                   \
                    (UINT8 *)d1_, (UINT8 *)s1_, n_, 4, &direct, pad         \
                );                                                          \
            } else {                                                        \
                ImagingLineBoxBlur32x2(                                     \
                    d0_, d1_, s0_, s1_, lastx, radius, edgeA, edgeB, ww, fw \
                );                                                          \
            }                                                               \
            s0_ = d0_;                                                      \
            s1_ = d1_;                                                      \
        }                                                                   \
        if (s0_ != (out0)) {                                                \
            memcpy((out0), s0_, imIn->xsize * sizeof(UINT32));              \
        }                                                                   \
        if (s1_ != (out1)) {                                                \
            memcpy((out1), s1_, imIn->xsize * sizeof(UINT32));              \
        }                                                                   \
    }
    for (p = 0; p < 4; p++) {
        pair[p / 2][p % 2] = malloc(imIn->xsize * sizeof(UINT32));
        if (!pair[p / 2][p % 2]) {
            for (p = 0; p < 4; p++) {
                free(pair[p / 2][p % 2]);
            }
            free(buf[0]);
            free(buf[1]);
            free(pad);
            return ImagingError_MemoryError();
        }
    }
#endif

    ImagingSectionEnter(&cookie);

    if (imIn->image8) {
        y = 0;
#if defined(IMAGING_SIMD_SSE4)
        UINT32 *lineIn4 = malloc(imIn->xsize * sizeof(UINT32));
        UINT32 *lineOut4 = malloc(imIn->xsize * sizeof(UINT32));
        UINT32 *lineIn4b = malloc(imIn->xsize * sizeof(UINT32));
        UINT32 *lineOut4b = malloc(imIn->xsize * sizeof(UINT32));
        if (!lineIn4 || !lineOut4 || !lineIn4b || !lineOut4b) {
            ImagingSectionLeave(&cookie);
            free(lineIn4);
            free(lineOut4);
            free(lineIn4b);
            free(lineOut4b);
            free(buf[0]);
            free(buf[1]);
            free(pad);
#if defined(IMAGING_SIMD_AVX2)
            for (p = 0; p < 4; p++) {
                free(pair[p / 2][p % 2]);
            }
#endif
            return ImagingError_MemoryError();
        }
#if defined(IMAGING_SIMD_AVX2)
        // Eight rows: two groups of four, each a row of 4-byte pixels.
        for (; y < imIn->ysize - 7; y += 8) {
            interleave4(
                lineIn4,
                imIn->image8[y],
                imIn->image8[y + 1],
                imIn->image8[y + 2],
                imIn->image8[y + 3],
                imIn->xsize
            );
            interleave4(
                lineIn4b,
                imIn->image8[y + 4],
                imIn->image8[y + 5],
                imIn->image8[y + 6],
                imIn->image8[y + 7],
                imIn->xsize
            );
            BLUR_PASSES_2(lineIn4, lineIn4b, lineOut4, lineOut4b);
            deinterleave4(
                imOut->image8[y],
                imOut->image8[y + 1],
                imOut->image8[y + 2],
                imOut->image8[y + 3],
                lineOut4,
                imIn->xsize
            );
            deinterleave4(
                imOut->image8[y + 4],
                imOut->image8[y + 5],
                imOut->image8[y + 6],
                imOut->image8[y + 7],
                lineOut4b,
                imIn->xsize
            );
        }
#endif
        // All four input rows are read before any output row is written,
        // so imOut may be imIn.
        for (; y < imIn->ysize - 3; y += 4) {
            interleave4(
                lineIn4,
                imIn->image8[y],
                imIn->image8[y + 1],
                imIn->image8[y + 2],
                imIn->image8[y + 3],
                imIn->xsize
            );
            BLUR_PASSES(UINT32, lineIn4, lineOut4, ImagingLineBoxBlur32);
            deinterleave4(
                imOut->image8[y],
                imOut->image8[y + 1],
                imOut->image8[y + 2],
                imOut->image8[y + 3],
                lineOut4,
                imIn->xsize
            );
        }
        free(lineIn4);
        free(lineOut4);
        free(lineIn4b);
        free(lineOut4b);
#endif
        for (; y < imIn->ysize; y++) {
            BLUR_PASSES(UINT8, imIn->image8[y], imOut->image8[y], ImagingLineBoxBlur8);
        }
    } else {
        y = 0;
#if defined(IMAGING_SIMD_AVX2)
        for (; y < imIn->ysize - 1; y += 2) {
            BLUR_PASSES_2(
                (UINT32 *)imIn->image32[y],
                (UINT32 *)imIn->image32[y + 1],
                (UINT32 *)imOut->image32[y],
                (UINT32 *)imOut->image32[y + 1]
            );
        }
#endif
        for (; y < imIn->ysize; y++) {
            BLUR_PASSES(
                UINT32, imIn->image32[y], imOut->image32[y], ImagingLineBoxBlur32
            );
        }
    }
#undef BLUR_PASSES

    ImagingSectionLeave(&cookie);

    free(buf[0]);
    free(buf[1]);
    free(pad);
#if defined(IMAGING_SIMD_AVX2)
#undef BLUR_PASSES_2
    for (p = 0; p < 4; p++) {
        free(pair[p / 2][p % 2]);
    }
#endif

    return imOut;
}

/* One output row of the vertical pass: acc holds a running sum per byte,
   updated with the rows entering and leaving the window. Same arithmetic
   per channel as the horizontal pass. */
static inline void
ImagingBoxBlurColumns(
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

    for (; b < size; b++) {
        acc[b] += add[b] - sub[b];
        UINT32 bulk = acc[b] * ww + (sub[b] + far_row[b]) * fw;
        out[b] = (UINT8)((bulk + (1 << 23)) >> 24);
    }
}

/* Blurs columns: each byte is averaged with the bytes above and below it,
   with edge rows repeated. This gives the same result as transposing,
   blurring rows and transposing back, but reads rows sequentially and is
   vectorised across the row. imOut and imIn must be distinct. */
static Imaging
ImagingVerticalBoxBlur(Imaging imOut, Imaging imIn, float floatRadius) {
    ImagingSectionCookie cookie;
    int y, i, b;

    int radius = (int)floatRadius;
    UINT32 ww = (UINT32)(1 << 24) / (floatRadius * 2 + 1);
    UINT32 fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2;
    int lasty = imIn->ysize - 1;
    int size = imIn->linesize;
    BoxDirect direct;
    BoxDirectInit(&direct, radius, ww, fw);

    UINT32 *acc = calloc(size, sizeof(UINT32));
    if (acc == NULL) {
        return ImagingError_MemoryError();
    }

/* Row index i (64-bit, as y + radius can exceed INT_MAX) clamped to the image. */
#define ROW(i) ((UINT8 *)imIn->image[(i) < 0 ? 0 : ((i) > lasty ? lasty : (int)(i))])

    ImagingSectionEnter(&cookie);

    /* Window for row -1: rows -radius-1 .. radius-1. The radius + 1 rows
       before the image repeat row 0 and those after it repeat the last row,
       so they are added by multiplication, and the cost doesn't grow with
       the radius. The sums are the same modulo 2^32. */
    {
        int inside = radius - 1 < lasty ? radius - 1 : lasty;
        UINT32 after = radius - 1 > lasty ? (UINT32)(radius - 1 - lasty) : 0;
        UINT8 *first = ROW(0), *last = ROW(lasty);
        for (b = 0; b < size; b++) {
            acc[b] = first[b] * (UINT32)(radius + 1) + last[b] * after;
        }
        for (i = 0; i <= inside; i++) {
            UINT8 *row = ROW(i);
            for (b = 0; b < size; b++) {
                acc[b] += row[b];
            }
        }
    }
    for (y = 0; y <= lasty; y++) {
        long long yl = y;
        if (direct.direct) {
            /* No running sum: the rows of the box directly. */
            const UINT8 *src[2 * BOX_DIRECT_MAX + 3];
            for (i = 0; i < direct.count; i++) {
                src[i] = ROW(yl - radius - 1 + i);
            }
            ImagingBoxBlurDirect((UINT8 *)imOut->image[y], src, size, &direct);
            continue;
        }
        ImagingBoxBlurColumns(
            (UINT8 *)imOut->image[y],
            acc,
            ROW(yl + radius),
            ROW(yl - radius - 1),
            ROW(yl + radius + 1),
            size,
            ww,
            fw
        );
    }

    ImagingSectionLeave(&cookie);

#undef ROW

    free(acc);
    return imOut;
}

/* passes vertical box blurs in one sweep down the image: pass k keeps the
   rows pass k + 1 still needs in a ring of 2 * radius + 3 rows, and only
   the last pass writes imOut. The same arithmetic as running the passes
   one after the other, with one read and one write of the image instead
   of one per pass. */
typedef struct {
    UINT32 *acc; /* running sums of the pass */
    UINT8 *ring; /* its output rows, row i in slot i % ringRows */
    int next;    /* next output row to compute */
} BoxBlurStage;

typedef struct {
    Imaging imOut, imIn;
    BoxBlurStage *stage;
    int passes, radius, lasty, size, ringRows;
    UINT32 ww, fw;
    BoxDirect direct;
} BoxBlurCascade;

/* Row i of the input of pass k, clamped to the image. */
static inline UINT8 *
BoxBlurInputRow(BoxBlurCascade *c, int k, long long i) {
    int row = i < 0 ? 0 : (i > c->lasty ? c->lasty : (int)i);
    if (k == 0) {
        return (UINT8 *)c->imIn->image[row];
    }
    return c->stage[k - 1].ring + (size_t)(row % c->ringRows) * c->size;
}

/* Computes the rows of pass k up to row upto, and those of the earlier
   passes they need. */
static void
BoxBlurEnsure(BoxBlurCascade *c, int k, int upto) {
    BoxBlurStage *st = &c->stage[k];
    int radius = c->radius, lasty = c->lasty, size = c->size, b, i;
    while (st->next <= upto) {
        long long y = st->next;
        if (y == 0) {
            /* Window for row -1, as in ImagingVerticalBoxBlur. */
            int inside = radius - 1 < lasty ? radius - 1 : lasty;
            UINT32 after = radius - 1 > lasty ? (UINT32)(radius - 1 - lasty) : 0;
            UINT8 *first, *last;
            if (k > 0) {
                BoxBlurEnsure(c, k - 1, after ? lasty : inside < 0 ? 0 : inside);
            }
            first = BoxBlurInputRow(c, k, 0);
            for (b = 0; b < size; b++) {
                st->acc[b] = first[b] * (UINT32)(radius + 1);
            }
            if (after) {
                last = BoxBlurInputRow(c, k, lasty);
                for (b = 0; b < size; b++) {
                    st->acc[b] += last[b] * after;
                }
            }
            for (i = 0; i <= inside; i++) {
                UINT8 *row = BoxBlurInputRow(c, k, i);
                for (b = 0; b < size; b++) {
                    st->acc[b] += row[b];
                }
            }
        }
        if (k > 0) {
            BoxBlurEnsure(
                c, k - 1, y + radius + 1 < lasty ? (int)(y + radius + 1) : lasty
            );
        }
        UINT8 *out = k == c->passes - 1 ? (UINT8 *)c->imOut->image[y]
                                        : st->ring + (size_t)(y % c->ringRows) * size;
        if (c->direct.direct) {
            /* As in ImagingVerticalBoxBlur. */
            const UINT8 *src[2 * BOX_DIRECT_MAX + 3];
            for (i = 0; i < c->direct.count; i++) {
                src[i] = BoxBlurInputRow(c, k, y - radius - 1 + i);
            }
            ImagingBoxBlurDirect(out, src, size, &c->direct);
        } else {
            ImagingBoxBlurColumns(
                out,
                st->acc,
                BoxBlurInputRow(c, k, y + radius),
                BoxBlurInputRow(c, k, y - radius - 1),
                BoxBlurInputRow(c, k, y + radius + 1),
                size,
                c->ww,
                c->fw
            );
        }
        st->next++;
    }
}

/* Largest total size of the rings; above it, passes run one by one. */
#define BOXBLUR_RINGS_MAX (8 * 1024 * 1024)
/* the ring of a fused blur, kept small enough to stay in cache */
#define BOXBLUR_FUSED_MAX (4 * 1024 * 1024)

/* Whether ImagingVerticalBoxBlurPasses should run the passes. */
static int
BoxBlurCascadeFits(Imaging im, float floatRadius, int passes) {
    long long ringRows = 2 * (long long)(int)floatRadius + 3;
    return passes > 1 && ringRows * im->linesize * (passes - 1) <= BOXBLUR_RINGS_MAX;
}

/* Returns imOut, or NULL if out of memory. imOut and imIn must be
   distinct. */
static Imaging
ImagingVerticalBoxBlurPasses(
    Imaging imOut, Imaging imIn, float floatRadius, int passes
) {
    ImagingSectionCookie cookie;
    BoxBlurCascade c;
    int k, ok = 1;
    long long ringRows = 2 * (long long)(int)floatRadius + 3;

    c.imOut = imOut;
    c.imIn = imIn;
    c.passes = passes;
    c.radius = (int)floatRadius;
    c.ww = (UINT32)(1 << 24) / (floatRadius * 2 + 1);
    c.fw = ((1 << 24) - (c.radius * 2 + 1) * c.ww) / 2;
    BoxDirectInit(&c.direct, c.radius, c.ww, c.fw);
    c.lasty = imIn->ysize - 1;
    c.size = imIn->linesize;
    c.ringRows = (int)ringRows;
    c.stage = calloc(passes, sizeof(BoxBlurStage));
    if (!c.stage) {
        return NULL;
    }
    for (k = 0; k < passes; k++) {
        c.stage[k].acc = calloc(c.size, sizeof(UINT32));
        if (k < passes - 1) {
            c.stage[k].ring = malloc((size_t)c.ringRows * c.size);
        }
        if (!c.stage[k].acc || (k < passes - 1 && !c.stage[k].ring)) {
            ok = 0;
        }
    }
    if (ok) {
        ImagingSectionEnter(&cookie);
        BoxBlurEnsure(&c, passes - 1, c.lasty);
        ImagingSectionLeave(&cookie);
    }
    for (k = 0; k < passes; k++) {
        free(c.stage[k].acc);
        free(c.stage[k].ring);
    }
    free(c.stage);
    return ok ? imOut : NULL;
}

/* The rows of a fused blur: row k of imIn blurred horizontally into slot
   k % rows of a ring the first time it is asked for. */
typedef struct {
    Imaging im;
    UINT8 *ring, *pad;
    int rows, made, n;
    const BoxDirect *d;
    int radius, lastx, edgeA, edgeB;
    UINT32 ww, fw;
} BoxFusedRows;

static inline const UINT8 *
box_fused_row(BoxFusedRows *f, long long k) {
    const int lasty = f->im->ysize - 1;
    k = k < 0 ? 0 : (k > lasty ? lasty : k);
    while (f->made < k) {
        UINT8 *dst, *src;
#if defined(IMAGING_SIMD_AVX2)
        /* running sums two rows at a time; the ring has a row to spare */
        if (!f->d->direct && f->made + 2 <= lasty) {
            UINT8 *dst2 = f->ring + (size_t)((f->made + 2) % f->rows) * f->im->linesize;
            dst = f->ring + (size_t)((f->made + 1) % f->rows) * f->im->linesize;
            ImagingLineBoxBlur32x2(
                (UINT32 *)dst,
                (UINT32 *)dst2,
                (UINT32 *)f->im->image[f->made + 1],
                (UINT32 *)f->im->image[f->made + 2],
                f->lastx,
                f->radius,
                f->edgeA,
                f->edgeB,
                f->ww,
                f->fw
            );
            f->made += 2;
            continue;
        }
#endif
        f->made++;
        dst = f->ring + (size_t)(f->made % f->rows) * f->im->linesize;
        src = (UINT8 *)f->im->image[f->made];
        if (f->d->direct) {
            ImagingLineBoxBlurDirect(dst, src, f->n, f->im->pixelsize, f->d, f->pad);
        } else {
            ImagingLineBoxBlur32(
                (UINT32 *)dst,
                (UINT32 *)src,
                f->lastx,
                f->radius,
                f->edgeA,
                f->edgeB,
                f->ww,
                f->fw
            );
        }
    }
    return f->ring + (size_t)(k % f->rows) * f->im->linesize;
}

/* One horizontal and one vertical pass: each row is blurred horizontally
   into a ring of 2 * yradius + 3 rows just before the vertical pass first
   reads it, so the horizontal result stays in cache and needs no image of
   its own; the same arithmetic as the two passes in turn. Each pass takes
   its boxes directly (integer radii up to BOX_DIRECT_MAX, or below 1) or
   with running sums, those of the horizontal pass only for 4-byte pixels
   (8-bit images blur four rows at a time in the unfused pass). Returns 0
   when it doesn't apply, -1 if out of memory. */
static int
ImagingBoxBlurFused(Imaging imOut, Imaging imIn, float xradius, float yradius) {
    ImagingSectionCookie cookie;
    BoxDirect dx, dy;
    BoxFusedRows f;
    const int rx = (int)xradius, ry = (int)yradius;
    const UINT32 wwx = (UINT32)(1 << 24) / (xradius * 2 + 1);
    const UINT32 wwy = (UINT32)(1 << 24) / (yradius * 2 + 1);
    const UINT32 fwy = ((1 << 24) - (ry * 2 + 1) * wwy) / 2;
    const int lasty = imIn->ysize - 1, size = imIn->linesize;
    /* the rows a vertical box reads, and one made ahead */
    long long ringBytes = (2 * (long long)ry + 4) * size;
    UINT32 *acc = NULL;
    int y, i;

    BoxDirectInit(&dx, rx, wwx, ((1 << 24) - (rx * 2 + 1) * wwx) / 2);
    BoxDirectInit(&dy, ry, wwy, fwy);
    if ((!dx.direct && !imIn->image32) || ringBytes > BOXBLUR_FUSED_MAX ||
        (dy.direct && dy.count > 2 * BOX_DIRECT_MAX + 3)) {
        return 0;
    }
    f.im = imIn;
    f.rows = 2 * ry + 4;
    f.made = -1;
    f.n = imIn->xsize * imIn->pixelsize;
    f.d = &dx;
    f.radius = rx;
    f.lastx = imIn->xsize - 1;
    f.edgeA = MIN(rx + 1, imIn->xsize);
    f.edgeB = MAX(imIn->xsize - rx - 1, 0);
    f.ww = wwx;
    f.fw = dx.fw;
    f.ring = malloc((size_t)ringBytes);
    f.pad = malloc((imIn->xsize + 2 * (BOX_DIRECT_MAX + 1)) * sizeof(UINT32));
    if (!dy.direct) {
        acc = calloc(size, sizeof(UINT32));
    }
    if (!f.ring || !f.pad || (!dy.direct && !acc)) {
        free(f.ring);
        free(f.pad);
        free(acc);
        return -1;
    }

    ImagingSectionEnter(&cookie);
    if (dy.direct) {
        const UINT8 *src[2 * BOX_DIRECT_MAX + 3];
        for (y = 0; y <= lasty; y++) {
            /* the rows a box reads span count rows, so none is
               overwritten while in use */
            for (i = 0; i < dy.count; i++) {
                src[i] = box_fused_row(&f, (long long)y - ry - 1 + i);
            }
            ImagingBoxBlurDirect((UINT8 *)imOut->image[y], src, size, &dy);
        }
    } else {
        /* ImagingVerticalBoxBlur's running sums, from the ring: the window
           for row -1, then a row in and a row out per output row */
        const int inside = ry - 1 < lasty ? ry - 1 : lasty;
        const UINT32 after = ry - 1 > lasty ? (UINT32)(ry - 1 - lasty) : 0;
        int b;
        {
            const UINT8 *first = box_fused_row(&f, 0);
            for (b = 0; b < size; b++) {
                acc[b] = first[b] * (UINT32)(ry + 1);
            }
            if (after) {
                /* a short image: the ring holds every row */
                const UINT8 *last = box_fused_row(&f, lasty);
                for (b = 0; b < size; b++) {
                    acc[b] += last[b] * after;
                }
            }
        }
        for (i = 0; i <= inside; i++) {
            const UINT8 *row = box_fused_row(&f, i);
            for (b = 0; b < size; b++) {
                acc[b] += row[b];
            }
        }
        for (y = 0; y <= lasty; y++) {
            /* rows y - ry - 1 to y + ry + 1, count rows */
            const UINT8 *add = box_fused_row(&f, (long long)y + ry);
            const UINT8 *far_row = box_fused_row(&f, (long long)y + ry + 1);
            const UINT8 *sub = box_fused_row(&f, (long long)y - ry - 1);
            ImagingBoxBlurColumns(
                (UINT8 *)imOut->image[y], acc, add, sub, far_row, size, wwy, fwy
            );
        }
    }
    ImagingSectionLeave(&cookie);

    free(acc);
    free(f.ring);
    free(f.pad);
    return 1;
}

Imaging
ImagingBoxBlur(Imaging imOut, Imaging imIn, float xradius, float yradius, int n) {
    int i;

    if (imOut->xsize == 0 || imOut->ysize == 0) {
        return imOut;
    }
    if (n < 1) {
        return ImagingError_ValueError("number of passes must be greater than zero");
    }
    /* Negated comparisons, so that NaN is rejected as well. */
    if (!(xradius >= 0) || !(yradius >= 0)) {
        return ImagingError_ValueError("radius must be >= 0");
    }
    /* 2**31 and above cannot be converted to an int. */
    if (xradius >= 2147483648.0f || yradius >= 2147483648.0f) {
        return ImagingError_ValueError("radius too large");
    }

    if (imIn->mode != imOut->mode || imIn->type != imOut->type ||
        imIn->bands != imOut->bands || imIn->xsize != imOut->xsize ||
        imIn->ysize != imOut->ysize) {
        return ImagingError_Mismatch();
    }

    if (imIn->type != IMAGING_TYPE_UINT8) {
        return ImagingError_ModeError();
    }

    if (imIn->mode != IMAGING_MODE_RGB && imIn->mode != IMAGING_MODE_RGBA &&
        imIn->mode != IMAGING_MODE_RGBa && imIn->mode != IMAGING_MODE_RGBX &&
        imIn->mode != IMAGING_MODE_CMYK && imIn->mode != IMAGING_MODE_L &&
        imIn->mode != IMAGING_MODE_LA && imIn->mode != IMAGING_MODE_La) {
        return ImagingError_ModeError();
    }

    if (n == 1 && xradius != 0 && yradius != 0) {
        int done = ImagingBoxBlurFused(imOut, imIn, xradius, yradius);
        if (done < 0) {
            return ImagingError_MemoryError();
        }
        if (done) {
            return imOut;
        }
    }

    /* Horizontal passes work in place on rows. Vertical passes need a
       separate destination, so they alternate between imOut and a
       temporary image; the horizontal result goes to whichever buffer
       makes the last vertical pass end in imOut. */
    Imaging imTemp = NULL;
    Imaging src = imIn;
    /* With the vertical passes in one sweep, it reads the horizontal
       result (or imIn) and writes imOut. */
    int cascade = yradius != 0 && BoxBlurCascadeFits(imIn, yradius, n);
    if (yradius != 0 && (!cascade || xradius != 0)) {
        imTemp = ImagingNewDirty(imIn->mode, imIn->xsize, imIn->ysize);
        if (!imTemp) {
            return NULL;
        }
    }
    if (xradius != 0) {
        Imaging dst = (yradius != 0 && (cascade || n % 2 == 1)) ? imTemp : imOut;
        if (!ImagingHorizontalBoxBlur(dst, imIn, xradius, n)) {
            goto error;
        }
        src = dst;
    }
    if (cascade) {
        if (!ImagingVerticalBoxBlurPasses(imOut, src, yradius, n)) {
            goto error;
        }
        if (imTemp) {
            ImagingDelete(imTemp);
        }
    } else if (yradius != 0) {
        for (i = 0; i < n; i++) {
            Imaging dst = (n - i) % 2 == 1 ? imOut : imTemp;
            if (!ImagingVerticalBoxBlur(dst, src, yradius)) {
                goto error;
            }
            src = dst;
        }
        ImagingDelete(imTemp);
    }
    if (xradius == 0 && yradius == 0) {
        if (!ImagingCopy2(imOut, imIn)) {
            return NULL;
        }
    }

    return imOut;

error:
    if (imTemp) {
        ImagingDelete(imTemp);
    }
    return NULL;
}

static float
_gaussian_blur_radius(float radius, int passes) {
    float sigma2, L, l, a;

    sigma2 = radius * radius / passes;
    // from https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf
    // [7] Box length.
    L = sqrt(12.0 * sigma2 + 1.0);
    // [11] Integer part of box radius.
    l = floor((L - 1.0) / 2.0);
    // [14], [Fig. 2] Fractional part of box radius.
    a = (2 * l + 1) * (l * (l + 1) - 3 * sigma2);
    a /= 6 * (sigma2 - (l + 1) * (l + 1));

    return l + a;
}

/* The extended box radius of each pass, for the fast kernel (GaussianFast.c),
   which applies the same boxes convolved together. */
float
ImagingGaussianBlurBoxRadius(float radius, int passes) {
    return _gaussian_blur_radius(radius, passes);
}

Imaging
ImagingGaussianBlur(
    Imaging imOut, Imaging imIn, float xradius, float yradius, int passes
) {
    /* fast mode, read once for the whole blur */
    if (passes == 3 && (ImagingGetFast() & IMAGING_FAST_ALL) &&
        ImagingGaussianBlurFastApplies(imOut, imIn, xradius, yradius)) {
        return ImagingGaussianBlurFast(imOut, imIn, xradius, yradius);
    }
    return ImagingBoxBlur(
        imOut,
        imIn,
        _gaussian_blur_radius(xradius, passes),
        _gaussian_blur_radius(yradius, passes),
        passes
    );
}
