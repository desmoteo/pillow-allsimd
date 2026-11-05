/* The rank filters' vector kernels (RankFilter.c), compiled into
   SIMDKernels.c: the 3x3 and 5x5 medians of 1-byte and 4-byte 8-bit images
   through sorting networks in registers (whole images), and one pass of a
   minimum or maximum over several source rows. */

#include "RankFilter.h"

#if defined(IMAGING_SIMD_AVX2)
#define RANK_VEC 32
typedef __m256i rank_vec;
#define RANK_LOAD(p) _mm256_loadu_si256((const __m256i *)(p))
#define RANK_STORE(p, v) _mm256_storeu_si256((__m256i *)(p), v)
#define RANK_VMIN(a, b) _mm256_min_epu8(a, b)
#define RANK_VMAX(a, b) _mm256_max_epu8(a, b)
#elif defined(IMAGING_SIMD_SSE4)
#define RANK_VEC 16
typedef __m128i rank_vec;
#define RANK_LOAD(p) _mm_loadu_si128((const __m128i *)(p))
#define RANK_STORE(p, v) _mm_storeu_si128((__m128i *)(p), v)
#define RANK_VMIN(a, b) _mm_min_epu8(a, b)
#define RANK_VMAX(a, b) _mm_max_epu8(a, b)
#elif defined(IMAGING_SIMD_NEON)
#define RANK_VEC 16
typedef uint8x16_t rank_vec;
#define RANK_LOAD(p) vld1q_u8(p)
#define RANK_STORE(p, v) vst1q_u8(p, v)
#define RANK_VMIN(a, b) vminq_u8(a, b)
#define RANK_VMAX(a, b) vmaxq_u8(a, b)
#endif

/* Medians of 3 x 3 (rank 4) and 5 x 5 (rank 12), RANK_VEC output bytes at
   a time: the neighbours are loaded from the rows and go through a median
   network in registers, where the generic path above copies them to
   buffers and sorts there: Paeth's 19 comparators for 9 values,
   Devillard's 99 for 25 (checked for all 2^25 inputs of 0 and 1, which by
   the 0-1 principle covers all inputs). The compiler drops the halves of
   comparators whose result isn't used. The median doesn't depend on the
   sorting method, so the output is the same. margin and ps are as for
   RankFilterNetwork_u8. */
#define RANK_SORT(a, b)           \
    t = RANK_VMIN(p[a], p[b]);    \
    p[b] = RANK_VMAX(p[a], p[b]); \
    p[a] = t;

static inline void
RankFilterMedianVec_u8(Imaging imOut, Imaging im, int size, int margin, int ps) {
    int rowbytes = im->xsize * ps, outbytes = imOut->xsize * ps;
    int x, y, i;
    UINT8 window[5][RANK_VEC + 16], result[RANK_VEC];

    for (y = 0; y < imOut->ysize; y++) {
        const UINT8 *rows[5];
        UINT8 *out = (UINT8 *)imOut->image[y];
        for (i = 0; i < size; i++) {
            int r = y + i - margin;
            rows[i] = (UINT8 *)im->image[r < 0 ? 0 : r < im->ysize ? r : im->ysize - 1];
        }
        for (x = 0; x < outbytes; x += RANK_VEC) {
            /* Byte x of the output takes bytes base + j * ps of each row,
               j = 0 .. size - 1. */
            int base = x - margin * ps;
            const UINT8 *src[5];
            rank_vec p[25], t;
            if (base >= 0 && base + (size - 1) * ps + RANK_VEC <= rowbytes) {
                for (i = 0; i < size; i++) {
                    src[i] = rows[i] + base;
                }
            } else {
                /* The first and last vectors of a row: copy the window,
                   repeating the edge pixels. Lanes past the end of the row
                   are computed too, but not stored. */
                int n = RANK_VEC + (size - 1) * ps;
                for (i = 0; i < size; i++) {
                    int lead = base < 0 ? -base : 0;
                    int end = base + n > rowbytes ? rowbytes - base : n;
                    if (lead > n) {
                        lead = n;
                    }
                    if (end < lead) {
                        end = lead;
                    }
                    RankFillPixels(window[i], rows[i], ps, lead);
                    memcpy(window[i] + lead, rows[i] + base + lead, end - lead);
                    RankFillPixels(
                        window[i] + end, rows[i] + rowbytes - ps, ps, n - end
                    );
                    src[i] = window[i];
                }
            }
            /* Constant indices into p only: with any variable one, p would
               live in memory instead of registers, as it does at -O2 if the
               loads are a loop that the compiler doesn't unroll. */
#define RANK_LOAD3(i)                        \
    p[(i) * 3] = RANK_LOAD(src[i]);          \
    p[(i) * 3 + 1] = RANK_LOAD(src[i] + ps); \
    p[(i) * 3 + 2] = RANK_LOAD(src[i] + 2 * ps);
#define RANK_LOAD5(i)                            \
    p[(i) * 5] = RANK_LOAD(src[i]);              \
    p[(i) * 5 + 1] = RANK_LOAD(src[i] + ps);     \
    p[(i) * 5 + 2] = RANK_LOAD(src[i] + 2 * ps); \
    p[(i) * 5 + 3] = RANK_LOAD(src[i] + 3 * ps); \
    p[(i) * 5 + 4] = RANK_LOAD(src[i] + 4 * ps);
            if (size == 3) {
                RANK_LOAD3(0) RANK_LOAD3(1) RANK_LOAD3(2)
            } else {
                RANK_LOAD5(0) RANK_LOAD5(1) RANK_LOAD5(2) RANK_LOAD5(3) RANK_LOAD5(4)
            }
#undef RANK_LOAD3
#undef RANK_LOAD5
            if (size == 3) {
                RANK_SORT(1, 2) RANK_SORT(4, 5) RANK_SORT(7, 8);
                RANK_SORT(0, 1) RANK_SORT(3, 4) RANK_SORT(6, 7);
                RANK_SORT(1, 2) RANK_SORT(4, 5) RANK_SORT(7, 8);
                RANK_SORT(0, 3) RANK_SORT(5, 8) RANK_SORT(4, 7);
                RANK_SORT(3, 6) RANK_SORT(1, 4) RANK_SORT(2, 5);
                RANK_SORT(4, 7) RANK_SORT(4, 2) RANK_SORT(6, 4);
                RANK_SORT(4, 2);
                p[12] = p[4];
            } else {
                RANK_SORT(0, 1) RANK_SORT(3, 4) RANK_SORT(2, 4);
                RANK_SORT(2, 3) RANK_SORT(6, 7) RANK_SORT(5, 7);
                RANK_SORT(5, 6) RANK_SORT(9, 10) RANK_SORT(8, 10);
                RANK_SORT(8, 9) RANK_SORT(12, 13) RANK_SORT(11, 13);
                RANK_SORT(11, 12) RANK_SORT(15, 16) RANK_SORT(14, 16);
                RANK_SORT(14, 15) RANK_SORT(18, 19) RANK_SORT(17, 19);
                RANK_SORT(17, 18) RANK_SORT(21, 22) RANK_SORT(20, 22);
                RANK_SORT(20, 21) RANK_SORT(23, 24) RANK_SORT(2, 5);
                RANK_SORT(3, 6) RANK_SORT(0, 6) RANK_SORT(0, 3);
                RANK_SORT(4, 7) RANK_SORT(1, 7) RANK_SORT(1, 4);
                RANK_SORT(11, 14) RANK_SORT(8, 14) RANK_SORT(8, 11);
                RANK_SORT(12, 15) RANK_SORT(9, 15) RANK_SORT(9, 12);
                RANK_SORT(13, 16) RANK_SORT(10, 16) RANK_SORT(10, 13);
                RANK_SORT(20, 23) RANK_SORT(17, 23) RANK_SORT(17, 20);
                RANK_SORT(21, 24) RANK_SORT(18, 24) RANK_SORT(18, 21);
                RANK_SORT(19, 22) RANK_SORT(8, 17) RANK_SORT(9, 18);
                RANK_SORT(0, 18) RANK_SORT(0, 9) RANK_SORT(10, 19);
                RANK_SORT(1, 19) RANK_SORT(1, 10) RANK_SORT(11, 20);
                RANK_SORT(2, 20) RANK_SORT(2, 11) RANK_SORT(12, 21);
                RANK_SORT(3, 21) RANK_SORT(3, 12) RANK_SORT(13, 22);
                RANK_SORT(4, 22) RANK_SORT(4, 13) RANK_SORT(14, 23);
                RANK_SORT(5, 23) RANK_SORT(5, 14) RANK_SORT(15, 24);
                RANK_SORT(6, 24) RANK_SORT(6, 15) RANK_SORT(7, 16);
                RANK_SORT(7, 19) RANK_SORT(13, 21) RANK_SORT(15, 23);
                RANK_SORT(7, 13) RANK_SORT(7, 15) RANK_SORT(1, 9);
                RANK_SORT(3, 11) RANK_SORT(5, 17) RANK_SORT(11, 17);
                RANK_SORT(9, 17) RANK_SORT(4, 10) RANK_SORT(6, 12);
                RANK_SORT(7, 14) RANK_SORT(4, 6) RANK_SORT(4, 7);
                RANK_SORT(12, 14) RANK_SORT(10, 14) RANK_SORT(6, 7);
                RANK_SORT(10, 12) RANK_SORT(6, 10) RANK_SORT(6, 17);
                RANK_SORT(12, 17) RANK_SORT(7, 17) RANK_SORT(7, 10);
                RANK_SORT(12, 18) RANK_SORT(7, 12) RANK_SORT(10, 18);
                RANK_SORT(12, 20) RANK_SORT(10, 20) RANK_SORT(10, 12);
            }
            if (x + RANK_VEC <= outbytes) {
                RANK_STORE(out + x, p[12]);
            } else {
                RANK_STORE(result, p[12]);
                memcpy(out + x, result, outbytes - x);
            }
        }
    }
}
#undef RANK_SORT

/* Constant sizes, so that the loops and the networks are unrolled. */
static void
RankFilterMedian3_u8(Imaging imOut, Imaging im, int margin, int ps) {
    RankFilterMedianVec_u8(imOut, im, 3, margin, ps);
}

static void
RankFilterMedian5_u8(Imaging imOut, Imaging im, int margin, int ps) {
    RankFilterMedianVec_u8(imOut, im, 5, margin, ps);
}

/* out[x] = the minimum or maximum of src[0][x] .. src[count - 1][x]: one
   pass over the line with all sources at once, where RankFilter.c's loops
   make count passes, and need -O3 to be vectorized at all. Does the bytes
   the vector loop can and returns how many. */
static int
rank_minmax_line_simd(
    UINT8 *restrict out, const UINT8 **src, int count, int n, int is_max
) {
    int x = 0, j;
    if (count == 3) {
        /* the 3x3 filters, without the loop over the sources */
        const UINT8 *s0 = src[0], *s1 = src[1], *s2 = src[2];
        if (is_max) {
            for (; x + RANK_VEC <= n; x += RANK_VEC) {
                rank_vec v = RANK_VMAX(RANK_LOAD(s0 + x), RANK_LOAD(s1 + x));
                RANK_STORE(out + x, RANK_VMAX(v, RANK_LOAD(s2 + x)));
            }
        } else {
            for (; x + RANK_VEC <= n; x += RANK_VEC) {
                rank_vec v = RANK_VMIN(RANK_LOAD(s0 + x), RANK_LOAD(s1 + x));
                RANK_STORE(out + x, RANK_VMIN(v, RANK_LOAD(s2 + x)));
            }
        }
    } else if (is_max) {
        for (; x + RANK_VEC <= n; x += RANK_VEC) {
            rank_vec v = RANK_LOAD(src[0] + x);
            for (j = 1; j < count; j++) {
                v = RANK_VMAX(v, RANK_LOAD(src[j] + x));
            }
            RANK_STORE(out + x, v);
        }
    } else {
        for (; x + RANK_VEC <= n; x += RANK_VEC) {
            rank_vec v = RANK_LOAD(src[0] + x);
            for (j = 1; j < count; j++) {
                v = RANK_VMIN(v, RANK_LOAD(src[j] + x));
            }
            RANK_STORE(out + x, v);
        }
    }
    return x;
}

/* The general rank filter's sorting network over a chunk: the loop of
   PlainLoops.h as this tier's compiler makes it. */
static void
rank_network_simd(UINT8 *v, const UINT8 (*network)[2], int count) {
    plain_rank_network(v, network, count);
}

static const ImagingSIMDRankKernels rank_simd_kernels = {
    .median3 = RankFilterMedian3_u8,
    .median5 = RankFilterMedian5_u8,
    .minmax_line = rank_minmax_line_simd,
    .network = rank_network_simd,
};
