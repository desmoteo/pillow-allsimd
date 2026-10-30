/*
 * The Python Imaging Library
 * $Id$
 *
 * min, max, median filters
 *
 * history:
 * 2002-06-08 fl    Created
 *
 * Copyright (c) Secret Labs AB 2002.  All rights reserved.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

/* Fast rank algorithm (due to Wirth), based on public domain code
   by Nicolas Devillard, available at http://ndevilla.free.fr */

#define SWAP(type, a, b)       \
    {                          \
        register type t = (a); \
        (a) = (b);             \
        (b) = t;               \
    }

#define MakeRankFunction(type)                       \
    static type Rank##type(type a[], int n, int k) { \
        register int i, j, l, m;                     \
        register type x;                             \
        l = 0;                                       \
        m = n - 1;                                   \
        while (l < m) {                              \
            x = a[k];                                \
            i = l;                                   \
            j = m;                                   \
            do {                                     \
                while (a[i] < x) {                   \
                    i++;                             \
                }                                    \
                while (x < a[j]) {                   \
                    j--;                             \
                }                                    \
                if (i <= j) {                        \
                    SWAP(type, a[i], a[j]);          \
                    i++;                             \
                    j--;                             \
                }                                    \
            } while (i <= j);                        \
            if (j < k) {                             \
                l = i;                               \
            }                                        \
            if (k < i) {                             \
                m = j;                               \
            }                                        \
        }                                            \
        return a[k];                                 \
    }

MakeRankFunction(UINT8) MakeRankFunction(INT32) MakeRankFunction(FLOAT32)
/* Fast paths for 1-byte images. Both give the same pixels as the
   selection above: any way of sorting a window gives the same k-th value.
   Their inner loops run along rows and are plain enough for compilers to
   vectorise. */
#ifndef RANK_CHUNK
#define RANK_CHUNK 256
#endif

    /* Puts the minimum of each pair in a and the maximum in b. */
    static inline void RankCompareExchange(UINT8 *restrict a, UINT8 *restrict b) {
    int x;
    for (x = 0; x < RANK_CHUNK; x++) {
        UINT8 lo = a[x] < b[x] ? a[x] : b[x];
        UINT8 hi = a[x] < b[x] ? b[x] : a[x];
        a[x] = lo;
        b[x] = hi;
    }
}

/* Largest window sorted by a network; larger ones use the selection above. */
#define RANK_NETWORK_MAX 11

/* Builds a comparator network that sorts n values, n <= 128: Batcher's
   odd-even merge sort for the next power of two, without the comparators
   that touch the wires past n. Those wires can be taken as holding the
   largest value, which no comparator moves, so the rest still sorts.
   Returns the number of comparators. */
static int
RankNetworkBatcher(int n, UINT8 (*network)[2]) {
    int count = 0, size = 1, p, k, j, i;
    while (size < n) {
        size <<= 1;
    }
    for (p = 1; p < size; p <<= 1) {
        for (k = p; k >= 1; k >>= 1) {
            for (j = k % p; j + k < size; j += 2 * k) {
                for (i = 0; i < k && i + j + k < size; i++) {
                    if ((i + j) / (2 * p) == (i + j + k) / (2 * p) && i + j + k < n) {
                        network[count][0] = (UINT8)(i + j);
                        network[count][1] = (UINT8)(i + j + k);
                        count++;
                    }
                }
            }
        }
    }
    return count;
}

/* Keeps only the comparators that the value ending on wire rank depends
   on, in order. Returns the new count. */
static int
RankNetworkPrune(UINT8 (*network)[2], int count, int rank) {
    UINT8 needed[RANK_NETWORK_MAX * RANK_NETWORK_MAX] = {0};
    int c, kept = 0;
    needed[rank] = 1;
    for (c = count - 1; c >= 0; c--) {
        if (needed[network[c][0]] || needed[network[c][1]]) {
            needed[network[c][0]] = needed[network[c][1]] = 1;
            network[c][0] |= 0x80;  // mark as kept
        }
    }
    for (c = 0; c < count; c++) {
        if (network[c][0] & 0x80) {
            network[kept][0] = network[c][0] & 0x7f;
            network[kept][1] = network[c][1];
            kept++;
        }
    }
    return kept;
}

/* n bytes of the ps-byte pixel at pixel, repeated: an edge of a row. */
static void
RankFillPixels(UINT8 *dst, const UINT8 *pixel, int ps, int n) {
    int k;
    if (ps == 1) {
        memset(dst, pixel[0], n);
        return;
    }
    for (k = 0; k < n; k++) {
        dst[k] = pixel[k % ps];
    }
}

/* Size 3 to RANK_NETWORK_MAX, any rank: the size * size neighbours of
   RANK_CHUNK output bytes go through a sorting network, pruned to what
   wire rank needs, and v[rank] is the result. Each compare-exchange is a
   minimum and a maximum over a row of bytes, which compilers vectorise.
   Pixels are ps bytes, 1 or 4; with 4, each byte is filtered with the
   same byte of its neighbours, so every band at once. Returns 0 if out
   of memory.
   With margin 0, im is the image expanded by (size - 1) / 2 on each side;
   otherwise im has imOut's size and pixels outside it repeat its edges, as
   the expanded image would. */
static int
RankFilterNetwork_u8(
    Imaging imOut, Imaging im, int size, int rank, int margin, int ps
) {
    /* An optimal network for 9 values. */
    static const UINT8 network3[25][2] = {{0, 1}, {3, 4}, {6, 7}, {1, 2}, {4, 5},
                                          {7, 8}, {0, 1}, {3, 4}, {6, 7}, {0, 3},
                                          {3, 6}, {0, 3}, {1, 4}, {4, 7}, {1, 4},
                                          {2, 5}, {5, 8}, {2, 5}, {1, 3}, {5, 7},
                                          {2, 6}, {4, 6}, {2, 4}, {2, 3}, {5, 6}};
    /* Batcher's network for 128 values has 1471 comparators. */
    UINT8 network[1471][2];
    int n = size * size, count, x0, y, i, j, c;
    int rowbytes = im->xsize * ps, outbytes = imOut->xsize * ps;
    UINT8(*v)[RANK_CHUNK];

    if (size == 3) {
        memcpy(network, network3, sizeof(network3));
        count = 25;
    } else {
        count = RankNetworkBatcher(n, network);
    }
    count = RankNetworkPrune(network, count, rank);

    v = malloc((size_t)n * RANK_CHUNK);
    if (!v) {
        return 0;
    }
    // Lanes past the end of the row are sorted too; keep them defined.
    memset(v, 0, (size_t)n * RANK_CHUNK);
    for (y = 0; y < imOut->ysize; y++) {
        // RANK_CHUNK is a multiple of ps, so chunks hold whole pixels.
        for (x0 = 0; x0 < outbytes; x0 += RANK_CHUNK) {
            int w = outbytes - x0 < RANK_CHUNK ? outbytes - x0 : RANK_CHUNK;
            for (i = 0; i < size; i++) {
                int yy = y + i - margin;
                const UINT8 *row = (UINT8 *)im->image
                                       [yy < 0           ? 0
                                        : yy < im->ysize ? yy
                                                         : im->ysize - 1];
                for (j = 0; j < size; j++) {
                    int x = x0 + (j - margin) * ps;
                    UINT8 *dst = v[i * size + j];
                    if (x >= 0 && x + w <= rowbytes) {
                        memcpy(dst, &row[x], w);
                    } else {
                        // the edge pixels repeated before and after the row
                        int lead = x < 0 ? (-x < w ? -x : w) : 0;
                        int tail = x + w > rowbytes ? x + w - rowbytes : 0;
                        if (tail > w - lead) {
                            tail = w - lead;
                        }
                        RankFillPixels(dst, row, ps, lead);
                        memcpy(dst + lead, &row[x + lead], w - lead - tail);
                        RankFillPixels(dst + w - tail, row + rowbytes - ps, ps, tail);
                    }
                }
            }
            for (c = 0; c < count; c++) {
                RankCompareExchange(v[network[c][0]], v[network[c][1]]);
            }
            memcpy((UINT8 *)imOut->image[y] + x0, v[rank], w);
        }
    }
    free(v);
    return 1;
}

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

#if defined(RANK_VEC)
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
#endif

#if defined(RANK_VEC)
/* out[x] = the minimum or maximum of src[0][x] .. src[count - 1][x]: one
   pass over the line with all sources at once, where the loops below make
   count passes, and need -O3 to be vectorized at all. */
static void
RankMinMaxLine(UINT8 *restrict out, const UINT8 **src, int count, int n, int is_max) {
    int x = 0, j;
    if (is_max) {
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
    for (; x < n; x++) {
        UINT8 v = src[0][x];
        for (j = 1; j < count; j++) {
            UINT8 s = src[j][x];
            v = is_max ? (s > v ? s : v) : (s < v ? s : v);
        }
        out[x] = v;
    }
}
#endif

/* Minimum (rank 0) or maximum (rank size * size - 1) of any size: of each
   row's size neighbours, then of size of those row results. The row
   results go to a ring of size rows, so the work stays in cache. Returns 0
   if out of memory. margin and ps are as for RankFilterNetwork_u8. */
static int
RankFilterMinMax_u8(
    Imaging imOut, Imaging im, int size, int is_max, int margin, int ps
) {
    int xsize = imOut->xsize * ps, ysize = imOut->ysize;
    int y, k, j;
    UINT8 *ring, *padded = NULL;
    const UINT8 **src;
    if (xsize == 0 || ysize == 0) {
        return 1;
    }
    ring = malloc((size_t)xsize * size);
    src = malloc(size * sizeof(*src));
    if (margin) {
        padded = malloc((size_t)xsize + 2 * margin * ps);
    }
    if (!ring || !src || (margin && !padded)) {
        free(ring);
        free(src);
        free(padded);
        return 0;
    }

/* Ring slot k holds the row result of source row k - margin, clipped to
   the image; k - margin is the source row of window row k. */
#define RANK_SLOT(k) (ring + (size_t)((k) % size) * xsize)
#define RANK_HROW(op, k)                                                       \
    do {                                                                       \
        int r_ = (k) - margin;                                                 \
        const UINT8 *in = (UINT8 *)im->image                                   \
                              [r_ < 0           ? 0                            \
                               : r_ < im->ysize ? r_                           \
                                                : im->ysize - 1];              \
        UINT8 *restrict row = RANK_SLOT(k);                                    \
        if (margin) {                                                          \
            /* the row with its edge pixels repeated, as expanded */           \
            RankFillPixels(padded, in, ps, margin * ps);                       \
            memcpy(padded + margin * ps, in, xsize);                           \
            RankFillPixels(                                                    \
                padded + margin * ps + xsize, in + xsize - ps, ps, margin * ps \
            );                                                                 \
            in = padded;                                                       \
        }                                                                      \
        RANK_LINE(op, row, in + j * ps);                                       \
    } while (0)
#define RANK_MINMAX(op)                                 \
    for (k = 0; k < size - 1; k++) {                    \
        RANK_HROW(op, k);                               \
    }                                                   \
    for (y = 0; y < ysize; y++) {                       \
        UINT8 *restrict out = (UINT8 *)imOut->image[y]; \
        RANK_HROW(op, y + size - 1);                    \
        RANK_LINE(op, out, RANK_SLOT(y + j));           \
    }
#if defined(RANK_VEC)
/* dst = op of source(j) for j = 0 .. size - 1 */
#define RANK_LINE(op, dst, source) \
    for (j = 0; j < size; j++) {   \
        src[j] = (source);         \
    }                              \
    RankMinMaxLine(dst, src, size, xsize, is_max);
#else
#define RANK_LINE(op, dst, source)                                          \
    j = 0;                                                                  \
    memcpy(dst, (source), xsize);                                           \
    for (j = 1; j < size; j++) {                                            \
        /* restrict: without it, GCC's cost model leaves the loop scalar */ \
        /* rather than check that the rows don't overlap */                 \
        const UINT8 *restrict s_ = (source);                                \
        UINT8 *restrict d_ = (dst);                                         \
        for (int x_ = 0; x_ < xsize; x_++) {                                \
            d_[x_] = op(d_[x_], s_[x_]);                                    \
        }                                                                   \
    }
#endif
#define RANK_MIN(a, b) ((a) < (b) ? (a) : (b))
#define RANK_MAX(a, b) ((a) > (b) ? (a) : (b))
    if (is_max) {
        RANK_MINMAX(RANK_MAX)
    } else {
        RANK_MINMAX(RANK_MIN)
    }
#undef RANK_MIN
#undef RANK_MAX
#undef RANK_LINE
#undef RANK_MINMAX
#undef RANK_HROW
#undef RANK_SLOT

    free(ring);
    free(src);
    free(padded);
    return 1;
}

Imaging
ImagingRankFilter(Imaging im, int size, int rank) {
    Imaging imOut = NULL;
    int x, y;
    int i, margin, size2;

    if (!im || im->bands != 1 || im->type == IMAGING_TYPE_I16) {
        return (Imaging)ImagingError_ModeError();
    }

    if (!(size & 1)) {
        return (Imaging)ImagingError_ValueError("bad filter size");
    }

    /* malloc check ok, for overflow in the define below */
    if (size > INT_MAX / (size * (int)sizeof(FLOAT32))) {
        return (Imaging)ImagingError_ValueError("filter size too large");
    }

    size2 = size * size;
    margin = (size - 1) / 2;

    if (rank < 0 || rank >= size2) {
        return (Imaging)ImagingError_ValueError("bad rank value");
    }

    // Every output pixel is written by the rank loop below
    imOut = ImagingNewDirty(im->mode, im->xsize - 2 * margin, im->ysize - 2 * margin);
    if (!imOut) {
        return NULL;
    }

    /* malloc check ok, checked above */
#define RANK_BODY(type)                                                           \
    do {                                                                          \
        type *buf = malloc(size2 * sizeof(type));                                 \
        if (!buf) {                                                               \
            ImagingSectionLeave(&cookie);                                         \
            goto nomemory;                                                        \
        }                                                                         \
        for (y = 0; y < imOut->ysize; y++) {                                      \
            for (x = 0; x < imOut->xsize; x++) {                                  \
                for (i = 0; i < size; i++) {                                      \
                    memcpy(                                                       \
                        buf + i * size,                                           \
                        &IMAGING_PIXEL_##type(im, x, y + i),                      \
                        size * sizeof(type)                                       \
                    );                                                            \
                }                                                                 \
                IMAGING_PIXEL_##type(imOut, x, y) = Rank##type(buf, size2, rank); \
            }                                                                     \
        }                                                                         \
        free(buf);                                                                \
    } while (0)

    ImagingSectionCookie cookie;
    ImagingSectionEnter(&cookie);
    if (im->image8 && (rank == 0 || rank == size2 - 1) && size > 1) {
        if (!RankFilterMinMax_u8(imOut, im, size, rank != 0, 0, 1)) {
            ImagingSectionLeave(&cookie);
            goto nomemory;
        }
#if defined(RANK_VEC)
    } else if (im->image8 && size == 3 && rank == 4) {
        RankFilterMedian3_u8(imOut, im, 0, 1);
    } else if (im->image8 && size == 5 && rank == 12) {
        RankFilterMedian5_u8(imOut, im, 0, 1);
#endif
    } else if (im->image8 && size >= 3 && size <= RANK_NETWORK_MAX) {
        if (!RankFilterNetwork_u8(imOut, im, size, rank, 0, 1)) {
            ImagingSectionLeave(&cookie);
            goto nomemory;
        }
    } else if (im->image8) {
        RANK_BODY(UINT8);
    } else if (im->type == IMAGING_TYPE_INT32) {
        RANK_BODY(INT32);
    } else if (im->type == IMAGING_TYPE_FLOAT32) {
        RANK_BODY(FLOAT32);
    } else {
        /* safety net (we shouldn't end up here) */
        ImagingSectionLeave(&cookie);
        ImagingDelete(imOut);
        return (Imaging)ImagingError_ModeError();
    }
    ImagingSectionLeave(&cookie);

    ImagingCopyPalette(imOut, im);

    return imOut;

nomemory:
    ImagingDelete(imOut);
    return (Imaging)ImagingError_MemoryError();
}

/* Whether ImagingRankFilterSameSize filters all bands of im at once
   faster than band by band. For 4-byte pixels it filters every byte,
   padding included, so it only pays off where the work per byte is small
   (minimum, maximum, the vector 3 x 3 and 5 x 5 medians) or there is no
   padding. */
int
ImagingRankFilterAllBands(Imaging im, int size, int rank) {
    if (!im || im->type != IMAGING_TYPE_UINT8 || !im->image32 || im->pixelsize != 4 ||
        size < 3 || !(size & 1) || size > RANK_NETWORK_MAX || rank < 0 ||
        rank >= size * size) {
        return 0;
    }
#if defined(RANK_VEC)
    if ((size == 3 && rank == 4) || (size == 5 && rank == 12)) {
        return 1;
    }
#endif
    return rank == 0 || rank == size * size - 1 || im->bands == 4;
}

/* As ImagingRankFilter on im expanded by (size - 1) / 2 pixels that repeat
   its edges, without making the expanded copy where the fast paths can
   read the edges themselves. Those also take images of several 8-bit
   bands in 4-byte pixels, filtering each band as ImagingRankFilter would
   filter it alone. */
Imaging
ImagingRankFilterSameSize(Imaging im, int size, int rank) {
    Imaging imOut, imExpanded;
    int ok, ps;
    ps = !im || im->type != IMAGING_TYPE_UINT8 ? 0
         : im->image8 && im->bands == 1        ? 1
         : im->image32 && im->pixelsize == 4   ? 4
                                               : 0;
    if (ps && im->xsize > 0 && im->ysize > 0 && size >= 3 && (size & 1) &&
        size <= RANK_NETWORK_MAX && rank >= 0 && rank < size * size) {
        ImagingSectionCookie cookie;
        imOut = ImagingNewDirty(im->mode, im->xsize, im->ysize);
        if (!imOut) {
            return NULL;
        }
        ImagingSectionEnter(&cookie);
        if (rank == 0 || rank == size * size - 1) {
            ok = RankFilterMinMax_u8(imOut, im, size, rank != 0, (size - 1) / 2, ps);
#if defined(RANK_VEC)
        } else if (size == 3 && rank == 4) {
            RankFilterMedian3_u8(imOut, im, 1, ps);
            ok = 1;
        } else if (size == 5 && rank == 12) {
            RankFilterMedian5_u8(imOut, im, 2, ps);
            ok = 1;
#endif
        } else {
            ok = RankFilterNetwork_u8(imOut, im, size, rank, (size - 1) / 2, ps);
        }
        ImagingSectionLeave(&cookie);
        if (!ok) {
            ImagingDelete(imOut);
            return (Imaging)ImagingError_MemoryError();
        }
        ImagingCopyPalette(imOut, im);
        return imOut;
    }
    if (!im) {
        return (Imaging)ImagingError_ModeError();
    }
    imExpanded = ImagingExpand(im, size > 0 ? size / 2 : 0);
    if (!imExpanded) {
        return NULL;
    }
    imOut = ImagingRankFilter(imExpanded, size, rank);
    ImagingDelete(imExpanded);
    return imOut;
}
