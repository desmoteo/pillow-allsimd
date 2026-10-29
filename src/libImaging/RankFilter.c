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

/* Size 3 to RANK_NETWORK_MAX, any rank: the size * size neighbours of
   RANK_CHUNK output pixels go through a sorting network, pruned to what
   wire rank needs, and v[rank] is the result. Each compare-exchange is a
   minimum and a maximum over a row of bytes, which compilers vectorise.
   Returns 0 if out of memory.
   With margin 0, im is the image expanded by (size - 1) / 2 on each side;
   otherwise im has imOut's size and pixels outside it repeat its edges, as
   the expanded image would. */
static int
RankFilterNetwork_u8(Imaging imOut, Imaging im, int size, int rank, int margin) {
    /* An optimal network for 9 values. */
    static const UINT8 network3[25][2] = {{0, 1}, {3, 4}, {6, 7}, {1, 2}, {4, 5},
                                          {7, 8}, {0, 1}, {3, 4}, {6, 7}, {0, 3},
                                          {3, 6}, {0, 3}, {1, 4}, {4, 7}, {1, 4},
                                          {2, 5}, {5, 8}, {2, 5}, {1, 3}, {5, 7},
                                          {2, 6}, {4, 6}, {2, 4}, {2, 3}, {5, 6}};
    /* Batcher's network for 128 values has 1471 comparators. */
    UINT8 network[1471][2];
    int n = size * size, count, x0, y, i, j, c;
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
        for (x0 = 0; x0 < imOut->xsize; x0 += RANK_CHUNK) {
            int w = imOut->xsize - x0 < RANK_CHUNK ? imOut->xsize - x0 : RANK_CHUNK;
            for (i = 0; i < size; i++) {
                int yy = y + i - margin;
                const UINT8 *row = im->image8
                                       [yy < 0           ? 0
                                        : yy < im->ysize ? yy
                                                         : im->ysize - 1];
                for (j = 0; j < size; j++) {
                    int x = x0 + j - margin;
                    UINT8 *dst = v[i * size + j];
                    if (x >= 0 && x + w <= im->xsize) {
                        memcpy(dst, &row[x], w);
                    } else {
                        // the edge pixels repeated before and after the row
                        int lead = x < 0 ? (-x < w ? -x : w) : 0;
                        int tail = x + w > im->xsize ? x + w - im->xsize : 0;
                        if (tail > w - lead) {
                            tail = w - lead;
                        }
                        memset(dst, row[0], lead);
                        memcpy(dst + lead, &row[x + lead], w - lead - tail);
                        memset(dst + w - tail, row[im->xsize - 1], tail);
                    }
                }
            }
            for (c = 0; c < count; c++) {
                RankCompareExchange(v[network[c][0]], v[network[c][1]]);
            }
            memcpy(&imOut->image8[y][x0], v[rank], w);
        }
    }
    free(v);
    return 1;
}

/* Minimum (rank 0) or maximum (rank size * size - 1) of any size: of each
   row's size neighbours, then of size of those row results. Returns 0 if
   out of memory. margin is as for RankFilterNetwork_u8. */
static int
RankFilterMinMax_u8(Imaging imOut, Imaging im, int size, int is_max, int margin) {
    int xsize = imOut->xsize, ysize = imOut->ysize;
    int x, y, k;
    UINT8 *rows, *padded = NULL;
    if (xsize == 0 || ysize == 0) {
        return 1;
    }
    rows = malloc((size_t)xsize * im->ysize);
    if (margin) {
        padded = malloc((size_t)xsize + 2 * margin);
    }
    if (!rows || (margin && !padded)) {
        free(rows);
        return 0;
    }

#define RANK_ROW(r) \
    (rows + (size_t)((r) < 0 ? 0 : (r) < im->ysize ? (r) : im->ysize - 1) * xsize)
#define RANK_MINMAX(op)                                             \
    for (y = 0; y < im->ysize; y++) {                               \
        const UINT8 *in = im->image8[y];                            \
        UINT8 *restrict row = rows + (size_t)y * xsize;             \
        if (margin) {                                               \
            /* the row with its edges repeated, as expanded */      \
            memset(padded, in[0], margin);                          \
            memcpy(padded + margin, in, xsize);                     \
            memset(padded + margin + xsize, in[xsize - 1], margin); \
            in = padded;                                            \
        }                                                           \
        memcpy(row, in, xsize);                                     \
        for (k = 1; k < size; k++) {                                \
            for (x = 0; x < xsize; x++) {                           \
                row[x] = op(row[x], in[x + k]);                     \
            }                                                       \
        }                                                           \
    }                                                               \
    for (y = 0; y < ysize; y++) {                                   \
        UINT8 *restrict out = imOut->image8[y];                     \
        memcpy(out, RANK_ROW(y - margin), xsize);                   \
        for (k = 1; k < size; k++) {                                \
            const UINT8 *restrict row = RANK_ROW(y + k - margin);   \
            for (x = 0; x < xsize; x++) {                           \
                out[x] = op(out[x], row[x]);                        \
            }                                                       \
        }                                                           \
    }
#define RANK_MIN(a, b) ((a) < (b) ? (a) : (b))
#define RANK_MAX(a, b) ((a) > (b) ? (a) : (b))
    if (is_max) {
        RANK_MINMAX(RANK_MAX)
    } else {
        RANK_MINMAX(RANK_MIN)
    }
#undef RANK_MIN
#undef RANK_MAX
#undef RANK_MINMAX
#undef RANK_ROW

    free(rows);
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
        if (!RankFilterMinMax_u8(imOut, im, size, rank != 0, 0)) {
            ImagingSectionLeave(&cookie);
            goto nomemory;
        }
    } else if (im->image8 && size >= 3 && size <= RANK_NETWORK_MAX) {
        if (!RankFilterNetwork_u8(imOut, im, size, rank, 0)) {
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

/* As ImagingRankFilter on im expanded by (size - 1) / 2 pixels that repeat
   its edges, without making the expanded copy where the fast paths for
   1-byte images can read the edges themselves. */
Imaging
ImagingRankFilterSameSize(Imaging im, int size, int rank) {
    Imaging imOut, imExpanded;
    int ok;
    if (im && im->image8 && im->type == IMAGING_TYPE_UINT8 && im->bands == 1 &&
        im->xsize > 0 && im->ysize > 0 && size >= 3 && (size & 1) &&
        size <= RANK_NETWORK_MAX && rank >= 0 && rank < size * size) {
        ImagingSectionCookie cookie;
        imOut = ImagingNewDirty(im->mode, im->xsize, im->ysize);
        if (!imOut) {
            return NULL;
        }
        ImagingSectionEnter(&cookie);
        if (rank == 0 || rank == size * size - 1) {
            ok = RankFilterMinMax_u8(imOut, im, size, rank != 0, (size - 1) / 2);
        } else {
            ok = RankFilterNetwork_u8(imOut, im, size, rank, (size - 1) / 2);
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
