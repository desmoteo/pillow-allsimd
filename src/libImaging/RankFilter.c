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

/* 3x3, any rank: the nine neighbours of RANK_CHUNK output pixels are
   sorted by a 25-comparator sorting network, then v[rank] is the result. */
static void
RankFilter3x3_u8(Imaging imOut, Imaging im, int rank) {
    static const UINT8 network[25][2] = {{0, 1}, {3, 4}, {6, 7}, {1, 2}, {4, 5},
                                         {7, 8}, {0, 1}, {3, 4}, {6, 7}, {0, 3},
                                         {3, 6}, {0, 3}, {1, 4}, {4, 7}, {1, 4},
                                         {2, 5}, {5, 8}, {2, 5}, {1, 3}, {5, 7},
                                         {2, 6}, {4, 6}, {2, 4}, {2, 3}, {5, 6}};
    UINT8 v[9][RANK_CHUNK];
    int x0, y, i, j, c;

    // Lanes past the end of the row are sorted too; keep them defined.
    memset(v, 0, sizeof(v));
    for (y = 0; y < imOut->ysize; y++) {
        for (x0 = 0; x0 < imOut->xsize; x0 += RANK_CHUNK) {
            int n = imOut->xsize - x0 < RANK_CHUNK ? imOut->xsize - x0 : RANK_CHUNK;
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    memcpy(v[i * 3 + j], &im->image8[y + i][x0 + j], n);
                }
            }
            for (c = 0; c < 25; c++) {
                RankCompareExchange(v[network[c][0]], v[network[c][1]]);
            }
            memcpy(&imOut->image8[y][x0], v[rank], n);
        }
    }
}

/* Minimum (rank 0) or maximum (rank size * size - 1) of any size: of each
   row's size neighbours, then of size of those row results. Returns 0 if
   out of memory. */
static int
RankFilterMinMax_u8(Imaging imOut, Imaging im, int size, int is_max) {
    int xsize = imOut->xsize, ysize = imOut->ysize;
    int x, y, k;
    UINT8 *rows;
    if (xsize == 0 || ysize == 0) {
        return 1;
    }
    rows = malloc((size_t)xsize * (ysize + size - 1));
    if (!rows) {
        return 0;
    }

#define RANK_MINMAX(op)                                                 \
    for (y = 0; y < ysize + size - 1; y++) {                            \
        const UINT8 *in = im->image8[y];                                \
        UINT8 *restrict row = rows + (size_t)y * xsize;                 \
        memcpy(row, in, xsize);                                         \
        for (k = 1; k < size; k++) {                                    \
            for (x = 0; x < xsize; x++) {                               \
                row[x] = op(row[x], in[x + k]);                         \
            }                                                           \
        }                                                               \
    }                                                                   \
    for (y = 0; y < ysize; y++) {                                       \
        UINT8 *restrict out = imOut->image8[y];                         \
        memcpy(out, rows + (size_t)y * xsize, xsize);                   \
        for (k = 1; k < size; k++) {                                    \
            const UINT8 *restrict row = rows + (size_t)(y + k) * xsize; \
            for (x = 0; x < xsize; x++) {                               \
                out[x] = op(out[x], row[x]);                            \
            }                                                           \
        }                                                               \
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

    free(rows);
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
        if (!RankFilterMinMax_u8(imOut, im, size, rank != 0)) {
            ImagingSectionLeave(&cookie);
            goto nomemory;
        }
    } else if (im->image8 && size == 3) {
        RankFilter3x3_u8(imOut, im, rank);
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
