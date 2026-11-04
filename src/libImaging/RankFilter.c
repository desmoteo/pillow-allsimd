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
#include "RankFilter.h"

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

/* out[x] = the minimum or maximum of src[0][x] .. src[count - 1][x]: the
   active tier's kernel (RankFilterSIMD.c) does the bytes it can, one pass
   with all sources at once, and the loop below finishes; without one, count
   passes over the line, which GCC vectorizes at -O3 (restrict: without it,
   its cost model leaves the loop scalar rather than check that the rows
   don't overlap). */
static void
rank_minmax_line(UINT8 *restrict out, const UINT8 **src, int count, int n, int is_max) {
    int x, j;
    if (!ImagingSIMD->rank->minmax_line) {
        memcpy(out, src[0], n);
        for (j = 1; j < count; j++) {
            const UINT8 *restrict s = src[j];
            UINT8 *restrict d = out;
            if (is_max) {
                for (x = 0; x < n; x++) {
                    d[x] = d[x] > s[x] ? d[x] : s[x];
                }
            } else {
                for (x = 0; x < n; x++) {
                    d[x] = d[x] < s[x] ? d[x] : s[x];
                }
            }
        }
        return;
    }
    x = ImagingSIMD->rank->minmax_line(out, src, count, n, is_max);
    for (; x < n; x++) {
        UINT8 v = src[0][x];
        for (j = 1; j < count; j++) {
            UINT8 s = src[j][x];
            v = is_max ? (s > v ? s : v) : (s < v ? s : v);
        }
        out[x] = v;
    }
}

/* Minimum (rank 0) or maximum (rank size * size - 1) of any size: of each
   row's size neighbours, then of size of those row results. The row
   results go to a ring of size rows, so the work stays in cache. Returns 0
   if out of memory. margin and ps are as for RankFilterNetwork_u8. */
static int
RankFilterMinMax_u8(
    Imaging imOut, Imaging im, int size, int is_max, int margin, int ps
) {
    int xsize = imOut->xsize * ps, ysize = imOut->ysize;
    /* with a margin, the bytes at each end whose neighbours are partly
       outside the row: edge pixels repeated, as expanded */
    int edge = margin * ps < xsize / 2 ? margin * ps : (xsize + 1) / 2;
    int inner = xsize - 2 * edge;
    int y, k, j;
    UINT8 *ring;
    const UINT8 **src;
    if (xsize == 0 || ysize == 0) {
        return 1;
    }
    ring = malloc((size_t)xsize * size);
    src = malloc(size * sizeof(*src));
    if (!ring || !src) {
        free(ring);
        free(src);
        return 0;
    }

/* Ring slot k holds the row result of source row k - margin, clipped to
   the image; k - margin is the source row of window row k. Without a
   margin, the source rows are expanded and the row results start at
   their first pixel; with one, the inner bytes read the row itself and
   the edges clamp the pixel index. */
#define RANK_SLOT(k) (ring + (size_t)((k) % size) * xsize)
#define RANK_HROW(op, k)                                          \
    do {                                                          \
        int r_ = (k) - margin;                                    \
        const UINT8 *in = (UINT8 *)im->image                      \
                              [r_ < 0           ? 0               \
                               : r_ < im->ysize ? r_              \
                                                : im->ysize - 1]; \
        UINT8 *restrict row = RANK_SLOT(k);                       \
        if (margin) {                                             \
            int x_, last_ = imOut->xsize - 1;                     \
            if (inner > 0) {                                      \
                RANK_LINE(op, row + edge, in + j * ps, inner);    \
            }                                                     \
            for (x_ = 0; x_ < xsize; x_++) {                      \
                int p_ = x_ / ps - margin, c_ = x_ % ps;          \
                UINT8 v_;                                         \
                if (x_ == edge && inner > 0) {                    \
                    x_ += inner - 1;                              \
                    continue;                                     \
                }                                                 \
                v_ = in[(p_ < 0 ? 0 : p_) * ps + c_];             \
                for (j = 1; j < size; j++) {                      \
                    int q_ = p_ + j > last_ ? last_ : p_ + j;     \
                    v_ = op(v_, in[(q_ < 0 ? 0 : q_) * ps + c_]); \
                }                                                 \
                row[x_] = v_;                                     \
            }                                                     \
        } else {                                                  \
            RANK_LINE(op, row, in + j * ps, xsize);               \
        }                                                         \
    } while (0)
#define RANK_MINMAX(op)                                 \
    for (k = 0; k < size - 1; k++) {                    \
        RANK_HROW(op, k);                               \
    }                                                   \
    for (y = 0; y < ysize; y++) {                       \
        UINT8 *restrict out = (UINT8 *)imOut->image[y]; \
        RANK_HROW(op, y + size - 1);                    \
        RANK_LINE(op, out, RANK_SLOT(y + j), xsize);    \
    }
/* dst = op of source(j) for j = 0 .. size - 1, n bytes */
#define RANK_LINE(op, dst, source, n) \
    for (j = 0; j < size; j++) {      \
        src[j] = (source);            \
    }                                 \
    rank_minmax_line(dst, src, size, n, is_max);
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
    } else if (im->image8 && size == 3 && rank == 4 && ImagingSIMD->rank->median3) {
        /* the active tier's medians in registers (RankFilterSIMD.c) */
        ImagingSIMD->rank->median3(imOut, im, 0, 1);
    } else if (im->image8 && size == 5 && rank == 12 && ImagingSIMD->rank->median5) {
        ImagingSIMD->rank->median5(imOut, im, 0, 1);
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
    if (ImagingSIMD->rank->median3 &&
        ((size == 3 && rank == 4) || (size == 5 && rank == 12))) {
        return 1;
    }
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
        } else if (size == 3 && rank == 4 && ImagingSIMD->rank->median3) {
            ImagingSIMD->rank->median3(imOut, im, 1, ps);
            ok = 1;
        } else if (size == 5 && rank == 12 && ImagingSIMD->rank->median5) {
            ImagingSIMD->rank->median5(imOut, im, 2, ps);
            ok = 1;
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
