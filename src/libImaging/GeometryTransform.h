/* The transforms' coordinate maps, tile loops and clamps (Geometry.c),
   shared with the SIMD filters' tile loops (GeometrySIMD.c), where the
   compiler inlines them. The tile macros expect imOut, imIn, x0, y0, x1,
   y1, transform_data, fill, and the locals tx, ty, x, y, xx, yy and out. */
#ifndef __GEOMETRY_TRANSFORM_H__
#define __GEOMETRY_TRANSFORM_H__

#include <math.h>
#include <string.h>

#define COORD(v) ((v) < 0.0 ? -1 : ((int)(v)))
#define FLOOR(v) ((v) < 0.0 ? ((int)floor(v)) : ((int)(v)))

#define XCLIP(x) (((x) < 0) ? 0 : ((x) < xsize) ? (x) : xsize - 1)
#define YCLIP(y) (((y) < 0) ? 0 : ((y) < ysize) ? (y) : ysize - 1)

/* transform primitives (ImagingTransformMap) */

static inline int
affine_transform(double *xout, double *yout, int x, int y, void *data) {
    /* full moon tonight.  your compiler will generate bogus code
       for simple expressions, unless you reorganize the code, or
       install Service Pack 3 */

    double *a = (double *)data;
    double a0 = a[0];
    double a1 = a[1];
    double a2 = a[2];
    double a3 = a[3];
    double a4 = a[4];
    double a5 = a[5];

    double xin = x + 0.5;
    double yin = y + 0.5;

    xout[0] = a0 * xin + a1 * yin + a2;
    yout[0] = a3 * xin + a4 * yin + a5;

    return 1;
}

static inline int
perspective_transform(double *xout, double *yout, int x, int y, void *data) {
    double *a = (double *)data;
    double a0 = a[0];
    double a1 = a[1];
    double a2 = a[2];
    double a3 = a[3];
    double a4 = a[4];
    double a5 = a[5];
    double a6 = a[6];
    double a7 = a[7];

    double xin = x + 0.5;
    double yin = y + 0.5;

    xout[0] = (a0 * xin + a1 * yin + a2) / (a6 * xin + a7 * yin + 1);
    yout[0] = (a3 * xin + a4 * yin + a5) / (a6 * xin + a7 * yin + 1);

    return 1;
}

static inline int
quad_transform(double *xout, double *yout, int x, int y, void *data) {
    /* quad warp: map quadrilateral to rectangle */

    double *a = (double *)data;
    double a0 = a[0];
    double a1 = a[1];
    double a2 = a[2];
    double a3 = a[3];
    double a4 = a[4];
    double a5 = a[5];
    double a6 = a[6];
    double a7 = a[7];

    double xin = x + 0.5;
    double yin = y + 0.5;

    xout[0] = a0 + a1 * xin + a2 * yin + a3 * xin * yin;
    yout[0] = a4 + a5 * xin + a6 * yin + a7 * xin * yin;

    return 1;
}

#ifndef TRANSFORM_TILE
#define TRANSFORM_TILE 64
#endif

/* perspective_transform for a row of TRANSFORM_TILE pixels from x: the
   same expressions, in a loop of fixed length that the compiler
   vectorizes, so that the divisions run several at once. */
static inline void
perspective_row(
    double *restrict xs, double *restrict ys, const double *a, int x, int y
) {
    double a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    double a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    double yin = y + 0.5;
    int i;
    for (i = 0; i < TRANSFORM_TILE; i++) {
        double xin = (x + i) + 0.5;
        double den = a6 * xin + a7 * yin + 1;
        xs[i] = (a0 * xin + a1 * yin + a2) / den;
        ys[i] = (a3 * xin + a4 * yin + a5) / den;
    }
}

/* Tiles of the output: a rotation crosses a new source row every few
   pixels of an output row, and a tile keeps those rows in cache. Each
   pixel is computed as before, so the order doesn't change results. */
#define TRANSFORM_TILES(transform, filter)                                      \
    for (ty = y0; ty < y1; ty += TRANSFORM_TILE) {                              \
        int ty1 = ty + TRANSFORM_TILE < y1 ? ty + TRANSFORM_TILE : y1;          \
        for (tx = x0; tx < x1; tx += TRANSFORM_TILE) {                          \
            int tx1 = tx + TRANSFORM_TILE < x1 ? tx + TRANSFORM_TILE : x1;      \
            for (y = ty; y < ty1; y++) {                                        \
                out = imOut->image[y] + tx * imOut->pixelsize;                  \
                for (x = tx; x < tx1; x++) {                                    \
                    if (!transform(&xx, &yy, x - x0, y - y0, transform_data) || \
                        !filter(out, imIn, xx, yy)) {                           \
                        if (fill) {                                             \
                            memset(out, 0, imOut->pixelsize);                   \
                        }                                                       \
                    }                                                           \
                    out += imOut->pixelsize;                                    \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }

/* Perspective transforms: each row of a tile gets its coordinates at
   once, from perspective_row. */
#define PERSPECTIVE_TILES(filter)                                             \
    for (ty = y0; ty < y1; ty += TRANSFORM_TILE) {                            \
        int ty1 = ty + TRANSFORM_TILE < y1 ? ty + TRANSFORM_TILE : y1;        \
        for (tx = x0; tx < x1; tx += TRANSFORM_TILE) {                        \
            int i, n = (tx + TRANSFORM_TILE < x1 ? TRANSFORM_TILE : x1 - tx); \
            for (y = ty; y < ty1; y++) {                                      \
                double xs[TRANSFORM_TILE], ys[TRANSFORM_TILE];                \
                perspective_row(xs, ys, transform_data, tx - x0, y - y0);     \
                out = imOut->image[y] + tx * imOut->pixelsize;                \
                for (i = 0; i < n; i++) {                                     \
                    if (!filter(out, imIn, xs[i], ys[i]) && fill) {           \
                        memset(out, 0, imOut->pixelsize);                     \
                    }                                                         \
                    out += imOut->pixelsize;                                  \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }

#endif
