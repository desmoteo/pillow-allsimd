/*
 * The Imaging library
 *
 * Fast mode affine, perspective and quad transforms with the bilinear and
 * bicubic filters (design/APPROXIMATE_KERNELS_PLAN.md, item 4;
 * design/SIMD_FAST_KERNELS.md).
 *
 * The exact transforms compute each pixel's source position in doubles and
 * filter it in doubles, truncating the result; the compilers fuse their
 * multiply-adds differently on x86 and arm64. Here the positions are fixed
 * point, computed in exact integer arithmetic (perspective: integer
 * numerators and denominators, then one IEEE division and two products, in
 * 32-bit floats for images below TF_FLOAT_MAX on both sides),
 * rounded to 1/256 pixel, and the filters are integer sums of products with
 * one rounding at the end (bicubic: one more between its passes), so every
 * build on every architecture gives the same pixels. The scalar functions
 * define the result; the SSE4 and NEON loops for pixels away from the image
 * edges compute the same integers.
 */

#include "Imaging.h"

#include <math.h>
#include <stdint.h>

#include "TransformFast.h"

static int
tf_round(double v, int64_t *out) {
    if (!(fabs(v) < 9.0e18)) {
        return 0;
    }
    *out = (int64_t)floor(v + 0.5);
    return 1;
}

static int
tf_map_init(TFMap *m, int method, const double *a) {
    int i, ok = 1;
    m->method = method;
    if (method == IMAGING_TRANSFORM_AFFINE) {
        ok &= tf_round(a[0] * 2147483648.0, &m->c[0]);
        ok &= tf_round(a[1] * 2147483648.0, &m->c[1]);
        ok &= tf_round(a[2] * 4294967296.0, &m->c[2]);
        ok &= tf_round(a[3] * 2147483648.0, &m->c[3]);
        ok &= tf_round(a[4] * 2147483648.0, &m->c[4]);
        ok &= tf_round(a[5] * 4294967296.0, &m->c[5]);
    } else if (method == IMAGING_TRANSFORM_QUAD) {
        for (i = 0; i < 2; i++) {
            ok &= tf_round(a[4 * i + 0] * 4294967296.0, &m->c[4 * i + 0]);
            ok &= tf_round(a[4 * i + 1] * 2147483648.0, &m->c[4 * i + 1]);
            ok &= tf_round(a[4 * i + 2] * 2147483648.0, &m->c[4 * i + 2]);
            ok &= tf_round(a[4 * i + 3] * 1073741824.0, &m->c[4 * i + 3]);
        }
    } else {
        const double k = 68719476736.0; /* 2^36: (2i+1) / 2 times 2^37 */
        ok &= tf_round(a[0] * k, &m->c[0]);
        ok &= tf_round(a[1] * k, &m->c[1]);
        ok &= tf_round(a[2] * 2 * k, &m->c[2]);
        ok &= tf_round(a[3] * k, &m->c[3]);
        ok &= tf_round(a[4] * k, &m->c[4]);
        ok &= tf_round(a[5] * 2 * k, &m->c[5]);
        ok &= tf_round(a[6] * k, &m->c[6]);
        ok &= tf_round(a[7] * k, &m->c[7]);
        m->c[8] = (int64_t)1 << 37;
    }
    return ok;
}

/* Whether the region's corners map to positions the fixed point holds:
   affine and quad positions are then in range everywhere in it (linear in
   each direction), perspective ones are checked per pixel. */
static int
tf_region_ok(const TFMap *m, int w, int h) {
    int k;
    if (m->method == IMAGING_TRANSFORM_PERSPECTIVE) {
        /* the integer numerators and denominator below 2^53 */
        const int64_t lim = (int64_t)1 << 52;
        const int64_t u[2] = {1, 2 * (int64_t)w + 1}, v[2] = {1, 2 * (int64_t)h + 1};
        for (k = 0; k < 4; k++) {
            const int64_t uu = u[k & 1], vv = v[k >> 1];
            const double nx = (double)m->c[0] * uu + (double)m->c[1] * vv + m->c[2];
            const double ny = (double)m->c[3] * uu + (double)m->c[4] * vv + m->c[5];
            const double d = (double)m->c[6] * uu + (double)m->c[7] * vv + m->c[8];
            if (!(fabs(nx) < lim && fabs(ny) < lim && fabs(d) < lim)) {
                return 0;
            }
        }
        return 1;
    }
    for (k = 0; k < 4; k++) {
        const double u = k & 1 ? 2.0 * w + 1 : 1, v = k & 2 ? 2.0 * h + 1 : 1;
        double x, y;
        if (m->method == IMAGING_TRANSFORM_AFFINE) {
            x = (double)m->c[0] * u + (double)m->c[1] * v + m->c[2];
            y = (double)m->c[3] * u + (double)m->c[4] * v + m->c[5];
        } else {
            x = m->c[0] + (double)m->c[1] * u + (double)m->c[2] * v +
                (double)m->c[3] * u * v;
            y = m->c[4] + (double)m->c[5] * u + (double)m->c[6] * v +
                (double)m->c[7] * u * v;
        }
        if (!(fabs(x) < TF_LIMIT * 4294967296.0 && fabs(y) < TF_LIMIT * 4294967296.0)) {
            return 0;
        }
    }
    return 1;
}

/* The bicubic filter's weights (Pillow's, a = -1) at d = k / 256, times
   4096 and rounded, with the remainder in the largest; the values at
   dyadic d are exact in doubles, so this is the same everywhere. */
static void
tf_cubic_table(INT16 (*w)[4]) {
    int k, i;
    for (k = 0; k < 256; k++) {
        const double d = k / 256.0, d2 = d * d, d3 = d2 * d;
        const double f[4] = {-d + 2 * d2 - d3, 1 - 2 * d2 + d3, d + d2 - d3, d3 - d2};
        int sum = 0, big = 0;
        for (i = 0; i < 4; i++) {
            w[k][i] = (INT16)floor(f[i] * 4096 + 0.5);
            sum += w[k][i];
            if (w[k][i] > w[k][big]) {
                big = i;
            }
        }
        w[k][big] += 4096 - sum;
    }
}

/* Transforms the region x0..x1, y0..y1 of imOut, as ImagingGenericTransform
   does with the bilinear or bicubic filter, in fast mode. Returns 1 if done,
   0 if this transform isn't one the fast kernels take (the caller then runs
   the exact one), -1 on error. */
int
ImagingTransformFast(
    Imaging imOut,
    Imaging imIn,
    int method,
    int x0,
    int y0,
    int x1,
    int y1,
    const double *a,
    int filterid,
    int fill
) {
    ImagingSectionCookie cookie;
    TFMap map;
    INT16 cubic[256][4];
    const int bicubic = filterid == IMAGING_TRANSFORM_BICUBIC;
    const int ps = imIn->pixelsize;
    const ModeID mode = imIn->mode;

    if (filterid != IMAGING_TRANSFORM_BILINEAR &&
        filterid != IMAGING_TRANSFORM_BICUBIC) {
        return 0;
    }
    if (method != IMAGING_TRANSFORM_AFFINE && method != IMAGING_TRANSFORM_PERSPECTIVE &&
        method != IMAGING_TRANSFORM_QUAD) {
        return 0;
    }
    if (imIn->type != IMAGING_TYPE_UINT8 || imOut->mode != mode ||
        !(mode == IMAGING_MODE_L ||
          (ps == 4 && (mode == IMAGING_MODE_RGB || mode == IMAGING_MODE_RGBA ||
                       mode == IMAGING_MODE_RGBa || mode == IMAGING_MODE_RGBX ||
                       mode == IMAGING_MODE_CMYK)))) {
        return 0;
    }
    /* positions in 1/512 pixel fit 31 bits */
    if (imIn->xsize >= (1 << 22) || imIn->ysize >= (1 << 22)) {
        return 0;
    }
    if (x0 < 0) {
        x0 = 0;
    }
    if (y0 < 0) {
        y0 = 0;
    }
    if (x1 > imOut->xsize) {
        x1 = imOut->xsize;
    }
    if (y1 > imOut->ysize) {
        y1 = imOut->ysize;
    }
    if (!tf_map_init(&map, method, a) || !tf_region_ok(&map, x1 - x0, y1 - y0)) {
        return 0;
    }
    if (bicubic) {
        tf_cubic_table(cubic);
    }

    ImagingCopyPalette(imOut, imIn);
    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

    /* the active tier's loops (TransformFastSIMD.c), or the scalar ones */
    if (!IMAGING_SIMD_CALL(
            transform_fast->tiles,
            imOut,
            imIn,
            &map,
            cubic,
            x0,
            y0,
            x1,
            y1,
            bicubic,
            fill
        )) {
        tf_transform_tiles(imOut, imIn, &map, cubic, x0, y0, x1, y1, bicubic, fill);
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);
    return 1;
}
