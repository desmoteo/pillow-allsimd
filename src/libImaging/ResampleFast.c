/* Fast mode (design/SIMD_FAST_KERNELS.md): resize of 8-bit images with 4-byte
 * pixels, the two passes fused, with 6 more bits between them and the
 * vertical pass in 16-bit lanes.
 *
 * Each source row goes through the horizontal kernels into a ring of rows
 * kept in the cache, in place of the exact path's whole intermediate
 * image. The kernels are the exact ones compiled again (RESAMPLE_OUT16 in
 * ResampleSIMDHorizontalConv.c): the same integer sums, rounded to 2^-6
 * levels instead of whole ones, clipped to 0 .. 255 as there, stored
 * doubled in 16 bits. Each output row is the vertical combination of ring
 * rows with weights in 2^-14 units (edge weights may exceed 1), each
 * product rounded,
 *
 *     rf_mul(a, w) = (a * w + 2^14) >> 15,
 *
 * in units of 2^-6 levels. That is pmulhrsw on x86 and vqrdmulh on arm64,
 * so the SSE4, AVX2 and NEON loops compute the portable loop's integers
 * exactly and every build gives the same pixels. The sums stay below 2^15
 * in magnitude (the weights' absolute values add up to RF_WSUM_MAX at most,
 * checked), so they need no saturation and their order doesn't matter. On
 * arm64 this doubles the lanes of the exact vertical kernel's 32-bit
 * products.
 *
 * Included by Resample.c. */

#include "ResampleFast.h"

/* The vertical pass's weights in 2^-14 units, ksize per output, from the
   exact path's coefficients. Returns 0 if the sums could overflow. */
static int
rf_weights(INT16 *w, int size, int ksize, const int *bounds, const double *kk) {
    int i, k;
    if (ksize > RF_TAPS_MAX) {
        return 0;
    }
    for (i = 0; i < size; i++) {
        double total = 0;
        for (k = 0; k < ksize; k++) {
            double v = k < bounds[i * 2 + 1] ? kk[i * ksize + k] : 0;
            total += fabs(v);
            w[i * ksize + k] = (INT16)lround(v * 16384);
        }
        if (total > RF_WSUM_MAX) {
            return 0;
        }
    }
    return 1;
}

/* An output row of n bytes from taps ring rows with weights w. */
static void
rf_vrow(UINT8 *restrict out, INT16 **rows, const INT16 *w, int taps, int n) {
    int x = IMAGING_SIMD_CALL(resample->fast_vrow, out, rows, w, taps, n), k;
    for (; x < n; x++) {
        int s = 0;
        for (k = 0; k < taps; k++) {
            s += rf_mul(rows[k][x], w[k]);
        }
        s = (s + (1 << (RF_FRAC - 1))) >> RF_FRAC;
        out[x] = (UINT8)(s < 0 ? 0 : s > 255 ? 255 : s);
    }
}

/* The horizontal pass of n source rows (up to 4) into ring rows, with the
   16-bit versions of the kernels _ImagingResampleHorizontal_8bpc uses. */
static void
rf_hrows(
    uint64_t **out,
    UINT32 **in,
    int n,
    int xsize,
    int bands,
    int *bounds,
    INT16 *kk,
    int ksize,
    int coefs_precision
) {
    const ImagingSIMDResampleKernels *simd = ImagingSIMD->resample;
    ImagingResampleRow16Fn row =
        simd->horizontal_8u_16 ? simd->horizontal_8u_16 : resample_horizontal_8u_16;
    ImagingResampleRows16Fn rows4 = simd->horizontal_8u4x_16
                                        ? simd->horizontal_8u4x_16
                                        : resample_horizontal_8u4x_16;
    int r = 0;
    if (bands == 3 && !simd->horizontal_8u4x_16) {
        /* the portable kernels have one for 3 bands */
        for (; r < n; r++) {
            resample_horizontal_8u3_16(
                out[r], in[r], xsize, bounds, kk, ksize, coefs_precision
            );
        }
    }
    if (n - r == 4) {
        rows4(
            out[0],
            out[1],
            out[2],
            out[3],
            in[0],
            in[1],
            in[2],
            in[3],
            xsize,
            bounds,
            kk,
            ksize,
            coefs_precision
        );
        r = n;
    }
    for (; r < n; r++) {
        row(out[r], in[r], xsize, bounds, kk, ksize, coefs_precision);
    }
}

/* The resize of an 8-bit image with 4-byte pixels (RGB, RGBA, CMYK, LA
   and the like) in fast mode, horizontal pass first, with a vertical pass;
   with alpha's row functions set, premultiplied by alpha. Returns 0 to
   leave the resize to the exact passes (weights whose sums could overflow;
   nothing is changed then), else 1 with *out the result, or NULL with an
   error set. The vertical bounds start at source row vbase; the
   horizontal coefficients are normalized in place. */
static int
resample_fast8(
    Imaging *out,
    Imaging imIn,
    int xsize,
    int ysize,
    int need_horizontal,
    int ksize_h,
    int *bounds_h,
    double *kk_h,
    int ksize_v,
    int *bounds_v,
    double *kk_v,
    int vbase,
    ImagingShuffler premultiply,
    ImagingShuffler unpremultiply
) {
    ImagingSectionCookie cookie;
    INT16 *wv, *kk = NULL;
    INT16 *ring = NULL, *rows[RF_TAPS_MAX];
    void *ring_block = NULL;
    UINT8 *scratch = NULL;
    Imaging imOut = NULL;
    /* 3 more rows than the most taps, to resample 4 rows ahead of need */
    int ring_rows = ksize_v + 3, coefs_precision = 0;
    /* the source rows the vertical pass reads, from vbase */
    int src_rows = bounds_v[ysize * 2 - 2] + bounds_v[ysize * 2 - 1];
    size_t values = (size_t)xsize * 4;
    /* ring rows in values: whole cache lines, one line apart from a
       multiple of 4 KB, so that rows don't share L1 sets at equal offsets
       (on x86, 2048-pixel rows of exactly 16 KB were 10% slower) */
    size_t stride = ((values + 31) & ~(size_t)31) + 32;
    int y, k, next;

    *out = NULL;
    wv = malloc((size_t)ysize * ksize_v * sizeof(INT16));
    if (!wv) {
        ImagingError_MemoryError();
        return 1;
    }
    if (!rf_weights(wv, ysize, ksize_v, bounds_v, kk_v)) {
        free(wv);
        return 0;
    }
    if (need_horizontal) {
        kk = (INT16 *)kk_h;
        coefs_precision = normalize_coeffs_8bpc(xsize, ksize_h, kk_h);
    }
    ring_block = malloc(ring_rows * stride * sizeof(INT16) + 63);
    ring = ring_block ? (INT16 *)(((uintptr_t)ring_block + 63) & ~(uintptr_t)63) : NULL;
    scratch = premultiply ? malloc(4 * (size_t)imIn->linesize) : NULL;
    imOut = ImagingNewDirty(imIn->mode, xsize, ysize);
    if (!ring || (premultiply && !scratch && imIn->linesize) || !imOut) {
        ImagingError_MemoryError();
        goto done;
    }

    MUTEX_LOCK(&imIn->mutex);
    ImagingSectionEnter(&cookie);
    next = 0; /* the next source row to resample, from vbase */
    for (y = 0; y < ysize; y++) {
        int ymin = bounds_v[y * 2], taps = bounds_v[y * 2 + 1];
        UINT8 *line = (UINT8 *)imOut->image32[y];
        while (next < ymin + taps) {
            /* up to 4 rows at a time; the ring's 3 extra rows keep every row
               still needed */
            UINT32 *in[4];
            uint64_t *h[4];
            int n = src_rows - next < 4 ? src_rows - next : 4, r;
            for (r = 0; r < n; r++) {
                in[r] = (UINT32 *)imIn->image32[vbase + next + r];
                h[r] = (uint64_t *)(ring + (size_t)((next + r) % ring_rows) * stride);
                if (premultiply) {
                    UINT8 *p = scratch + (size_t)r * imIn->linesize;
                    premultiply(p, (UINT8 *)in[r], imIn->xsize);
                    in[r] = (UINT32 *)p;
                }
                if (!need_horizontal) {
                    INT16 *v = (INT16 *)h[r];
                    size_t x;
                    for (x = 0; x < values; x++) {
                        v[x] = (INT16)(((UINT8 *)in[r])[x] << (RF_FRAC + 1));
                    }
                }
            }
            if (need_horizontal) {
                rf_hrows(
                    h,
                    in,
                    n,
                    xsize,
                    imIn->bands,
                    bounds_h,
                    kk,
                    ksize_h,
                    coefs_precision - RF_FRAC
                );
            }
            next += n;
        }
        for (k = 0; k < taps; k++) {
            rows[k] = ring + (size_t)((ymin + k) % ring_rows) * stride;
        }
        rf_vrow(line, rows, wv + (size_t)y * ksize_v, taps, xsize * 4);
        if (unpremultiply) {
            unpremultiply(line, line, xsize);
        }
    }
    ImagingSectionLeave(&cookie);
    MUTEX_UNLOCK(&imIn->mutex);

    *out = imOut;
    imOut = NULL;
done:
    if (imOut) {
        ImagingDelete(imOut);
    }
    free(wv);
    free(ring_block);
    free(scratch);
    return 1;
}
