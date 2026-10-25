/*
 * The Python Imaging Library
 * $Id$
 *
 * Alpha composite imSrc over imDst.
 * https://en.wikipedia.org/wiki/Alpha_compositing
 *
 * See the README file for details on usage and redistribution.
 */

#include "Imaging.h"

#define PRECISION_BITS 7

typedef struct {
    UINT8 r;
    UINT8 g;
    UINT8 b;
    UINT8 a;
} rgba8;

#if defined(IMAGING_SIMD_SSE4)

#define MM_SHIFTDIV255_epi16(src) \
    _mm_srli_epi16(_mm_add_epi16(src, _mm_srli_epi16(src, 8)), 8)

/* Composites four pixels. Every pixel of a row, including the last
   xsize % 4, goes through this code, so results don't depend on x. */
static inline __m128i
alpha_composite_4px(__m128i mm_dst, __m128i mm_src) {
    __m128i mm_max_alpha = _mm_set1_epi32(255);
    __m128i mm_max_alpha2 = _mm_set1_epi32(255 * 255);
    __m128i mm_zero = _mm_setzero_si128();
    __m128i mm_half = _mm_set1_epi16(128);
    __m128i mm_get_lo =
        _mm_set_epi8(-1, -1, 5, 4, 5, 4, 5, 4, -1, -1, 1, 0, 1, 0, 1, 0);
    __m128i mm_get_hi =
        _mm_set_epi8(-1, -1, 13, 12, 13, 12, 13, 12, -1, -1, 9, 8, 9, 8, 9, 8);
    __m128i mm_dst_lo, mm_dst_hi, mm_src_lo, mm_src_hi;
    __m128i mm_dst_a, mm_src_a, mm_out_a, mm_blend;
    __m128i mm_coef1, mm_coef2, mm_out_lo, mm_out_hi;

    // [16] a1 b1 g1 r1 a0 b0 g0 r0
    mm_dst_lo = _mm_unpacklo_epi8(mm_dst, mm_zero);
    // [16] a3 b3 g3 r3 a2 b2 g2 r2
    mm_dst_hi = _mm_unpackhi_epi8(mm_dst, mm_zero);
    mm_src_lo = _mm_unpacklo_epi8(mm_src, mm_zero);
    mm_src_hi = _mm_unpackhi_epi8(mm_src, mm_zero);

    // [32] a3 a2 a1 a0
    mm_dst_a = _mm_srli_epi32(mm_dst, 24);
    mm_src_a = _mm_srli_epi32(mm_src, 24);

    // Compute coefficients
    // blend = dst->a * (255 - src->a)
    // [16] xx b3 xx b2 xx b1 xx b0
    mm_blend = _mm_mullo_epi16(mm_dst_a, _mm_sub_epi32(mm_max_alpha, mm_src_a));
    // outa = src->a * 255 + blend
    // [16] xx a3 xx a2 xx a1 xx a0
    mm_out_a = _mm_add_epi32(_mm_mullo_epi16(mm_src_a, mm_max_alpha), mm_blend);
    // coef1 = src->a * 255 * 255 / outa
    mm_coef1 = _mm_mullo_epi32(mm_src_a, mm_max_alpha2);
    // [8]  xx xx xx c3 xx xx xx c2 xx xx xx c1 xx xx xx c0
    mm_coef1 = _mm_cvtps_epi32(
        _mm_mul_ps(_mm_cvtepi32_ps(mm_coef1), _mm_rcp_ps(_mm_cvtepi32_ps(mm_out_a)))
    );
    // [8]  xx xx xx c3 xx xx xx c2 xx xx xx c1 xx xx xx c0
    mm_coef2 = _mm_sub_epi32(mm_max_alpha, mm_coef1);

    mm_out_lo = _mm_add_epi16(
        _mm_mullo_epi16(mm_src_lo, _mm_shuffle_epi8(mm_coef1, mm_get_lo)),
        _mm_mullo_epi16(mm_dst_lo, _mm_shuffle_epi8(mm_coef2, mm_get_lo))
    );
    mm_out_lo = _mm_or_si128(
        mm_out_lo, _mm_slli_epi64(_mm_unpacklo_epi32(mm_out_a, mm_zero), 48)
    );
    mm_out_lo = _mm_add_epi16(mm_out_lo, mm_half);
    mm_out_lo = MM_SHIFTDIV255_epi16(mm_out_lo);

    mm_out_hi = _mm_add_epi16(
        _mm_mullo_epi16(mm_src_hi, _mm_shuffle_epi8(mm_coef1, mm_get_hi)),
        _mm_mullo_epi16(mm_dst_hi, _mm_shuffle_epi8(mm_coef2, mm_get_hi))
    );
    mm_out_hi = _mm_or_si128(
        mm_out_hi, _mm_slli_epi64(_mm_unpackhi_epi32(mm_out_a, mm_zero), 48)
    );
    mm_out_hi = _mm_add_epi16(mm_out_hi, mm_half);
    mm_out_hi = MM_SHIFTDIV255_epi16(mm_out_hi);

    return _mm_packus_epi16(mm_out_lo, mm_out_hi);
}

#undef MM_SHIFTDIV255_epi16

#endif

#if defined(IMAGING_SIMD_AVX2)

#define MM_SHIFTDIV255_epi16(src) \
    _mm256_srli_epi16(_mm256_add_epi16(src, _mm256_srli_epi16(src, 8)), 8)

/* Same arithmetic as alpha_composite_4px, for eight pixels. */
static inline __m256i
alpha_composite_8px(__m256i mm_dst, __m256i mm_src) {
    __m256i vmm_max_alpha = _mm256_set1_epi32(255);
    __m256i vmm_max_alpha2 = _mm256_set1_epi32(255 * 255);
    __m256i vmm_zero = _mm256_setzero_si256();
    __m256i vmm_half = _mm256_set1_epi16(128);
    __m256i vmm_get_lo = _mm256_set_epi8(
        -1,
        -1,
        5,
        4,
        5,
        4,
        5,
        4,
        -1,
        -1,
        1,
        0,
        1,
        0,
        1,
        0,
        -1,
        -1,
        5,
        4,
        5,
        4,
        5,
        4,
        -1,
        -1,
        1,
        0,
        1,
        0,
        1,
        0
    );
    __m256i vmm_get_hi = _mm256_set_epi8(
        -1,
        -1,
        13,
        12,
        13,
        12,
        13,
        12,
        -1,
        -1,
        9,
        8,
        9,
        8,
        9,
        8,
        -1,
        -1,
        13,
        12,
        13,
        12,
        13,
        12,
        -1,
        -1,
        9,
        8,
        9,
        8,
        9,
        8
    );
    __m256i mm_dst_lo, mm_dst_hi, mm_src_lo, mm_src_hi;
    __m256i mm_dst_a, mm_src_a, mm_out_a, mm_blend;
    __m256i mm_coef1, mm_coef2, mm_out_lo, mm_out_hi;

    mm_dst_lo = _mm256_unpacklo_epi8(mm_dst, vmm_zero);
    mm_dst_hi = _mm256_unpackhi_epi8(mm_dst, vmm_zero);
    mm_src_lo = _mm256_unpacklo_epi8(mm_src, vmm_zero);
    mm_src_hi = _mm256_unpackhi_epi8(mm_src, vmm_zero);

    mm_dst_a = _mm256_srli_epi32(mm_dst, 24);
    mm_src_a = _mm256_srli_epi32(mm_src, 24);

    // Compute coefficients
    // blend = dst->a * (255 - src->a);  16 bits
    mm_blend = _mm256_mullo_epi16(mm_dst_a, _mm256_sub_epi32(vmm_max_alpha, mm_src_a));
    // outa = src->a * 255 + dst->a * (255 - src->a);  16 bits
    mm_out_a = _mm256_add_epi32(_mm256_mullo_epi16(mm_src_a, vmm_max_alpha), mm_blend);
    mm_coef1 = _mm256_mullo_epi32(mm_src_a, vmm_max_alpha2);
    // 8 bits
    mm_coef1 = _mm256_cvtps_epi32(_mm256_mul_ps(
        _mm256_cvtepi32_ps(mm_coef1), _mm256_rcp_ps(_mm256_cvtepi32_ps(mm_out_a))
    ));
    // 8 bits
    mm_coef2 = _mm256_sub_epi32(vmm_max_alpha, mm_coef1);

    mm_out_lo = _mm256_add_epi16(
        _mm256_mullo_epi16(mm_src_lo, _mm256_shuffle_epi8(mm_coef1, vmm_get_lo)),
        _mm256_mullo_epi16(mm_dst_lo, _mm256_shuffle_epi8(mm_coef2, vmm_get_lo))
    );
    mm_out_lo = _mm256_or_si256(
        mm_out_lo, _mm256_slli_epi64(_mm256_unpacklo_epi32(mm_out_a, vmm_zero), 48)
    );
    mm_out_lo = _mm256_add_epi16(mm_out_lo, vmm_half);
    mm_out_lo = MM_SHIFTDIV255_epi16(mm_out_lo);

    mm_out_hi = _mm256_add_epi16(
        _mm256_mullo_epi16(mm_src_hi, _mm256_shuffle_epi8(mm_coef1, vmm_get_hi)),
        _mm256_mullo_epi16(mm_dst_hi, _mm256_shuffle_epi8(mm_coef2, vmm_get_hi))
    );
    mm_out_hi = _mm256_or_si256(
        mm_out_hi, _mm256_slli_epi64(_mm256_unpackhi_epi32(mm_out_a, vmm_zero), 48)
    );
    mm_out_hi = _mm256_add_epi16(mm_out_hi, vmm_half);
    mm_out_hi = MM_SHIFTDIV255_epi16(mm_out_hi);

    return _mm256_packus_epi16(mm_out_lo, mm_out_hi);
}

#undef MM_SHIFTDIV255_epi16

#endif

#if defined(IMAGING_SIMD_SSE4)
#define ALPHA_COMPOSITE_VECTOR
#define ALPHA_COMPOSITE_4PX(out, dst, src)                                       \
    _mm_storeu_si128(                                                            \
        (__m128i *)(out),                                                        \
        alpha_composite_4px(                                                     \
            _mm_loadu_si128((__m128i *)(dst)), _mm_loadu_si128((__m128i *)(src)) \
        )                                                                        \
    )
#endif

/* Composites one row of xsize pixels: out may be dst, for in place. */
static void
alpha_composite_row(
    rgba8 *out, const rgba8 *dst, const rgba8 *restrict src, int xsize
) {
    int x = 0;

#if defined(IMAGING_SIMD_AVX2)
    for (; x < xsize - 7; x += 8) {
        _mm256_storeu_si256(
            (__m256i *)&out[x],
            alpha_composite_8px(
                _mm256_loadu_si256((__m256i *)&dst[x]),
                _mm256_loadu_si256((__m256i *)&src[x])
            )
        );
    }
#endif

#if defined(ALPHA_COMPOSITE_VECTOR)
    for (; x < xsize - 3; x += 4) {
        ALPHA_COMPOSITE_4PX(&out[x], &dst[x], &src[x]);
    }
    if (x < xsize) {
        // Run the remaining pixels through the same vector code.
        rgba8 tmp_dst[4] = {{0}}, tmp_src[4] = {{0}}, tmp_out[4];
        int n = xsize - x;
        memcpy(tmp_dst, &dst[x], n * sizeof(rgba8));
        memcpy(tmp_src, &src[x], n * sizeof(rgba8));
        ALPHA_COMPOSITE_4PX(tmp_out, tmp_dst, tmp_src);
        memcpy(&out[x], tmp_out, n * sizeof(rgba8));
    }
#else
    for (; x < xsize; x += 1) {
        if (src[x].a == 0) {
            // Copy 4 bytes at once.
            out[x] = dst[x];
        } else {
            // Integer implementation with increased precision.
            // Each variable has extra meaningful bits.
            // Divisions are rounded.

            UINT32 tmpr, tmpg, tmpb;
            UINT32 blend = dst[x].a * (255 - src[x].a);
            UINT32 outa255 = src[x].a * 255 + blend;
            // There we use 7 bits for precision.
            // We could use more, but we go beyond 32 bits.
            UINT32 coef1 = src[x].a * 255 * 255 * (1 << PRECISION_BITS) / outa255;
            UINT32 coef2 = 255 * (1 << PRECISION_BITS) - coef1;

            tmpr = src[x].r * coef1 + dst[x].r * coef2;
            tmpg = src[x].g * coef1 + dst[x].g * coef2;
            tmpb = src[x].b * coef1 + dst[x].b * coef2;
            out[x].r =
                SHIFTFORDIV255(tmpr + (0x80 << PRECISION_BITS)) >> PRECISION_BITS;
            out[x].g =
                SHIFTFORDIV255(tmpg + (0x80 << PRECISION_BITS)) >> PRECISION_BITS;
            out[x].b =
                SHIFTFORDIV255(tmpb + (0x80 << PRECISION_BITS)) >> PRECISION_BITS;
            out[x].a = SHIFTFORDIV255(outa255 + 0x80);
        }
    }
#endif
}

/**
 * Alpha-composite imSrc over imDst, returning a newly allocated result.
 *
 * Contract: imDst and imSrc are read-only and may alias each other.
 */
Imaging
ImagingAlphaComposite(Imaging imDst, Imaging imSrc) {
    Imaging imOut;
    ImagingSectionCookie cookie;

    /* Check arguments */
    if (!imDst || !imSrc ||
        (imDst->mode != IMAGING_MODE_RGBA && imDst->mode != IMAGING_MODE_LA)) {
        return ImagingError_ModeError();
    }

    if (imDst->mode != imSrc->mode || imDst->xsize != imSrc->xsize ||
        imDst->ysize != imSrc->ysize) {
        return ImagingError_Mismatch();
    }

    imOut = ImagingNewDirty(imDst->mode, imDst->xsize, imDst->ysize);
    if (!imOut) {
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    for (int y = 0; y < imDst->ysize; y++) {
        alpha_composite_row(
            (rgba8 *)imOut->image[y],
            (rgba8 *)imDst->image[y],
            (rgba8 *)imSrc->image[y],
            imDst->xsize
        );
    }
    ImagingSectionLeave(&cookie);

    return imOut;
}

/* The lowest and the highest address of an image's rows. */
static void
image_span(Imaging im, const char **lo, const char **hi) {
    int y;
    *lo = *hi = NULL;
    for (y = 0; y < im->ysize; y++) {
        const char *row = im->image[y];
        if (!*lo || row < *lo) {
            *lo = row;
        }
        if (!*hi || row + im->linesize > *hi) {
            *hi = row + im->linesize;
        }
    }
}

/**
 * Alpha-composite the xsize x ysize pixels of imSrc at (sx, sy) over imDst
 * at (dx, dy), in place, as compositing them onto a copy of that area and
 * pasting it back would: each pixel's result depends only on the two
 * pixels. The area is clipped to imDst, and must lie within imSrc.
 *
 * Returns 0 if the images share memory, for the caller to composite
 * through copies; -1 on error.
 */
int
ImagingAlphaCompositeOnto(
    Imaging imDst, Imaging imSrc, int dx, int dy, int sx, int sy, int xsize, int ysize
) {
    ImagingSectionCookie cookie;
    const char *dlo, *dhi, *slo, *shi;
    int y;

    if (!imDst || !imSrc ||
        (imDst->mode != IMAGING_MODE_RGBA && imDst->mode != IMAGING_MODE_LA) ||
        imDst->mode != imSrc->mode) {
        (void)ImagingError_ModeError();
        return -1;
    }
    if (sx < 0 || sy < 0 || xsize < 0 || ysize < 0 || xsize > imSrc->xsize - sx ||
        ysize > imSrc->ysize - sy) {
        (void)ImagingError_ValueError("source area outside the image");
        return -1;
    }
    image_span(imDst, &dlo, &dhi);
    image_span(imSrc, &slo, &shi);
    if (dlo < shi && slo < dhi) {
        return 0;
    }

    /* Clip to imDst. */
    if (dx < 0) {
        sx -= dx, xsize += dx, dx = 0;
    }
    if (dy < 0) {
        sy -= dy, ysize += dy, dy = 0;
    }
    if (xsize > imDst->xsize - dx) {
        xsize = imDst->xsize - dx;
    }
    if (ysize > imDst->ysize - dy) {
        ysize = imDst->ysize - dy;
    }
    if (xsize <= 0 || ysize <= 0) {
        return 1;
    }

    ImagingSectionEnter(&cookie);
    for (y = 0; y < ysize; y++) {
        rgba8 *dst = (rgba8 *)imDst->image[dy + y] + dx;
        alpha_composite_row(dst, dst, (rgba8 *)imSrc->image[sy + y] + sx, xsize);
    }
    ImagingSectionLeave(&cookie);
    return 1;
}
