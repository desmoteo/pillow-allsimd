/*
 * The Imaging library
 *
 * Fast mode Gaussian blur (design/APPROXIMATE_KERNELS_PLAN.md, item 1). The exact
 * Gaussian (BoxBlur.c) runs three extended box passes, each rounded to 8
 * bits and each a sweep over the image. Here the same three boxes are
 * convolved into one kernel, its weights rounded once to 8 bits summing to
 * 256, and applied in one sweep down the image: each output row takes the
 * vertical taps of its source rows into one row rounded to 8 bits, then the
 * horizontal taps of that row (KleidiCV's scheme: every sum fits 16 bits, so
 * vectors hold 8 or 16 lanes). The blur is Pillow's, its rounding differs.
 *
 * gauss_taps() in portable C defines the result; the SSE4, AVX2 and NEON
 * loops compute the same integer sums, so every build gives the same pixels.
 */

#include "Imaging.h"

#include <math.h>

/* The largest kernel radius: three boxes of up to 15 pixels. */
#define GAUSS_FAST_MAX_RADIUS 48
/* Above this sigma the box passes, which cost the same for any sigma, are
   faster than the direct taps (both machines measured cross over between
   6 and 8). */
#ifndef GAUSS_FAST_MAX_SIGMA
#define GAUSS_FAST_MAX_SIGMA 6.0f
#endif

typedef struct {
    int radius;
    /* w[0] is the centre weight, w[j] that of the taps at distance j */
    UINT16 w[GAUSS_FAST_MAX_RADIUS + 1];
    /* the reach of the three boxes before trailing zeros were dropped, and
       the radius of each box, for the exact edges */
    int reach;
    float box_radius;
} GaussKernel;

/* The three extended boxes of a Gaussian blur of the given sigma, with the
   weights ImagingBoxBlur uses (ww inside, fw on the fractional edge), are
   convolved in double; the result is rounded to multiples of 1/256 with the
   remainder in the centre, trailing zero weights dropped. Returns 0 if the
   kernel is too long. */
static int
gauss_kernel(GaussKernel *k, float sigma) {
    double box[GAUSS_FAST_MAX_RADIUS + 1], once[2 * GAUSS_FAST_MAX_RADIUS + 1];
    double twice[2 * GAUSS_FAST_MAX_RADIUS + 1], c[2 * GAUSS_FAST_MAX_RADIUS + 1];
    float floatRadius;
    int radius, br, r, i, j, sum = 0;
    UINT32 ww, fw;

    if (!(sigma > 0)) {
        k->radius = k->reach = 0;
        k->w[0] = 256;
        k->box_radius = 0;
        return 1;
    }
    floatRadius = ImagingGaussianBlurBoxRadius(sigma, 3);
    if (!(floatRadius >= 0 && floatRadius < GAUSS_FAST_MAX_RADIUS / 3 - 1)) {
        return 0;
    }
    /* as ImagingHorizontalBoxBlur and ImagingVerticalBoxBlur compute them */
    radius = (int)floatRadius;
    ww = (UINT32)(1 << 24) / (floatRadius * 2 + 1);
    fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2;
    br = radius + 1; /* box half-width, the fractional edge included */
    for (i = 0; i <= br; i++) {
        box[i] = (i <= radius ? ww : fw) / (double)(1 << 24);
    }
    /* full symmetric kernels, index offset by their radius */
    for (i = -br; i <= br; i++) {
        once[i + br] = box[i < 0 ? -i : i];
    }
    r = 2 * br;
    for (i = -r; i <= r; i++) {
        double v = 0;
        for (j = -br; j <= br; j++) {
            if (i - j >= -br && i - j <= br) {
                v += once[j + br] * once[i - j + br];
            }
        }
        twice[i + r] = v;
    }
    r = 3 * br;
    for (i = -r; i <= r; i++) {
        double v = 0;
        for (j = -br; j <= br; j++) {
            if (i - j >= -2 * br && i - j <= 2 * br) {
                v += once[j + br] * twice[i - j + 2 * br];
            }
        }
        c[i + r] = v;
    }
    for (j = 1; j <= r; j++) {
        k->w[j] = (UINT16)floor(256.0 * c[j + r] + 0.5);
        sum += 2 * k->w[j];
    }
    if (sum > 256) {
        return 0;
    }
    k->w[0] = (UINT16)(256 - sum);
    k->reach = r;
    k->box_radius = floatRadius;
    while (r > 0 && k->w[r] == 0) {
        r--;
    }
    k->radius = r;
    return 1;
}

/* Whether the fast kernel takes this blur: the images ImagingBoxBlur
   accepts, radii it accepts, up to GAUSS_FAST_MAX_SIGMA. Anything else goes
   to the exact path, which also reports the errors. */
int
ImagingGaussianBlurFastApplies(
    Imaging imOut, Imaging imIn, float xradius, float yradius
) {
    GaussKernel k;
    ModeID mode = imIn->mode;
    if (imOut == imIn || imOut->mode != mode || imOut->xsize != imIn->xsize ||
        imOut->ysize != imIn->ysize || imIn->type != IMAGING_TYPE_UINT8) {
        return 0;
    }
    if (mode != IMAGING_MODE_RGB && mode != IMAGING_MODE_RGBA &&
        mode != IMAGING_MODE_RGBa && mode != IMAGING_MODE_RGBX &&
        mode != IMAGING_MODE_CMYK && mode != IMAGING_MODE_L &&
        mode != IMAGING_MODE_LA && mode != IMAGING_MODE_La) {
        return 0;
    }
    /* negated, so that NaN goes to the exact path */
    if (!(xradius >= 0 && xradius <= GAUSS_FAST_MAX_SIGMA) ||
        !(yradius >= 0 && yradius <= GAUSS_FAST_MAX_SIGMA)) {
        return 0;
    }
    return gauss_kernel(&k, xradius) && gauss_kernel(&k, yradius);
}

/* out[i] = (128 + w[0] src[r][i] + sum over j of w[j] (src[r - j][i] +
   src[r + j][i])) >> 8 for i < n: the taps are 2r + 1 rows (vertical) or
   shifted pointers into one row (horizontal). The largest sum is
   128 + 255 * 256, so 16 bits hold it, and every product does too
   (w[j] <= 128 for j > 0, the pairs are at most 510). */
static void
gauss_taps(UINT8 *restrict out, const UINT8 **src, const GaussKernel *k, int n) {
    const int r = k->radius;
    int i = 0, j;

#if defined(IMAGING_SIMD_AVX2)
    for (; i + 32 <= n; i += 32) {
        __m256i c0 = mm256_cvtepu8_epi16(src[r] + i);
        __m256i c1 = mm256_cvtepu8_epi16(src[r] + i + 16);
        __m256i w = _mm256_set1_epi16((short)k->w[0]);
        __m256i a0 =
            _mm256_add_epi16(_mm256_set1_epi16(128), _mm256_mullo_epi16(c0, w));
        __m256i a1 =
            _mm256_add_epi16(_mm256_set1_epi16(128), _mm256_mullo_epi16(c1, w));
        for (j = 1; j <= r; j++) {
            __m256i p0 = _mm256_add_epi16(
                mm256_cvtepu8_epi16(src[r - j] + i), mm256_cvtepu8_epi16(src[r + j] + i)
            );
            __m256i p1 = _mm256_add_epi16(
                mm256_cvtepu8_epi16(src[r - j] + i + 16),
                mm256_cvtepu8_epi16(src[r + j] + i + 16)
            );
            w = _mm256_set1_epi16((short)k->w[j]);
            a0 = _mm256_add_epi16(a0, _mm256_mullo_epi16(p0, w));
            a1 = _mm256_add_epi16(a1, _mm256_mullo_epi16(p1, w));
        }
        __m256i packed =
            _mm256_packus_epi16(_mm256_srli_epi16(a0, 8), _mm256_srli_epi16(a1, 8));
        _mm256_storeu_si256(
            (__m256i *)(out + i), _mm256_permute4x64_epi64(packed, 0xd8)
        );
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    for (; i + 16 <= n; i += 16) {
        __m128i zero = _mm_setzero_si128();
        __m128i c = _mm_loadu_si128((const __m128i *)(src[r] + i));
        __m128i w = _mm_set1_epi16((short)k->w[0]);
        __m128i a0 = _mm_add_epi16(
            _mm_set1_epi16(128), _mm_mullo_epi16(_mm_unpacklo_epi8(c, zero), w)
        );
        __m128i a1 = _mm_add_epi16(
            _mm_set1_epi16(128), _mm_mullo_epi16(_mm_unpackhi_epi8(c, zero), w)
        );
        for (j = 1; j <= r; j++) {
            __m128i lo = _mm_loadu_si128((const __m128i *)(src[r - j] + i));
            __m128i hi = _mm_loadu_si128((const __m128i *)(src[r + j] + i));
            w = _mm_set1_epi16((short)k->w[j]);
            a0 = _mm_add_epi16(
                a0,
                _mm_mullo_epi16(
                    _mm_add_epi16(
                        _mm_unpacklo_epi8(lo, zero), _mm_unpacklo_epi8(hi, zero)
                    ),
                    w
                )
            );
            a1 = _mm_add_epi16(
                a1,
                _mm_mullo_epi16(
                    _mm_add_epi16(
                        _mm_unpackhi_epi8(lo, zero), _mm_unpackhi_epi8(hi, zero)
                    ),
                    w
                )
            );
        }
        _mm_storeu_si128(
            (__m128i *)(out + i),
            _mm_packus_epi16(_mm_srli_epi16(a0, 8), _mm_srli_epi16(a1, 8))
        );
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; i + 16 <= n; i += 16) {
        uint8x16_t c = vld1q_u8(src[r] + i);
        uint16x8_t a0 =
            vmlaq_n_u16(vdupq_n_u16(128), vmovl_u8(vget_low_u8(c)), k->w[0]);
        uint16x8_t a1 = vmlaq_n_u16(vdupq_n_u16(128), vmovl_high_u8(c), k->w[0]);
        for (j = 1; j <= r; j++) {
            uint8x16_t lo = vld1q_u8(src[r - j] + i);
            uint8x16_t hi = vld1q_u8(src[r + j] + i);
            a0 = vmlaq_n_u16(a0, vaddl_u8(vget_low_u8(lo), vget_low_u8(hi)), k->w[j]);
            a1 = vmlaq_n_u16(a1, vaddl_high_u8(lo, hi), k->w[j]);
        }
        vst1q_u8(out + i, vcombine_u8(vshrn_n_u16(a0, 8), vshrn_n_u16(a1, 8)));
    }
#endif
    for (; i < n; i++) {
        unsigned int acc = 128 + (unsigned int)k->w[0] * src[r][i];
        for (j = 1; j <= r; j++) {
            acc += (unsigned int)k->w[j] * (src[r - j][i] + src[r + j][i]);
        }
        out[i] = (UINT8)(acc >> 8);
    }
}

/* Copies rows y0..y1 of src, columns x0..x1, to dst at (dx, dy). */
static void
gauss_copy(Imaging dst, int dx, int dy, Imaging src, int x0, int y0, int x1, int y1) {
    const int ps = src->pixelsize;
    int y;
    for (y = y0; y < y1; y++) {
        memcpy(
            (UINT8 *)dst->image[dy + y - y0] + (size_t)dx * ps,
            (UINT8 *)src->image[y] + (size_t)x0 * ps,
            (size_t)(x1 - x0) * ps
        );
    }
}

/* The bands within the boxes' reach (bx, by) of the image's edges, where
   the exact blur clamps after each pass, which one kernel can't do, are
   taken from the exact blur of strips twice as wide: the top and bottom
   strips of 2 by rows, then the left and right strips of 2 bx columns side
   by side in one image. Each pass clamps at a strip's inner edge (or mixes
   the two strips at the seam), which changes the strip by at most the
   reach from there: the outer halves are the exact blur of the image. */
static int
gauss_exact_edges(
    Imaging imOut, Imaging imIn, const GaussKernel *kx, const GaussKernel *ky
) {
    const int xs = imIn->xsize, ys = imIn->ysize, bx = kx->reach, by = ky->reach;
    Imaging part, blurred;
    int ok;

    if (by > 0) {
        part = ImagingCrop(imIn, 0, 0, xs, 2 * by);
        blurred = part ? ImagingNewDirty(imIn->mode, xs, 2 * by) : NULL;
        ok =
            blurred && ImagingBoxBlur(blurred, part, kx->box_radius, ky->box_radius, 3);
        if (ok) {
            gauss_copy(imOut, 0, 0, blurred, 0, 0, xs, by);
            ImagingDelete(part);
            part = ImagingCrop(imIn, 0, ys - 2 * by, xs, ys);
            ok = part &&
                 ImagingBoxBlur(blurred, part, kx->box_radius, ky->box_radius, 3);
            if (ok) {
                gauss_copy(imOut, 0, ys - by, blurred, 0, by, xs, 2 * by);
            }
        }
        if (part) {
            ImagingDelete(part);
        }
        if (blurred) {
            ImagingDelete(blurred);
        }
        if (!ok) {
            return 0;
        }
    }
    if (bx > 0) {
        part = ImagingNewDirty(imIn->mode, 4 * bx, ys);
        blurred = part ? ImagingNewDirty(imIn->mode, 4 * bx, ys) : NULL;
        ok = blurred != NULL;
        if (ok) {
            gauss_copy(part, 0, 0, imIn, 0, 0, 2 * bx, ys);
            gauss_copy(part, 2 * bx, 0, imIn, xs - 2 * bx, 0, xs, ys);
            ok = ImagingBoxBlur(blurred, part, kx->box_radius, ky->box_radius, 3) !=
                 NULL;
        }
        if (ok) {
            gauss_copy(imOut, 0, 0, blurred, 0, 0, bx, ys);
            gauss_copy(imOut, xs - bx, 0, blurred, 3 * bx, 0, 4 * bx, ys);
        }
        if (part) {
            ImagingDelete(part);
        }
        if (blurred) {
            ImagingDelete(blurred);
        }
        if (!ok) {
            return 0;
        }
    }
    return 1;
}

Imaging
ImagingGaussianBlurFast(Imaging imOut, Imaging imIn, float xradius, float yradius) {
    ImagingSectionCookie cookie;
    GaussKernel kx, ky;
    const UINT8 *src[2 * GAUSS_FAST_MAX_RADIUS + 1];
    UINT8 *row, *padded;
    int x, y, j, c;
    const int ps = imIn->pixelsize;
    const int n = imIn->xsize * ps;

    if (!gauss_kernel(&kx, xradius) || !gauss_kernel(&ky, yradius)) {
        return (Imaging)ImagingError_ValueError("radius too large for fast mode");
    }
    if (imIn->xsize == 0 || imIn->ysize == 0) {
        return imOut;
    }
    /* The bands near the edges are exact (gauss_exact_edges); images too
       small for that are blurred exactly altogether. */
    if (imIn->xsize <= 4 * kx.reach || imIn->ysize <= 4 * ky.reach) {
        return ImagingBoxBlur(imOut, imIn, kx.box_radius, ky.box_radius, 3);
    }
    row = malloc(n > 0 ? n : 1);
    padded = malloc((size_t)(imIn->xsize + 2 * kx.radius) * ps + 1);
    if (!row || !padded) {
        free(row);
        free(padded);
        return (Imaging)ImagingError_MemoryError();
    }

    ImagingSectionEnter(&cookie);
    for (y = 0; y < imIn->ysize; y++) {
        const UINT8 *vrow;
        /* vertical taps, rows beyond the edges repeat the edge row */
        if (ky.radius == 0) {
            vrow = (const UINT8 *)imIn->image[y];
        } else {
            for (j = 0; j <= 2 * ky.radius; j++) {
                int sy = y - ky.radius + j;
                sy = sy < 0 ? 0 : (sy >= imIn->ysize ? imIn->ysize - 1 : sy);
                src[j] = (const UINT8 *)imIn->image[sy];
            }
            gauss_taps(row, src, &ky, n);
            vrow = row;
        }
        /* horizontal taps, pixels beyond the edges repeat the edge pixel */
        if (kx.radius == 0) {
            memcpy(imOut->image[y], vrow, n);
            continue;
        }
        for (x = 0; x < kx.radius; x++) {
            for (c = 0; c < ps; c++) {
                padded[x * ps + c] = vrow[c];
                padded[(kx.radius + imIn->xsize + x) * ps + c] = vrow[n - ps + c];
            }
        }
        memcpy(padded + kx.radius * ps, vrow, n);
        for (j = 0; j <= 2 * kx.radius; j++) {
            src[j] = padded + j * ps;
        }
        gauss_taps((UINT8 *)imOut->image[y], src, &kx, n);
    }
    ImagingSectionLeave(&cookie);

    free(row);
    free(padded);

    if (!gauss_exact_edges(imOut, imIn, &kx, &ky)) {
        return NULL;
    }
    return imOut;
}
