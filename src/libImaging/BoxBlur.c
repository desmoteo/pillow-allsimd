#include "Imaging.h"
#include "ImagingPx4.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void static inline ImagingLineBoxBlur32(
    UINT32 *lineOut,
    UINT32 *lineIn,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
) {
    int x;
    px4 acc;
    px4 bulk;
    px4 wws = px4_set1(ww);
    px4 fws = px4_set1(fw);

    // #define MOVE_ACC(acc, subtract, add)
    //     acc[0] += lineIn[add][0] - lineIn[subtract][0];

#define MOVE_ACC(acc, subtract, add) \
    acc = px4_sub(px4_add(acc, px4_load(&lineIn[add])), px4_load(&lineIn[subtract]));

    // #define ADD_FAR(bulk, acc, left, right)
    //     bulk[0] = (acc[0] * ww) + (lineIn[left][0] + lineIn[right][0]) * fw;

#define ADD_FAR(bulk, acc, left, right)                                      \
    bulk = px4_add(                                                          \
        px4_mul(px4_add(px4_load(&lineIn[left]), px4_load(&lineIn[right])), fws), \
        px4_mul(acc, wws));

    // #define SAVE(x, bulk)
    //    lineOut[x][0] = (UINT8)((bulk[0] + (1 << 23)) >> 24);

#define SAVE(x, bulk) lineOut[x] = px4_round24(bulk)

    /* Compute acc for -1 pixel (outside of image):
       From "-radius-1" to "-1" get first pixel,
       then from "0" to "radius-1". */
    // acc[0] = lineIn[0][0] * (radius + 1);
    acc = px4_mul(px4_load(&lineIn[0]), px4_set1(radius + 1));

    /* As radius can be bigger than xsize, iterate to edgeA -1. */
    for (x = 0; x < edgeA - 1; x++) {
        // acc[0] += lineIn[x][0];
        acc = px4_add(acc, px4_load(&lineIn[x]));
    }
    /* Then multiply remainder to last x. */
    // acc[0] += lineIn[lastx][0] * (radius - edgeA + 1);
    acc = px4_add(acc, px4_mul(px4_load(&lineIn[x]), px4_set1(radius - edgeA + 1)));

    if (edgeA <= edgeB) {
        /* Subtract pixel from left ("0").
           Add pixels from radius. */
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        /* Subtract previous pixel from "-radius".
           Add pixels from radius. */
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        /* Subtract previous pixel from "-radius".
           Add last pixel. */
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}

/* Radius below 1: each pixel and its two neighbours, edges repeated. */
void static inline ImagingLineBoxBlurZero32(
    UINT32 *lineOut, UINT32 *lineIn, int lastx, UINT32 ww, UINT32 fw
) {
    int x;
    px4 wws = px4_set1(ww);
    px4 fws = px4_set1(fw);

    for (x = 0; x <= lastx; x++) {
        int left = x > 0 ? x - 1 : 0;
        int right = x < lastx ? x + 1 : lastx;
        px4 bulk = px4_add(
            px4_mul(px4_add(px4_load(&lineIn[left]), px4_load(&lineIn[right])), fws),
            px4_mul(px4_load(&lineIn[x]), wws)
        );
        lineOut[x] = px4_round24(bulk);
    }
}

void static inline ImagingLineBoxBlur8(
    UINT8 *lineOut,
    UINT8 *lineIn,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
) {
    int x;
    UINT32 acc;
    UINT32 bulk;

#define MOVE_ACC(acc, subtract, add) acc += lineIn[add] - lineIn[subtract];

#define ADD_FAR(bulk, acc, left, right) \
    bulk = (acc * ww) + (lineIn[left] + lineIn[right]) * fw;

#define SAVE(x, bulk) lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24)

    acc = lineIn[0] * (radius + 1);
    for (x = 0; x < edgeA - 1; x++) {
        acc += lineIn[x];
    }
    acc += lineIn[lastx] * (radius - edgeA + 1);

    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}

/* Radius below 1: each pixel and its two neighbours, edges repeated. */
void static inline ImagingLineBoxBlurZero8(
    UINT8 *lineOut, UINT8 *lineIn, int lastx, UINT32 ww, UINT32 fw
) {
    int x;

    for (x = 0; x <= lastx; x++) {
        int left = x > 0 ? x - 1 : 0;
        int right = x < lastx ? x + 1 : lastx;
        UINT32 bulk = lineIn[x] * ww + (lineIn[left] + lineIn[right]) * fw;
        lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24);
    }
}

Imaging
ImagingHorizontalBoxBlur(Imaging imOut, Imaging imIn, float floatRadius) {
    ImagingSectionCookie cookie;

    int y;

    int radius = (int)floatRadius;
    UINT32 ww = (UINT32)(1 << 24) / (floatRadius * 2 + 1);
    UINT32 fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2;

    int edgeA = MIN(radius + 1, imIn->xsize);
    int edgeB = MAX(imIn->xsize - radius - 1, 0);

    UINT32 *lineOut = calloc(imIn->xsize, sizeof(UINT32));
    if (lineOut == NULL) {
        return ImagingError_MemoryError();
    }

    // printf(">>> radius: %d edges: %d %d, weights: %d %d\n",
    //     radius, edgeA, edgeB, ww, fw);

    ImagingSectionEnter(&cookie);

    if (imIn->image8) {
        if (radius) {
            for (y = 0; y < imIn->ysize; y++) {
                ImagingLineBoxBlur8(
                    (imIn == imOut ? (UINT8 *)lineOut : imOut->image8[y]),
                    imIn->image8[y],
                    imIn->xsize - 1,
                    radius,
                    edgeA,
                    edgeB,
                    ww,
                    fw
                );
                if (imIn == imOut) {
                    // Commit.
                    memcpy(imOut->image8[y], lineOut, imIn->xsize);
                }
            }
        } else {
            for (y = 0; y < imIn->ysize; y++) {
                ImagingLineBoxBlurZero8(
                    (imIn == imOut ? (UINT8 *)lineOut : imOut->image8[y]),
                    imIn->image8[y],
                    imIn->xsize - 1,
                    ww,
                    fw
                );
                if (imIn == imOut) {
                    // Commit.
                    memcpy(imOut->image8[y], lineOut, imIn->xsize);
                }
            }
        }
    } else {
        if (radius) {
            for (y = 0; y < imIn->ysize; y++) {
                ImagingLineBoxBlur32(
                    imIn == imOut ? (UINT32 *)lineOut : (UINT32 *)imOut->image32[y],
                    (UINT32 *)imIn->image32[y],
                    imIn->xsize - 1,
                    radius,
                    edgeA,
                    edgeB,
                    ww,
                    fw
                );
                if (imIn == imOut) {
                    // Commit.
                    memcpy(imOut->image32[y], lineOut, imIn->xsize * 4);
                }
            }
        } else {
            for (y = 0; y < imIn->ysize; y++) {
                ImagingLineBoxBlurZero32(
                    imIn == imOut ? (UINT32 *)lineOut : (UINT32 *)imOut->image32[y],
                    (UINT32 *)imIn->image32[y],
                    imIn->xsize - 1,
                    ww,
                    fw
                );
                if (imIn == imOut) {
                    // Commit.
                    memcpy(imOut->image32[y], lineOut, imIn->xsize * 4);
                }
            }
        }
    }

    ImagingSectionLeave(&cookie);

    free(lineOut);

    return imOut;
}

/* One output row of the vertical pass: acc holds a running sum per byte,
   updated with the rows entering and leaving the window. Same arithmetic
   per channel as the horizontal pass. */
static inline void
ImagingBoxBlurColumns(
    UINT8 *out,
    UINT32 *acc,
    const UINT8 *add,
    const UINT8 *sub,
    const UINT8 *far,
    int size,
    UINT32 ww,
    UINT32 fw
) {
    int b = 0;

#if defined(IMAGING_SIMD_SSE4)
    __m128i wws = _mm_set1_epi32(ww);
    __m128i fws = _mm_set1_epi32(fw);
    __m128i b23 = _mm_set1_epi32(1 << 23);
    for (; b < size - 15; b += 16) {
        __m128i add8 = _mm_loadu_si128((__m128i *)&add[b]);
        __m128i sub8 = _mm_loadu_si128((__m128i *)&sub[b]);
        __m128i far8 = _mm_loadu_si128((__m128i *)&far[b]);
        __m128i r[4];
#define COLUMNS_4(k)                                                           \
    {                                                                          \
        __m128i a = _mm_cvtepu8_epi32(_mm_srli_si128(add8, k * 4));            \
        __m128i s = _mm_cvtepu8_epi32(_mm_srli_si128(sub8, k * 4));            \
        __m128i f = _mm_cvtepu8_epi32(_mm_srli_si128(far8, k * 4));            \
        __m128i c = _mm_loadu_si128((__m128i *)&acc[b + k * 4]);               \
        c = _mm_sub_epi32(_mm_add_epi32(c, a), s);                             \
        _mm_storeu_si128((__m128i *)&acc[b + k * 4], c);                       \
        c = _mm_add_epi32(                                                     \
            _mm_mullo_epi32(c, wws), _mm_mullo_epi32(_mm_add_epi32(s, f), fws)); \
        r[k] = _mm_srli_epi32(_mm_add_epi32(c, b23), 24);                      \
    }
        COLUMNS_4(0);
        COLUMNS_4(1);
        COLUMNS_4(2);
        COLUMNS_4(3);
#undef COLUMNS_4
        _mm_storeu_si128(
            (__m128i *)&out[b],
            _mm_packus_epi16(_mm_packs_epi32(r[0], r[1]), _mm_packs_epi32(r[2], r[3])));
    }
#elif defined(IMAGING_SIMD_NEON)
    uint32x4_t wws = vdupq_n_u32(ww);
    uint32x4_t fws = vdupq_n_u32(fw);
    for (; b < size - 15; b += 16) {
        uint8x16_t add8 = vld1q_u8(&add[b]);
        uint8x16_t sub8 = vld1q_u8(&sub[b]);
        uint8x16_t far8 = vld1q_u8(&far[b]);
        uint16x8_t a16[2] = {vmovl_u8(vget_low_u8(add8)), vmovl_high_u8(add8)};
        uint16x8_t s16[2] = {vmovl_u8(vget_low_u8(sub8)), vmovl_high_u8(sub8)};
        uint16x8_t f16[2] = {vmovl_u8(vget_low_u8(far8)), vmovl_high_u8(far8)};
        uint16x4_t r[4];
        for (int k = 0; k < 4; k++) {
            uint32x4_t a = k & 1 ? vmovl_high_u16(a16[k / 2]) : vmovl_u16(vget_low_u16(a16[k / 2]));
            uint32x4_t s = k & 1 ? vmovl_high_u16(s16[k / 2]) : vmovl_u16(vget_low_u16(s16[k / 2]));
            uint32x4_t f = k & 1 ? vmovl_high_u16(f16[k / 2]) : vmovl_u16(vget_low_u16(f16[k / 2]));
            uint32x4_t c = vld1q_u32(&acc[b + k * 4]);
            c = vsubq_u32(vaddq_u32(c, a), s);
            vst1q_u32(&acc[b + k * 4], c);
            c = vmlaq_u32(vmulq_u32(c, wws), vaddq_u32(s, f), fws);
            r[k] = vmovn_u32(vshrq_n_u32(vaddq_u32(c, vdupq_n_u32(1 << 23)), 24));
        }
        vst1q_u8(&out[b], vcombine_u8(
            vmovn_u16(vcombine_u16(r[0], r[1])), vmovn_u16(vcombine_u16(r[2], r[3]))));
    }
#endif

    for (; b < size; b++) {
        acc[b] += add[b] - sub[b];
        UINT32 bulk = acc[b] * ww + (sub[b] + far[b]) * fw;
        out[b] = (UINT8)((bulk + (1 << 23)) >> 24);
    }
}

/* Blurs columns: each byte is averaged with the bytes above and below it,
   with edge rows repeated. This gives the same result as transposing,
   blurring rows and transposing back, but reads rows sequentially and is
   vectorised across the row. imOut and imIn must be distinct. */
static Imaging
ImagingVerticalBoxBlur(Imaging imOut, Imaging imIn, float floatRadius) {
    ImagingSectionCookie cookie;
    int y, i, b;

    int radius = (int)floatRadius;
    UINT32 ww = (UINT32)(1 << 24) / (floatRadius * 2 + 1);
    UINT32 fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2;
    int lasty = imIn->ysize - 1;
    int size = imIn->linesize;

    UINT32 *acc = calloc(size, sizeof(UINT32));
    if (acc == NULL) {
        return ImagingError_MemoryError();
    }

#define ROW(i) ((UINT8 *)imIn->image[(i) < 0 ? 0 : ((i) > lasty ? lasty : (i))])

    ImagingSectionEnter(&cookie);

    /* Window for row -1: rows -radius-1 .. radius-1. */
    for (i = -radius - 1; i < radius; i++) {
        UINT8 *row = ROW(i);
        for (b = 0; b < size; b++) {
            acc[b] += row[b];
        }
    }
    for (y = 0; y <= lasty; y++) {
        ImagingBoxBlurColumns(
            (UINT8 *)imOut->image[y],
            acc,
            ROW(y + radius),
            ROW(y - radius - 1),
            ROW(y + radius + 1),
            size,
            ww,
            fw
        );
    }

    ImagingSectionLeave(&cookie);

#undef ROW

    free(acc);
    return imOut;
}

Imaging
ImagingBoxBlur(Imaging imOut, Imaging imIn, float xradius, float yradius, int n) {
    int i;

    if (imOut->xsize == 0 || imOut->ysize == 0) {
        return imOut;
    }
    if (n < 1) {
        return ImagingError_ValueError("number of passes must be greater than zero");
    }
    if (xradius < 0 || yradius < 0) {
        return ImagingError_ValueError("radius must be >= 0");
    }

    if (imIn->mode != imOut->mode || imIn->type != imOut->type ||
        imIn->bands != imOut->bands || imIn->xsize != imOut->xsize ||
        imIn->ysize != imOut->ysize) {
        return ImagingError_Mismatch();
    }

    if (imIn->type != IMAGING_TYPE_UINT8) {
        return ImagingError_ModeError();
    }

    if (imIn->mode != IMAGING_MODE_RGB && imIn->mode != IMAGING_MODE_RGBA &&
        imIn->mode != IMAGING_MODE_RGBa && imIn->mode != IMAGING_MODE_RGBX &&
        imIn->mode != IMAGING_MODE_CMYK && imIn->mode != IMAGING_MODE_L &&
        imIn->mode != IMAGING_MODE_LA && imIn->mode != IMAGING_MODE_La) {
        return ImagingError_ModeError();
    }

    /* Horizontal passes work in place on rows. Vertical passes need a
       separate destination, so they alternate between imOut and a
       temporary image; the horizontal result goes to whichever buffer
       makes the last vertical pass end in imOut. */
    Imaging imTemp = NULL;
    Imaging src = imIn;
    if (yradius != 0) {
        imTemp = ImagingNewDirty(imIn->mode, imIn->xsize, imIn->ysize);
        if (!imTemp) {
            return NULL;
        }
    }

    if (xradius != 0) {
        Imaging dst = (yradius != 0 && n % 2 == 1) ? imTemp : imOut;
        if (!ImagingHorizontalBoxBlur(dst, imIn, xradius)) {
            goto error;
        }
        for (i = 1; i < n; i++) {
            if (!ImagingHorizontalBoxBlur(dst, dst, xradius)) {
                goto error;
            }
        }
        src = dst;
    }
    if (yradius != 0) {
        for (i = 0; i < n; i++) {
            Imaging dst = (n - i) % 2 == 1 ? imOut : imTemp;
            if (!ImagingVerticalBoxBlur(dst, src, yradius)) {
                goto error;
            }
            src = dst;
        }
        ImagingDelete(imTemp);
    }
    if (xradius == 0 && yradius == 0) {
        if (!ImagingCopy2(imOut, imIn)) {
            return NULL;
        }
    }

    return imOut;

error:
    if (imTemp) {
        ImagingDelete(imTemp);
    }
    return NULL;
}

static float
_gaussian_blur_radius(float radius, int passes) {
    float sigma2, L, l, a;

    sigma2 = radius * radius / passes;
    // from https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf
    // [7] Box length.
    L = sqrt(12.0 * sigma2 + 1.0);
    // [11] Integer part of box radius.
    l = floor((L - 1.0) / 2.0);
    // [14], [Fig. 2] Fractional part of box radius.
    a = (2 * l + 1) * (l * (l + 1) - 3 * sigma2);
    a /= 6 * (sigma2 - (l + 1) * (l + 1));

    return l + a;
}

Imaging
ImagingGaussianBlur(
    Imaging imOut, Imaging imIn, float xradius, float yradius, int passes
) {
    return ImagingBoxBlur(
        imOut,
        imIn,
        _gaussian_blur_radius(xradius, passes),
        _gaussian_blur_radius(yradius, passes),
        passes
    );
}
