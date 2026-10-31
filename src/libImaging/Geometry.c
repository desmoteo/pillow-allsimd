#include "Imaging.h"

/* For large images rotation is an inefficient operation in terms of CPU cache.
   One row in the source image affects each column in destination.
   Rotating in chunks that fit in the cache can speed up rotation
   8x on a modern CPU. A chunk size of 128 requires only 65k and is large enough
   that the overhead from the extra loops are not apparent. */
#define ROTATE_CHUNK 512
#define ROTATE_SMALL_CHUNK 8

#define COORD(v) ((v) < 0.0 ? -1 : ((int)(v)))
#define FLOOR(v) ((v) < 0.0 ? ((int)floor(v)) : ((int)(v)))

/* -------------------------------------------------------------------- */
/* Transpose operations                                                 */

#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
#define TRANSPOSE_SIMD

/* Column k of the 4 x 4 block of 32-bit pixels whose rows start at in[]
   to out[k]. */
static inline void
transpose_4x4_32(INT32 *const in[4], INT32 *const out[4]) {
#if defined(IMAGING_SIMD_SSE4)
    __m128i a = _mm_loadu_si128((const __m128i *)in[0]);
    __m128i b = _mm_loadu_si128((const __m128i *)in[1]);
    __m128i c = _mm_loadu_si128((const __m128i *)in[2]);
    __m128i d = _mm_loadu_si128((const __m128i *)in[3]);
    __m128i ab01 = _mm_unpacklo_epi32(a, b), cd01 = _mm_unpacklo_epi32(c, d);
    __m128i ab23 = _mm_unpackhi_epi32(a, b), cd23 = _mm_unpackhi_epi32(c, d);
    _mm_storeu_si128((__m128i *)out[0], _mm_unpacklo_epi64(ab01, cd01));
    _mm_storeu_si128((__m128i *)out[1], _mm_unpackhi_epi64(ab01, cd01));
    _mm_storeu_si128((__m128i *)out[2], _mm_unpacklo_epi64(ab23, cd23));
    _mm_storeu_si128((__m128i *)out[3], _mm_unpackhi_epi64(ab23, cd23));
#else
    uint32x4x2_t ab = vtrnq_u32(
        vld1q_u32((const uint32_t *)in[0]), vld1q_u32((const uint32_t *)in[1])
    );
    uint32x4x2_t cd = vtrnq_u32(
        vld1q_u32((const uint32_t *)in[2]), vld1q_u32((const uint32_t *)in[3])
    );
    vst1q_u32(
        (uint32_t *)out[0],
        vcombine_u32(vget_low_u32(ab.val[0]), vget_low_u32(cd.val[0]))
    );
    vst1q_u32(
        (uint32_t *)out[1],
        vcombine_u32(vget_low_u32(ab.val[1]), vget_low_u32(cd.val[1]))
    );
    vst1q_u32(
        (uint32_t *)out[2],
        vcombine_u32(vget_high_u32(ab.val[0]), vget_high_u32(cd.val[0]))
    );
    vst1q_u32(
        (uint32_t *)out[3],
        vcombine_u32(vget_high_u32(ab.val[1]), vget_high_u32(cd.val[1]))
    );
#endif
}

/* The same for an 8 x 8 block of 8-bit pixels. */
static inline void
transpose_8x8_8(UINT8 *const in[8], UINT8 *const out[8]) {
#if defined(IMAGING_SIMD_SSE4)
    __m128i ab = _mm_unpacklo_epi8(
        _mm_loadl_epi64((const __m128i *)in[0]), _mm_loadl_epi64((const __m128i *)in[1])
    );
    __m128i cd = _mm_unpacklo_epi8(
        _mm_loadl_epi64((const __m128i *)in[2]), _mm_loadl_epi64((const __m128i *)in[3])
    );
    __m128i ef = _mm_unpacklo_epi8(
        _mm_loadl_epi64((const __m128i *)in[4]), _mm_loadl_epi64((const __m128i *)in[5])
    );
    __m128i gh = _mm_unpacklo_epi8(
        _mm_loadl_epi64((const __m128i *)in[6]), _mm_loadl_epi64((const __m128i *)in[7])
    );
    /* Columns 0-3 and 4-7 of rows a-d, then of rows e-h. */
    __m128i ad03 = _mm_unpacklo_epi16(ab, cd), ad47 = _mm_unpackhi_epi16(ab, cd);
    __m128i eh03 = _mm_unpacklo_epi16(ef, gh), eh47 = _mm_unpackhi_epi16(ef, gh);
    __m128i c01 = _mm_unpacklo_epi32(ad03, eh03), c23 = _mm_unpackhi_epi32(ad03, eh03);
    __m128i c45 = _mm_unpacklo_epi32(ad47, eh47), c67 = _mm_unpackhi_epi32(ad47, eh47);
    _mm_storel_epi64((__m128i *)out[0], c01);
    _mm_storel_epi64((__m128i *)out[1], _mm_unpackhi_epi64(c01, c01));
    _mm_storel_epi64((__m128i *)out[2], c23);
    _mm_storel_epi64((__m128i *)out[3], _mm_unpackhi_epi64(c23, c23));
    _mm_storel_epi64((__m128i *)out[4], c45);
    _mm_storel_epi64((__m128i *)out[5], _mm_unpackhi_epi64(c45, c45));
    _mm_storel_epi64((__m128i *)out[6], c67);
    _mm_storel_epi64((__m128i *)out[7], _mm_unpackhi_epi64(c67, c67));
#else
    uint8x8x2_t ab = vtrn_u8(vld1_u8(in[0]), vld1_u8(in[1]));
    uint8x8x2_t cd = vtrn_u8(vld1_u8(in[2]), vld1_u8(in[3]));
    uint8x8x2_t ef = vtrn_u8(vld1_u8(in[4]), vld1_u8(in[5]));
    uint8x8x2_t gh = vtrn_u8(vld1_u8(in[6]), vld1_u8(in[7]));
    /* Columns 0 and 4, 2 and 6, 1 and 5, 3 and 7 of rows a-d, then e-h. */
    uint16x4x2_t ad0 =
        vtrn_u16(vreinterpret_u16_u8(ab.val[0]), vreinterpret_u16_u8(cd.val[0]));
    uint16x4x2_t ad1 =
        vtrn_u16(vreinterpret_u16_u8(ab.val[1]), vreinterpret_u16_u8(cd.val[1]));
    uint16x4x2_t eh0 =
        vtrn_u16(vreinterpret_u16_u8(ef.val[0]), vreinterpret_u16_u8(gh.val[0]));
    uint16x4x2_t eh1 =
        vtrn_u16(vreinterpret_u16_u8(ef.val[1]), vreinterpret_u16_u8(gh.val[1]));
    uint32x2x2_t c04 =
        vtrn_u32(vreinterpret_u32_u16(ad0.val[0]), vreinterpret_u32_u16(eh0.val[0]));
    uint32x2x2_t c15 =
        vtrn_u32(vreinterpret_u32_u16(ad1.val[0]), vreinterpret_u32_u16(eh1.val[0]));
    uint32x2x2_t c26 =
        vtrn_u32(vreinterpret_u32_u16(ad0.val[1]), vreinterpret_u32_u16(eh0.val[1]));
    uint32x2x2_t c37 =
        vtrn_u32(vreinterpret_u32_u16(ad1.val[1]), vreinterpret_u32_u16(eh1.val[1]));
    vst1_u8(out[0], vreinterpret_u8_u32(c04.val[0]));
    vst1_u8(out[1], vreinterpret_u8_u32(c15.val[0]));
    vst1_u8(out[2], vreinterpret_u8_u32(c26.val[0]));
    vst1_u8(out[3], vreinterpret_u8_u32(c37.val[0]));
    vst1_u8(out[4], vreinterpret_u8_u32(c04.val[1]));
    vst1_u8(out[5], vreinterpret_u8_u32(c15.val[1]));
    vst1_u8(out[6], vreinterpret_u8_u32(c26.val[1]));
    vst1_u8(out[7], vreinterpret_u8_u32(c37.val[1]));
#endif
}

/* The 90 degree transposes of 8-bit and 32-bit images: imOut's row x,
   column y is imIn's column x, row y, mirrored on each axis whose flip
   is set (rotate 90 mirrors x, rotate 270 y, transverse both). In bands
   of 64 bytes' worth of rows, each left to right in columns of BS x BS
   blocks of INT pixels: a column of blocks fills 64 bytes of each of its
   output rows, whole cache lines when they are aligned, and the blocks
   next to it read the rest of the input's cache lines. */
#define TRANSPOSE_BLOCKS(name, INT, image, BS, kernel)                            \
    static void name(Imaging imOut, Imaging imIn, int flipx, int flipy) {         \
        enum { BAND = 64 / sizeof(INT) };                                         \
        int xsize = imIn->xsize, ysize = imIn->ysize;                             \
        int xb = xsize / BS * BS, yb = ysize / BS * BS;                           \
        int x, y, yy, h, k;                                                       \
                                                                                  \
        for (y = 0; y < yb; y += h) {                                             \
            INT *rows[BAND];                                                      \
            h = yb - y < BAND ? yb - y : BAND;                                    \
            for (k = 0; k < h; k++) {                                             \
                rows[k] = imIn->image[y + k];                                     \
            }                                                                     \
            for (x = 0; x < xb; x += BS) {                                        \
                INT *outrows[BS], *in[BS], *out[BS];                              \
                for (k = 0; k < BS; k++) {                                        \
                    outrows[k] = imOut->image[flipx ? xsize - 1 - x - k : x + k]; \
                }                                                                 \
                for (yy = 0; yy < h; yy += BS) {                                  \
                    /* The block's output columns start at oy. With flipy,        \
                       the rows are read last first, to come in order. */         \
                    int oy = flipy ? ysize - y - yy - BS : y + yy;                \
                    for (k = 0; k < BS; k++) {                                    \
                        in[k] = rows[flipy ? yy + BS - 1 - k : yy + k] + x;       \
                        out[k] = outrows[k] + oy;                                 \
                    }                                                             \
                    kernel(in, out);                                              \
                }                                                                 \
            }                                                                     \
        }                                                                         \
        /* The pixels left over at the image's edges. */                          \
        for (y = 0; y < ysize; y++) {                                             \
            int oy = flipy ? ysize - 1 - y : y;                                   \
            for (x = y < yb ? xb : 0; x < xsize; x++) {                           \
                int ox = flipx ? xsize - 1 - x : x;                               \
                imOut->image[ox][oy] = imIn->image[y][x];                         \
            }                                                                     \
        }                                                                         \
    }

TRANSPOSE_BLOCKS(transpose_blocks8, UINT8, image8, 8, transpose_8x8_8)
TRANSPOSE_BLOCKS(transpose_blocks32, INT32, image32, 4, transpose_4x4_32)
#undef TRANSPOSE_BLOCKS

static void
transpose_blocks(Imaging imOut, Imaging imIn, int flipx, int flipy) {
    if (imIn->pixelsize == 1) {
        transpose_blocks8(imOut, imIn, flipx, flipy);
    } else {
        transpose_blocks32(imOut, imIn, flipx, flipy);
    }
}

/* Each row of an 8-bit or 32-bit image reversed, into the same row of
   imOut or, with flipy, the mirrored one (flip left right, rotate 180). */
static void
flip_rows(Imaging imOut, Imaging imIn, int flipy) {
    int xsize = imIn->xsize, ysize = imIn->ysize;
    int x, y;

    for (y = 0; y < ysize; y++) {
        int yo = flipy ? ysize - 1 - y : y;
        x = 0;
        if (imIn->pixelsize == 1) {
            UINT8 *in = imIn->image8[y], *out = imOut->image8[yo] + xsize;
#if defined(IMAGING_SIMD_SSE4)
            __m128i reverse =
                _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            for (; x + 16 <= xsize; x += 16) {
                __m128i v = _mm_loadu_si128((const __m128i *)(in + x));
                _mm_storeu_si128(
                    (__m128i *)(out - x - 16), _mm_shuffle_epi8(v, reverse)
                );
            }
#else
            for (; x + 16 <= xsize; x += 16) {
                uint8x16_t v = vrev64q_u8(vld1q_u8(in + x));
                vst1q_u8(out - x - 16, vextq_u8(v, v, 8));
            }
#endif
            for (; x < xsize; x++) {
                out[-1 - x] = in[x];
            }
        } else {
            INT32 *in = imIn->image32[y], *out = imOut->image32[yo] + xsize;
#if defined(IMAGING_SIMD_SSE4)
            for (; x + 4 <= xsize; x += 4) {
                __m128i v = _mm_loadu_si128((const __m128i *)(in + x));
                _mm_storeu_si128(
                    (__m128i *)(out - x - 4),
                    _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 1, 2, 3))
                );
            }
#else
            for (; x + 4 <= xsize; x += 4) {
                uint32x4_t v = vrev64q_u32(vld1q_u32((const uint32_t *)(in + x)));
                vst1q_u32((uint32_t *)(out - x - 4), vextq_u32(v, v, 2));
            }
#endif
            for (; x < xsize; x++) {
                out[-1 - x] = in[x];
            }
        }
    }
}
#endif

Imaging
ImagingFlipLeftRight(Imaging imOut, Imaging imIn) {
    ImagingSectionCookie cookie;
    int x, y, xr;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        return (Imaging)ImagingError_ModeError();
    }

    int xsize = imIn->xsize, ysize = imIn->ysize;
    if (xsize != imOut->xsize || ysize != imOut->ysize) {
        return (Imaging)ImagingError_Mismatch();
    }

    ImagingCopyPalette(imOut, imIn);

#define FLIP_LEFT_RIGHT(INT, image)         \
    for (y = 0; y < ysize; y++) {           \
        INT *in = (INT *)imIn->image[y];    \
        INT *out = (INT *)imOut->image[y];  \
        xr = xsize - 1;                     \
        for (x = 0; x < xsize; x++, xr--) { \
            out[xr] = in[x];                \
        }                                   \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

#if defined(TRANSPOSE_SIMD)
    if (imIn->pixelsize == 1 || imIn->pixelsize == 4) {
        flip_rows(imOut, imIn, 0);
    } else
#endif
    {
        if (imIn->image8) {
            if (isModeI16(imIn->mode)) {
                FLIP_LEFT_RIGHT(UINT16, image8)
            } else {
                FLIP_LEFT_RIGHT(UINT8, image8)
            }
        } else {
            FLIP_LEFT_RIGHT(INT32, image32)
        }
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef FLIP_LEFT_RIGHT

    return imOut;
}

Imaging
ImagingFlipTopBottom(Imaging imOut, Imaging imIn) {
    ImagingSectionCookie cookie;
    int y, yr;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        return (Imaging)ImagingError_ModeError();
    }

    int ysize = imIn->ysize;
    if (imIn->xsize != imOut->xsize || ysize != imOut->ysize) {
        return (Imaging)ImagingError_Mismatch();
    }

    ImagingCopyPalette(imOut, imIn);

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

    yr = ysize - 1;
    for (y = 0; y < ysize; y++, yr--) {
        memcpy(imOut->image[yr], imIn->image[y], imIn->linesize);
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

    return imOut;
}

Imaging
ImagingRotate90(Imaging imOut, Imaging imIn) {
    ImagingSectionCookie cookie;
    int x, y, xx, yy, xr, xxsize, yysize;
    int xxx, yyy, xxxsize, yyysize;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        return (Imaging)ImagingError_ModeError();
    }

    int xsize = imIn->xsize, ysize = imIn->ysize;
    if (xsize != imOut->ysize || ysize != imOut->xsize) {
        return (Imaging)ImagingError_Mismatch();
    }

    ImagingCopyPalette(imOut, imIn);

#define ROTATE_90(INT, image)                                             \
    for (y = 0; y < ysize; y += ROTATE_CHUNK) {                           \
        for (x = 0; x < xsize; x += ROTATE_CHUNK) {                       \
            yysize = y + ROTATE_CHUNK < ysize ? y + ROTATE_CHUNK : ysize; \
            xxsize = x + ROTATE_CHUNK < xsize ? x + ROTATE_CHUNK : xsize; \
            for (yy = y; yy < yysize; yy += ROTATE_SMALL_CHUNK) {         \
                for (xx = x; xx < xxsize; xx += ROTATE_SMALL_CHUNK) {     \
                    yyysize = yy + ROTATE_SMALL_CHUNK < ysize             \
                                  ? yy + ROTATE_SMALL_CHUNK               \
                                  : ysize;                                \
                    xxxsize = xx + ROTATE_SMALL_CHUNK < xsize             \
                                  ? xx + ROTATE_SMALL_CHUNK               \
                                  : xsize;                                \
                    for (yyy = yy; yyy < yyysize; yyy++) {                \
                        INT *in = (INT *)imIn->image[yyy];                \
                        xr = xsize - 1 - xx;                              \
                        for (xxx = xx; xxx < xxxsize; xxx++, xr--) {      \
                            INT *out = (INT *)imOut->image[xr];           \
                            out[yyy] = in[xxx];                           \
                        }                                                 \
                    }                                                     \
                }                                                         \
            }                                                             \
        }                                                                 \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

#if defined(TRANSPOSE_SIMD)
    if (imIn->pixelsize == 1 || imIn->pixelsize == 4) {
        transpose_blocks(imOut, imIn, 1, 0);
    } else
#endif
    {
        if (imIn->image8) {
            if (isModeI16(imIn->mode)) {
                ROTATE_90(UINT16, image8);
            } else {
                ROTATE_90(UINT8, image8);
            }
        } else {
            ROTATE_90(INT32, image32);
        }
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef ROTATE_90

    return imOut;
}

Imaging
ImagingTranspose(Imaging imOut, Imaging imIn) {
    ImagingSectionCookie cookie;
    int x, y, xx, yy, xxsize, yysize;
    int xxx, yyy, xxxsize, yyysize;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        return (Imaging)ImagingError_ModeError();
    }

    int xsize = imIn->xsize, ysize = imIn->ysize;
    if (xsize != imOut->ysize || ysize != imOut->xsize) {
        return (Imaging)ImagingError_Mismatch();
    }

    ImagingCopyPalette(imOut, imIn);

#define TRANSPOSE(INT, image)                                             \
    for (y = 0; y < ysize; y += ROTATE_CHUNK) {                           \
        for (x = 0; x < xsize; x += ROTATE_CHUNK) {                       \
            yysize = y + ROTATE_CHUNK < ysize ? y + ROTATE_CHUNK : ysize; \
            xxsize = x + ROTATE_CHUNK < xsize ? x + ROTATE_CHUNK : xsize; \
            for (yy = y; yy < yysize; yy += ROTATE_SMALL_CHUNK) {         \
                for (xx = x; xx < xxsize; xx += ROTATE_SMALL_CHUNK) {     \
                    yyysize = yy + ROTATE_SMALL_CHUNK < ysize             \
                                  ? yy + ROTATE_SMALL_CHUNK               \
                                  : ysize;                                \
                    xxxsize = xx + ROTATE_SMALL_CHUNK < xsize             \
                                  ? xx + ROTATE_SMALL_CHUNK               \
                                  : xsize;                                \
                    for (yyy = yy; yyy < yyysize; yyy++) {                \
                        INT *in = (INT *)imIn->image[yyy];                \
                        for (xxx = xx; xxx < xxxsize; xxx++) {            \
                            INT *out = (INT *)imOut->image[xxx];          \
                            out[yyy] = in[xxx];                           \
                        }                                                 \
                    }                                                     \
                }                                                         \
            }                                                             \
        }                                                                 \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

#if defined(TRANSPOSE_SIMD)
    if (imIn->pixelsize == 1 || imIn->pixelsize == 4) {
        transpose_blocks(imOut, imIn, 0, 0);
    } else
#endif
    {
        if (imIn->image8) {
            if (isModeI16(imIn->mode)) {
                TRANSPOSE(UINT16, image8);
            } else {
                TRANSPOSE(UINT8, image8);
            }
        } else {
            TRANSPOSE(INT32, image32);
        }
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef TRANSPOSE

    return imOut;
}

Imaging
ImagingTransverse(Imaging imOut, Imaging imIn) {
    ImagingSectionCookie cookie;
    int x, y, xr, yr, xx, yy, xxsize, yysize;
    int xxx, yyy, xxxsize, yyysize;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        return (Imaging)ImagingError_ModeError();
    }

    int xsize = imIn->xsize, ysize = imIn->ysize;
    if (xsize != imOut->ysize || ysize != imOut->xsize) {
        return (Imaging)ImagingError_Mismatch();
    }

    ImagingCopyPalette(imOut, imIn);

#define TRANSVERSE(INT, image)                                            \
    for (y = 0; y < ysize; y += ROTATE_CHUNK) {                           \
        for (x = 0; x < xsize; x += ROTATE_CHUNK) {                       \
            yysize = y + ROTATE_CHUNK < ysize ? y + ROTATE_CHUNK : ysize; \
            xxsize = x + ROTATE_CHUNK < xsize ? x + ROTATE_CHUNK : xsize; \
            for (yy = y; yy < yysize; yy += ROTATE_SMALL_CHUNK) {         \
                for (xx = x; xx < xxsize; xx += ROTATE_SMALL_CHUNK) {     \
                    yyysize = yy + ROTATE_SMALL_CHUNK < ysize             \
                                  ? yy + ROTATE_SMALL_CHUNK               \
                                  : ysize;                                \
                    xxxsize = xx + ROTATE_SMALL_CHUNK < xsize             \
                                  ? xx + ROTATE_SMALL_CHUNK               \
                                  : xsize;                                \
                    yr = ysize - 1 - yy;                                  \
                    for (yyy = yy; yyy < yyysize; yyy++, yr--) {          \
                        INT *in = (INT *)imIn->image[yyy];                \
                        xr = xsize - 1 - xx;                              \
                        for (xxx = xx; xxx < xxxsize; xxx++, xr--) {      \
                            INT *out = (INT *)imOut->image[xr];           \
                            out[yr] = in[xxx];                            \
                        }                                                 \
                    }                                                     \
                }                                                         \
            }                                                             \
        }                                                                 \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

#if defined(TRANSPOSE_SIMD)
    if (imIn->pixelsize == 1 || imIn->pixelsize == 4) {
        transpose_blocks(imOut, imIn, 1, 1);
    } else
#endif
    {
        if (imIn->image8) {
            if (isModeI16(imIn->mode)) {
                TRANSVERSE(UINT16, image8);
            } else {
                TRANSVERSE(UINT8, image8);
            }
        } else {
            TRANSVERSE(INT32, image32);
        }
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef TRANSVERSE

    return imOut;
}

Imaging
ImagingRotate180(Imaging imOut, Imaging imIn) {
    ImagingSectionCookie cookie;
    int x, y, xr, yr;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        return (Imaging)ImagingError_ModeError();
    }

    int xsize = imIn->xsize, ysize = imIn->ysize;
    if (xsize != imOut->xsize || ysize != imOut->ysize) {
        return (Imaging)ImagingError_Mismatch();
    }

    ImagingCopyPalette(imOut, imIn);

#define ROTATE_180(INT, image)              \
    for (y = 0; y < ysize; y++, yr--) {     \
        INT *in = (INT *)imIn->image[y];    \
        INT *out = (INT *)imOut->image[yr]; \
        xr = xsize - 1;                     \
        for (x = 0; x < xsize; x++, xr--) { \
            out[xr] = in[x];                \
        }                                   \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

#if defined(TRANSPOSE_SIMD)
    if (imIn->pixelsize == 1 || imIn->pixelsize == 4) {
        flip_rows(imOut, imIn, 1);
    } else
#endif
    {
        yr = ysize - 1;
        if (imIn->image8) {
            if (isModeI16(imIn->mode)) {
                ROTATE_180(UINT16, image8)
            } else {
                ROTATE_180(UINT8, image8)
            }
        } else {
            ROTATE_180(INT32, image32)
        }
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef ROTATE_180

    return imOut;
}

Imaging
ImagingRotate270(Imaging imOut, Imaging imIn) {
    ImagingSectionCookie cookie;
    int x, y, xx, yy, yr, xxsize, yysize;
    int xxx, yyy, xxxsize, yyysize;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        return (Imaging)ImagingError_ModeError();
    }

    int xsize = imIn->xsize, ysize = imIn->ysize;
    if (xsize != imOut->ysize || ysize != imOut->xsize) {
        return (Imaging)ImagingError_Mismatch();
    }

    ImagingCopyPalette(imOut, imIn);

#define ROTATE_270(INT, image)                                            \
    for (y = 0; y < ysize; y += ROTATE_CHUNK) {                           \
        for (x = 0; x < xsize; x += ROTATE_CHUNK) {                       \
            yysize = y + ROTATE_CHUNK < ysize ? y + ROTATE_CHUNK : ysize; \
            xxsize = x + ROTATE_CHUNK < xsize ? x + ROTATE_CHUNK : xsize; \
            for (yy = y; yy < yysize; yy += ROTATE_SMALL_CHUNK) {         \
                for (xx = x; xx < xxsize; xx += ROTATE_SMALL_CHUNK) {     \
                    yyysize = yy + ROTATE_SMALL_CHUNK < ysize             \
                                  ? yy + ROTATE_SMALL_CHUNK               \
                                  : ysize;                                \
                    xxxsize = xx + ROTATE_SMALL_CHUNK < xsize             \
                                  ? xx + ROTATE_SMALL_CHUNK               \
                                  : xsize;                                \
                    yr = ysize - 1 - yy;                                  \
                    for (yyy = yy; yyy < yyysize; yyy++, yr--) {          \
                        INT *in = (INT *)imIn->image[yyy];                \
                        for (xxx = xx; xxx < xxxsize; xxx++) {            \
                            INT *out = (INT *)imOut->image[xxx];          \
                            out[yr] = in[xxx];                            \
                        }                                                 \
                    }                                                     \
                }                                                         \
            }                                                             \
        }                                                                 \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

#if defined(TRANSPOSE_SIMD)
    if (imIn->pixelsize == 1 || imIn->pixelsize == 4) {
        transpose_blocks(imOut, imIn, 0, 1);
    } else
#endif
    {
        if (imIn->image8) {
            if (isModeI16(imIn->mode)) {
                ROTATE_270(UINT16, image8);
            } else {
                ROTATE_270(UINT8, image8);
            }
        } else {
            ROTATE_270(INT32, image32);
        }
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef ROTATE_270

    return imOut;
}

/* -------------------------------------------------------------------- */
/* Transforms                                                           */

/* transform primitives (ImagingTransformMap) */

static int
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

static int
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

static int
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

/* transform filters (ImagingTransformFilter) */

static int
nearest_filter8(void *out, Imaging im, double xin, double yin) {
    int x = COORD(xin);
    int y = COORD(yin);
    if (x < 0 || x >= im->xsize || y < 0 || y >= im->ysize) {
        return 0;
    }
    ((UINT8 *)out)[0] = im->image8[y][x];
    return 1;
}

static int
nearest_filter16(void *out, Imaging im, double xin, double yin) {
    int x = COORD(xin);
    int y = COORD(yin);
    if (x < 0 || x >= im->xsize || y < 0 || y >= im->ysize) {
        return 0;
    }
    memcpy(out, im->image8[y] + x * sizeof(INT16), sizeof(INT16));
    return 1;
}

static int
nearest_filter32(void *out, Imaging im, double xin, double yin) {
    int x = COORD(xin);
    int y = COORD(yin);
    if (x < 0 || x >= im->xsize || y < 0 || y >= im->ysize) {
        return 0;
    }
    memcpy(out, &im->image32[y][x], sizeof(INT32));
    return 1;
}

#define XCLIP(x) (((x) < 0) ? 0 : ((x) < xsize) ? (x) : xsize - 1)
#define YCLIP(y) (((y) < 0) ? 0 : ((y) < ysize) ? (y) : ysize - 1)

/* Vector filters for 3- and 4-band 8-bit images, below. Not on arm64: a
   NEON version (two float64x2_t, contracted to FMA like the scalar code)
   was 4% faster for bilinear but 12% slower for bicubic and 9% slower for
   perspective transforms than GCC's code for the scalar filters on a
   Cortex-A78AE. */
#if defined(IMAGING_SIMD_SSE4)
#define TRANSFORM_SIMD
#endif

#define BILINEAR(v, a, b, d) (v = (a) + ((b) - (a)) * (d))

#define BILINEAR_HEAD(type)                                       \
    int x, y;                                                     \
    int x0, x1;                                                   \
    double v1, v2;                                                \
    double dx, dy;                                                \
    type *in;                                                     \
    int xsize = im->xsize, ysize = im->ysize;                     \
    if (xin < 0.0 || xin >= xsize || yin < 0.0 || yin >= ysize) { \
        return 0;                                                 \
    }                                                             \
    xin -= 0.5;                                                   \
    yin -= 0.5;                                                   \
    x = FLOOR(xin);                                               \
    y = FLOOR(yin);                                               \
    dx = xin - x;                                                 \
    dy = yin - y;

#define BILINEAR_BODY(type, image, step, offset)    \
    {                                               \
        in = (type *)((image)[YCLIP(y)] + offset);  \
        x0 = XCLIP(x + 0) * step;                   \
        x1 = XCLIP(x + 1) * step;                   \
        BILINEAR(v1, in[x0], in[x1], dx);           \
        if (y + 1 >= 0 && y + 1 < ysize) {          \
            in = (type *)((image)[y + 1] + offset); \
            BILINEAR(v2, in[x0], in[x1], dx);       \
        } else {                                    \
            v2 = v1;                                \
        }                                           \
        BILINEAR(v1, v1, v2, dy);                   \
    }

static int
bilinear_filter8(void *out, Imaging im, double xin, double yin) {
    BILINEAR_HEAD(UINT8);
    BILINEAR_BODY(UINT8, im->image8, 1, 0);
    ((UINT8 *)out)[0] = (UINT8)v1;
    return 1;
}

static int
bilinear_filter32I(void *out, Imaging im, double xin, double yin) {
    INT32 k;
    BILINEAR_HEAD(INT32);
    BILINEAR_BODY(INT32, im->image32, 1, 0);
    k = v1;
    memcpy(out, &k, sizeof(k));
    return 1;
}

static int
bilinear_filter32F(void *out, Imaging im, double xin, double yin) {
    FLOAT32 k;
    BILINEAR_HEAD(FLOAT32);
    BILINEAR_BODY(FLOAT32, im->image32, 1, 0);
    k = v1;
    memcpy(out, &k, sizeof(k));
    return 1;
}

static int
bilinear_filter32LA(void *out, Imaging im, double xin, double yin) {
    BILINEAR_HEAD(UINT8);
    BILINEAR_BODY(UINT8, im->image, 4, 0);
    ((UINT8 *)out)[0] = (UINT8)v1;
    ((UINT8 *)out)[1] = (UINT8)v1;
    ((UINT8 *)out)[2] = (UINT8)v1;
    BILINEAR_BODY(UINT8, im->image, 4, 3);
    ((UINT8 *)out)[3] = (UINT8)v1;
    return 1;
}

#if !defined(TRANSFORM_SIMD)
static int
bilinear_filter32RGB(void *out, Imaging im, double xin, double yin) {
    int b;
    BILINEAR_HEAD(UINT8);
    for (b = 0; b < im->bands; b++) {
        BILINEAR_BODY(UINT8, im->image, 4, b);
        ((UINT8 *)out)[b] = (UINT8)v1;
    }
    return 1;
}
#endif

#undef BILINEAR
#undef BILINEAR_HEAD
#undef BILINEAR_BODY

#define BICUBIC(v, v1, v2, v3, v4, d)                \
    {                                                \
        double p1 = v2;                              \
        double p2 = -v1 + v3;                        \
        double p3 = 2 * (v1 - v2) + v3 - v4;         \
        double p4 = -v1 + v2 - v3 + v4;              \
        v = p1 + (d) * (p2 + (d) * (p3 + (d) * p4)); \
    }

#define BICUBIC_HEAD(type)                                        \
    int x = FLOOR(xin);                                           \
    int y = FLOOR(yin);                                           \
    int x0, x1, x2, x3;                                           \
    double v1, v2, v3, v4;                                        \
    double dx, dy;                                                \
    type *in;                                                     \
    int xsize = im->xsize, ysize = im->ysize;                     \
    if (xin < 0.0 || xin >= xsize || yin < 0.0 || yin >= ysize) { \
        return 0;                                                 \
    }                                                             \
    xin -= 0.5;                                                   \
    yin -= 0.5;                                                   \
    x = FLOOR(xin);                                               \
    y = FLOOR(yin);                                               \
    dx = xin - x;                                                 \
    dy = yin - y;                                                 \
    x--;                                                          \
    y--;

#define BICUBIC_BODY(type, image, step, offset)              \
    {                                                        \
        in = (type *)((image)[YCLIP(y)] + offset);           \
        x0 = XCLIP(x + 0) * step;                            \
        x1 = XCLIP(x + 1) * step;                            \
        x2 = XCLIP(x + 2) * step;                            \
        x3 = XCLIP(x + 3) * step;                            \
        BICUBIC(v1, in[x0], in[x1], in[x2], in[x3], dx);     \
        if (y + 1 >= 0 && y + 1 < ysize) {                   \
            in = (type *)((image)[y + 1] + offset);          \
            BICUBIC(v2, in[x0], in[x1], in[x2], in[x3], dx); \
        } else {                                             \
            v2 = v1;                                         \
        }                                                    \
        if (y + 2 >= 0 && y + 2 < ysize) {                   \
            in = (type *)((image)[y + 2] + offset);          \
            BICUBIC(v3, in[x0], in[x1], in[x2], in[x3], dx); \
        } else {                                             \
            v3 = v2;                                         \
        }                                                    \
        if (y + 3 >= 0 && y + 3 < ysize) {                   \
            in = (type *)((image)[y + 3] + offset);          \
            BICUBIC(v4, in[x0], in[x1], in[x2], in[x3], dx); \
        } else {                                             \
            v4 = v3;                                         \
        }                                                    \
        BICUBIC(v1, v1, v2, v3, v4, dy);                     \
    }

static int
bicubic_filter8(void *out, Imaging im, double xin, double yin) {
    BICUBIC_HEAD(UINT8);
    BICUBIC_BODY(UINT8, im->image8, 1, 0);
    if (v1 <= 0.0) {
        ((UINT8 *)out)[0] = 0;
    } else if (v1 >= 255.0) {
        ((UINT8 *)out)[0] = 255;
    } else {
        ((UINT8 *)out)[0] = (UINT8)v1;
    }
    return 1;
}

static int
bicubic_filter32I(void *out, Imaging im, double xin, double yin) {
    INT32 k;
    BICUBIC_HEAD(INT32);
    BICUBIC_BODY(INT32, im->image32, 1, 0);
    k = v1;
    memcpy(out, &k, sizeof(k));
    return 1;
}

static int
bicubic_filter32F(void *out, Imaging im, double xin, double yin) {
    FLOAT32 k;
    BICUBIC_HEAD(FLOAT32);
    BICUBIC_BODY(FLOAT32, im->image32, 1, 0);
    k = v1;
    memcpy(out, &k, sizeof(k));
    return 1;
}

static int
bicubic_filter32LA(void *out, Imaging im, double xin, double yin) {
    BICUBIC_HEAD(UINT8);
    BICUBIC_BODY(UINT8, im->image, 4, 0);
    if (v1 <= 0.0) {
        ((UINT8 *)out)[0] = 0;
        ((UINT8 *)out)[1] = 0;
        ((UINT8 *)out)[2] = 0;
    } else if (v1 >= 255.0) {
        ((UINT8 *)out)[0] = 255;
        ((UINT8 *)out)[1] = 255;
        ((UINT8 *)out)[2] = 255;
    } else {
        ((UINT8 *)out)[0] = (UINT8)v1;
        ((UINT8 *)out)[1] = (UINT8)v1;
        ((UINT8 *)out)[2] = (UINT8)v1;
    }
    BICUBIC_BODY(UINT8, im->image, 4, 3);
    if (v1 <= 0.0) {
        ((UINT8 *)out)[3] = 0;
    } else if (v1 >= 255.0) {
        ((UINT8 *)out)[3] = 255;
    } else {
        ((UINT8 *)out)[3] = (UINT8)v1;
    }
    return 1;
}

#if !defined(TRANSFORM_SIMD)
static int
bicubic_filter32RGB(void *out, Imaging im, double xin, double yin) {
    int b;
    BICUBIC_HEAD(UINT8);
    for (b = 0; b < im->bands; b++) {
        BICUBIC_BODY(UINT8, im->image, 4, b);
        if (v1 <= 0.0) {
            ((UINT8 *)out)[b] = 0;
        } else if (v1 >= 255.0) {
            ((UINT8 *)out)[b] = 255;
        } else {
            ((UINT8 *)out)[b] = (UINT8)v1;
        }
    }
    return 1;
}
#endif

#undef BICUBIC
#undef BICUBIC_HEAD
#undef BICUBIC_BODY

#if defined(TRANSFORM_SIMD)
/* Bilinear and bicubic filters for 3- and 4-band 8-bit images with the
   channels of a pixel side by side: one 256-bit vector of doubles with AVX2,
   two 128-bit vectors otherwise. Each channel goes through the same
   operations in the same order as the scalar filters above, in double, so
   the results are identical; x86 builds don't contract them to FMA. */
/* Constant sizes, so that the copies are inlined. */
static inline void
tv_store_bands(UINT8 *out, INT32 pixel, int bands) {
    if (bands == 4) {
        memcpy(out, &pixel, 4);
    } else {
        memcpy(out, &pixel, 3);
    }
}

#if defined(IMAGING_SIMD_AVX2)
typedef __m256d transform_vec;

static inline transform_vec
tv_load(const UINT8 *p) {
    return _mm256_cvtepi32_pd(mm_cvtepu8_epi32(p));
}

static inline transform_vec
tv_set1(double v) {
    return _mm256_set1_pd(v);
}

static inline transform_vec
tv_add(transform_vec a, transform_vec b) {
    return _mm256_add_pd(a, b);
}

static inline transform_vec
tv_sub(transform_vec a, transform_vec b) {
    return _mm256_sub_pd(a, b);
}

static inline transform_vec
tv_mul(transform_vec a, transform_vec b) {
    return _mm256_mul_pd(a, b);
}

/* Truncates to integers in 0..255 and stores 3 or 4 bytes. */
static inline void
tv_store(UINT8 *out, transform_vec v, int bands) {
    __m256d clamped =
        _mm256_min_pd(_mm256_max_pd(v, _mm256_setzero_pd()), _mm256_set1_pd(255.0));
    __m128i i = _mm256_cvttpd_epi32(clamped);
    INT32 pixel;
    i = _mm_packus_epi16(_mm_packus_epi32(i, i), i);
    pixel = _mm_cvtsi128_si32(i);
    tv_store_bands(out, pixel, bands);
}
#else
typedef struct {
    __m128d lo, hi;
} transform_vec;

static inline transform_vec
tv_load(const UINT8 *p) {
    __m128i i = mm_cvtepu8_epi32(p);
    transform_vec v = {_mm_cvtepi32_pd(i), _mm_cvtepi32_pd(_mm_srli_si128(i, 8))};
    return v;
}

static inline transform_vec
tv_set1(double d) {
    transform_vec v = {_mm_set1_pd(d), _mm_set1_pd(d)};
    return v;
}

static inline transform_vec
tv_add(transform_vec a, transform_vec b) {
    transform_vec v = {_mm_add_pd(a.lo, b.lo), _mm_add_pd(a.hi, b.hi)};
    return v;
}

static inline transform_vec
tv_sub(transform_vec a, transform_vec b) {
    transform_vec v = {_mm_sub_pd(a.lo, b.lo), _mm_sub_pd(a.hi, b.hi)};
    return v;
}

static inline transform_vec
tv_mul(transform_vec a, transform_vec b) {
    transform_vec v = {_mm_mul_pd(a.lo, b.lo), _mm_mul_pd(a.hi, b.hi)};
    return v;
}

/* Truncates to integers in 0..255 and stores 3 or 4 bytes. */
static inline void
tv_store(UINT8 *out, transform_vec v, int bands) {
    __m128d zero = _mm_setzero_pd(), max = _mm_set1_pd(255.0);
    __m128i lo = _mm_cvttpd_epi32(_mm_min_pd(_mm_max_pd(v.lo, zero), max));
    __m128i hi = _mm_cvttpd_epi32(_mm_min_pd(_mm_max_pd(v.hi, zero), max));
    __m128i i = _mm_unpacklo_epi64(lo, hi);
    INT32 pixel;
    i = _mm_packus_epi16(_mm_packus_epi32(i, i), i);
    pixel = _mm_cvtsi128_si32(i);
    tv_store_bands(out, pixel, bands);
}
#endif

/* BILINEAR: a + (b - a) * d */
static inline transform_vec
tv_bilinear(transform_vec a, transform_vec b, transform_vec d) {
    return tv_add(a, tv_mul(tv_sub(b, a), d));
}

/* BICUBIC, term by term: -a + b is b - a, which rounds the same. */
static inline transform_vec
tv_bicubic(
    transform_vec v1,
    transform_vec v2,
    transform_vec v3,
    transform_vec v4,
    transform_vec d
) {
    transform_vec p2 = tv_sub(v3, v1);
    transform_vec p3 = tv_sub(tv_add(tv_mul(tv_set1(2.0), tv_sub(v1, v2)), v3), v4);
    transform_vec p4 = tv_add(tv_sub(tv_sub(v2, v1), v3), v4);
    return tv_add(v2, tv_mul(d, tv_add(p2, tv_mul(d, tv_add(p3, tv_mul(d, p4))))));
}

static int
bilinear_filter32RGB_simd(void *out, Imaging im, double xin, double yin) {
    int xsize = im->xsize, ysize = im->ysize;
    int x, y, x0, x1;
    double dx, dy;
    const UINT8 *in;
    transform_vec d, v1, v2;
    if (xin < 0.0 || xin >= im->xsize || yin < 0.0 || yin >= im->ysize) {
        return 0;
    }
    xin -= 0.5;
    yin -= 0.5;
    x = FLOOR(xin);
    y = FLOOR(yin);
    dx = xin - x;
    dy = yin - y;
    x0 = XCLIP(x + 0) * 4;
    x1 = XCLIP(x + 1) * 4;
    d = tv_set1(dx);
    in = (UINT8 *)im->image[YCLIP(y)];
    v1 = tv_bilinear(tv_load(in + x0), tv_load(in + x1), d);
    if (y + 1 >= 0 && y + 1 < im->ysize) {
        in = (UINT8 *)im->image[y + 1];
        v2 = tv_bilinear(tv_load(in + x0), tv_load(in + x1), d);
    } else {
        v2 = v1;
    }
    tv_store(out, tv_bilinear(v1, v2, tv_set1(dy)), im->bands);
    return 1;
}

static int
bicubic_filter32RGB_simd(void *out, Imaging im, double xin, double yin) {
    int xsize = im->xsize, ysize = im->ysize;
    int x, y, x0, x1, x2, x3;
    double dx, dy;
    const UINT8 *in;
    transform_vec d, v1, v2, v3, v4;
    if (xin < 0.0 || xin >= im->xsize || yin < 0.0 || yin >= im->ysize) {
        return 0;
    }
    xin -= 0.5;
    yin -= 0.5;
    x = FLOOR(xin);
    y = FLOOR(yin);
    dx = xin - x;
    dy = yin - y;
    x--;
    y--;
    x0 = XCLIP(x + 0) * 4;
    x1 = XCLIP(x + 1) * 4;
    x2 = XCLIP(x + 2) * 4;
    x3 = XCLIP(x + 3) * 4;
    d = tv_set1(dx);
#define TV_ROW(in)                                                                \
    tv_bicubic(                                                                   \
        tv_load(in + x0), tv_load(in + x1), tv_load(in + x2), tv_load(in + x3), d \
    )
    in = (UINT8 *)im->image[YCLIP(y)];
    v1 = TV_ROW(in);
    if (y + 1 >= 0 && y + 1 < im->ysize) {
        in = (UINT8 *)im->image[y + 1];
        v2 = TV_ROW(in);
    } else {
        v2 = v1;
    }
    if (y + 2 >= 0 && y + 2 < im->ysize) {
        in = (UINT8 *)im->image[y + 2];
        v3 = TV_ROW(in);
    } else {
        v3 = v2;
    }
    if (y + 3 >= 0 && y + 3 < im->ysize) {
        in = (UINT8 *)im->image[y + 3];
        v4 = TV_ROW(in);
    } else {
        v4 = v3;
    }
#undef TV_ROW
    tv_store(out, tv_bicubic(v1, v2, v3, v4, tv_set1(dy)), im->bands);
    return 1;
}
#endif

static ImagingTransformFilter
getfilter(Imaging im, int filterid) {
    switch (filterid) {
        case IMAGING_TRANSFORM_NEAREST:
            if (im->image8) {
                switch (im->type) {
                    case IMAGING_TYPE_UINT8:
                        return nearest_filter8;
                    case IMAGING_TYPE_I16:
                        return nearest_filter16;
                }
            } else {
                return nearest_filter32;
            }
            break;
        case IMAGING_TRANSFORM_BILINEAR:
            if (im->image8) {
                return bilinear_filter8;
            } else if (im->image32) {
                switch (im->type) {
                    case IMAGING_TYPE_UINT8:
                        if (im->bands == 2) {
                            return bilinear_filter32LA;
                        } else {
#if defined(TRANSFORM_SIMD)
                            return bilinear_filter32RGB_simd;
#else
                            return bilinear_filter32RGB;
#endif
                        }
                    case IMAGING_TYPE_INT32:
                        return bilinear_filter32I;
                    case IMAGING_TYPE_FLOAT32:
                        return bilinear_filter32F;
                }
            }
            break;
        case IMAGING_TRANSFORM_BICUBIC:
            if (im->image8) {
                return bicubic_filter8;
            } else if (im->image32) {
                switch (im->type) {
                    case IMAGING_TYPE_UINT8:
                        if (im->bands == 2) {
                            return bicubic_filter32LA;
                        } else {
#if defined(TRANSFORM_SIMD)
                            return bicubic_filter32RGB_simd;
#else
                            return bicubic_filter32RGB;
#endif
                        }
                    case IMAGING_TYPE_INT32:
                        return bicubic_filter32I;
                    case IMAGING_TYPE_FLOAT32:
                        return bicubic_filter32F;
                }
            }
            break;
    }
    /* no such filter */
    return NULL;
}

/* transformation engines */

#ifndef TRANSFORM_TILE
#define TRANSFORM_TILE 64
#endif

/* perspective_transform for a row of TRANSFORM_TILE pixels from x: the
   same expressions, in a loop of fixed length that the compiler
   vectorizes, so that the divisions run several at once. */
static void
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

int
ImagingGenericTransform(
    Imaging imOut,
    Imaging imIn,
    int x0,
    int y0,
    int x1,
    int y1,
    ImagingTransformMap transform,
    void *transform_data,
    int filterid,
    int fill
) {
    /* slow generic transformation.  use ImagingTransformAffine or
       ImagingScaleAffine where possible. */

    ImagingSectionCookie cookie;
    int x, y, tx, ty;
    char *out;
    double xx, yy;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        ImagingError_ModeError();
        return -1;
    }

    ImagingTransformFilter filter = getfilter(imIn, filterid);
    if (!filter) {
        ImagingError_ValueError("bad filter number");
        return -1;
    }

    /* fast mode, read once for the whole transform */
    if (ImagingGetFast() & IMAGING_FAST_ALL) {
        int method = transform == affine_transform ? IMAGING_TRANSFORM_AFFINE
                     : transform == perspective_transform
                         ? IMAGING_TRANSFORM_PERSPECTIVE
                     : transform == quad_transform ? IMAGING_TRANSFORM_QUAD
                                                   : -1;
        if (method >= 0) {
            int done = ImagingTransformFast(
                imOut, imIn, method, x0, y0, x1, y1, transform_data, filterid, fill
            );
            if (done) {
                return done < 0 ? -1 : 0;
            }
        }
    }

    ImagingCopyPalette(imOut, imIn);

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

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

#if defined(TRANSFORM_SIMD)
    /* Direct calls, which the compiler inlines, for the common cases. */
    if (transform == affine_transform && filter == bilinear_filter32RGB_simd) {
        TRANSFORM_TILES(affine_transform, bilinear_filter32RGB_simd)
    } else if (transform == affine_transform && filter == bicubic_filter32RGB_simd) {
        TRANSFORM_TILES(affine_transform, bicubic_filter32RGB_simd)
    } else if (transform == quad_transform && filter == bilinear_filter32RGB_simd) {
        TRANSFORM_TILES(quad_transform, bilinear_filter32RGB_simd)
    } else if (
        transform == perspective_transform && filter == bilinear_filter32RGB_simd
    ) {
        PERSPECTIVE_TILES(bilinear_filter32RGB_simd)
    } else if (
        transform == perspective_transform && filter == bicubic_filter32RGB_simd
    ) {
        PERSPECTIVE_TILES(bicubic_filter32RGB_simd)
    } else
#endif
        if (transform == perspective_transform) {
        PERSPECTIVE_TILES(filter)
    } else {
        TRANSFORM_TILES(transform, filter)
    }
#undef TRANSFORM_TILES
#undef PERSPECTIVE_TILES

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

    return 0;
}

static int
ImagingScaleAffine(
    Imaging imOut, Imaging imIn, int x0, int y0, int x1, int y1, double a[6], int fill
) {
    /* scale, nearest neighbour resampling */

    ImagingSectionCookie cookie;
    int x, y;
    int xin;
    double xo, yo;
    int xmin, xmax;
    int *xintab;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        ImagingError_ModeError();
        return -1;
    }

    ImagingCopyPalette(imOut, imIn);

    if (x0 < 0) {
        x0 = 0;
    }
    if (y0 < 0) {
        y0 = 0;
    }

    int in_xsize = imIn->xsize, in_ysize = imIn->ysize;
    int out_xsize = imOut->xsize, out_ysize = imOut->ysize;
    if (x1 > out_xsize) {
        x1 = out_xsize;
    }
    if (y1 > out_ysize) {
        y1 = out_ysize;
    }

    /* malloc check ok, uses calloc for overflow */
    xintab = (int *)calloc(out_xsize, sizeof(int));
    if (!xintab) {
        ImagingError_MemoryError();
        return -1;
    }

    xo = a[2] + a[0] * 0.5;
    yo = a[5] + a[4] * 0.5;

    xmin = x1;
    xmax = x0;

    /* Pretabulate horizontal pixel positions */
    for (x = x0; x < x1; x++) {
        xin = COORD(xo);
        if (xin >= 0 && xin < (int)in_xsize) {
            xmax = x + 1;
            if (x < xmin) {
                xmin = x;
            }
            xintab[x] = xin;
        }
        xo += a[0];
    }

#define AFFINE_SCALE(pixel, image)                          \
    for (y = y0; y < y1; y++) {                             \
        int yi = COORD(yo);                                 \
        pixel *in, *out;                                    \
        out = imOut->image[y];                              \
        if (fill && x1 > x0) {                              \
            memset(out + x0, 0, (x1 - x0) * sizeof(pixel)); \
        }                                                   \
        if (yi >= 0 && yi < in_ysize) {                     \
            in = imIn->image[yi];                           \
            for (x = xmin; x < xmax; x++) {                 \
                out[x] = in[xintab[x]];                     \
            }                                               \
        }                                                   \
        yo += a[4];                                         \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

    if (imIn->image8) {
        AFFINE_SCALE(UINT8, image8);
    } else {
        AFFINE_SCALE(INT32, image32);
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef AFFINE_SCALE

    free(xintab);

    return 0;
}

static inline int
check_fixed(double a[6], int x, int y) {
    return (
        fabs(x * a[0] + y * a[1] + a[2]) < 32768.0 &&
        fabs(x * a[3] + y * a[4] + a[5]) < 32768.0
    );
}

static inline int
affine_fixed(
    Imaging imOut,
    Imaging imIn,
    int x0,
    int y0,
    int x1,
    int y1,
    double a[6],
    int filterid,
    int fill
) {
    /* affine transform, nearest neighbour resampling, fixed point
       arithmetics */

    ImagingSectionCookie cookie;
    int x, y, tx, ty;
    int xin, yin;
    int xsize, ysize;
    int xx, yy;
    int a0, a1, a2, a3, a4, a5;

    ImagingCopyPalette(imOut, imIn);

    xsize = (int)imIn->xsize;
    ysize = (int)imIn->ysize;

/* use 16.16 fixed point arithmetics */
#define FIX(v) FLOOR((v) * 65536.0 + 0.5)

    a0 = FIX(a[0]);
    a1 = FIX(a[1]);
    a3 = FIX(a[3]);
    a4 = FIX(a[4]);
    a2 = FIX(a[2] + a[0] * 0.5 + a[1] * 0.5);
    a5 = FIX(a[5] + a[3] * 0.5 + a[4] * 0.5);

#undef FIX

/* In tiles of the output, as ImagingGenericTransform. The coordinates of a
   tile's first pixel are what the per-pixel additions reach there: they fit
   an int, since the corners were checked, but the products may not. */
#define AFFINE_TRANSFORM_FIXED(pixel, image)                                       \
    for (ty = y0; ty < y1; ty += TRANSFORM_TILE) {                                 \
        int ty1 = ty + TRANSFORM_TILE < y1 ? ty + TRANSFORM_TILE : y1;             \
        for (tx = x0; tx < x1; tx += TRANSFORM_TILE) {                             \
            int tx1 = tx + TRANSFORM_TILE < x1 ? tx + TRANSFORM_TILE : x1;         \
            for (y = ty; y < ty1; y++) {                                           \
                pixel *out = (pixel *)imOut->image[y] + tx;                        \
                xx = (int)(a2 + (int64_t)(y - y0) * a1 + (int64_t)(tx - x0) * a0); \
                yy = (int)(a5 + (int64_t)(y - y0) * a4 + (int64_t)(tx - x0) * a3); \
                if (fill) {                                                        \
                    memset(out, 0, (tx1 - tx) * sizeof(pixel));                    \
                }                                                                  \
                for (x = tx; x < tx1; x++, out++) {                                \
                    xin = xx >> 16;                                                \
                    if (xin >= 0 && xin < xsize) {                                 \
                        yin = yy >> 16;                                            \
                        if (yin >= 0 && yin < ysize) {                             \
                            *out = imIn->image[yin][xin];                          \
                        }                                                          \
                    }                                                              \
                    xx += a0;                                                      \
                    yy += a3;                                                      \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

    if (imIn->image8) {
        AFFINE_TRANSFORM_FIXED(UINT8, image8)
    } else {
        AFFINE_TRANSFORM_FIXED(INT32, image32)
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef AFFINE_TRANSFORM_FIXED

    return 0;
}

int
ImagingTransformAffine(
    Imaging imOut,
    Imaging imIn,
    int x0,
    int y0,
    int x1,
    int y1,
    double a[6],
    int filterid,
    int fill
) {
    /* affine transform, nearest neighbour resampling, floating point
       arithmetics*/

    ImagingSectionCookie cookie;
    int x, y;
    int xin, yin;
    int xsize, ysize;
    double xx, yy;
    double xo, yo;

    if (!imOut || !imIn || imIn->mode != imOut->mode) {
        ImagingError_ModeError();
        return -1;
    }

    if (filterid || imIn->type == IMAGING_TYPE_I16) {
        return ImagingGenericTransform(
            imOut, imIn, x0, y0, x1, y1, affine_transform, a, filterid, fill
        );
    }

    if (a[1] == 0 && a[3] == 0) {
        /* Scaling */
        return ImagingScaleAffine(imOut, imIn, x0, y0, x1, y1, a, fill);
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

    /* translate all four corners to check if they are within the
       range that can be represented by the fixed point arithmetics */

    if (check_fixed(a, 0, 0) && check_fixed(a, x1 - x0, y1 - y0) &&
        check_fixed(a, 0, y1 - y0) && check_fixed(a, x1 - x0, 0)) {
        return affine_fixed(imOut, imIn, x0, y0, x1, y1, a, filterid, fill);
    }

    /* FIXME: cannot really think of any reasonable case when the
       following code is used.  maybe we should fall back on the slow
       generic transform engine in this case? */

    ImagingCopyPalette(imOut, imIn);

    xsize = (int)imIn->xsize;
    ysize = (int)imIn->ysize;

    xo = a[2] + a[1] * 0.5 + a[0] * 0.5;
    yo = a[5] + a[4] * 0.5 + a[3] * 0.5;

#define AFFINE_TRANSFORM(pixel, image)                      \
    for (y = y0; y < y1; y++) {                             \
        pixel *out;                                         \
        xx = xo;                                            \
        yy = yo;                                            \
        out = imOut->image[y];                              \
        if (fill && x1 > x0) {                              \
            memset(out + x0, 0, (x1 - x0) * sizeof(pixel)); \
        }                                                   \
        for (x = x0; x < x1; x++, out++) {                  \
            xin = COORD(xx);                                \
            if (xin >= 0 && xin < xsize) {                  \
                yin = COORD(yy);                            \
                if (yin >= 0 && yin < ysize) {              \
                    *out = imIn->image[yin][xin];           \
                }                                           \
            }                                               \
            xx += a[0];                                     \
            yy += a[3];                                     \
        }                                                   \
        xo += a[1];                                         \
        yo += a[4];                                         \
    }

    ImagingSectionEnter(&cookie);
    MUTEX_LOCK(&imIn->mutex);

    if (imIn->image8) {
        AFFINE_TRANSFORM(UINT8, image8)
    } else {
        AFFINE_TRANSFORM(INT32, image32)
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);

#undef AFFINE_TRANSFORM

    return 0;
}

/**
 * Runs a transformation over the data from imIn, placing the result in imOut.
 *
 * @param imOut     Existing image to write into
 * @param imIn      Source image to transform
 * @param method    Which method to use
 * @param x0        The left co-ordinate of the area to transform
 * @param y0        The upper co-ordinate of the area to transform
 * @param x1        The right co-ordinate of the area to transform
 * @param y1        The lower co-ordinate of the area to transform
 * @param a         An array of extra data for the transformation method
 * @param filterid  Which filter to use
 * @param fill      1 to fill the area with zeros
 * @return          -1 in case of an error, 0 otherwise
 */
int
ImagingTransform(
    Imaging imOut,
    Imaging imIn,
    int method,
    int x0,
    int y0,
    int x1,
    int y1,
    double a[8],
    int filterid,
    int fill
) {
    ImagingTransformMap transform;

    switch (method) {
        case IMAGING_TRANSFORM_AFFINE:
            return ImagingTransformAffine(
                imOut, imIn, x0, y0, x1, y1, a, filterid, fill
            );
            break;
        case IMAGING_TRANSFORM_PERSPECTIVE:
            transform = perspective_transform;
            break;
        case IMAGING_TRANSFORM_QUAD:
            transform = quad_transform;
            break;
        default:
            ImagingError_ValueError("bad transform method");
            return -1;
    }

    return ImagingGenericTransform(
        imOut, imIn, x0, y0, x1, y1, transform, a, filterid, fill
    );
}
