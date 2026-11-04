/* The geometry kernels (Geometry.c), compiled into SIMDKernels.c: the 90
   degree transposes and the row flips of 8-bit and 32-bit images in
   register blocks, and the bilinear and bicubic filters of the transforms
   for 8-bit images with 3 or 4 bands, with tile loops of their own so the
   filters are inlined. */

#include "GeometryTransform.h"

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

/* Returns 0 for pixel sizes other than 1 and 4 (16-bit images). */
static int
transpose_blocks(Imaging imOut, Imaging imIn, int flipx, int flipy) {
    if (imIn->pixelsize == 1) {
        transpose_blocks8(imOut, imIn, flipx, flipy);
    } else if (imIn->pixelsize == 4) {
        transpose_blocks32(imOut, imIn, flipx, flipy);
    } else {
        return 0;
    }
    return 1;
}

/* Each row of an 8-bit or 32-bit image reversed, into the same row of
   imOut or, with flipy, the mirrored one (flip left right, rotate 180). */
static int
flip_rows(Imaging imOut, Imaging imIn, int flipy) {
    int xsize = imIn->xsize, ysize = imIn->ysize;
    int x, y;

    if (imIn->pixelsize != 1 && imIn->pixelsize != 4) {
        return 0;
    }
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
    return 1;
}

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

#if defined(IMAGING_SIMD_SSE4)
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
#else
/* NEON. BILINEAR and BICUBIC get bytes in the scalar filters, so b - a and
   p1 to p4 are integers there, converted to double; the rest is double, with
   the fused multiply-adds that GCC and clang make of those expressions on
   arm64. Here the integers are computed for the 4 bands of two rows at
   once, in 16-bit lanes, and converted through float, which holds them
   exactly. Converting each byte to double instead, as above, was slower
   than the scalar filters on a Cortex-A78AE. */

/* Eight 16-bit lanes to four pairs of doubles. */
static inline void
tn_double(int16x8_t v, float64x2_t out[4]) {
    float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v)));
    float32x4_t hi = vcvtq_f32_s32(vmovl_high_s16(v));
    out[0] = vcvt_f64_f32(vget_low_f32(lo));
    out[1] = vcvt_high_f64_f32(lo);
    out[2] = vcvt_f64_f32(vget_low_f32(hi));
    out[3] = vcvt_high_f64_f32(hi);
}

/* The pixels at byte offsets x0 and x1, or x0 to x3, of a row. */
static inline uint8x8_t
tn_load2(const UINT8 *in, int x0, int x1) {
    UINT32 p[2];
    if (x1 == x0 + 4) {
        return vld1_u8(in + x0);
    }
    memcpy(&p[0], in + x0, 4);
    memcpy(&p[1], in + x1, 4);
    return vreinterpret_u8_u32(vld1_u32(p));
}

static inline uint8x16_t
tn_load4(const UINT8 *in, int x0, int x1, int x2, int x3) {
    UINT32 p[4];
    if (x3 == x0 + 12) {
        return vld1q_u8(in + x0);
    }
    memcpy(&p[0], in + x0, 4);
    memcpy(&p[1], in + x1, 4);
    memcpy(&p[2], in + x2, 4);
    memcpy(&p[3], in + x3, 4);
    return vreinterpretq_u8_u32(vld1q_u32(p));
}

/* 3 or 4 bands of the pixel in bytes 0-3 of b, or of two pixels, in bytes
   0-3 and 4-7, stored from the vector register: 3 bytes copied from a
   general register went through the stack, which stalled. */
static inline void
tn_store_bytes(UINT8 *out, uint8x8_t b, int bands, int pixels) {
    if (bands == 4) {
        if (pixels == 2) {
            vst1_u8(out, b);
        } else {
            vst1_lane_u32((uint32_t *)out, vreinterpret_u32_u8(b), 0);
        }
        return;
    }
    vst1_lane_u16((uint16_t *)out, vreinterpret_u16_u8(b), 0);
    vst1_lane_u8(out + 2, b, 2);
    if (pixels == 2) {
        vst1_lane_u16((uint16_t *)(out + 4), vreinterpret_u16_u8(b), 2);
        vst1_lane_u8(out + 6, b, 6);
    }
}

/* Bands 0-1 and 2-3 of a pixel, in 0..255, truncated as (UINT8) does, as
   32-bit lanes. */
static inline uint32x4_t
tn_bands(float64x2_t lo, float64x2_t hi) {
    return vuzp1q_u32(
        vreinterpretq_u32_u64(vcvtq_u64_f64(lo)),
        vreinterpretq_u32_u64(vcvtq_u64_f64(hi))
    );
}

static inline void
tn_store(UINT8 *out, float64x2_t lo, float64x2_t hi, int bands) {
    uint16x8_t h = vreinterpretq_u16_u32(tn_bands(lo, hi));
    tn_store_bytes(out, vmovn_u16(vuzp1q_u16(h, h)), bands, 1);
}

/* The same for two pixels: a table lookup, faster here than narrowing
   as tn_store does. */
static inline void
tn_store2(
    UINT8 *out,
    float64x2_t lo0,
    float64x2_t hi0,
    float64x2_t lo1,
    float64x2_t hi1,
    int bands
) {
    uint8x16x4_t t = {
        {vreinterpretq_u8_u64(vcvtq_u64_f64(lo0)),
         vreinterpretq_u8_u64(vcvtq_u64_f64(hi0)),
         vreinterpretq_u8_u64(vcvtq_u64_f64(lo1)),
         vreinterpretq_u8_u64(vcvtq_u64_f64(hi1))}
    };
    tn_store_bytes(out, vqtbl4_u8(t, vcreate_u8(0x3830282018100800)), bands, 2);
}

/* BILINEAR across two pixels of rows a and b: r[0] and r[1] for bands 0-1
   and 2-3 of a, r[2] and r[3] of b. */
static inline void
tn_bilinear2(uint8x8_t a, uint8x8_t b, float64x2_t d, float64x2_t r[4]) {
    uint32x2_t ua = vreinterpret_u32_u8(a), ub = vreinterpret_u32_u8(b);
    /* the first pixels of both rows, then the second ones */
    uint8x8_t v1 = vreinterpret_u8_u32(vtrn1_u32(ua, ub));
    uint8x8_t v2 = vreinterpret_u8_u32(vtrn2_u32(ua, ub));
    float64x2_t f1[4], f2[4];
    int i;
    tn_double(vreinterpretq_s16_u16(vmovl_u8(v1)), f1);
    tn_double(vreinterpretq_s16_u16(vsubl_u8(v2, v1)), f2);
    for (i = 0; i < 4; i++) {
        r[i] = vfmaq_f64(f1[i], f2[i], d);
    }
}

/* BICUBIC across four pixels of rows a and b, as tn_bilinear2. */
static inline void
tn_bicubic2(uint8x16_t a, uint8x16_t b, float64x2_t d, float64x2_t r[4]) {
    uint32x4_t ua = vreinterpretq_u32_u8(a), ub = vreinterpretq_u32_u8(b);
    /* pixels 0 and 2 of both rows, then pixels 1 and 3 */
    uint8x16_t e = vreinterpretq_u8_u32(vtrn1q_u32(ua, ub));
    uint8x16_t o = vreinterpretq_u8_u32(vtrn2q_u32(ua, ub));
    int16x8_t v1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(e)));
    int16x8_t v2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(o)));
    int16x8_t v3 = vreinterpretq_s16_u16(vmovl_high_u8(e));
    int16x8_t v4 = vreinterpretq_s16_u16(vmovl_high_u8(o));
    float64x2_t p1[4], p2[4], p3[4], p4[4];
    int i;
    tn_double(v2, p1);
    tn_double(vsubq_s16(v3, v1), p2);
    tn_double(vsubq_s16(vaddq_s16(vshlq_n_s16(vsubq_s16(v1, v2), 1), v3), v4), p3);
    tn_double(vaddq_s16(vsubq_s16(vsubq_s16(v2, v1), v3), v4), p4);
    for (i = 0; i < 4; i++) {
        r[i] = vfmaq_f64(p1[i], d, vfmaq_f64(p2[i], d, vfmaq_f64(p3[i], d, p4[i])));
    }
}

/* The vertical BICUBIC, of doubles; -a + b is b - a, which rounds the same,
   and 2 (a - b) + c is the same fused or not. */
static inline float64x2_t
tn_bicubic_v(
    float64x2_t v1, float64x2_t v2, float64x2_t v3, float64x2_t v4, float64x2_t d
) {
    float64x2_t p2 = vsubq_f64(v3, v1);
    float64x2_t p3 = vsubq_f64(vaddq_f64(vmulq_n_f64(vsubq_f64(v1, v2), 2.0), v3), v4);
    float64x2_t p4 = vaddq_f64(vsubq_f64(vsubq_f64(v2, v1), v3), v4);
    float64x2_t v = vfmaq_f64(v2, d, vfmaq_f64(p2, d, vfmaq_f64(p3, d, p4)));
    return vminq_f64(vmaxq_f64(v, vdupq_n_f64(0.0)), vdupq_n_f64(255.0));
}

static int
bilinear_filter32RGB_simd(void *out, Imaging im, double xin, double yin) {
    int xsize = im->xsize, ysize = im->ysize;
    int x, y, x0, x1;
    double dx, dy;
    const UINT8 *in0, *in1;
    float64x2_t r[4], d;
    if (xin < 0.0 || xin >= xsize || yin < 0.0 || yin >= ysize) {
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
    /* a missing second row gives the first row's values, as in the scalar
       filter */
    in0 = (UINT8 *)im->image[YCLIP(y)];
    in1 = y + 1 >= 0 && y + 1 < ysize ? (UINT8 *)im->image[y + 1] : in0;
    tn_bilinear2(tn_load2(in0, x0, x1), tn_load2(in1, x0, x1), vdupq_n_f64(dx), r);
    d = vdupq_n_f64(dy);
    tn_store(
        out,
        vfmaq_f64(r[0], vsubq_f64(r[2], r[0]), d),
        vfmaq_f64(r[1], vsubq_f64(r[3], r[1]), d),
        im->bands
    );
    return 1;
}

static int
bicubic_filter32RGB_simd(void *out, Imaging im, double xin, double yin) {
    int xsize = im->xsize, ysize = im->ysize;
    int x, y, x0, x1, x2, x3, k;
    double dx, dy;
    const UINT8 *in[4];
    float64x2_t ra[4], rb[4], d;
    if (xin < 0.0 || xin >= xsize || yin < 0.0 || yin >= ysize) {
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
    /* a missing row gives the values of the row before, as in the scalar
       filter */
    in[0] = (UINT8 *)im->image[YCLIP(y)];
    for (k = 1; k < 4; k++) {
        in[k] = y + k >= 0 && y + k < ysize ? (UINT8 *)im->image[y + k] : in[k - 1];
    }
    d = vdupq_n_f64(dx);
    tn_bicubic2(
        tn_load4(in[0], x0, x1, x2, x3), tn_load4(in[1], x0, x1, x2, x3), d, ra
    );
    tn_bicubic2(
        tn_load4(in[2], x0, x1, x2, x3), tn_load4(in[3], x0, x1, x2, x3), d, rb
    );
    d = vdupq_n_f64(dy);
    tn_store(
        out,
        tn_bicubic_v(ra[0], ra[2], rb[0], rb[2], d),
        tn_bicubic_v(ra[1], ra[3], rb[1], rb[3], d),
        im->bands
    );
    return 1;
}

/* Two pixels at once, so that their long chains of dependent operations
   overlap, when both have all their source pixels inside the image: 1.3 to
   1.6 times as fast as the filters above, one pixel at a time, on a
   Cortex-A78AE. tn_coords gives the coordinates as those filters do, if
   source pixels x to x + span - 1 and y to y + span - 1, x and y less
   margin, are inside the image: compared in double first, so that the
   conversions don't overflow and (int) floors. */
static inline int
tn_coords(
    Imaging im,
    double xin,
    double yin,
    int margin,
    int span,
    int *x,
    int *y,
    double *dx,
    double *dy
) {
    xin -= 0.5;
    yin -= 0.5;
    if (!(xin >= margin && xin < im->xsize + margin - span + 1 && yin >= margin &&
          yin < im->ysize + margin - span + 1)) {
        return 0;
    }
    *x = (int)xin;
    *y = (int)yin;
    *dx = xin - *x;
    *dy = yin - *y;
    *x -= margin;
    *y -= margin;
    return 1;
}

static inline void
tn_bilinear_rows(Imaging im, int x, int y, double dx, float64x2_t r[4]) {
    const UINT8 *in0 = (UINT8 *)im->image[y] + x * 4;
    const UINT8 *in1 = (UINT8 *)im->image[y + 1] + x * 4;
    tn_bilinear2(vld1_u8(in0), vld1_u8(in1), vdupq_n_f64(dx), r);
}

static inline void
tn_bicubic_rows(
    Imaging im, int x, int y, double dx, float64x2_t ra[4], float64x2_t rb[4]
) {
    const UINT8 *in0 = (UINT8 *)im->image[y] + x * 4;
    const UINT8 *in1 = (UINT8 *)im->image[y + 1] + x * 4;
    const UINT8 *in2 = (UINT8 *)im->image[y + 2] + x * 4;
    const UINT8 *in3 = (UINT8 *)im->image[y + 3] + x * 4;
    float64x2_t d = vdupq_n_f64(dx);
    tn_bicubic2(vld1q_u8(in0), vld1q_u8(in1), d, ra);
    tn_bicubic2(vld1q_u8(in2), vld1q_u8(in3), d, rb);
}

/* The filters above for the pixels at out and out + 4, if both have all
   their source pixels inside the image; otherwise returns 0, having
   written nothing. Written with loops over the two pixels for bilinear and
   without for bicubic, which GCC then interleaves better: 4% to 8% faster
   each. */
static inline int
bilinear_filter32RGB_simd2(
    UINT8 *out, Imaging im, const double *xin, const double *yin
) {
    int x[2], y[2], k;
    double dx[2], dy[2];
    float64x2_t r[2][4], v[2][2];
    for (k = 0; k < 2; k++) {
        if (!tn_coords(im, xin[k], yin[k], 0, 2, &x[k], &y[k], &dx[k], &dy[k])) {
            return 0;
        }
    }
    for (k = 0; k < 2; k++) {
        tn_bilinear_rows(im, x[k], y[k], dx[k], r[k]);
    }
    for (k = 0; k < 2; k++) {
        float64x2_t dv = vdupq_n_f64(dy[k]);
        v[k][0] = vfmaq_f64(r[k][0], vsubq_f64(r[k][2], r[k][0]), dv);
        v[k][1] = vfmaq_f64(r[k][1], vsubq_f64(r[k][3], r[k][1]), dv);
    }
    tn_store2(out, v[0][0], v[0][1], v[1][0], v[1][1], im->bands);
    return 1;
}

static inline int
bicubic_filter32RGB_simd2(
    UINT8 *out, Imaging im, const double *xin, const double *yin
) {
    int xa, ya, xb, yb;
    double dxa, dya, dxb, dyb;
    float64x2_t ra[4], rb[4], sa[4], sb[4], da, db;
    if (!tn_coords(im, xin[0], yin[0], 1, 4, &xa, &ya, &dxa, &dya) ||
        !tn_coords(im, xin[1], yin[1], 1, 4, &xb, &yb, &dxb, &dyb)) {
        return 0;
    }
    tn_bicubic_rows(im, xa, ya, dxa, ra, rb);
    tn_bicubic_rows(im, xb, yb, dxb, sa, sb);
    da = vdupq_n_f64(dya);
    db = vdupq_n_f64(dyb);
    tn_store2(
        out,
        tn_bicubic_v(ra[0], ra[2], rb[0], rb[2], da),
        tn_bicubic_v(ra[1], ra[3], rb[1], rb[3], da),
        tn_bicubic_v(sa[0], sa[2], sb[0], sb[2], db),
        tn_bicubic_v(sa[1], sa[3], sb[1], sb[3], db),
        im->bands
    );
    return 1;
}
#endif

/* ImagingGenericTransform's tiles for the filters above. Returns 0 for the
   images and filters they don't take. */
static int
geometry_transform_simd(
    Imaging imOut,
    Imaging imIn,
    int x0,
    int y0,
    int x1,
    int y1,
    int method,
    void *transform_data,
    int filterid,
    int fill
) {
    ImagingTransformMap transform =
        method == IMAGING_TRANSFORM_AFFINE ? affine_transform
        : method == IMAGING_TRANSFORM_QUAD ? quad_transform
                                           : perspective_transform;
    ImagingTransformFilter filter;
    int x, y, tx, ty;
    char *out;
#if !defined(IMAGING_SIMD_NEON)
    double xx, yy;
#endif

    if (!imIn->image32 || imIn->type != IMAGING_TYPE_UINT8 || imIn->bands == 2) {
        return 0;
    }
    if (filterid == IMAGING_TRANSFORM_BILINEAR) {
        filter = bilinear_filter32RGB_simd;
    } else if (filterid == IMAGING_TRANSFORM_BICUBIC) {
        filter = bicubic_filter32RGB_simd;
    } else {
        return 0;
    }
#if defined(IMAGING_SIMD_NEON)
    /* Pairs of pixels through the filters for two when both are inside the
       image, else one at a time. */
#define FILTER_PAIR(filter, xs, ys, ok, n)                                           \
    if ((n) < 2 || !(ok)[0] || !(ok)[1] || !filter##2((UINT8 *)out, imIn, xs, ys)) { \
        int k;                                                                       \
        for (k = 0; k < (n); k++) {                                                  \
            char *p = out + k * imOut->pixelsize;                                    \
            if ((!(ok)[k] || !filter(p, imIn, (xs)[k], (ys)[k])) && fill) {          \
                memset(p, 0, imOut->pixelsize);                                      \
            }                                                                        \
        }                                                                            \
    }

#define TRANSFORM_PAIRS(transform, filter)                                     \
    for (ty = y0; ty < y1; ty += TRANSFORM_TILE) {                             \
        int ty1 = ty + TRANSFORM_TILE < y1 ? ty + TRANSFORM_TILE : y1;         \
        for (tx = x0; tx < x1; tx += TRANSFORM_TILE) {                         \
            int tx1 = tx + TRANSFORM_TILE < x1 ? tx + TRANSFORM_TILE : x1;     \
            for (y = ty; y < ty1; y++) {                                       \
                out = imOut->image[y] + tx * imOut->pixelsize;                 \
                for (x = tx; x < tx1; x += 2) {                                \
                    double xs[2], ys[2];                                       \
                    int ok[2] = {1, 1}, i, n = x + 1 < tx1 ? 2 : 1;            \
                    for (i = 0; i < n; i++) {                                  \
                        ok[i] = transform(                                     \
                            &xs[i], &ys[i], x + i - x0, y - y0, transform_data \
                        );                                                     \
                    }                                                          \
                    FILTER_PAIR(filter, xs, ys, ok, n)                         \
                    out += n * imOut->pixelsize;                               \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

#define PERSPECTIVE_PAIRS(filter)                                             \
    for (ty = y0; ty < y1; ty += TRANSFORM_TILE) {                            \
        int ty1 = ty + TRANSFORM_TILE < y1 ? ty + TRANSFORM_TILE : y1;        \
        for (tx = x0; tx < x1; tx += TRANSFORM_TILE) {                        \
            int i, n = (tx + TRANSFORM_TILE < x1 ? TRANSFORM_TILE : x1 - tx); \
            for (y = ty; y < ty1; y++) {                                      \
                double xs[TRANSFORM_TILE], ys[TRANSFORM_TILE];                \
                static const int ok[2] = {1, 1};                              \
                perspective_row(xs, ys, transform_data, tx - x0, y - y0);     \
                out = imOut->image[y] + tx * imOut->pixelsize;                \
                for (i = 0; i < n; i += 2) {                                  \
                    int m = i + 1 < n ? 2 : 1;                                \
                    FILTER_PAIR(filter, xs + i, ys + i, ok, m)                \
                    out += m * imOut->pixelsize;                              \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }

    if (transform == affine_transform && filter == bilinear_filter32RGB_simd) {
        TRANSFORM_PAIRS(affine_transform, bilinear_filter32RGB_simd)
    } else if (transform == affine_transform && filter == bicubic_filter32RGB_simd) {
        TRANSFORM_PAIRS(affine_transform, bicubic_filter32RGB_simd)
    } else if (transform == quad_transform && filter == bilinear_filter32RGB_simd) {
        TRANSFORM_PAIRS(quad_transform, bilinear_filter32RGB_simd)
    } else if (transform == quad_transform && filter == bicubic_filter32RGB_simd) {
        TRANSFORM_PAIRS(quad_transform, bicubic_filter32RGB_simd)
    } else if (
        transform == perspective_transform && filter == bilinear_filter32RGB_simd
    ) {
        PERSPECTIVE_PAIRS(bilinear_filter32RGB_simd)
    } else if (
        transform == perspective_transform && filter == bicubic_filter32RGB_simd
    ) {
        PERSPECTIVE_PAIRS(bicubic_filter32RGB_simd)
    } else {
        return 0;
    }
#undef FILTER_PAIR
#undef TRANSFORM_PAIRS
#undef PERSPECTIVE_PAIRS
#else
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
    } else if (transform == perspective_transform) {
        PERSPECTIVE_TILES(filter)
    } else {
        TRANSFORM_TILES(transform, filter)
    }
#endif
    return 1;
}

static const ImagingSIMDGeometryKernels geometry_simd_kernels = {
    .transpose = transpose_blocks,
    .flip_rows = flip_rows,
    .bilinear_rgb = bilinear_filter32RGB_simd,
    .bicubic_rgb = bicubic_filter32RGB_simd,
    .transform = geometry_transform_simd,
};
