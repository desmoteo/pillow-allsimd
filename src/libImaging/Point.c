/*
 * The Python Imaging Library
 * $Id$
 *
 * point (pixel) translation
 *
 * history:
 * 1995-11-27 fl   Created
 * 1996-03-31 fl   Fixed colour support
 * 1996-08-13 fl   Support 8-bit to "1" thresholding
 * 1997-05-31 fl   Added floating point transform
 * 1998-07-02 fl   Added integer point transform
 * 1998-07-17 fl   Support L to anything lookup
 * 2004-12-18 fl   Refactored; added I to L lookup
 *
 * Copyright (c) 1997-2004 by Secret Labs AB.
 * Copyright (c) 1995-2004 by Fredrik Lundh.
 *
 * See the README file for information on usage and redistribution.
 */

#include "Imaging.h"

typedef struct {
    const void *table;
} im_point_context;

#if defined(IMAGING_SIMD_NEON)
/* 16 lookups in a 256-entry byte table, held as four 64-byte groups of
   registers: tbl for indices 0-63 (others give 0), tbx for the next
   groups (indices out of a group's range keep the byte). */
static inline uint8x16_t
neon_lut16(const uint8x16x4_t *t, uint8x16_t idx) {
    uint8x16_t r = vqtbl4q_u8(t[0], idx);
    r = vqtbx4q_u8(r, t[1], vsubq_u8(idx, vdupq_n_u8(64)));
    r = vqtbx4q_u8(r, t[2], vsubq_u8(idx, vdupq_n_u8(128)));
    return vqtbx4q_u8(r, t[3], vsubq_u8(idx, vdupq_n_u8(192)));
}

static inline void
neon_lut_load(uint8x16x4_t *t, const UINT8 *table) {
    int k;
    for (k = 0; k < 4; k++) {
        t[k] = vld1q_u8_x4(table + 64 * k);
    }
}

/* A row of 4-byte pixels through bands tables, 16 pixels at a time: the
   row split into band planes (in cache), each plane looked up with its
   table loaded once, then interleaved back; a fourth band beyond bands is
   0. Returns the pixels done. */
static int
neon_lut_row(
    UINT8 *out, const UINT8 *in, int xsize, int bands, const UINT8 *table, UINT8 *planes
) {
    const int n = xsize & ~15;
    uint8x16x4_t t[4];
    int x, b;
    for (x = 0; x < n; x += 16) {
        const uint8x16x4_t px = vld4q_u8(in + 4 * x);
        for (b = 0; b < bands; b++) {
            vst1q_u8(planes + b * xsize + x, px.val[b]);
        }
    }
    for (b = 0; b < bands; b++) {
        UINT8 *plane = planes + b * xsize;
        neon_lut_load(t, table + 256 * b);
        for (x = 0; x < n; x += 16) {
            vst1q_u8(plane + x, neon_lut16(t, vld1q_u8(plane + x)));
        }
    }
    for (x = 0; x < n; x += 16) {
        uint8x16x4_t px;
        for (b = 0; b < 4; b++) {
            px.val[b] = b < bands ? vld1q_u8(planes + b * xsize + x) : vdupq_n_u8(0);
        }
        vst4q_u8(out + 4 * x, px);
    }
    return n;
}
#endif

static void
im_point_8_8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 8-bit source, 8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
#if defined(IMAGING_SIMD_NEON)
    uint8x16x4_t t[4];
    neon_lut_load(t, table);
#endif
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *in = imIn->image8[y];
        UINT8 *out = imOut->image8[y];
        x = 0;
#if defined(IMAGING_SIMD_NEON)
        for (; x + 16 <= imIn->xsize; x += 16) {
            vst1q_u8(out + x, neon_lut16(t, vld1q_u8(in + x)));
        }
#endif
        for (; x < imIn->xsize; x++) {
            out[x] = table[in[x]];
        }
    }
}

static void
im_point_2x8_2x8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 2x8-bit source, 2x8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *in = (UINT8 *)imIn->image[y];
        UINT8 *out = (UINT8 *)imOut->image[y];
        for (x = 0; x < imIn->xsize; x++) {
            out[0] = table[in[0]];
            out[1] = out[2] = 0;  // the padding, as a zero-filled image has it
            out[3] = table[in[3] + 256];
            in += 4;
            out += 4;
        }
    }
}

static void
im_point_3x8_3x8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 3x8-bit source, 3x8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
#if defined(IMAGING_SIMD_NEON)
    UINT8 *planes = malloc((size_t)imIn->xsize * 4);
#endif
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *in = (UINT8 *)imIn->image[y];
        UINT8 *out = (UINT8 *)imOut->image[y];
        x = 0;
#if defined(IMAGING_SIMD_NEON)
        if (planes) {
            x = neon_lut_row(out, in, imIn->xsize, 3, table, planes);
            in += 4 * x;
            out += 4 * x;
        }
#endif
        for (; x < imIn->xsize; x++) {
            out[0] = table[in[0]];
            out[1] = table[in[1] + 256];
            out[2] = table[in[2] + 512];
            out[3] = 0;  // the padding, as a zero-filled image has it
            in += 4;
            out += 4;
        }
    }
#if defined(IMAGING_SIMD_NEON)
    free(planes);
#endif
}

static void
im_point_4x8_4x8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 4x8-bit source, 4x8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
#if defined(IMAGING_SIMD_NEON)
    UINT8 *planes = malloc((size_t)imIn->xsize * 4);
#endif
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *in = (UINT8 *)imIn->image[y];
        UINT8 *out = (UINT8 *)imOut->image[y];
        x = 0;
#if defined(IMAGING_SIMD_NEON)
        if (planes) {
            x = neon_lut_row(out, in, imIn->xsize, 4, table, planes);
            in += 4 * x;
            out += 4 * x;
        }
#endif
        for (; x < imIn->xsize; x++) {
            out[0] = table[in[0]];
            out[1] = table[in[1] + 256];
            out[2] = table[in[2] + 512];
            out[3] = table[in[3] + 768];
            in += 4;
            out += 4;
        }
    }
#if defined(IMAGING_SIMD_NEON)
    free(planes);
#endif
}

static void
im_point_8_32(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 8-bit source, 32-bit destination */
    char *table = (char *)context->table;
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *in = imIn->image8[y];
        INT32 *out = imOut->image32[y];
        for (x = 0; x < imIn->xsize; x++) {
            memcpy(out + x, table + in[x] * sizeof(INT32), sizeof(INT32));
        }
    }
}

static void
im_point_32_8(Imaging imOut, Imaging imIn, im_point_context *context) {
    int x, y;
    /* 32-bit source, 8-bit destination */
    UINT8 *table = (UINT8 *)context->table;
    for (y = 0; y < imIn->ysize; y++) {
        INT32 *in = imIn->image32[y];
        UINT8 *out = imOut->image8[y];
        for (x = 0; x < imIn->xsize; x++) {
            int v = in[x];
            if (v < 0) {
                v = 0;
            } else if (v > 65535) {
                v = 65535;
            }
            out[x] = table[v];
        }
    }
}

/* Tables that are one simple operation on every byte, as ImageOps
   invert, posterize and solarize make. When all bands use the same one,
   whole rows are transformed with a loop compilers vectorise. */
enum point_byte_op { POINT_TABLE, POINT_XOR, POINT_AND, POINT_SOLARIZE };

static enum point_byte_op
point_byte_op(const UINT8 *table, int bands, UINT8 *arg) {
    int i, b, t;
    for (b = 1; b < bands; b++) {
        if (memcmp(table, table + 256 * b, 256)) {
            return POINT_TABLE;
        }
    }
    for (i = 0; i < 256 && table[i] == (i ^ table[0]); i++) {
    }
    if (i == 256) {
        *arg = table[0];
        return POINT_XOR;
    }
    for (i = 0; i < 256 && table[i] == (i & table[255]); i++) {
    }
    if (i == 256) {
        *arg = table[255];
        return POINT_AND;
    }
    // i below the threshold t, 255 - i from it
    for (t = 0; t < 256 && table[t] == t; t++) {
    }
    for (i = t; i < 256 && table[i] == 255 - i; i++) {
    }
    if (i == 256) {
        *arg = (UINT8)t;
        return POINT_SOLARIZE;
    }
    return POINT_TABLE;
}

/* One row. Compilers don't vectorise these loops at -O2, which would need
   a scalar epilogue; the SIMD loops leave the last bytes to them. */
static void
point_byte_row(UINT8 *out, const UINT8 *in, int n, enum point_byte_op op, UINT8 arg) {
    int x = 0;
#if defined(IMAGING_SIMD_AVX2)
    __m256i a32 = _mm256_set1_epi8((char)arg);
    for (; x < n - 31; x += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(in + x));
        if (op == POINT_XOR) {
            v = _mm256_xor_si256(v, a32);
        } else if (op == POINT_AND) {
            v = _mm256_and_si256(v, a32);
        } else {
            // v >= arg as max(v, arg) == v, then v ^ 255 there
            v = _mm256_xor_si256(v, _mm256_cmpeq_epi8(_mm256_max_epu8(v, a32), v));
        }
        _mm256_storeu_si256((__m256i *)(out + x), v);
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    __m128i a16 = _mm_set1_epi8((char)arg);
    for (; x < n - 15; x += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(in + x));
        if (op == POINT_XOR) {
            v = _mm_xor_si128(v, a16);
        } else if (op == POINT_AND) {
            v = _mm_and_si128(v, a16);
        } else {
            v = _mm_xor_si128(v, _mm_cmpeq_epi8(_mm_max_epu8(v, a16), v));
        }
        _mm_storeu_si128((__m128i *)(out + x), v);
    }
#elif defined(IMAGING_SIMD_NEON)
    uint8x16_t a16 = vdupq_n_u8(arg);
    for (; x < n - 15; x += 16) {
        uint8x16_t v = vld1q_u8(in + x);
        if (op == POINT_XOR) {
            v = veorq_u8(v, a16);
        } else if (op == POINT_AND) {
            v = vandq_u8(v, a16);
        } else {
            v = veorq_u8(v, vcgeq_u8(v, a16));
        }
        vst1q_u8(out + x, v);
    }
#endif
    for (; x < n; x++) {
        if (op == POINT_XOR) {
            out[x] = in[x] ^ arg;
        } else if (op == POINT_AND) {
            out[x] = in[x] & arg;
        } else {
            // 255 - v is v ^ 255
            out[x] = in[x] ^ (in[x] >= arg ? 255 : 0);
        }
    }
}

static void
point_byte_rows(Imaging imOut, Imaging imIn, enum point_byte_op op, UINT8 arg) {
    int y;
    for (y = 0; y < imIn->ysize; y++) {
        point_byte_row(
            (UINT8 *)imOut->image[y], (UINT8 *)imIn->image[y], imIn->linesize, op, arg
        );
    }
}

Imaging
ImagingPoint(Imaging imIn, ModeID mode, const void *table) {
    /* lookup table transform */

    ImagingSectionCookie cookie;
    Imaging imOut;
    im_point_context context;
    void (*point)(Imaging imIn, Imaging imOut, im_point_context *context);

    if (!imIn) {
        return (Imaging)ImagingError_ModeError();
    }

    if (mode == IMAGING_MODE_UNKNOWN) {
        mode = imIn->mode;
    }

    if (imIn->type == IMAGING_TYPE_UINT8) {
        if (imIn->mode != mode && !imIn->image8) {
            goto mode_mismatch;
        }
    } else if (imIn->type == IMAGING_TYPE_INT32) {
        if (mode != IMAGING_MODE_L) {
            // I images can only be used with L output mode
            goto mode_mismatch;
        }
    } else {
        // Reject F modes and modes based on I;16
        goto mode_mismatch;
    }

    if (imIn->type == IMAGING_TYPE_UINT8 && imIn->mode == mode) {
        UINT8 arg;
        enum point_byte_op op = point_byte_op(table, imIn->bands, &arg);
        if (op != POINT_TABLE) {
            // Every byte of every row is written, padding included.
            imOut = ImagingNewDirty(mode, imIn->xsize, imIn->ysize);
            if (!imOut) {
                return NULL;
            }
            ImagingCopyPalette(imOut, imIn);
            ImagingSectionEnter(&cookie);
            point_byte_rows(imOut, imIn, op, arg);
            ImagingSectionLeave(&cookie);
            return imOut;
        }
    }

    // Every handler below writes every byte, padding included.
    imOut = ImagingNewDirty(mode, imIn->xsize, imIn->ysize);
    if (!imOut) {
        return NULL;
    }

    /* find appropriate handler */
    if (imIn->type == IMAGING_TYPE_UINT8) {
        if (imIn->bands == imOut->bands && imIn->type == imOut->type) {
            switch (imIn->bands) {
                case 1:
                    point = im_point_8_8;
                    break;
                case 2:
                    point = im_point_2x8_2x8;
                    break;
                case 3:
                    point = im_point_3x8_3x8;
                    break;
                case 4:
                    point = im_point_4x8_4x8;
                    break;
                default:
                    /* this cannot really happen */
                    point = im_point_8_8;
                    break;
            }
        } else {
            point = im_point_8_32;
        }
    } else {
        point = im_point_32_8;
    }

    ImagingCopyPalette(imOut, imIn);

    ImagingSectionEnter(&cookie);

    context.table = table;
    point(imOut, imIn, &context);

    ImagingSectionLeave(&cookie);

    return imOut;

mode_mismatch:
    return (Imaging)ImagingError_ValueError(
        "point operation not supported for this mode"
    );
}

Imaging
ImagingPointTransform(Imaging imIn, double scale, double offset) {
    /* scale/offset transform */

    ImagingSectionCookie cookie;
    Imaging imOut;
    int x, y;

    if (!imIn || (imIn->mode != IMAGING_MODE_I && imIn->mode != IMAGING_MODE_I_16 &&
                  imIn->mode != IMAGING_MODE_F)) {
        return (Imaging)ImagingError_ModeError();
    }

    imOut = ImagingNew(imIn->mode, imIn->xsize, imIn->ysize);
    if (!imOut) {
        return NULL;
    }

    switch (imIn->type) {
        case IMAGING_TYPE_INT32:
            ImagingSectionEnter(&cookie);
            for (y = 0; y < imIn->ysize; y++) {
                INT32 *in = imIn->image32[y];
                INT32 *out = imOut->image32[y];
                /* FIXME: add clipping? */
                for (x = 0; x < imIn->xsize; x++) {
                    out[x] = in[x] * scale + offset;
                }
            }
            ImagingSectionLeave(&cookie);
            break;
        case IMAGING_TYPE_FLOAT32:
            ImagingSectionEnter(&cookie);
            for (y = 0; y < imIn->ysize; y++) {
                FLOAT32 *in = (FLOAT32 *)imIn->image32[y];
                FLOAT32 *out = (FLOAT32 *)imOut->image32[y];
                for (x = 0; x < imIn->xsize; x++) {
                    out[x] = in[x] * scale + offset;
                }
            }
            ImagingSectionLeave(&cookie);
            break;
        case IMAGING_TYPE_I16:
            if (imIn->mode == IMAGING_MODE_I_16) {
                ImagingSectionEnter(&cookie);
                for (y = 0; y < imIn->ysize; y++) {
                    char *in = (char *)imIn->image[y];
                    char *out = (char *)imOut->image[y];
                    /* FIXME: add clipping? */
                    for (x = 0; x < imIn->xsize; x++) {
                        UINT16 v;
                        memcpy(&v, in + x * sizeof(v), sizeof(v));
                        v = v * scale + offset;
                        memcpy(out + x * sizeof(UINT16), &v, sizeof(v));
                    }
                }
                ImagingSectionLeave(&cookie);
                break;
            }
            /* FALL THROUGH */
        default:
            ImagingDelete(imOut);
            return (Imaging)ImagingError_ValueError("internal error");
    }

    return imOut;
}
