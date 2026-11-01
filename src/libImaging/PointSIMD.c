/* Point operations (Point.c), compiled into SIMDKernels.c: lookups in
   256-entry tables with NEON's table instructions, whole images, and the
   byte operations (xor, and, solarize) for the bytes the vector loop can,
   their count back. */

#include "Point.h"

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

/* 8-bit source, 8-bit destination: the whole image. */
static int
point_8_8_neon(Imaging imOut, Imaging imIn, const UINT8 *table) {
    int x, y;
    uint8x16x4_t t[4];
    neon_lut_load(t, table);
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *in = imIn->image8[y];
        UINT8 *out = imOut->image8[y];
        x = 0;
        for (; x + 16 <= imIn->xsize; x += 16) {
            vst1q_u8(out + x, neon_lut16(t, vld1q_u8(in + x)));
        }
        for (; x < imIn->xsize; x++) {
            out[x] = table[in[x]];
        }
    }
    return 1;
}

/* 3 or 4 8-bit bands in 4-byte pixels, a table per band: the whole image,
   or 0 if out of memory. */
static int
point_nx8_neon(Imaging imOut, Imaging imIn, const UINT8 *table, int bands) {
    UINT8 *planes = malloc((size_t)imIn->xsize * 4);
    int x, y, b;
    if (!planes) {
        return 0;
    }
    for (y = 0; y < imIn->ysize; y++) {
        UINT8 *in = (UINT8 *)imIn->image[y];
        UINT8 *out = (UINT8 *)imOut->image[y];
        x = neon_lut_row(out, in, imIn->xsize, bands, table, planes);
        in += 4 * x;
        out += 4 * x;
        for (; x < imIn->xsize; x++) {
            for (b = 0; b < bands; b++) {
                out[b] = table[in[b] + 256 * b];
            }
            if (bands == 3) {
                out[3] = 0;  // the padding, as a zero-filled image has it
            }
            in += 4;
            out += 4;
        }
    }
    free(planes);
    return 1;
}
#endif

static int
point_byte_row_simd(UINT8 *out, const UINT8 *in, int n, int op, UINT8 arg) {
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
    return x;
}

static const ImagingSIMDPointKernels point_simd_kernels = {
#if defined(IMAGING_SIMD_NEON)
    .lut_8_8 = point_8_8_neon,
    .lut_nx8 = point_nx8_neon,
#endif
    .byte_row = point_byte_row_simd,
};
