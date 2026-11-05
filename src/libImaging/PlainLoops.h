/*
 * Pillow-AllSIMD: loops in plain C that are compiled once per instruction
 * set.
 *
 * A compiler turns these loops into vector code when it may use AVX2 or
 * NEON, and leaves them scalar, or makes poor vector code of them, for
 * baseline x86-64. A dispatch build (-C simd=dispatch, the x86-64 wheels)
 * compiles Pillow's own files for the baseline, where each of them took
 * 1.6 to 3.4 times as long as in a build for AVX2. So they are defined
 * here once and compiled several times: by the file that owns the
 * operation, for the scalar table, and by SIMDKernels.c for every tier,
 * where the area's companion file puts them in the kernel table.
 *
 * It is the same C every time, and no tier is compiled with FMA
 * (setup.py), so the results are the same in every tier. A loop belongs
 * here only if it is called once per row or per image: through the table
 * a call per pixel or per span costs more than the instruction set gains
 * (design/NEON_ROUND_2026-11-04.md).
 */

#ifndef PILLOW_PLAIN_LOOPS_H
#define PILLOW_PLAIN_LOOPS_H

/* ImageChops' soft light and hard light of two bytes, in int arithmetic.
   Overlay is hard light with the operands exchanged. */
#define CHOP_SOFT_LIGHT(a, b)                \
    ((((255 - (a)) * ((a) * (b))) / 65536) + \
     ((a) * (255 - ((255 - (a)) * (255 - (b)) / 255))) / 255)
#define CHOP_HARD_LIGHT(a, b) \
    (((b) < 128) ? (((a) * (b)) / 127) : 255 - (((255 - (b)) * (255 - (a))) / 127))

/* Chops.c. The output is a fresh image; the inputs may be one image. */
static inline void
plain_soft_light_row(UINT8 *restrict out, const UINT8 *in1, const UINT8 *in2, int n) {
    int x;
    for (x = 0; x < n; x++) {
        out[x] = CHOP_SOFT_LIGHT(in1[x], in2[x]);
    }
}

static inline void
plain_hard_light_row(UINT8 *restrict out, const UINT8 *in1, const UINT8 *in2, int n) {
    int x;
    for (x = 0; x < n; x++) {
        out[x] = CHOP_HARD_LIGHT(in1[x], in2[x]);
    }
}

/* Blend.c: between two images (alpha in 0..1, the exact mode), and beyond
   them, clipped (ImageEnhance with a factor above 1). */
static inline void
plain_blend_interpolate_row(
    UINT8 *restrict out, const UINT8 *in1, const UINT8 *in2, int n, float alpha
) {
    int x;
    for (x = 0; x < n; x++) {
        out[x] = (UINT8)((int)in1[x] + alpha * ((int)in2[x] - (int)in1[x]));
    }
}

static inline void
plain_blend_extrapolate_row(
    UINT8 *restrict out, const UINT8 *in1, const UINT8 *in2, int n, float alpha
) {
    int x;
    for (x = 0; x < n; x++) {
        float temp = (float)((int)in1[x] + alpha * ((int)in2[x] - (int)in1[x]));
        if (temp <= 0.0) {
            out[x] = 0;
        } else if (temp >= 255.0) {
            out[x] = 255;
        } else {
            out[x] = (UINT8)temp;
        }
    }
}

/* Matrix.c: a row of 4-byte pixels through a colour matrix, to L (m is
   the matrix's first row) and to RGB (all three). */
#define PLAIN_CLIPF(v) ((v <= 0.0) ? 0 : (v >= 255.0F) ? 255 : (UINT8)v)

static inline void
plain_matrix_l_row(UINT8 *restrict out, const UINT8 *in, int n, const float *m) {
    int x;
    for (x = 0; x < n; x++) {
        float v = m[0] * in[0] + m[1] * in[1] + m[2] * in[2] + m[3] + 0.5;
        out[x] = PLAIN_CLIPF(v);
        in += 4;
    }
}

static inline void
plain_matrix_rgb_row(UINT8 *restrict out, const UINT8 *in, int n, const float *m) {
    int x;
    for (x = 0; x < n; x++) {
        float v0 = m[0] * in[0] + m[1] * in[1] + m[2] * in[2] + m[3] + 0.5;
        float v1 = m[4] * in[0] + m[5] * in[1] + m[6] * in[2] + m[7] + 0.5;
        float v2 = m[8] * in[0] + m[9] * in[1] + m[10] * in[2] + m[11] + 0.5;
        out[0] = PLAIN_CLIPF(v0);
        out[1] = PLAIN_CLIPF(v1);
        out[2] = PLAIN_CLIPF(v2);
        in += 4;
        out += 4;
    }
}

/* Convert.c: L to I, and I to L with saturation. */
static inline void
plain_l2i_row(UINT8 *restrict out_, const UINT8 *in, int n) {
    int x;
    for (x = 0; x < n; x++, out_ += 4) {
        INT32 v = in[x];
        memcpy(out_, &v, sizeof(v));
    }
}

static inline void
plain_i2l_row(UINT8 *restrict out, const UINT8 *in_, int n) {
    int x;
    for (x = 0; x < n; x++, in_ += 4) {
        INT32 v;
        memcpy(&v, in_, sizeof(v));
        out[x] = (UINT8)(v <= 0 ? 0 : (v >= 255 ? 255 : v));
    }
}

/* RankFilter.c: a sorting network over rows of PLAIN_RANK_CHUNK bytes, the
   wires at v + wire * PLAIN_RANK_CHUNK. Each comparator leaves the minimum
   of every pair of bytes on its first wire and the maximum on its second. */
#define PLAIN_RANK_CHUNK 256

static inline void
plain_rank_network(UINT8 *v, const UINT8 (*network)[2], int count) {
    int c, x;
    for (c = 0; c < count; c++) {
        UINT8 *restrict a = v + (size_t)network[c][0] * PLAIN_RANK_CHUNK;
        UINT8 *restrict b = v + (size_t)network[c][1] * PLAIN_RANK_CHUNK;
        for (x = 0; x < PLAIN_RANK_CHUNK; x++) {
            UINT8 lo = a[x] < b[x] ? a[x] : b[x];
            UINT8 hi = a[x] < b[x] ? b[x] : a[x];
            a[x] = lo;
            b[x] = hi;
        }
    }
}

#endif
