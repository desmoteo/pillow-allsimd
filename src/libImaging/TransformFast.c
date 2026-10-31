/*
 * The Imaging library
 *
 * Fast mode affine, perspective and quad transforms with the bilinear and
 * bicubic filters (APPROXIMATE_KERNELS_PLAN.md, item 4; SIMD_FAST_KERNELS.md).
 *
 * The exact transforms compute each pixel's source position in doubles and
 * filter it in doubles, truncating the result; the compilers fuse their
 * multiply-adds differently on x86 and arm64. Here the positions are fixed
 * point, computed in exact integer arithmetic (perspective: integer
 * numerators and denominators, then one IEEE reciprocal and two products),
 * rounded to 1/256 pixel, and the filters are integer sums of products with
 * one rounding at the end (bicubic: one more between its passes), so every
 * build on every architecture gives the same pixels. The scalar functions
 * define the result; the SSE4 and NEON loops for pixels away from the image
 * edges compute the same integers.
 */

#include "Imaging.h"

#include <math.h>
#include <stdint.h>

#define TF_TILE 64
/* a position as a 32.32 fixed-point number of pixels */
#define TF_ONE ((int64_t)1 << 32)
#define TF_LIMIT ((double)(1 << 30))

typedef struct {
    int method;
    /* affine: x = c[0] (2i+1) + c[1] (2j+1) + c[2], y likewise c[3..5], in
       units of 2^-32 pixel (c[0], c[1] are a0, a1 times 2^31).
       quad: x = c[0] + c[1] (2i+1) + c[2] (2j+1) + c[3] (2i+1)(2j+1),
       y = c[4] + ... c[7].
       perspective: n_x = c[0] (2i+1) + c[1] (2j+1) + c[2], n_y = c[3..5],
       d = c[6] (2i+1) + c[7] (2j+1) + c[8], each times 2^37: x = n_x / d. */
    int64_t c[9];
} TFMap;

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

/* Perspective lanes: base + k step (exact integers), whether 0 <= p < lim
   (false for NaN), and (t + 1) >> 1 - 128 for the truncated p. */
#if defined(IMAGING_SIMD_SSE4)
#define TF_Q_SSE(t)                                                                   \
    _mm_sub_epi32(                                                                    \
        _mm_srai_epi32(_mm_add_epi32((t), _mm_set1_epi32(1)), 1), _mm_set1_epi32(128) \
    )
#define TF_LINE_SSE(base, step, k) \
    _mm_add_pd(_mm_set1_pd(base), _mm_mul_pd((k), _mm_set1_pd(step)))
#define TF_INSIDE_SSE(p, lim) \
    _mm_and_pd(_mm_cmpge_pd((p), _mm_setzero_pd()), _mm_cmplt_pd((p), _mm_set1_pd(lim)))
#endif
#if defined(IMAGING_SIMD_AVX2)
#define TF_LINE_AVX2(base, step, k) \
    _mm256_add_pd(_mm256_set1_pd(base), _mm256_mul_pd((k), _mm256_set1_pd(step)))
#define TF_INSIDE_AVX2(p, lim)                               \
    _mm256_and_pd(                                           \
        _mm256_cmp_pd((p), _mm256_setzero_pd(), _CMP_GE_OQ), \
        _mm256_cmp_pd((p), _mm256_set1_pd(lim), _CMP_LT_OQ)  \
    )
#endif
#if defined(IMAGING_SIMD_NEON)
#define TF_Q_NEON(t) \
    vsub_s32(vshr_n_s32(vadd_s32((t), vdup_n_s32(1)), 1), vdup_n_s32(128))
#define TF_LINE_NEON(base, step, k) \
    vaddq_f64(vdupq_n_f64(base), vmulq_n_f64((k), (step)))
#define TF_INSIDE_NEON(p, lim) \
    vandq_u64(vcgeq_f64((p), vdupq_n_f64(0)), vcltq_f64((p), vdupq_n_f64(lim)))
#endif

/* The source positions of output pixels i0 .. i0 + n - 1 of row j of the
   region, minus half a pixel and rounded to 1/256 pixel (qx, qy), and their
   kind: 0 outside the image, 1 near an edge (a neighbour is clamped), 2
   away from the edges.

   Affine and quad positions are exact integer sums in 2^-32 pixel.
   Perspective ones are p = n * (512 / d) for the integer numerators n and
   denominator d, which are exact in doubles (below 2^53, the steps along a
   row too, so a fused multiply-add gives the same sums): one IEEE division
   and one product, truncated to an integer, the same in vectors and in
   scalars; p is in 1/512 pixel, so (p + 1) >> 1 rounds it to 1/256. */
static void
tf_row(
    const TFMap *m,
    int i0,
    int n,
    int j,
    Imaging im,
    int bicubic,
    INT32 *qx,
    INT32 *qy,
    char *kind
) {
    const int lo = bicubic ? 1 : 0, hi = bicubic ? 2 : 1;
    const int64_t u = 2 * (int64_t)i0 + 1, v = 2 * (int64_t)j + 1;
    char in[TF_TILE];
    int i;

    if (m->method == IMAGING_TRANSFORM_PERSPECTIVE) {
        const double nx0 = (double)(m->c[0] * u + m->c[1] * v + m->c[2]);
        const double ny0 = (double)(m->c[3] * u + m->c[4] * v + m->c[5]);
        const double d0 = (double)(m->c[6] * u + m->c[7] * v + m->c[8]);
        const double sx = (double)(2 * m->c[0]), sy = (double)(2 * m->c[3]);
        const double sd = (double)(2 * m->c[6]);
        const double xlim = 512.0 * im->xsize, ylim = 512.0 * im->ysize;
        i = 0;
#if defined(IMAGING_SIMD_AVX2)
        for (; i + 4 <= n; i += 4) {
            const __m256d k =
                _mm256_add_pd(_mm256_set1_pd(i), _mm256_setr_pd(0, 1, 2, 3));
            const __m256d r =
                _mm256_div_pd(_mm256_set1_pd(512.0), TF_LINE_AVX2(d0, sd, k));
            const __m256d px = _mm256_mul_pd(TF_LINE_AVX2(nx0, sx, k), r);
            const __m256d py = _mm256_mul_pd(TF_LINE_AVX2(ny0, sy, k), r);
            const int bits = _mm256_movemask_pd(
                _mm256_and_pd(TF_INSIDE_AVX2(px, xlim), TF_INSIDE_AVX2(py, ylim))
            );
            _mm_storeu_si128((__m128i *)(qx + i), TF_Q_SSE(_mm256_cvttpd_epi32(px)));
            _mm_storeu_si128((__m128i *)(qy + i), TF_Q_SSE(_mm256_cvttpd_epi32(py)));
            in[i] = bits & 1;
            in[i + 1] = (bits >> 1) & 1;
            in[i + 2] = (bits >> 2) & 1;
            in[i + 3] = (bits >> 3) & 1;
        }
#elif defined(IMAGING_SIMD_SSE4)
        for (; i + 2 <= n; i += 2) {
            const __m128d k = _mm_add_pd(_mm_set1_pd(i), _mm_setr_pd(0, 1));
            const __m128d r = _mm_div_pd(_mm_set1_pd(512.0), TF_LINE_SSE(d0, sd, k));
            const __m128d px = _mm_mul_pd(TF_LINE_SSE(nx0, sx, k), r);
            const __m128d py = _mm_mul_pd(TF_LINE_SSE(ny0, sy, k), r);
            const int bits = _mm_movemask_pd(
                _mm_and_pd(TF_INSIDE_SSE(px, xlim), TF_INSIDE_SSE(py, ylim))
            );
            _mm_storel_epi64((__m128i *)(qx + i), TF_Q_SSE(_mm_cvttpd_epi32(px)));
            _mm_storel_epi64((__m128i *)(qy + i), TF_Q_SSE(_mm_cvttpd_epi32(py)));
            in[i] = bits & 1;
            in[i + 1] = (bits >> 1) & 1;
        }
#elif defined(IMAGING_SIMD_NEON)
        for (; i + 2 <= n; i += 2) {
            const float64x2_t k =
                vaddq_f64(vdupq_n_f64(i), vcombine_f64(vdup_n_f64(0), vdup_n_f64(1)));
            const float64x2_t r =
                vdivq_f64(vdupq_n_f64(512.0), TF_LINE_NEON(d0, sd, k));
            const float64x2_t px = vmulq_f64(TF_LINE_NEON(nx0, sx, k), r);
            const float64x2_t py = vmulq_f64(TF_LINE_NEON(ny0, sy, k), r);
            const uint64x2_t ok =
                vandq_u64(TF_INSIDE_NEON(px, xlim), TF_INSIDE_NEON(py, ylim));
            vst1_s32(qx + i, TF_Q_NEON(vmovn_s64(vcvtq_s64_f64(px))));
            vst1_s32(qy + i, TF_Q_NEON(vmovn_s64(vcvtq_s64_f64(py))));
            in[i] = (char)(vgetq_lane_u64(ok, 0) & 1);
            in[i + 1] = (char)(vgetq_lane_u64(ok, 1) & 1);
        }
#endif
        for (; i < n; i++) {
            const double r = 512.0 / (d0 + i * sd);
            const double px = (nx0 + i * sx) * r, py = (ny0 + i * sy) * r;
            /* a zero denominator gives infinities or NaN: outside */
            in[i] = px >= 0 && px < xlim && py >= 0 && py < ylim;
            if (in[i]) {
                qx[i] = (((INT32)px + 1) >> 1) - 128;
                qy[i] = (((INT32)py + 1) >> 1) - 128;
            }
        }
    } else {
        const int64_t xlim = (int64_t)im->xsize << 32, ylim = (int64_t)im->ysize << 32;
        const int64_t off = ((int64_t)1 << 23) - (TF_ONE >> 1);
        int64_t x, y, dx, dy;
        if (m->method == IMAGING_TRANSFORM_AFFINE) {
            x = m->c[0] * u + m->c[1] * v + m->c[2];
            y = m->c[3] * u + m->c[4] * v + m->c[5];
            dx = 2 * m->c[0];
            dy = 2 * m->c[3];
        } else {
            const int64_t cx = m->c[1] + m->c[3] * v, cy = m->c[5] + m->c[7] * v;
            x = m->c[0] + m->c[2] * v + cx * u;
            y = m->c[4] + m->c[6] * v + cy * u;
            dx = 2 * cx;
            dy = 2 * cy;
        }
        for (i = 0; i < n; i++) {
            in[i] = x >= 0 && x < xlim && y >= 0 && y < ylim;
            qx[i] = (INT32)((x + off) >> 24);
            qy[i] = (INT32)((y + off) >> 24);
            x += dx;
            y += dy;
        }
    }
    for (i = 0; i < n; i++) {
        const int ix = qx[i] >> 8, iy = qy[i] >> 8;
        kind[i] =
            !in[i] ? 0
            : ix - lo >= 0 && ix + hi < im->xsize && iy - lo >= 0 && iy + hi < im->ysize
                ? 2
                : 1;
    }
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

static inline int
tf_clip(int v, int hi) {
    return v < 0 ? 0 : (v > hi ? hi : v);
}

/* One pixel, any position inside the image: the reference. qx, qy are the
   sample position minus half a pixel, in 1/256 pixel. */
static inline void
tf_bilinear_pixel(UINT8 *out, Imaging im, int ps, INT32 qx, INT32 qy) {
    const int ix = (int)(qx >> 8), iy = (int)(qy >> 8);
    const int fx = (int)(qx & 255), fy = (int)(qy & 255);
    const int x0 = tf_clip(ix, im->xsize - 1), x1 = tf_clip(ix + 1, im->xsize - 1);
    const UINT8 *r0 = (UINT8 *)im->image[tf_clip(iy, im->ysize - 1)];
    const UINT8 *r1 = (UINT8 *)im->image[tf_clip(iy + 1, im->ysize - 1)];
    int c;
    for (c = 0; c < ps; c++) {
        const unsigned top = r0[x0 * ps + c] * (256 - fx) + r0[x1 * ps + c] * fx;
        const unsigned bot = r1[x0 * ps + c] * (256 - fx) + r1[x1 * ps + c] * fx;
        out[c] = (UINT8)((top * (256 - fy) + bot * fy + 32768) >> 16);
    }
}

static inline void
tf_bicubic_pixel(UINT8 *out, Imaging im, int ps, INT32 qx, INT32 qy, INT16 (*w)[4]) {
    const int ix = (int)(qx >> 8) - 1, iy = (int)(qy >> 8) - 1;
    const INT16 *wx = w[qx & 255], *wy = w[qy & 255];
    int xs[4], c, i, j;
    for (i = 0; i < 4; i++) {
        xs[i] = tf_clip(ix + i, im->xsize - 1) * ps;
    }
    for (c = 0; c < ps; c++) {
        int acc = 0;
        for (j = 0; j < 4; j++) {
            const UINT8 *row = (UINT8 *)im->image[tf_clip(iy + j, im->ysize - 1)];
            int h = 0;
            for (i = 0; i < 4; i++) {
                h += wx[i] * row[xs[i] + c];
            }
            acc += wy[j] * ((h + 128) >> 8);
        }
        out[c] = (UINT8)tf_clip((acc + 32768) >> 16, 255);
    }
}

#if defined(IMAGING_SIMD_SSE4)
/* Two pixels of 4 bytes away from the edges, bilinear. */
static inline __m128i
tf_bilinear2_sse(Imaging im, const INT32 *qx, const INT32 *qy) {
    /* per channel, (p0, p1) as word pairs; weights (256 - f, f) */
    const __m128i pairs =
        _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    __m128i top, bot, res[2];
    int k;
    for (k = 0; k < 2; k++) {
        const int ix = (int)(qx[k] >> 8), iy = (int)(qy[k] >> 8);
        const int fx = (int)(qx[k] & 255), fy = (int)(qy[k] & 255);
        const __m128i wx = _mm_set1_epi32((fx << 16) | (256 - fx));
        __m128i p = _mm_shuffle_epi8(
            _mm_unpacklo_epi64(
                _mm_loadl_epi64((const __m128i *)((UINT8 *)im->image[iy] + ix * 4)),
                _mm_loadl_epi64((const __m128i *)((UINT8 *)im->image[iy + 1] + ix * 4))
            ),
            pairs
        );
        top = _mm_madd_epi16(_mm_cvtepu8_epi16(p), wx);
        bot = _mm_madd_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(p, 8)), wx);
        res[k] = _mm_srli_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(top, _mm_set1_epi32(256 - fy)),
                    _mm_mullo_epi32(bot, _mm_set1_epi32(fy))
                ),
                _mm_set1_epi32(32768)
            ),
            16
        );
    }
    /* two pixels' 4 channels, in 0..255 */
    return _mm_packus_epi16(_mm_packus_epi32(res[0], res[1]), _mm_setzero_si128());
}

/* One pixel of 4 bytes away from the edges, bicubic. */
static inline __m128i
tf_bicubic1_sse(Imaging im, INT32 qx, INT32 qy, INT16 (*w)[4]) {
    /* per channel, pixel pairs (0, 1) and (2, 3) as words */
    const __m128i order =
        _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    const int ix = (int)(qx >> 8) - 1, iy = (int)(qy >> 8) - 1;
    const INT16 *wx = w[qx & 255], *wy = w[qy & 255];
    const __m128i w01 = _mm_set1_epi32(((UINT16)wx[1] << 16) | (UINT16)wx[0]);
    const __m128i w23 = _mm_set1_epi32(((UINT16)wx[3] << 16) | (UINT16)wx[2]);
    __m128i acc = _mm_setzero_si128();
    int j;
    for (j = 0; j < 4; j++) {
        __m128i p = _mm_shuffle_epi8(
            _mm_loadu_si128((const __m128i *)((UINT8 *)im->image[iy + j] + ix * 4)),
            order
        );
        __m128i h = _mm_add_epi32(
            _mm_madd_epi16(_mm_cvtepu8_epi16(p), w01),
            _mm_madd_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(p, 8)), w23)
        );
        h = _mm_srai_epi32(_mm_add_epi32(h, _mm_set1_epi32(128)), 8);
        acc = _mm_add_epi32(acc, _mm_mullo_epi32(h, _mm_set1_epi32(wy[j])));
    }
    acc = _mm_srai_epi32(_mm_add_epi32(acc, _mm_set1_epi32(32768)), 16);
    return _mm_packus_epi16(_mm_packus_epi32(acc, acc), _mm_setzero_si128());
}
#endif

#if defined(IMAGING_SIMD_NEON)
/* Two pixels of 4 bytes away from the edges, bilinear: the same integers
   (p0 (256 - f) as (p0 << 8) - p0 f, so the weights fit 8 bits). */
static inline uint8x8_t
tf_bilinear2_neon(Imaging im, const INT32 *qx, const INT32 *qy) {
    const int ixa = (int)(qx[0] >> 8), iya = (int)(qy[0] >> 8);
    const int ixb = (int)(qx[1] >> 8), iyb = (int)(qy[1] >> 8);
    const unsigned fxa = (unsigned)(qx[0] & 255), fxb = (unsigned)(qx[1] & 255);
    const UINT16 fya = (UINT16)(qy[0] & 255), fyb = (UINT16)(qy[1] & 255);
    /* [p00 p01] of each pixel, then [p00 of a, p00 of b] and so on */
    const uint32x2_t a0 =
        vreinterpret_u32_u8(vld1_u8((UINT8 *)im->image[iya] + ixa * 4));
    const uint32x2_t a1 =
        vreinterpret_u32_u8(vld1_u8((UINT8 *)im->image[iya + 1] + ixa * 4));
    const uint32x2_t b0 =
        vreinterpret_u32_u8(vld1_u8((UINT8 *)im->image[iyb] + ixb * 4));
    const uint32x2_t b1 =
        vreinterpret_u32_u8(vld1_u8((UINT8 *)im->image[iyb + 1] + ixb * 4));
    const uint8x8_t p00 = vreinterpret_u8_u32(vuzp1_u32(a0, b0));
    const uint8x8_t p01 = vreinterpret_u8_u32(vuzp2_u32(a0, b0));
    const uint8x8_t p10 = vreinterpret_u8_u32(vuzp1_u32(a1, b1));
    const uint8x8_t p11 = vreinterpret_u8_u32(vuzp2_u32(a1, b1));
    const uint8x8_t fx = vreinterpret_u8_u32(
        vset_lane_u32(fxb * 0x01010101u, vdup_n_u32(fxa * 0x01010101u), 1)
    );
    const uint16x8_t top = vmlal_u8(vmlsl_u8(vshll_n_u8(p00, 8), p00, fx), p01, fx);
    const uint16x8_t bot = vmlal_u8(vmlsl_u8(vshll_n_u8(p10, 8), p10, fx), p11, fx);
    const uint32x4_t va = vmlal_n_u16(
        vmull_n_u16(vget_low_u16(top), (UINT16)(256 - fya)), vget_low_u16(bot), fya
    );
    const uint32x4_t vb = vmlal_n_u16(
        vmull_n_u16(vget_high_u16(top), (UINT16)(256 - fyb)), vget_high_u16(bot), fyb
    );
    return vmovn_u16(vcombine_u16(vrshrn_n_u32(va, 16), vrshrn_n_u32(vb, 16)));
}

/* One pixel of 4 bytes away from the edges, bicubic: rounding shifts are the
   scalar (v + half) >> bits, the narrows clip. */
static inline UINT32
tf_bicubic1_neon(Imaging im, INT32 qx, INT32 qy, INT16 (*w)[4]) {
    const int ix = (qx >> 8) - 1, iy = (qy >> 8) - 1;
    const INT16 *wx = w[qx & 255], *wy = w[qy & 255];
    int32x4_t acc = vdupq_n_s32(0);
    int j;
    for (j = 0; j < 4; j++) {
        const uint8x16_t p = vld1q_u8((UINT8 *)im->image[iy + j] + ix * 4);
        const int16x8_t p01 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(p)));
        const int16x8_t p23 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(p)));
        int32x4_t h = vmull_n_s16(vget_low_s16(p01), wx[0]);
        h = vmlal_n_s16(h, vget_high_s16(p01), wx[1]);
        h = vmlal_n_s16(h, vget_low_s16(p23), wx[2]);
        h = vmlal_n_s16(h, vget_high_s16(p23), wx[3]);
        acc = vmlaq_n_s32(acc, vrshrq_n_s32(h, 8), wy[j]);
    }
    return vget_lane_u32(
        vreinterpret_u32_u8(
            vqmovn_u16(vcombine_u16(vqmovun_s32(vrshrq_n_s32(acc, 16)), vdup_n_u16(0)))
        ),
        0
    );
}
#endif

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
    int tx, ty;

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

    for (ty = y0; ty < y1; ty += TF_TILE) {
        const int ty1 = ty + TF_TILE < y1 ? ty + TF_TILE : y1;
        for (tx = x0; tx < x1; tx += TF_TILE) {
            const int n = tx + TF_TILE < x1 ? TF_TILE : x1 - tx;
            int y;
            for (y = ty; y < ty1; y++) {
                INT32 qx[TF_TILE], qy[TF_TILE];
                char kind[TF_TILE];
                UINT8 *out = (UINT8 *)imOut->image[y] + (size_t)tx * ps;
                int i;
                tf_row(&map, tx - x0, n, y - y0, imIn, bicubic, qx, qy, kind);
                for (i = 0; i < n; i++) {
                    UINT8 *o = out + (size_t)i * ps;
                    if (kind[i] == 0) {
                        if (fill) {
                            memset(o, 0, ps);
                        }
                        continue;
                    }
#if defined(IMAGING_SIMD_SSE4)
                    if (ps == 4 && kind[i] == 2) {
                        if (!bicubic && i + 1 < n && kind[i + 1] == 2) {
                            _mm_storel_epi64(
                                (__m128i *)o, tf_bilinear2_sse(imIn, qx + i, qy + i)
                            );
                            i++;
                            continue;
                        }
                        if (bicubic) {
                            INT32 v = _mm_cvtsi128_si32(
                                tf_bicubic1_sse(imIn, qx[i], qy[i], cubic)
                            );
                            memcpy(o, &v, 4);
                            continue;
                        }
                    }
#elif defined(IMAGING_SIMD_NEON)
                    if (ps == 4 && kind[i] == 2) {
                        if (!bicubic && i + 1 < n && kind[i + 1] == 2) {
                            vst1_u8(o, tf_bilinear2_neon(imIn, qx + i, qy + i));
                            i++;
                            continue;
                        }
                        if (bicubic) {
                            const UINT32 v =
                                tf_bicubic1_neon(imIn, qx[i], qy[i], cubic);
                            memcpy(o, &v, 4);
                            continue;
                        }
                    }
#endif
                    if (bicubic) {
                        tf_bicubic_pixel(o, imIn, ps, qx[i], qy[i], cubic);
                    } else {
                        tf_bilinear_pixel(o, imIn, ps, qx[i], qy[i]);
                    }
                }
            }
        }
    }

    MUTEX_UNLOCK(&imIn->mutex);
    ImagingSectionLeave(&cookie);
    return 1;
}
