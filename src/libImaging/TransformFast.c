/*
 * The Imaging library
 *
 * Fast mode affine, perspective and quad transforms with the bilinear and
 * bicubic filters (design/APPROXIMATE_KERNELS_PLAN.md, item 4;
 * design/SIMD_FAST_KERNELS.md).
 *
 * The exact transforms compute each pixel's source position in doubles and
 * filter it in doubles, truncating the result; the compilers fuse their
 * multiply-adds differently on x86 and arm64. Here the positions are fixed
 * point, computed in exact integer arithmetic (perspective: integer
 * numerators and denominators, then one IEEE division and two products, in
 * 32-bit floats for images below TF_FLOAT_MAX on both sides),
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
/* perspective positions in floats for images below this on both sides */
#define TF_FLOAT_MAX 32768
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
   row too, so a fused multiply-add gives the same sums), converted to
   floats below TF_FLOAT_MAX: one IEEE division and one product, truncated
   to an integer, the same in vectors and in scalars; p is in 1/512 pixel,
   so (p + 1) >> 1 rounds it to 1/256. */
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

    if (m->method == IMAGING_TRANSFORM_PERSPECTIVE && im->xsize < TF_FLOAT_MAX &&
        im->ysize < TF_FLOAT_MAX) {
        /* The same in 32-bit floats from the exact sums, converted once:
           p below 2^24, where a float's step is at most 1/512 pixel, and
           four lanes per division. Conversion, division and product are
           each rounded once, alike everywhere. */
        const double nx0 = (double)(m->c[0] * u + m->c[1] * v + m->c[2]);
        const double ny0 = (double)(m->c[3] * u + m->c[4] * v + m->c[5]);
        const double d0 = (double)(m->c[6] * u + m->c[7] * v + m->c[8]);
        const double sx = (double)(2 * m->c[0]), sy = (double)(2 * m->c[3]);
        const double sd = (double)(2 * m->c[6]);
        const float xlim = 512.0f * im->xsize, ylim = 512.0f * im->ysize;
        i = 0;
#if defined(IMAGING_SIMD_AVX2)
        for (; i + 8 <= n; i += 8) {
            const __m256d k0 =
                _mm256_add_pd(_mm256_set1_pd(i), _mm256_setr_pd(0, 1, 2, 3));
            const __m256d k1 =
                _mm256_add_pd(_mm256_set1_pd(i), _mm256_setr_pd(4, 5, 6, 7));
#define TF_F32_AVX2(base, step)                        \
    _mm256_set_m128(                                   \
        _mm256_cvtpd_ps(TF_LINE_AVX2(base, step, k1)), \
        _mm256_cvtpd_ps(TF_LINE_AVX2(base, step, k0))  \
    )
            const __m256 r = _mm256_div_ps(_mm256_set1_ps(512.0f), TF_F32_AVX2(d0, sd));
            const __m256 px = _mm256_mul_ps(TF_F32_AVX2(nx0, sx), r);
            const __m256 py = _mm256_mul_ps(TF_F32_AVX2(ny0, sy), r);
#undef TF_F32_AVX2
            const __m256 zero = _mm256_setzero_ps();
            const int bits = _mm256_movemask_ps(_mm256_and_ps(
                _mm256_and_ps(
                    _mm256_cmp_ps(px, zero, _CMP_GE_OQ),
                    _mm256_cmp_ps(px, _mm256_set1_ps(xlim), _CMP_LT_OQ)
                ),
                _mm256_and_ps(
                    _mm256_cmp_ps(py, zero, _CMP_GE_OQ),
                    _mm256_cmp_ps(py, _mm256_set1_ps(ylim), _CMP_LT_OQ)
                )
            ));
            const __m256i one = _mm256_set1_epi32(1), half = _mm256_set1_epi32(128);
            int b;
            _mm256_storeu_si256(
                (__m256i *)(qx + i),
                _mm256_sub_epi32(
                    _mm256_srai_epi32(
                        _mm256_add_epi32(_mm256_cvttps_epi32(px), one), 1
                    ),
                    half
                )
            );
            _mm256_storeu_si256(
                (__m256i *)(qy + i),
                _mm256_sub_epi32(
                    _mm256_srai_epi32(
                        _mm256_add_epi32(_mm256_cvttps_epi32(py), one), 1
                    ),
                    half
                )
            );
            for (b = 0; b < 8; b++) {
                in[i + b] = (bits >> b) & 1;
            }
        }
#endif
#if defined(IMAGING_SIMD_SSE4)
        for (; i + 4 <= n; i += 4) {
            const __m128d k0 = _mm_add_pd(_mm_set1_pd(i), _mm_setr_pd(0, 1));
            const __m128d k1 = _mm_add_pd(_mm_set1_pd(i), _mm_setr_pd(2, 3));
#define TF_F32_SSE(base, step)                     \
    _mm_movelh_ps(                                 \
        _mm_cvtpd_ps(TF_LINE_SSE(base, step, k0)), \
        _mm_cvtpd_ps(TF_LINE_SSE(base, step, k1))  \
    )
            const __m128 r = _mm_div_ps(_mm_set1_ps(512.0f), TF_F32_SSE(d0, sd));
            const __m128 px = _mm_mul_ps(TF_F32_SSE(nx0, sx), r);
            const __m128 py = _mm_mul_ps(TF_F32_SSE(ny0, sy), r);
#undef TF_F32_SSE
            const int bits = _mm_movemask_ps(_mm_and_ps(
                _mm_and_ps(
                    _mm_cmpge_ps(px, _mm_setzero_ps()),
                    _mm_cmplt_ps(px, _mm_set1_ps(xlim))
                ),
                _mm_and_ps(
                    _mm_cmpge_ps(py, _mm_setzero_ps()),
                    _mm_cmplt_ps(py, _mm_set1_ps(ylim))
                )
            ));
            _mm_storeu_si128((__m128i *)(qx + i), TF_Q_SSE(_mm_cvttps_epi32(px)));
            _mm_storeu_si128((__m128i *)(qy + i), TF_Q_SSE(_mm_cvttps_epi32(py)));
            in[i] = bits & 1;
            in[i + 1] = (bits >> 1) & 1;
            in[i + 2] = (bits >> 2) & 1;
            in[i + 3] = (bits >> 3) & 1;
        }
#elif defined(IMAGING_SIMD_NEON)
        for (; i + 4 <= n; i += 4) {
            const float64x2_t k0 =
                vaddq_f64(vdupq_n_f64(i), vcombine_f64(vdup_n_f64(0), vdup_n_f64(1)));
            const float64x2_t k1 =
                vaddq_f64(vdupq_n_f64(i), vcombine_f64(vdup_n_f64(2), vdup_n_f64(3)));
#define TF_F32_NEON(base, step)                                                  \
    vcvt_high_f32_f64(                                                           \
        vcvt_f32_f64(TF_LINE_NEON(base, step, k0)), TF_LINE_NEON(base, step, k1) \
    )
            const float32x4_t r = vdivq_f32(vdupq_n_f32(512.0f), TF_F32_NEON(d0, sd));
            const float32x4_t px = vmulq_f32(TF_F32_NEON(nx0, sx), r);
            const float32x4_t py = vmulq_f32(TF_F32_NEON(ny0, sy), r);
#undef TF_F32_NEON
            const uint32x4_t ok = vandq_u32(
                vandq_u32(
                    vcgeq_f32(px, vdupq_n_f32(0)), vcltq_f32(px, vdupq_n_f32(xlim))
                ),
                vandq_u32(
                    vcgeq_f32(py, vdupq_n_f32(0)), vcltq_f32(py, vdupq_n_f32(ylim))
                )
            );
            const int32x4_t one = vdupq_n_s32(1), half = vdupq_n_s32(128);
            vst1q_s32(
                qx + i,
                vsubq_s32(vshrq_n_s32(vaddq_s32(vcvtq_s32_f32(px), one), 1), half)
            );
            vst1q_s32(
                qy + i,
                vsubq_s32(vshrq_n_s32(vaddq_s32(vcvtq_s32_f32(py), one), 1), half)
            );
            in[i] = (char)(vgetq_lane_u32(ok, 0) & 1);
            in[i + 1] = (char)(vgetq_lane_u32(ok, 1) & 1);
            in[i + 2] = (char)(vgetq_lane_u32(ok, 2) & 1);
            in[i + 3] = (char)(vgetq_lane_u32(ok, 3) & 1);
        }
#endif
        for (; i < n; i++) {
            const float r = 512.0f / (float)(d0 + i * sd);
            const float px = (float)(nx0 + i * sx) * r, py = (float)(ny0 + i * sy) * r;
            /* a zero denominator gives infinities or NaN: outside */
            in[i] = px >= 0 && px < xlim && py >= 0 && py < ylim;
            qx[i] = in[i] ? (((INT32)px + 1) >> 1) - 128 : 0;
            qy[i] = in[i] ? (((INT32)py + 1) >> 1) - 128 : 0;
        }
    } else if (m->method == IMAGING_TRANSFORM_PERSPECTIVE) {
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
            qx[i] = in[i] ? (((INT32)px + 1) >> 1) - 128 : 0;
            qy[i] = in[i] ? (((INT32)py + 1) >> 1) - 128 : 0;
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
        i = 0;
#if defined(IMAGING_SIMD_AVX2)
        {
            /* the low 32 bits of a logical and an arithmetic shift agree */
            const __m256i vdx = _mm256_set1_epi64x(4 * dx);
            const __m256i vdy = _mm256_set1_epi64x(4 * dy);
            const __m256i vxlim = _mm256_set1_epi64x(xlim);
            const __m256i vylim = _mm256_set1_epi64x(ylim);
            const __m256i voff = _mm256_set1_epi64x(off);
            const __m256i neg = _mm256_set1_epi64x(-1);
            const __m256i low = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
            __m256i vx = _mm256_setr_epi64x(x, x + dx, x + 2 * dx, x + 3 * dx);
            __m256i vy = _mm256_setr_epi64x(y, y + dy, y + 2 * dy, y + 3 * dy);
            for (; i + 4 <= n; i += 4) {
                const __m256i ok = _mm256_and_si256(
                    _mm256_and_si256(
                        _mm256_cmpgt_epi64(vx, neg), _mm256_cmpgt_epi64(vxlim, vx)
                    ),
                    _mm256_and_si256(
                        _mm256_cmpgt_epi64(vy, neg), _mm256_cmpgt_epi64(vylim, vy)
                    )
                );
                const int bits = _mm256_movemask_pd(_mm256_castsi256_pd(ok));
                _mm_storeu_si128(
                    (__m128i *)(qx + i),
                    _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
                        _mm256_srli_epi64(_mm256_add_epi64(vx, voff), 24), low
                    ))
                );
                _mm_storeu_si128(
                    (__m128i *)(qy + i),
                    _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
                        _mm256_srli_epi64(_mm256_add_epi64(vy, voff), 24), low
                    ))
                );
                in[i] = bits & 1;
                in[i + 1] = (bits >> 1) & 1;
                in[i + 2] = (bits >> 2) & 1;
                in[i + 3] = (bits >> 3) & 1;
                vx = _mm256_add_epi64(vx, vdx);
                vy = _mm256_add_epi64(vy, vdy);
            }
            x += i * dx;
            y += i * dy;
        }
#endif
        for (; i < n; i++) {
            in[i] = x >= 0 && x < xlim && y >= 0 && y < ylim;
            qx[i] = (INT32)((x + off) >> 24);
            qy[i] = (INT32)((y + off) >> 24);
            x += dx;
            y += dy;
        }
    }
    i = 0;
#if defined(IMAGING_SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        const __m256i ix =
            _mm256_srai_epi32(_mm256_loadu_si256((__m256i *)(qx + i)), 8);
        const __m256i iy =
            _mm256_srai_epi32(_mm256_loadu_si256((__m256i *)(qy + i)), 8);
        const __m256i inner = _mm256_and_si256(
            _mm256_and_si256(
                _mm256_cmpgt_epi32(ix, _mm256_set1_epi32(lo - 1)),
                _mm256_cmpgt_epi32(_mm256_set1_epi32(im->xsize - hi), ix)
            ),
            _mm256_and_si256(
                _mm256_cmpgt_epi32(iy, _mm256_set1_epi32(lo - 1)),
                _mm256_cmpgt_epi32(_mm256_set1_epi32(im->ysize - hi), iy)
            )
        );
        /* in is 0 or 1: kind = in + (in and inner) */
        const __m256i inv = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i *)(in + i)));
        const __m256i k32 = _mm256_add_epi32(inv, _mm256_and_si256(inv, inner));
        const __m128i k16 = _mm_packs_epi32(
            _mm256_castsi256_si128(k32), _mm256_extracti128_si256(k32, 1)
        );
        _mm_storel_epi64((__m128i *)(kind + i), _mm_packs_epi16(k16, k16));
    }
#endif
    for (; i < n; i++) {
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
   sample position minus half a pixel, in 1/256 pixel. Vertical first:
   v = p0 (256 - fy) + p1 fy fits 16 bits, and halved it fits a signed
   16-bit multiply-add for the horizontal step, which rounds once. */
static inline void
tf_bilinear_pixel(UINT8 *out, Imaging im, int ps, INT32 qx, INT32 qy) {
    const int ix = (int)(qx >> 8), iy = (int)(qy >> 8);
    const int fx = (int)(qx & 255), fy = (int)(qy & 255);
    const int x0 = tf_clip(ix, im->xsize - 1), x1 = tf_clip(ix + 1, im->xsize - 1);
    const UINT8 *r0 = (UINT8 *)im->image[tf_clip(iy, im->ysize - 1)];
    const UINT8 *r1 = (UINT8 *)im->image[tf_clip(iy + 1, im->ysize - 1)];
    int c;
    for (c = 0; c < ps; c++) {
        const unsigned v0 = r0[x0 * ps + c] * (256 - fy) + r1[x0 * ps + c] * fy;
        const unsigned v1 = r0[x1 * ps + c] * (256 - fy) + r1[x1 * ps + c] * fy;
        out[c] = (UINT8)(((v0 >> 1) * (256 - fx) + (v1 >> 1) * fx + 16384) >> 15);
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

/* An L pixel away from the edges: the references without the clamps. */
static inline UINT8
tf_bilinear_l(Imaging im, INT32 qx, INT32 qy) {
    const int fx = qx & 255, fy = qy & 255;
    const UINT8 *r0 = im->image8[qy >> 8] + (qx >> 8);
    const UINT8 *r1 = im->image8[(qy >> 8) + 1] + (qx >> 8);
    const unsigned v0 = r0[0] * (256 - fy) + r1[0] * fy;
    const unsigned v1 = r0[1] * (256 - fy) + r1[1] * fy;
    return (UINT8)(((v0 >> 1) * (256 - fx) + (v1 >> 1) * fx + 16384) >> 15);
}

static inline UINT8
tf_bicubic_l(Imaging im, INT32 qx, INT32 qy, INT16 (*w)[4]) {
    const int ix = (qx >> 8) - 1, iy = (qy >> 8) - 1;
    const INT16 *wx = w[qx & 255], *wy = w[qy & 255];
    int acc = 0, j;
    for (j = 0; j < 4; j++) {
        const UINT8 *row = im->image8[iy + j] + ix;
        const int h = wx[0] * row[0] + wx[1] * row[1] + wx[2] * row[2] + wx[3] * row[3];
        acc += wy[j] * ((h + 128) >> 8);
    }
    return (UINT8)tf_clip((acc + 32768) >> 16, 255);
}

#if defined(IMAGING_SIMD_SSE4)
/* The 8 bytes at (x, y) and (x + 1, y) of a pixel's row, its p0 and p1 as
   (p0, p1) byte pairs per channel. */
static inline __m128i
tf_pair_bytes(const UINT8 *row, int ix) {
    return _mm_shuffle_epi8(
        _mm_loadl_epi64((const __m128i *)(row + ix * 4)),
        _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1)
    );
}

/* Two pixels of 4 bytes away from the edges, bilinear: the reference's
   integers, with 16-bit multiplies for the vertical step. */
static inline __m128i
tf_bilinear2_sse(Imaging im, const INT32 *qx, const INT32 *qy) {
    __m128i res[2];
    int k;
    for (k = 0; k < 2; k++) {
        const int ix = qx[k] >> 8, iy = qy[k] >> 8;
        const int fx = qx[k] & 255, fy = qy[k] & 255;
        const __m128i t = _mm_cvtepu8_epi16(tf_pair_bytes((UINT8 *)im->image[iy], ix));
        const __m128i b =
            _mm_cvtepu8_epi16(tf_pair_bytes((UINT8 *)im->image[iy + 1], ix));
        const __m128i v = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(t, _mm_set1_epi16((short)(256 - fy))),
                _mm_mullo_epi16(b, _mm_set1_epi16((short)fy))
            ),
            1
        );
        res[k] = _mm_srli_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(v, _mm_set1_epi32((fx << 16) | (256 - fx))),
                _mm_set1_epi32(16384)
            ),
            15
        );
    }
    return _mm_packus_epi16(_mm_packus_epi32(res[0], res[1]), _mm_setzero_si128());
}

/* Eight L pixels away from the edges, bilinear: each pixel's (p0, p1) as
   one 16-bit word per row, then the reference's integers in 16-bit lanes. */
static inline __m128i
tf_bilinear8_l_sse(Imaging im, const INT32 *qx, const INT32 *qy) {
    const __m128i m = _mm_set1_epi32(255);
    const __m128i fx = _mm_packs_epi32(
        _mm_and_si128(_mm_loadu_si128((const __m128i *)qx), m),
        _mm_and_si128(_mm_loadu_si128((const __m128i *)(qx + 4)), m)
    );
    const __m128i fy = _mm_packs_epi32(
        _mm_and_si128(_mm_loadu_si128((const __m128i *)qy), m),
        _mm_and_si128(_mm_loadu_si128((const __m128i *)(qy + 4)), m)
    );
    const __m128i full = _mm_set1_epi16(256), low = _mm_set1_epi16(255);
    __m128i t = _mm_setzero_si128(), b = _mm_setzero_si128(), v0, v1, h0, h1;
    UINT16 pt[8], pb[8];
    int k;
    for (k = 0; k < 8; k++) {
        const UINT8 *r0 = im->image8[qy[k] >> 8] + (qx[k] >> 8);
        const UINT8 *r1 = im->image8[(qy[k] >> 8) + 1] + (qx[k] >> 8);
        memcpy(pt + k, r0, 2);
        memcpy(pb + k, r1, 2);
    }
    t = _mm_loadu_si128((const __m128i *)pt);
    b = _mm_loadu_si128((const __m128i *)pb);
    /* p0 is the low byte of each word, p1 the high one */
    v0 = _mm_add_epi16(
        _mm_mullo_epi16(_mm_and_si128(t, low), _mm_sub_epi16(full, fy)),
        _mm_mullo_epi16(_mm_and_si128(b, low), fy)
    );
    v1 = _mm_add_epi16(
        _mm_mullo_epi16(_mm_srli_epi16(t, 8), _mm_sub_epi16(full, fy)),
        _mm_mullo_epi16(_mm_srli_epi16(b, 8), fy)
    );
    v0 = _mm_srli_epi16(v0, 1);
    v1 = _mm_srli_epi16(v1, 1);
    h0 = _mm_madd_epi16(
        _mm_unpacklo_epi16(v0, v1), _mm_unpacklo_epi16(_mm_sub_epi16(full, fx), fx)
    );
    h1 = _mm_madd_epi16(
        _mm_unpackhi_epi16(v0, v1), _mm_unpackhi_epi16(_mm_sub_epi16(full, fx), fx)
    );
    h0 = _mm_srli_epi32(_mm_add_epi32(h0, _mm_set1_epi32(16384)), 15);
    h1 = _mm_srli_epi32(_mm_add_epi32(h1, _mm_set1_epi32(16384)), 15);
    return _mm_packus_epi16(_mm_packus_epi32(h0, h1), _mm_setzero_si128());
}

#if defined(IMAGING_SIMD_AVX2)
/* Four pixels of 4 bytes away from the edges, bilinear, the same integers:
   pixels a and b in the lanes of one register, c and d in another. */
static inline __m128i
tf_bilinear4_avx2(Imaging im, const INT32 *qx, const INT32 *qy) {
    const __m128i q = _mm_loadu_si128((const __m128i *)qx);
    const __m128i r = _mm_loadu_si128((const __m128i *)qy);
    const __m128i fx = _mm_and_si128(q, _mm_set1_epi32(255));
    const __m128i fy = _mm_and_si128(r, _mm_set1_epi32(255));
    /* words 256 - fy of a, b, c, d, then fy of a, b, c, d */
    const __m256i wy = _mm256_broadcastsi128_si256(
        _mm_packs_epi32(_mm_sub_epi32(_mm_set1_epi32(256), fy), fy)
    );
    /* (256 - fx, fx) word pairs of a, b, c, d */
    const __m256i wx = _mm256_castsi128_si256(
        _mm_or_si128(_mm_slli_epi32(fx, 16), _mm_sub_epi32(_mm_set1_epi32(256), fx))
    );
    __m256i out[2];
    int k;
    for (k = 0; k < 2; k++) {
        const int a = 2 * k, b = 2 * k + 1;
        const int ixa = qx[a] >> 8, iya = qy[a] >> 8;
        const int ixb = qx[b] >> 8, iyb = qy[b] >> 8;
        const __m256i t = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(
            tf_pair_bytes((UINT8 *)im->image[iya], ixa),
            tf_pair_bytes((UINT8 *)im->image[iyb], ixb)
        ));
        const __m256i bt = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(
            tf_pair_bytes((UINT8 *)im->image[iya + 1], ixa),
            tf_pair_bytes((UINT8 *)im->image[iyb + 1], ixb)
        ));
        /* each lane's pixel weight in all 8 words */
        const __m256i w0 = _mm256_shuffle_epi8(
            wy,
            _mm256_setr_epi8(
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3
            )
        );
        const __m256i w1 = _mm256_shuffle_epi8(
            _mm256_srli_si256(wy, 8),
            _mm256_setr_epi8(
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k,
                4 * k + 1,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3,
                4 * k + 2,
                4 * k + 3
            )
        );
        const __m256i v = _mm256_srli_epi16(
            _mm256_add_epi16(_mm256_mullo_epi16(t, w0), _mm256_mullo_epi16(bt, w1)), 1
        );
        const __m256i h =
            _mm256_permutevar8x32_epi32(wx, _mm256_setr_epi32(a, a, a, a, b, b, b, b));
        out[k] = _mm256_srli_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(v, h), _mm256_set1_epi32(16384)), 15
        );
    }
    {
        /* lanes [a, c] and [b, d] after the packs: dwords 0, 4, 1, 5 */
        const __m256i p = _mm256_packus_epi16(
            _mm256_packus_epi32(out[0], out[1]), _mm256_setzero_si256()
        );
        return _mm256_castsi256_si128(
            _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 0, 0, 0, 0))
        );
    }
}
#endif

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
/* One pixel of 4 bytes away from the edges, bilinear: the same integers,
   p0 (256 - fy) as (p0 << 8) - p0 fy so the weights fit 8 bits; vrshrn
   rounds as (v + 2^14) >> 15. */
static inline uint16x4_t
tf_bilinear1_neon(Imaging im, INT32 qx, INT32 qy) {
    const int ix = qx >> 8, iy = qy >> 8;
    const UINT16 fx = (UINT16)(qx & 255);
    const uint8x8_t fy = vdup_n_u8((UINT8)(qy & 255));
    /* [p0 p1] of the pixel's two rows */
    const uint8x8_t t = vld1_u8((UINT8 *)im->image[iy] + ix * 4);
    const uint8x8_t b = vld1_u8((UINT8 *)im->image[iy + 1] + ix * 4);
    const uint16x8_t v =
        vshrq_n_u16(vmlal_u8(vmlsl_u8(vshll_n_u8(t, 8), t, fy), b, fy), 1);
    return vrshrn_n_u32(
        vmlal_n_u16(vmull_n_u16(vget_low_u16(v), 256 - fx), vget_high_u16(v), fx), 15
    );
}

static inline uint8x8_t
tf_bilinear2_neon(Imaging im, const INT32 *qx, const INT32 *qy) {
    return vmovn_u16(vcombine_u16(
        tf_bilinear1_neon(im, qx[0], qy[0]), tf_bilinear1_neon(im, qx[1], qy[1])
    ));
}

/* Eight L pixels away from the edges, bilinear, the same integers. */
static inline uint8x8_t
tf_bilinear8_l_neon(Imaging im, const INT32 *qx, const INT32 *qy) {
    const uint32x4_t m = vdupq_n_u32(255);
    const uint16x8_t fx = vcombine_u16(
        vmovn_u32(vandq_u32(vreinterpretq_u32_s32(vld1q_s32(qx)), m)),
        vmovn_u32(vandq_u32(vreinterpretq_u32_s32(vld1q_s32(qx + 4)), m))
    );
    const uint16x8_t fy = vcombine_u16(
        vmovn_u32(vandq_u32(vreinterpretq_u32_s32(vld1q_s32(qy)), m)),
        vmovn_u32(vandq_u32(vreinterpretq_u32_s32(vld1q_s32(qy + 4)), m))
    );
    const uint16x8_t gx = vsubq_u16(vdupq_n_u16(256), fx);
    const uint16x8_t gy = vsubq_u16(vdupq_n_u16(256), fy);
    UINT16 pt[8], pb[8];
    uint16x8_t t, b, v0, v1;
    int k;
    for (k = 0; k < 8; k++) {
        const UINT8 *r0 = im->image8[qy[k] >> 8] + (qx[k] >> 8);
        const UINT8 *r1 = im->image8[(qy[k] >> 8) + 1] + (qx[k] >> 8);
        memcpy(pt + k, r0, 2);
        memcpy(pb + k, r1, 2);
    }
    t = vld1q_u16(pt);
    b = vld1q_u16(pb);
    /* p0 is the low byte of each word (little-endian), p1 the high one */
    v0 = vshrq_n_u16(
        vmlaq_u16(
            vmulq_u16(vandq_u16(t, vdupq_n_u16(255)), gy),
            vandq_u16(b, vdupq_n_u16(255)),
            fy
        ),
        1
    );
    v1 = vshrq_n_u16(
        vmlaq_u16(vmulq_u16(vshrq_n_u16(t, 8), gy), vshrq_n_u16(b, 8), fy), 1
    );
    return vmovn_u16(vcombine_u16(
        vrshrn_n_u32(
            vmlal_u16(
                vmull_u16(vget_low_u16(v0), vget_low_u16(gx)),
                vget_low_u16(v1),
                vget_low_u16(fx)
            ),
            15
        ),
        vrshrn_n_u32(
            vmlal_u16(
                vmull_u16(vget_high_u16(v0), vget_high_u16(gx)),
                vget_high_u16(v1),
                vget_high_u16(fx)
            ),
            15
        )
    ));
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
#if defined(IMAGING_SIMD_AVX2)
                    if (ps == 4 && !bicubic && i + 3 < n &&
                        !memcmp(kind + i, "\2\2\2\2", 4)) {
                        _mm_storeu_si128(
                            (__m128i *)o, tf_bilinear4_avx2(imIn, qx + i, qy + i)
                        );
                        i += 3;
                        continue;
                    }
#endif
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
#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
                    if (ps == 1 && !bicubic && i + 7 < n &&
                        !memcmp(kind + i, "\2\2\2\2\2\2\2\2", 8)) {
#if defined(IMAGING_SIMD_SSE4)
                        _mm_storel_epi64(
                            (__m128i *)o, tf_bilinear8_l_sse(imIn, qx + i, qy + i)
                        );
#else
                        vst1_u8(o, tf_bilinear8_l_neon(imIn, qx + i, qy + i));
#endif
                        i += 7;
                        continue;
                    }
#endif
                    if (ps == 1 && kind[i] == 2) {
                        *o = bicubic ? tf_bicubic_l(imIn, qx[i], qy[i], cubic)
                                     : tf_bilinear_l(imIn, qx[i], qy[i]);
                    } else if (bicubic) {
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
