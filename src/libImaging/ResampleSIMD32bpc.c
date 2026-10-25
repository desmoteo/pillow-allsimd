/* Resampling of I, F and I;16 images in vectors of doubles.

   The scalar code sums the taps of each output pixel in order, in double
   precision. Here each lane does the same for a pixel of its own, so the
   results are the same. In the vertical pass the lanes are neighbouring
   pixels of a row, which share their taps; in the horizontal pass they
   are the same pixel of neighbouring rows, read from a copy of the rows
   with their pixels interleaved. Compilers for arm64 fuse the scalar
   multiply-adds, so the NEON code does too. */

#if defined(IMAGING_SIMD_SSE4) || defined(IMAGING_SIMD_NEON)
#define RESAMPLE_WIDE

#if defined(IMAGING_SIMD_AVX2)
#define VD_LANES 4
typedef __m256d vd;
#define vd_zero() _mm256_setzero_pd()
#define vd_set1(v) _mm256_set1_pd(v)
#define vd_madd(acc, a, b) _mm256_add_pd(acc, _mm256_mul_pd(a, b))

static inline vd
vd_load_f32(const FLOAT32 *p) {
    return _mm256_cvtps_pd(_mm_loadu_ps(p));
}

static inline vd
vd_load_i32(const INT32 *p) {
    return _mm256_cvtepi32_pd(_mm_loadu_si128((const __m128i *)p));
}

static inline vd
vd_load_u16(const UINT8 *p, int bigendian) {
    __m128i v = _mm_loadl_epi64((const __m128i *)p);
    if (bigendian) {
        v = _mm_shuffle_epi8(
            v, _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1)
        );
    }
    return _mm256_cvtepi32_pd(_mm_cvtepu16_epi32(v));
}

static inline void
vd_store_f32(FLOAT32 *p, vd v) {
    _mm_storeu_ps(p, _mm256_cvtpd_ps(v));
}

/* ROUND_UP of each lane. */
static inline __m128i
vd_round(vd v) {
    vd half = _mm256_set1_pd(0.5);
    vd ge = _mm256_cmp_pd(v, _mm256_setzero_pd(), _CMP_GE_OQ);
    return _mm256_cvttpd_epi32(
        _mm256_blendv_pd(_mm256_sub_pd(v, half), _mm256_add_pd(v, half), ge)
    );
}

static inline void
vd_store_i32(INT32 *p, vd v) {
    _mm_storeu_si128((__m128i *)p, vd_round(v));
}

/* CLIP16(ROUND_UP()) of each lane, in the byte order given. */
static inline void
vd_store_u16(UINT8 *p, vd v, int bigendian) {
    __m128i i = vd_round(v);
    i = _mm_packus_epi32(i, i);
    if (bigendian) {
        i = _mm_shuffle_epi8(
            i, _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14)
        );
    }
    _mm_storel_epi64((__m128i *)p, i);
}

/* Lane r of v to pixel x of rows[r], as vd_store_f32 etc. would. */
static inline void
vd_scatter_f32(UINT8 *const *rows, int x, vd v) {
    __m128i f = _mm_castps_si128(_mm256_cvtpd_ps(v));
    INT32 w[4] = {
        _mm_cvtsi128_si32(f),
        _mm_extract_epi32(f, 1),
        _mm_extract_epi32(f, 2),
        _mm_extract_epi32(f, 3)
    };
    int r;
    for (r = 0; r < 4; r++) {
        memcpy((FLOAT32 *)rows[r] + x, &w[r], sizeof(w[r]));
    }
}

static inline void
vd_scatter_i32(UINT8 *const *rows, int x, vd v) {
    __m128i i = vd_round(v);
    ((INT32 *)rows[0])[x] = _mm_cvtsi128_si32(i);
    ((INT32 *)rows[1])[x] = _mm_extract_epi32(i, 1);
    ((INT32 *)rows[2])[x] = _mm_extract_epi32(i, 2);
    ((INT32 *)rows[3])[x] = _mm_extract_epi32(i, 3);
}

static inline void
vd_scatter_u16(UINT8 *const *rows, int x, vd v, int bigendian) {
    __m128i i = vd_round(v);
    int w[4], r;
    i = _mm_packus_epi32(i, i);
    w[0] = _mm_extract_epi16(i, 0), w[1] = _mm_extract_epi16(i, 1);
    w[2] = _mm_extract_epi16(i, 2), w[3] = _mm_extract_epi16(i, 3);
    for (r = 0; r < 4; r++) {
        rows[r][x * 2 + (bigendian ? 1 : 0)] = w[r] & 0xFF;
        rows[r][x * 2 + (bigendian ? 0 : 1)] = w[r] >> 8;
    }
}

#elif defined(IMAGING_SIMD_SSE4)
#define VD_LANES 2
typedef __m128d vd;
#define vd_zero() _mm_setzero_pd()
#define vd_set1(v) _mm_set1_pd(v)
#define vd_madd(acc, a, b) _mm_add_pd(acc, _mm_mul_pd(a, b))

static inline vd
vd_load_f32(const FLOAT32 *p) {
    return _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)p)));
}

static inline vd
vd_load_i32(const INT32 *p) {
    return _mm_cvtepi32_pd(_mm_loadl_epi64((const __m128i *)p));
}

static inline vd
vd_load_u16(const UINT8 *p, int bigendian) {
    __m128i v = _mm_cvtsi32_si128(mm_load_si32(p));
    if (bigendian) {
        v = _mm_shuffle_epi8(
            v, _mm_setr_epi8(1, 0, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
        );
    }
    return _mm_cvtepi32_pd(_mm_cvtepu16_epi32(v));
}

static inline void
vd_store_f32(FLOAT32 *p, vd v) {
    _mm_storel_epi64((__m128i *)p, _mm_castps_si128(_mm_cvtpd_ps(v)));
}

/* ROUND_UP of each lane, in the low two lanes. */
static inline __m128i
vd_round(vd v) {
    vd half = _mm_set1_pd(0.5);
    vd ge = _mm_cmpge_pd(v, _mm_setzero_pd());
    return _mm_cvttpd_epi32(
        _mm_blendv_pd(_mm_sub_pd(v, half), _mm_add_pd(v, half), ge)
    );
}

static inline void
vd_store_i32(INT32 *p, vd v) {
    _mm_storel_epi64((__m128i *)p, vd_round(v));
}

static inline void
vd_store_u16(UINT8 *p, vd v, int bigendian) {
    __m128i i = vd_round(v);
    INT32 w;
    i = _mm_packus_epi32(i, i);
    if (bigendian) {
        i = _mm_shuffle_epi8(
            i, _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14)
        );
    }
    w = _mm_cvtsi128_si32(i);
    memcpy(p, &w, sizeof(w));
}

/* Lane r of v to pixel x of rows[r], as vd_store_f32 etc. would. */
static inline void
vd_scatter_f32(UINT8 *const *rows, int x, vd v) {
    __m128i f = _mm_castps_si128(_mm_cvtpd_ps(v));
    INT32 w[2] = {_mm_cvtsi128_si32(f), _mm_extract_epi32(f, 1)};
    memcpy((FLOAT32 *)rows[0] + x, &w[0], sizeof(w[0]));
    memcpy((FLOAT32 *)rows[1] + x, &w[1], sizeof(w[1]));
}

static inline void
vd_scatter_i32(UINT8 *const *rows, int x, vd v) {
    __m128i i = vd_round(v);
    ((INT32 *)rows[0])[x] = _mm_cvtsi128_si32(i);
    ((INT32 *)rows[1])[x] = _mm_extract_epi32(i, 1);
}

static inline void
vd_scatter_u16(UINT8 *const *rows, int x, vd v, int bigendian) {
    __m128i i = vd_round(v);
    int w[2], r;
    i = _mm_packus_epi32(i, i);
    w[0] = _mm_extract_epi16(i, 0), w[1] = _mm_extract_epi16(i, 1);
    for (r = 0; r < 2; r++) {
        rows[r][x * 2 + (bigendian ? 1 : 0)] = w[r] & 0xFF;
        rows[r][x * 2 + (bigendian ? 0 : 1)] = w[r] >> 8;
    }
}

#else /* NEON */
#define VD_LANES 2
typedef float64x2_t vd;
#define vd_zero() vdupq_n_f64(0.0)
#define vd_set1(v) vdupq_n_f64(v)
#define vd_madd(acc, a, b) vfmaq_f64(acc, a, b)

static inline vd
vd_load_f32(const FLOAT32 *p) {
    return vcvt_f64_f32(vld1_f32(p));
}

static inline vd
vd_load_i32(const INT32 *p) {
    return vcvtq_f64_s64(vmovl_s32(vld1_s32(p)));
}

static inline vd
vd_load_u16(const UINT8 *p, int bigendian) {
    uint32_t w;
    uint16x4_t h;
    memcpy(&w, p, sizeof(w));
    h = vreinterpret_u16_u32(vdup_n_u32(w));
    if (bigendian) {
        h = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(h)));
    }
    return vcvtq_f64_u64(vmovl_u32(vget_low_u32(vmovl_u16(h))));
}

static inline void
vd_store_f32(FLOAT32 *p, vd v) {
    vst1_f32(p, vcvt_f32_f64(v));
}

/* ROUND_UP of each lane: the conversions saturate, as the scalar one. */
static inline int32x2_t
vd_round(vd v) {
    vd half = vdupq_n_f64(0.5);
    uint64x2_t ge = vcgeq_f64(v, vdupq_n_f64(0.0));
    return vqmovn_s64(
        vcvtq_s64_f64(vbslq_f64(ge, vaddq_f64(v, half), vsubq_f64(v, half)))
    );
}

static inline void
vd_store_i32(INT32 *p, vd v) {
    vst1_s32(p, vd_round(v));
}

static inline void
vd_store_u16(UINT8 *p, vd v, int bigendian) {
    int32x2_t i = vd_round(v);
    uint16x4_t h = vqmovun_s32(vcombine_s32(i, i));
    uint32_t w;
    if (bigendian) {
        h = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(h)));
    }
    w = vget_lane_u32(vreinterpret_u32_u16(h), 0);
    memcpy(p, &w, sizeof(w));
}
/* Lane r of v to pixel x of rows[r], as vd_store_f32 etc. would. */
static inline void
vd_scatter_f32(UINT8 *const *rows, int x, vd v) {
    float32x2_t f = vcvt_f32_f64(v);
    ((FLOAT32 *)rows[0])[x] = vget_lane_f32(f, 0);
    ((FLOAT32 *)rows[1])[x] = vget_lane_f32(f, 1);
}

static inline void
vd_scatter_i32(UINT8 *const *rows, int x, vd v) {
    int32x2_t i = vd_round(v);
    ((INT32 *)rows[0])[x] = vget_lane_s32(i, 0);
    ((INT32 *)rows[1])[x] = vget_lane_s32(i, 1);
}

static inline void
vd_scatter_u16(UINT8 *const *rows, int x, vd v, int bigendian) {
    int32x2_t i = vd_round(v);
    uint16x4_t h = vqmovun_s32(vcombine_s32(i, i));
    int w[2] = {vget_lane_u16(h, 0), vget_lane_u16(h, 1)}, r;
    for (r = 0; r < 2; r++) {
        rows[r][x * 2 + (bigendian ? 1 : 0)] = w[r] & 0xFF;
        rows[r][x * 2 + (bigendian ? 0 : 1)] = w[r] >> 8;
    }
}
#endif

#if defined(IMAGING_SIMD_SSE4)
/* Pixels x to x + 3 of row as 32-bit lanes: their bits (F, I) or values
   (I;16). */
#define F32_ROW4(row, x) _mm_loadu_si128((const __m128i *)((row) + (x) * 4))
#define I32_ROW4(row, x) F32_ROW4(row, x)
#define U16LE_ROW4(row, x) \
    _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)((row) + (x) * 2)))
#define U16BE_ROW4(row, x)                                                    \
    _mm_cvtepu16_epi32(_mm_shuffle_epi8(                                      \
        _mm_loadl_epi64((const __m128i *)((row) + (x) * 2)),                  \
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1) \
    ))

/* Lane i of a, b, c, d to dst[i * 4 + 0..3]. */
static inline void
interleave4(void *dst, __m128i a, __m128i b, __m128i c, __m128i d) {
    __m128i ab01 = _mm_unpacklo_epi32(a, b), cd01 = _mm_unpacklo_epi32(c, d);
    __m128i ab23 = _mm_unpackhi_epi32(a, b), cd23 = _mm_unpackhi_epi32(c, d);
    __m128i *p = (__m128i *)dst;
    _mm_storeu_si128(p + 0, _mm_unpacklo_epi64(ab01, cd01));
    _mm_storeu_si128(p + 1, _mm_unpackhi_epi64(ab01, cd01));
    _mm_storeu_si128(p + 2, _mm_unpacklo_epi64(ab23, cd23));
    _mm_storeu_si128(p + 3, _mm_unpackhi_epi64(ab23, cd23));
}
#else
#define F32_ROW4(row, x) vld1q_u32((const uint32_t *)(row) + (x))
#define I32_ROW4(row, x) F32_ROW4(row, x)
#define U16LE_ROW4(row, x) vmovl_u16(vld1_u16((const uint16_t *)((row) + (x) * 2)))
#define U16BE_ROW4(row, x) \
    vmovl_u16(vreinterpret_u16_u8(vrev16_u8(vld1_u8((row) + (x) * 2))))

static inline void
interleave4(void *dst, uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d) {
    uint32x4x4_t v = {{a, b, c, d}};
    vst4q_u32((uint32_t *)dst, v);
}
#endif

/* Pixel access by kind of image: F, I and I;16 little or big endian. A
   scalar pixel as a double, a vector of VD_LANES pixels, a result back. */
#define F32_GET(row, x) ((double)((FLOAT32 *)(row))[x])
#define F32_LOAD(row, x) vd_load_f32((FLOAT32 *)(row) + (x))
#define F32_PUT(row, x, ss) (((FLOAT32 *)(row))[x] = (ss))
#define F32_STORE(row, x, v) vd_store_f32((FLOAT32 *)(row) + (x), v)
#define F32_SCATTER(rows, x, v) vd_scatter_f32(rows, x, v)

#define I32_GET(row, x) ((double)((INT32 *)(row))[x])
#define I32_LOAD(row, x) vd_load_i32((INT32 *)(row) + (x))
#define I32_PUT(row, x, ss) (((INT32 *)(row))[x] = ROUND_UP(ss))
#define I32_STORE(row, x, v) vd_store_i32((INT32 *)(row) + (x), v)
#define I32_SCATTER(rows, x, v) vd_scatter_i32(rows, x, v)

#define U16_GET(row, x, be) \
    ((row)[(x) * 2 + ((be) ? 1 : 0)] + ((row)[(x) * 2 + ((be) ? 0 : 1)] << 8))
#define U16_PUT(row, x, ss, be)                          \
    do {                                                 \
        int ss_int = CLIP16(ROUND_UP(ss));               \
        (row)[(x) * 2 + ((be) ? 1 : 0)] = ss_int & 0xFF; \
        (row)[(x) * 2 + ((be) ? 0 : 1)] = ss_int >> 8;   \
    } while (0)

#define U16LE_GET(row, x) ((double)U16_GET(row, x, 0))
#define U16LE_LOAD(row, x) vd_load_u16((row) + (x) * 2, 0)
#define U16LE_PUT(row, x, ss) U16_PUT(row, x, ss, 0)
#define U16LE_STORE(row, x, v) vd_store_u16((row) + (x) * 2, v, 0)
#define U16LE_SCATTER(rows, x, v) vd_scatter_u16(rows, x, v, 0)

#define U16BE_GET(row, x) ((double)U16_GET(row, x, 1))
#define U16BE_LOAD(row, x) vd_load_u16((row) + (x) * 2, 1)
#define U16BE_PUT(row, x, ss) U16_PUT(row, x, ss, 1)
#define U16BE_STORE(row, x, v) vd_store_u16((row) + (x) * 2, v, 1)
#define U16BE_SCATTER(rows, x, v) vd_scatter_u16(rows, x, v, 1)

/* The vertical pass: two vectors of a row's pixels at a time. */
#define RESAMPLE_VERTICAL_WIDE(name, KIND)                                        \
    static void name(                                                             \
        Imaging imOut, Imaging imIn, int ksize, int *bounds, double *kk           \
    ) {                                                                           \
        int xsize = imOut->xsize, ysize = imOut->ysize;                           \
        int xx, yy, y;                                                            \
        for (yy = 0; yy < ysize; yy++) {                                          \
            UINT8 *lineOut = (UINT8 *)imOut->image[yy];                           \
            UINT8 **lines = (UINT8 **)imIn->image + bounds[yy * 2 + 0];           \
            int ymax = bounds[yy * 2 + 1];                                        \
            double *k = &kk[yy * ksize];                                          \
            for (xx = 0; xx + 2 * VD_LANES <= xsize; xx += 2 * VD_LANES) {        \
                vd ss0 = vd_zero(), ss1 = vd_zero();                              \
                for (y = 0; y < ymax; y++) {                                      \
                    vd ky = vd_set1(k[y]);                                        \
                    ss0 = vd_madd(ss0, KIND##_LOAD(lines[y], xx), ky);            \
                    ss1 = vd_madd(ss1, KIND##_LOAD(lines[y], xx + VD_LANES), ky); \
                }                                                                 \
                KIND##_STORE(lineOut, xx, ss0);                                   \
                KIND##_STORE(lineOut, xx + VD_LANES, ss1);                        \
            }                                                                     \
            for (; xx < xsize; xx++) {                                            \
                double ss = 0.0;                                                  \
                for (y = 0; y < ymax; y++) {                                      \
                    ss += KIND##_GET(lines[y], xx) * k[y];                        \
                }                                                                 \
                KIND##_PUT(lineOut, xx, ss);                                      \
            }                                                                     \
        }                                                                         \
    }

/* The horizontal pass: four rows at a time, from a copy of them with
   their pixels interleaved, as ELEM; the rows left over one at a time.
   Returns 0 if out of memory. */
#define RESAMPLE_HORIZONTAL_WIDE(name, KIND, ELEM, elem_load)                       \
    static int name(                                                                \
        Imaging imOut, Imaging imIn, int offset, int ksize, int *bounds, double *kk \
    ) {                                                                             \
        int xsizeIn = imIn->xsize, xsize = imOut->xsize, ysize = imOut->ysize;      \
        int xx, yy, x, r;                                                           \
        ELEM *rows = malloc(sizeof(ELEM) * 4 * (xsizeIn > 0 ? xsizeIn : 1));        \
        if (!rows) {                                                                \
            return 0;                                                               \
        }                                                                           \
        for (yy = 0; yy + 4 <= ysize; yy += 4) {                                    \
            UINT8 *lineIn[4], *lineOut[4];                                          \
            for (r = 0; r < 4; r++) {                                               \
                lineIn[r] = (UINT8 *)imIn->image[yy + r + offset];                  \
                lineOut[r] = (UINT8 *)imOut->image[yy + r];                         \
            }                                                                       \
            for (x = 0; x + 4 <= xsizeIn; x += 4) {                                 \
                interleave4(                                                        \
                    rows + x * 4,                                                   \
                    KIND##_ROW4(lineIn[0], x),                                      \
                    KIND##_ROW4(lineIn[1], x),                                      \
                    KIND##_ROW4(lineIn[2], x),                                      \
                    KIND##_ROW4(lineIn[3], x)                                       \
                );                                                                  \
            }                                                                       \
            for (; x < xsizeIn; x++) {                                              \
                for (r = 0; r < 4; r++) {                                           \
                    rows[x * 4 + r] = (ELEM)KIND##_GET(lineIn[r], x);               \
                }                                                                   \
            }                                                                       \
            /* Two columns at a time, for two independent sums: their               \
               common taps together, then the rest of each. */                      \
            for (xx = 0; xx < xsize; xx += 2) {                                     \
                int last = xx + 1 == xsize;                                         \
                int xmax0 = bounds[xx * 2 + 1];                                     \
                int xmax1 = last ? 0 : bounds[xx * 2 + 3];                          \
                int n = xmax0 < xmax1 ? xmax0 : xmax1;                              \
                const ELEM *p0 = rows + bounds[xx * 2 + 0] * 4;                     \
                const ELEM *p1 = last ? p0 : rows + bounds[xx * 2 + 2] * 4;         \
                double *k0 = &kk[xx * ksize], *k1 = k0 + ksize;                     \
                vd ss0[4 / VD_LANES], ss1[4 / VD_LANES];                            \
                for (r = 0; r < 4 / VD_LANES; r++) {                                \
                    ss0[r] = ss1[r] = vd_zero();                                    \
                }                                                                   \
                for (x = 0; x < n; x++) {                                           \
                    vd kx0 = vd_set1(k0[x]), kx1 = vd_set1(k1[x]);                  \
                    for (r = 0; r < 4 / VD_LANES; r++) {                            \
                        ss0[r] = vd_madd(                                           \
                            ss0[r], elem_load(p0 + x * 4 + r * VD_LANES), kx0       \
                        );                                                          \
                        ss1[r] = vd_madd(                                           \
                            ss1[r], elem_load(p1 + x * 4 + r * VD_LANES), kx1       \
                        );                                                          \
                    }                                                               \
                }                                                                   \
                for (x = n; x < xmax0; x++) {                                       \
                    vd kx0 = vd_set1(k0[x]);                                        \
                    for (r = 0; r < 4 / VD_LANES; r++) {                            \
                        ss0[r] = vd_madd(                                           \
                            ss0[r], elem_load(p0 + x * 4 + r * VD_LANES), kx0       \
                        );                                                          \
                    }                                                               \
                }                                                                   \
                for (x = n; x < xmax1; x++) {                                       \
                    vd kx1 = vd_set1(k1[x]);                                        \
                    for (r = 0; r < 4 / VD_LANES; r++) {                            \
                        ss1[r] = vd_madd(                                           \
                            ss1[r], elem_load(p1 + x * 4 + r * VD_LANES), kx1       \
                        );                                                          \
                    }                                                               \
                }                                                                   \
                for (r = 0; r < 4 / VD_LANES; r++) {                                \
                    KIND##_SCATTER(lineOut + r * VD_LANES, xx, ss0[r]);             \
                    if (!last) {                                                    \
                        KIND##_SCATTER(lineOut + r * VD_LANES, xx + 1, ss1[r]);     \
                    }                                                               \
                }                                                                   \
            }                                                                       \
        }                                                                           \
        for (; yy < ysize; yy++) {                                                  \
            UINT8 *lineIn = (UINT8 *)imIn->image[yy + offset];                      \
            UINT8 *lineOut = (UINT8 *)imOut->image[yy];                             \
            for (xx = 0; xx < xsize; xx++) {                                        \
                int xmin = bounds[xx * 2 + 0], xmax = bounds[xx * 2 + 1];           \
                double *k = &kk[xx * ksize], ss = 0.0;                              \
                for (x = 0; x < xmax; x++) {                                        \
                    ss += KIND##_GET(lineIn, x + xmin) * k[x];                      \
                }                                                                   \
                KIND##_PUT(lineOut, xx, ss);                                        \
            }                                                                       \
        }                                                                           \
        free(rows);                                                                 \
        return 1;                                                                   \
    }

RESAMPLE_VERTICAL_WIDE(resample_vertical_f32, F32)
RESAMPLE_VERTICAL_WIDE(resample_vertical_i32, I32)
RESAMPLE_VERTICAL_WIDE(resample_vertical_u16le, U16LE)
RESAMPLE_VERTICAL_WIDE(resample_vertical_u16be, U16BE)

/* I;16 values fit 32-bit integers, and F values floats, exactly. */
RESAMPLE_HORIZONTAL_WIDE(resample_horizontal_f32, F32, FLOAT32, vd_load_f32)
RESAMPLE_HORIZONTAL_WIDE(resample_horizontal_i32, I32, INT32, vd_load_i32)
RESAMPLE_HORIZONTAL_WIDE(resample_horizontal_u16le, U16LE, INT32, vd_load_i32)
RESAMPLE_HORIZONTAL_WIDE(resample_horizontal_u16be, U16BE, INT32, vd_load_i32)

#endif
