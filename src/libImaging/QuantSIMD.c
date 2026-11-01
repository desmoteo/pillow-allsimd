/* The quantizer's search for the palette entry nearest to a pixel
   (Quant.c), compiled into SIMDKernels.c. */

#include "QuantNearest.h"
#include "QuantTypes.h"

/* The palette entry nearest to each pixel. The search below starts from
   entry 0 and scans entries in the order of their distance from it, as far
   as one can be nearer; it keeps the first strictly nearer one. So the
   result is the nearest entry, ties going to entry 0, then to the entry
   first in that order. The vector search finds the same entry directly:
   the smallest of dist * 512 + rank over all entries, rank being 0 for
   entry 0 and 1 + the position in the order for the others. Distances are
   at most 3 * 255^2, so these keys fit 27 bits. */
static uint32_t
nearest_entry_simd(const NearestPalette *np, uint32_t v) {
    uint32_t j, key;
    Pixel p;
    p.v = v;
#if defined(IMAGING_SIMD_AVX2)
    __m256i r = _mm256_set1_epi32(p.c.r), g = _mm256_set1_epi32(p.c.g);
    __m256i b = _mm256_set1_epi32(p.c.b), best = _mm256_set1_epi32(-1);
    for (j = 0; j < np->n; j += 8) {
        __m256i dr =
            _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)&np->r[j]), r);
        __m256i dg =
            _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)&np->g[j]), g);
        __m256i db =
            _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)&np->b[j]), b);
        __m256i d = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(dr, dr), _mm256_mullo_epi32(dg, dg)),
            _mm256_mullo_epi32(db, db)
        );
        __m256i k = _mm256_or_si256(
            _mm256_slli_epi32(d, 9), _mm256_loadu_si256((const __m256i *)&np->rank[j])
        );
        best = _mm256_min_epu32(best, k);
    }
    __m128i m =
        _mm_min_epu32(_mm256_castsi256_si128(best), _mm256_extracti128_si256(best, 1));
#elif defined(IMAGING_SIMD_SSE4)
    __m128i r = _mm_set1_epi32(p.c.r), g = _mm_set1_epi32(p.c.g);
    __m128i b = _mm_set1_epi32(p.c.b), m = _mm_set1_epi32(-1);
    for (j = 0; j < np->n; j += 4) {
        __m128i dr = _mm_sub_epi32(_mm_loadu_si128((const __m128i *)&np->r[j]), r);
        __m128i dg = _mm_sub_epi32(_mm_loadu_si128((const __m128i *)&np->g[j]), g);
        __m128i db = _mm_sub_epi32(_mm_loadu_si128((const __m128i *)&np->b[j]), b);
        __m128i d = _mm_add_epi32(
            _mm_add_epi32(_mm_mullo_epi32(dr, dr), _mm_mullo_epi32(dg, dg)),
            _mm_mullo_epi32(db, db)
        );
        m = _mm_min_epu32(
            m,
            _mm_or_si128(
                _mm_slli_epi32(d, 9), _mm_loadu_si128((const __m128i *)&np->rank[j])
            )
        );
    }
#endif
#if defined(IMAGING_SIMD_SSE4)
    m = _mm_min_epu32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(1, 0, 3, 2)));
    m = _mm_min_epu32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(2, 3, 0, 1)));
    key = (uint32_t)_mm_cvtsi128_si32(m);
#elif defined(IMAGING_SIMD_NEON)
    int32x4_t r = vdupq_n_s32(p.c.r), g = vdupq_n_s32(p.c.g), b = vdupq_n_s32(p.c.b);
    uint32x4_t m = vdupq_n_u32(0xffffffff);
    for (j = 0; j < np->n; j += 4) {
        int32x4_t dr = vsubq_s32(vld1q_s32(&np->r[j]), r);
        int32x4_t dg = vsubq_s32(vld1q_s32(&np->g[j]), g);
        int32x4_t db = vsubq_s32(vld1q_s32(&np->b[j]), b);
        int32x4_t d = vmlaq_s32(vmlaq_s32(vmulq_s32(dr, dr), dg, dg), db, db);
        m = vminq_u32(
            m,
            vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(d), 9), vld1q_u32(&np->rank[j]))
        );
    }
    key = vminvq_u32(m);
#endif
    return np->entry[key & 511];
}

/* The fast octree quantizer's two passes over the pixels (QuantOctree.c):
   the loops of PlainLoops.h as this tier's compiler makes them. */
static int
octree_count_simd(const void *cube, const UINT8 *in, int n, unsigned int alpha_or) {
    plain_octree_count((const ImagingOctreeIndex *)cube, in, n, alpha_or);
    return n;
}

/* An RGB pixel into its bucket's four sums, r g b and the count, with one
   addition of four 32-bit lanes: the pixel's bytes widened, the fourth
   replaced by 1. The bucket's number as plain_octree_count_rgb() has it. */
static int
octree_count_rgb_simd(const UINT8 *in, int n, UINT32 *sums) {
    int x;
#if defined(IMAGING_SIMD_SSE4)
    const __m128i rgb = _mm_set_epi32(0, -1, -1, -1), one = _mm_set_epi32(1, 0, 0, 0);
#endif
    for (x = 0; x < n; x++, in += 4) {
        UINT32 *s = sums + 4 * PLAIN_OCTREE_RGB_BUCKET(in);
#if defined(IMAGING_SIMD_SSE4)
        __m128i v = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(mm_load_si32(in)));
        v = _mm_or_si128(_mm_and_si128(v, rgb), one);
        _mm_storeu_si128(
            (__m128i *)s, _mm_add_epi32(_mm_loadu_si128((const __m128i *)s), v)
        );
#elif defined(IMAGING_SIMD_NEON)
        uint32_t p;
        uint32x4_t v;
        memcpy(&p, in, 4);
        v = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(p))));
        vst1q_u32(s, vaddq_u32(vld1q_u32(s), vsetq_lane_u32(1, v, 3)));
#endif
    }
    return n;
}

static int
octree_map_rgb_simd(const UINT32 *lut, const UINT8 *in, int n, UINT8 *out) {
    plain_octree_map_rgb(lut, in, n, out);
    return n;
}

static int
octree_map_simd(const void *cube, const UINT8 *in, int n, UINT8 *out) {
    plain_octree_map((const ImagingOctreeIndex *)cube, in, n, out);
    return n;
}

static const ImagingSIMDQuantKernels quant_simd_kernels = {
    .nearest_entry = nearest_entry_simd,
    .octree_count = octree_count_simd,
    .octree_count_rgb = octree_count_rgb_simd,
    .octree_map_rgb = octree_map_rgb_simd,
    .octree_map = octree_map_simd,
};
