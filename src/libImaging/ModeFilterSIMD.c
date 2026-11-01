/* The mode filter's vector kernel (ModeFilter.c), compiled into
   SIMDKernels.c: the 3x3 window of an 8-bit image, MODE_VEC pixels of one
   row at a time, from its three source rows. For each of the nine window
   bytes, how many of the other eight equal it (the 36 pairs compared once,
   each result subtracted from both counts); then the byte with the highest
   count, the lowest value on a tie, as the scalar scan chooses it: the
   count and the inverted value packed into a 16-bit lane, count high, and
   the maximum over the nine. Two or more equal others (three in all) keep
   the choice, else the centre pixel. The same pixels as ModeFilter.c's
   histogram, which counts the same window. */

#if defined(IMAGING_SIMD_AVX2)
#define MODE_VEC 32
typedef __m256i mode_vec;
#define MODE_LOAD(p) _mm256_loadu_si256((const __m256i *)(p))
#define MODE_STORE(p, v) _mm256_storeu_si256((__m256i *)(p), v)
#define MODE_SET1(c) _mm256_set1_epi8(c)
#define MODE_SET1_16(c) _mm256_set1_epi16(c)
#define MODE_CMPEQ(a, b) _mm256_cmpeq_epi8(a, b)
#define MODE_SUB(a, b) _mm256_sub_epi8(a, b)
#define MODE_XOR(a, b) _mm256_xor_si256(a, b)
#define MODE_AND(a, b) _mm256_and_si256(a, b)
#define MODE_UNPACKLO(a, b) _mm256_unpacklo_epi8(a, b)
#define MODE_UNPACKHI(a, b) _mm256_unpackhi_epi8(a, b)
#define MODE_MAX16(a, b) _mm256_max_epu16(a, b)
#define MODE_SRL16(a, n) _mm256_srli_epi16(a, n)
#define MODE_PACKUS(lo, hi) _mm256_packus_epi16(lo, hi)
#define MODE_CMPGT(a, b) _mm256_cmpgt_epi8(a, b)
#define MODE_BLEND(no, yes, mask) _mm256_blendv_epi8(no, yes, mask)
#elif defined(IMAGING_SIMD_SSE4)
#define MODE_VEC 16
typedef __m128i mode_vec;
#define MODE_LOAD(p) _mm_loadu_si128((const __m128i *)(p))
#define MODE_STORE(p, v) _mm_storeu_si128((__m128i *)(p), v)
#define MODE_SET1(c) _mm_set1_epi8(c)
#define MODE_SET1_16(c) _mm_set1_epi16(c)
#define MODE_CMPEQ(a, b) _mm_cmpeq_epi8(a, b)
#define MODE_SUB(a, b) _mm_sub_epi8(a, b)
#define MODE_XOR(a, b) _mm_xor_si128(a, b)
#define MODE_AND(a, b) _mm_and_si128(a, b)
#define MODE_UNPACKLO(a, b) _mm_unpacklo_epi8(a, b)
#define MODE_UNPACKHI(a, b) _mm_unpackhi_epi8(a, b)
#define MODE_MAX16(a, b) _mm_max_epu16(a, b)
#define MODE_SRL16(a, n) _mm_srli_epi16(a, n)
#define MODE_PACKUS(lo, hi) _mm_packus_epi16(lo, hi)
#define MODE_CMPGT(a, b) _mm_cmpgt_epi8(a, b)
#define MODE_BLEND(no, yes, mask) _mm_blendv_epi8(no, yes, mask)
#elif defined(IMAGING_SIMD_NEON)
#define MODE_VEC 16
typedef uint8x16_t mode_vec;
#define MODE_LOAD(p) vld1q_u8(p)
#define MODE_STORE(p, v) vst1q_u8(p, v)
#define MODE_SET1(c) vdupq_n_u8(c)
#define MODE_CMPEQ(a, b) vceqq_u8(a, b)
#define MODE_SUB(a, b) vsubq_u8(a, b)
#define MODE_XOR(a, b) veorq_u8(a, b)
/* count high, inverted value low, in each 16-bit lane */
#define MODE_UNPACKLO(a, b) vzip1q_u8(a, b)
#define MODE_UNPACKHI(a, b) vzip2q_u8(a, b)
#define MODE_MAX16(a, b) \
    vreinterpretq_u8_u16(vmaxq_u16(vreinterpretq_u16_u8(a), vreinterpretq_u16_u8(b)))
#define MODE_LOWBYTES(lo, hi) vuzp1q_u8(lo, hi)
#define MODE_HIGHBYTES(lo, hi) vuzp2q_u8(lo, hi)
#define MODE_CMPGT(a, b) vcgtq_u8(a, b)
#define MODE_BLEND(no, yes, mask) vbslq_u8(mask, yes, no)
#endif

#if !defined(IMAGING_SIMD_NEON)
/* The low and the high bytes of the 16-bit lanes, as bytes in order. */
#define MODE_LOWBYTES(lo, hi) \
    MODE_PACKUS(MODE_AND(lo, MODE_SET1_16(0xff)), MODE_AND(hi, MODE_SET1_16(0xff)))
#define MODE_HIGHBYTES(lo, hi) MODE_PACKUS(MODE_SRL16(lo, 8), MODE_SRL16(hi, 8))
#endif

static int
mode3_u8_simd(
    UINT8 *out, const UINT8 *above, const UINT8 *row, const UINT8 *below, int xsize
) {
    int x;
    const mode_vec ones = MODE_SET1(1), all = MODE_SET1(-1);
    for (x = 1; x + MODE_VEC <= xsize - 1; x += MODE_VEC) {
        mode_vec p[9], c[9], e, lo, hi, best_lo, best_hi, best, count;
        p[0] = MODE_LOAD(above + x - 1);
        p[1] = MODE_LOAD(above + x);
        p[2] = MODE_LOAD(above + x + 1);
        p[3] = MODE_LOAD(row + x - 1);
        p[4] = MODE_LOAD(row + x);
        p[5] = MODE_LOAD(row + x + 1);
        p[6] = MODE_LOAD(below + x - 1);
        p[7] = MODE_LOAD(below + x);
        p[8] = MODE_LOAD(below + x + 1);
        /* c[i]: how many of the other bytes equal p[i], 0 to 8 */
        c[0] = c[1] = c[2] = c[3] = c[4] = c[5] = c[6] = c[7] = c[8] =
            MODE_SUB(ones, ones);
#define MODE_PAIR(i, j)         \
    e = MODE_CMPEQ(p[i], p[j]); \
    c[i] = MODE_SUB(c[i], e);   \
    c[j] = MODE_SUB(c[j], e);
        MODE_PAIR(0, 1) MODE_PAIR(0, 2) MODE_PAIR(0, 3) MODE_PAIR(0, 4);
        MODE_PAIR(0, 5) MODE_PAIR(0, 6) MODE_PAIR(0, 7) MODE_PAIR(0, 8);
        MODE_PAIR(1, 2) MODE_PAIR(1, 3) MODE_PAIR(1, 4) MODE_PAIR(1, 5);
        MODE_PAIR(1, 6) MODE_PAIR(1, 7) MODE_PAIR(1, 8) MODE_PAIR(2, 3);
        MODE_PAIR(2, 4) MODE_PAIR(2, 5) MODE_PAIR(2, 6) MODE_PAIR(2, 7);
        MODE_PAIR(2, 8) MODE_PAIR(3, 4) MODE_PAIR(3, 5) MODE_PAIR(3, 6);
        MODE_PAIR(3, 7) MODE_PAIR(3, 8) MODE_PAIR(4, 5) MODE_PAIR(4, 6);
        MODE_PAIR(4, 7) MODE_PAIR(4, 8) MODE_PAIR(5, 6) MODE_PAIR(5, 7);
        MODE_PAIR(5, 8) MODE_PAIR(6, 7) MODE_PAIR(6, 8) MODE_PAIR(7, 8);
#undef MODE_PAIR
        /* the key of each byte: count above, inverted value below, so the
           maximum is the highest count and, on a tie, the lowest value */
#define MODE_KEY(i)                                \
    lo = MODE_UNPACKLO(MODE_XOR(p[i], all), c[i]); \
    hi = MODE_UNPACKHI(MODE_XOR(p[i], all), c[i]); \
    best_lo = MODE_MAX16(best_lo, lo);             \
    best_hi = MODE_MAX16(best_hi, hi);
        best_lo = MODE_UNPACKLO(MODE_XOR(p[0], all), c[0]);
        best_hi = MODE_UNPACKHI(MODE_XOR(p[0], all), c[0]);
        MODE_KEY(1) MODE_KEY(2) MODE_KEY(3) MODE_KEY(4);
        MODE_KEY(5) MODE_KEY(6) MODE_KEY(7) MODE_KEY(8);
#undef MODE_KEY
        best = MODE_XOR(MODE_LOWBYTES(best_lo, best_hi), all);
        count = MODE_HIGHBYTES(best_lo, best_hi);
        /* three equal in all, the centre pixel otherwise */
        MODE_STORE(out + x, MODE_BLEND(p[4], best, MODE_CMPGT(count, ones)));
    }
    return x;
}

static const ImagingSIMDModeKernels mode_simd_kernels = {
    .mode3 = mode3_u8_simd,
};
