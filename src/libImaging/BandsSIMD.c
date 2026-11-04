/* Band operations (Bands.c), compiled into SIMDKernels.c: one band of a
   4-byte image into a 1-byte row, the split of a row into its band planes
   (an output of NULL skips that band) and NEON's merge of planes into
   4-byte pixels (an input of NULL is a band of zeros); each does the pixels
   its vector loop can and returns how many. */

static int
bands_get_band_simd(UINT8 *out, const UINT8 *in, int xsize, int band) {
    int x = 0;
#if defined(IMAGING_SIMD_SSE4)
    __m128i shuffle_mask = _mm_set_epi8(
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        12 + band,
        8 + band,
        4 + band,
        band
    );
#endif
#if defined(IMAGING_SIMD_SSE4)
    for (; x < xsize - 3; x += 4) {
        __m128i source = _mm_loadu_si128((__m128i *)in);
        UINT32 v = _mm_cvtsi128_si32(_mm_shuffle_epi8(source, shuffle_mask));
        memcpy(out + x, &v, sizeof(v));
        in += 16;
    }
#elif defined(IMAGING_SIMD_NEON)
    for (; x < xsize - 15; x += 16) {
        uint8x16x4_t source = vld4q_u8(in);
        vst1q_u8(out + x, source.val[band]);
        in += 64;
    }
#endif
    return x;
}

#if defined(IMAGING_SIMD_SSE4)
#define SPLIT_STEP 4
#define SPLIT_LOAD(in)                                                     \
    __m128i source = _mm_shuffle_epi8(                                     \
        _mm_loadu_si128((__m128i *)(in)),                                  \
        _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0) \
    );
#define SPLIT_STORE(out, band)                                          \
    {                                                                   \
        UINT32 v = _mm_cvtsi128_si32(_mm_srli_si128(source, 4 * band)); \
        memcpy(out, &v, sizeof(v));                                     \
    }
#elif defined(IMAGING_SIMD_NEON)
#define SPLIT_STEP 16
#define SPLIT_LOAD(in) uint8x16x4_t source = vld4q_u8(in);
#define SPLIT_STORE(out, band) vst1q_u8(out, source.val[band]);
#endif

static int
bands_split_simd(
    UINT8 *out0, UINT8 *out1, UINT8 *out2, UINT8 *out3, const UINT8 *in, int xsize
) {
    int x = 0;
    for (; x < xsize - (SPLIT_STEP - 1); x += SPLIT_STEP) {
        SPLIT_LOAD(in);
        if (out0) {
            SPLIT_STORE(out0 + x, 0);
        }
        if (out1) {
            SPLIT_STORE(out1 + x, 1);
        }
        if (out2) {
            SPLIT_STORE(out2 + x, 2);
        }
        if (out3) {
            SPLIT_STORE(out3 + x, 3);
        }
        in += SPLIT_STEP * 4;
    }
    return x;
}
#undef SPLIT_STEP
#undef SPLIT_LOAD
#undef SPLIT_STORE

#if defined(IMAGING_SIMD_NEON)
static int
bands_merge_neon(
    UINT8 *out,
    const UINT8 *in0,
    const UINT8 *in1,
    const UINT8 *in2,
    const UINT8 *in3,
    int xsize
) {
    int x = 0;
    for (; x < xsize - 15; x += 16) {
        uint8x16x4_t px = {
            {vld1q_u8(in0 + x),
             in1 ? vld1q_u8(in1 + x) : vdupq_n_u8(0),
             in2 ? vld1q_u8(in2 + x) : vdupq_n_u8(0),
             in3 ? vld1q_u8(in3 + x) : vdupq_n_u8(0)}
        };
        vst4q_u8(out + 4 * x, px);
    }
    return x;
}
#endif

static const ImagingSIMDBandsKernels bands_simd_kernels = {
    .get_band = bands_get_band_simd,
    .split = bands_split_simd,
#if defined(IMAGING_SIMD_NEON)
    .merge = bands_merge_neon,
#endif
};
